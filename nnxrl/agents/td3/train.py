from __future__ import annotations

from typing import Dict, Tuple, Callable, Protocol
import jax
import jax.numpy as jnp
from flax import nnx, struct
from ...utils.normalization import RMSState, rms_normalize, rms_update
from ...utils.replaybuffer import Batch, ReplayBufferState, add, sample
from .network import Actor
from ...utils.network import DoubleCritic, soft_update
from mujoco_playground import MjxEnv, State
from ...utils.evaluate import evaluate_policy


class TD3Config(Protocol):
    # This protocol describes the minimal config interface required by TD3 training code.
    # Any object with these attributes (e.g., a dataclass from a script) can be used.
    seed: int
    total_timesteps: int
    num_envs: int
    learning_starts: int
    num_evals: int

    gamma: float
    tau: float

    batch_size: int
    grad_step_per_env_step: int
    policy_frequency: int

    policy_lr: float
    actor_noise: float | None

    normalize_observation: bool
    exploration_noise: float

    policy_noise: float
    noise_clip: float

@struct.dataclass
class TrainState:
    """Mutable TD3 training state (models + optimizer states)."""

    actor: Actor
    critic: DoubleCritic
    target_actor: Actor
    target_critic: DoubleCritic
    actor_opt: nnx.Optimizer
    critic_opt: nnx.Optimizer

    rb: ReplayBufferState = None
    rms: RMSState | dict[str, RMSState] | None = None
    grad_updates: int = 0  # number of critic gradient updates  







def _add_gaussian_noise_to_grads(grads, *, key: jax.Array, std: float):
    """Adds N(0, std^2) noise to every array leaf in a gradient pytree."""
    std = float(std)
    if std <= 0.0:
        return grads

    leaves, treedef = jax.tree_util.tree_flatten(grads)
    keys = jax.random.split(key, len(leaves)) if leaves else ()
    noisy_leaves = []
    for leaf, k in zip(leaves, keys):
        if isinstance(leaf, jax.Array):
            noise = jax.random.normal(k, shape=leaf.shape, dtype=leaf.dtype) * std
            noisy_leaves.append(leaf + noise)
        else:
            noisy_leaves.append(leaf)
    return jax.tree_util.tree_unflatten(treedef, noisy_leaves)




def _target_policy_smoothing(
    target_actor: Actor,
    next_observations: jax.Array,
    *,
    key: jax.Array,
    policy_noise: float,
    noise_clip: float,
) -> jax.Array:
    """Compute smoothed target actions for TD3 critic targets.

    TD3 target policy smoothing:
        a' = clip(pi'(s') + clip(eps, -c, c) * action_scale, action_low, action_high)
    where eps ~ Normal(0, policy_noise).
    """
    target_actions = target_actor.get_action(next_observations)
    noise = jax.random.normal(key, target_actions.shape) * policy_noise
    clipped_noise = jnp.clip(noise, -noise_clip, noise_clip) * target_actor.action_scale
    a = jnp.clip(target_actions + clipped_noise, target_actor.action_low, target_actor.action_high)
    return a


def update_critic(
    train_state: TrainState,
    config: TD3Config,
    batch: Batch,
    key: jax.Array,
) -> Tuple[TrainState, Dict[str, jax.Array]]:
    """Update TD3 critics and return the updated TrainState."""
    target_key, _ = jax.random.split(key)
    smoothed_actions = _target_policy_smoothing(
        train_state.target_actor,
        batch.next_observations,
        key=target_key,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
    )
    target_q = train_state.target_critic(batch.next_observations, smoothed_actions)
    min_q = jnp.min(target_q, axis=1)
    target_values = batch.rewards + config.gamma * (1.0 - batch.dones) * min_q
    target_values = jax.lax.stop_gradient(target_values)[:, None, :]

    def critic_loss_fn(critic: DoubleCritic):
        q = critic(batch.observations, batch.actions)
        q_loss = jnp.mean((q - target_values) ** 2)

        info = {
            "training/critic_loss": q_loss,
            "training/q_mean": jnp.mean(q),
        }
        return q_loss, info

    (_loss, info), grads = nnx.value_and_grad(
        critic_loss_fn, has_aux=True
    )(train_state.critic)
    train_state.critic_opt.update(grads)
    return train_state, info





def update_actor(
    train_state: TrainState,
    batch: Batch,
) -> Tuple[TrainState, Dict[str, jax.Array]]:
    """Update TD3 actor and return the updated TrainState.
    """
    def actor_loss_fn(actor: Actor, critic: DoubleCritic):
        actions = actor.get_action(batch.observations)
        q = critic(batch.observations, actions)[:, 0, :]
        actor_loss = -jnp.mean(q)
        return actor_loss, {"training/actor_loss": actor_loss}

    (_loss, info), grads = nnx.value_and_grad(actor_loss_fn, has_aux=True, argnums=0)(train_state.actor, train_state.critic)
    train_state.actor_opt.update(grads)

    return train_state, info


def update_targets(train_state: TrainState, tau: float) -> TrainState:
    """Soft-update target actor and critic networks."""
    soft_update(train_state.actor, train_state.target_actor, tau)
    soft_update(train_state.critic, train_state.target_critic, tau)
    return train_state



def update_td3(
    train_state: TrainState,
    config: TD3Config,
    key: jax.Array,
    big_batch: Batch     
) -> Tuple[TrainState, Dict[str, jax.Array]]:
    """(multiple SGD steps per env step)."""

    if train_state.rms is not None and config.normalize_observation:
        stacked_obs = jnp.concatenate(
            [big_batch.observations, big_batch.next_observations],
            axis=0,
        )
        normalized_obs, rms = rms_normalize(stacked_obs, train_state.rms, update=True)
        train_state = train_state.replace(rms=rms)

        batch_size = big_batch.observations.shape[0]
        big_batch = big_batch._replace(
            observations=normalized_obs[:batch_size],
            next_observations=normalized_obs[batch_size:],
        )

    batches = jax.tree.map(
        lambda x: x.reshape(
            config.grad_step_per_env_step, config.batch_size, *x.shape[1:]),
        big_batch,
    )

    update_keys = jax.random.split(key, config.grad_step_per_env_step)
    @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
    def update_td3_mininbatch(train_state, batch, key):
        train_state, critic_info = update_critic(train_state, config, batch, key)
        train_state = train_state.replace(grad_updates=train_state.grad_updates + 1)

        zero = jnp.array(0.0, dtype=jnp.float32)
        policy_info_template = {"training/actor_loss": zero}

        def _do_policy_update(ts: TrainState):
            ts, actor_info = update_actor(
                ts,
                batch
            )
            ts = update_targets(ts, config.tau)
            return ts, actor_info

        def _skip_policy_update(ts: TrainState):
            return ts, policy_info_template

        train_state, policy_info = nnx.cond(train_state.grad_updates % config.policy_frequency == 0, _do_policy_update, _skip_policy_update, train_state)
        return train_state, {**critic_info, **policy_info}

    ts, infos = update_td3_mininbatch(train_state, batches, update_keys)

    avg_info = jax.tree.map(jnp.mean, infos)

    return ts, avg_info



def sample_and_update_td3(
    train_state: TrainState,
    config: TD3Config,
    key: jax.Array,
) -> Tuple[TrainState, Dict[str, jax.Array]]:
    """(multiple SGD steps per env step)."""
    assert train_state.rb is not None, "Replay buffer state must be initialized"
    sample_key, update_key = jax.random.split(key, 2)

    # 1) sample one big batch, then reshape to (grad_step_per_env_step, batch_size, ...)
    big_batch = sample(
        train_state.rb,
        sample_key,
        config.batch_size * config.grad_step_per_env_step
    )
    ts, info = update_td3(train_state, config, update_key, big_batch)

    return ts, info

#####################################################################
################## JAX ENV ##########################################
#####################################################################

    

def env_step(train_state: TrainState, config: TD3Config, state: State, envs: MjxEnv, step_idx: jax.Array, key: jax.Array):
    if config.normalize_observation and train_state.rms is not None:
        obs_for_policy, new_rms = rms_normalize(state.obs, train_state.rms)
        train_state = train_state.replace(rms=new_rms)
    else:
        obs_for_policy = state.obs

    def _action_with_exploration_noise(ts: TrainState):
        actions = ts.actor.get_action(obs_for_policy)
        noise = (
            jax.random.normal(key, shape=actions.shape)
            * config.exploration_noise
            * ts.actor.action_scale
        )
        
        return jnp.clip(noise + actions, min=ts.actor.action_low, max=ts.actor.action_high)

    actions = nnx.cond(
        step_idx * config.num_envs < config.learning_starts,
        lambda _: jax.random.uniform(
            key,
            shape=(config.num_envs, envs.action_size),
            minval=train_state.actor.action_low,
            maxval=train_state.actor.action_high,
        ),
        lambda ts: _action_with_exploration_noise(ts),
        train_state,
    )

    next_state = envs.step(state, actions)
    terminated = jnp.logical_and(
        next_state.done.astype(bool),
        jnp.logical_not(next_state.info["truncation"].astype(bool)),
    )
    new_rb = add(
        train_state.rb,
        state.obs,
        actions,
        next_state.reward,
        next_state.info['true_obs'],
        terminated
    )
    train_state = train_state.replace(rb=new_rb)

    return train_state, next_state





def make_train(
    envs: MjxEnv,
    config: TD3Config,
    train_log_fn: Callable,
):
    def train(train_state: TrainState):
        key = jax.random.PRNGKey(config.seed)
        env_key, update_key = jax.random.split(key, 2)
        init_state = envs.reset(jax.random.split(env_key, config.num_envs))

        num_env_step = config.total_timesteps // config.num_envs
        eval_indices = jnp.floor(jnp.linspace(1, num_env_step, config.num_evals)).astype(jnp.int32)

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry))
        def step(carry, env_step_idx):
            state, train_state = carry
            step_key = jax.random.fold_in(update_key, env_step_idx)
            step_key, train_key, eval_key = jax.random.split(step_key, 3)
            train_state, next_state = env_step(
                train_state, config, state, envs, env_step_idx, step_key
            )

            should_log = jnp.any(eval_indices == env_step_idx)

            def _train_td3(args):
                ts, key = args
                ts, info = sample_and_update_td3(ts, config, key)
                nnx.cond(
                    should_log,
                    lambda: jax.debug.callback(train_log_fn, info, env_step_idx * config.num_envs),
                    lambda: None,
                )
                return ts

            train_state = nnx.cond(
                env_step_idx * config.num_envs >= config.learning_starts,
                _train_td3,
                lambda args: args[0],
                (train_state, train_key),
            )

            def _eval_and_log():
                eval_metrics = evaluate_policy(envs, lambda obs, key: train_state.actor.get_action(obs), eval_key, train_state.rms)
                jax.debug.callback(train_log_fn, eval_metrics, env_step_idx * config.num_envs)
                return None

            nnx.cond(should_log, _eval_and_log, lambda : None)

            return next_state, train_state

        _, final_train_state = step(
            (init_state, train_state),
            jnp.arange(1, num_env_step + 1, dtype=jnp.int32)
        )
        return final_train_state

    return train
