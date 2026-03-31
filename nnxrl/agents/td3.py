from copy import deepcopy
from typing import Dict, Tuple, Callable, Protocol
import jax
import jax.numpy as jnp
from flax import nnx, struct
from ..utils.normalization import RMS
from ..utils.replaybuffer import Batch, JAXReplayBuffer
from ..utils.network import TanhDetActor, soft_update, EnsembleCritic
from mujoco_playground import MjxEnv, State
from ..utils.evaluate import evaluate_policy
from ..utils.checkpoint import load_states, save_states


class TD3Config(Protocol):
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

    normalize_observation: bool
    exploration_noise: float

    policy_noise: float
    noise_clip: float

@struct.dataclass
class TrainState:
    """Mutable TD3 training state (models + optimizer states)."""

    actor: TanhDetActor
    critic: EnsembleCritic
    target_actor: TanhDetActor
    target_critic: EnsembleCritic
    actor_opt: nnx.Optimizer
    critic_opt: nnx.Optimizer

    rms: RMS | None = None
    grad_updates: int = 0  

    @classmethod
    def create(cls,
    actor,
    critic,
    actor_opt,
    critic_opt,
    rb=None,
    rms=None):
        target_actor = deepcopy(actor)
        target_critic = deepcopy(critic)
        return cls(
            actor=actor,
            critic=critic,
            target_actor=target_actor,
            target_critic=target_critic,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            rms=rms,
            grad_updates=0
        )

    def save(self, path: str):
        save_states(path, {
            "actor" : self.actor,
            "critic": self.critic,
            "actor_opt": self.actor_opt,
            "critic_opt": self.critic_opt,
            "target_critic": self.target_critic,
            "target_actor": self.target_actor,
            "rms": self.rms,
            "grad_updates": self.grad_updates
        })

    def load(self, path: str):
        state_map = load_states(path, {
            "actor" : self.actor,
            "critic": self.critic,
            "actor_opt": self.actor_opt,
            "critic_opt": self.critic_opt,
            "target_critic": self.target_critic,
            "target_actor": self.target_actor,
            "rms": self.rms,
            "grad_updates": self.grad_updates
        })

        self.grad_updates = state_map["grad_updates"]






def _target_policy_smoothing(
    target_actor: TanhDetActor,
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
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Update TD3 critics and return the updated TrainState."""
    target_key, _ = jax.random.split(key)
    smoothed_actions = _target_policy_smoothing(
        train_state.target_actor,
        batch.next_observations,
        key=target_key,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
    )
    target_q = train_state.target_critic(batch.next_observations, smoothed_actions)    # [num_q, B, 1]
    min_q = jnp.min(target_q, axis=0)                                                  # [B, 1]
    target_values = batch.rewards + config.gamma * (1.0 - batch.dones) * min_q
    target_values = jax.lax.stop_gradient(target_values)

    def critic_loss_fn(critic):
        q = critic(batch.observations, batch.actions)
        # [num_q, B, 1] - [B, 1]
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
    config: TD3Config
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Update TD3 actor and return the updated TrainState.
    """
    def actor_loss_fn(actor, critic):
        actions = actor.get_action(batch.observations)
        q = critic(batch.observations, actions)[0]
        actor_loss = -jnp.mean(q)
        return actor_loss, {"training/actor_loss": actor_loss}

    (_loss, info), grads = nnx.value_and_grad(actor_loss_fn, has_aux=True, argnums=0)(train_state.actor, train_state.critic)
    train_state.actor_opt.update(grads)

    soft_update(train_state.actor, train_state.target_actor, config.tau)
    soft_update(train_state.critic, train_state.target_critic, config.tau)

    return train_state, info




def update_td3(
    train_state: TrainState,
    config: TD3Config,
    key: jax.Array,
    big_batch: Batch     
) -> tuple[TrainState, dict[str, jax.Array]]:
    """(multiple SGD steps per env step)."""

    if train_state.rms is not None and config.normalize_observation:
        stacked_obs = jnp.concatenate(
            [big_batch.observations, big_batch.next_observations],
            axis=0,
        )
        normalized_obs, rms = train_state.rms.normalize(stacked_obs, update=True)
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

    @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
    def update_td3_mininbatch(ts, batch, step):
        ts, critic_info = update_critic(ts, config, batch, jax.random.fold_in(key, step))
        ts = ts.replace(grad_updates=ts.grad_updates + 1)


        ts, policy_info = nnx.cond(
            train_state.grad_updates % config.policy_frequency == 0, 
            lambda ts: update_actor(ts, batch, config), 
            lambda ts: (ts, {"training/actor_loss": jnp.array(0.0)}),
            ts)
        return ts, {**critic_info, **policy_info}

    ts, infos = update_td3_mininbatch(train_state, batches, jnp.arange(config.grad_step_per_env_step))

    avg_info = jax.tree.map(jnp.mean, infos)

    return ts, avg_info



def sample_and_update_td3(
    train_state: TrainState,
    config: TD3Config,
    key: jax.Array,
    rb: JAXReplayBuffer
) -> Tuple[TrainState, Dict[str, jax.Array]]:
    """(multiple SGD steps per env step)."""
    sample_key, update_key = jax.random.split(key, 2)

    # 1) sample one big batch, then reshape to (grad_step_per_env_step, batch_size, ...)
    big_batch = rb.sample(
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
