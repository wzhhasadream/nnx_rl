from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
import flax.struct as struct
from mujoco_playground import MjxEnv
from typing import Any, Callable
from .network import Actor, Alpha
from ...utils.network import DoubleCritic, soft_update
from ...utils.replaybuffer import Batch, sample, add, ReplayBufferState
from ...utils.normalization import RMSState, rms_normalize, rms_update
from ...utils.evaluate import evaluate_policy


SACConfig = Any

@struct.dataclass
class TrainState:
    actor: Actor
    critic: DoubleCritic
    actor_opt: nnx.Optimizer
    critic_opt: nnx.Optimizer
    target_critic: DoubleCritic

    
    rb: ReplayBufferState
    rms: RMSState | dict[str, RMSState] | None = None
    grad_updates: int = 0  # number of critic gradient updates  
    alpha: Alpha | None = None
    alpha_opt: nnx.Optimizer | None = None




def update_critic(
    train_state: TrainState,
    config: SACConfig,
    batch: Batch,
    key: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Update critic parameters and return the updated TrainState."""

    def critic_loss_fn(critic_model: DoubleCritic):
        # Target: r + gamma * (min(Q') - alpha * log pi(a'|s'))
        next_actions, next_log_pi = train_state.actor.get_action(batch.next_observations, key)
        min_next_q = train_state.target_critic(batch.next_observations, next_actions)

        alpha_value = train_state.alpha() if config.autotune else config.alpha
        target_q = batch.rewards + (1.0 - batch.dones) * config.gamma * (
            min_next_q - alpha_value * next_log_pi
        )
        target_q = jax.lax.stop_gradient(target_q)

        q1 = critic_model.critic1(batch.observations, batch.actions)
        q2 = critic_model.critic2(batch.observations, batch.actions)

        q1_loss = jnp.mean((q1 - target_q) ** 2)
        q2_loss = jnp.mean((q2 - target_q) ** 2)
        critic_loss = q1_loss + q2_loss

        info = {
            "training/q1_mean": jnp.mean(q1),
            "training/q2_mean": jnp.mean(q2),
            "training/q1_loss": q1_loss,
            "training/q2_loss": q2_loss,
        }
        return critic_loss, info

    (_loss, info), grads = nnx.value_and_grad(critic_loss_fn, has_aux=True)(train_state.critic)
    train_state.critic_opt.update(grads)
    return train_state, info


def update_actor(
    train_state: TrainState,
    config: SACConfig,
    batch: Batch,
    key: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Update actor parameters and return the updated TrainState."""

    def actor_loss_fn(actor_model: Actor):
        actions, log_pi = actor_model.get_action(batch.observations, key)
        min_q = train_state.critic(batch.observations, actions)
        alpha_value = train_state.alpha() if config.autotune else config.alpha
        actor_loss = -jnp.mean(min_q - alpha_value * log_pi)
        return actor_loss, {"training/actor_loss": actor_loss}

    (_loss, info), grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(train_state.actor)
    train_state.actor_opt.update(grads)
    return train_state, info


def update_alpha(
    train_state: TrainState,
    config: SACConfig,
    batch: Batch,
    key: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Update entropy temperature (alpha) and return the updated TrainState."""

    def alpha_loss_fn(alpha_model: Alpha):
        _, log_pi = train_state.actor.get_action(batch.observations, key)
        alpha_loss = (-alpha_model() * (log_pi + config.target_entropy)).mean()
        return alpha_loss, {"training/alpha_loss": alpha_loss, "training/alpha_value": alpha_model()}

    (_loss, info), grads = nnx.value_and_grad(alpha_loss_fn, has_aux=True)(train_state.alpha)
    train_state.alpha_opt.update(grads)
    return train_state, info


def update_policy(
    train_state: TrainState,
    config: SACConfig,
    batch: Batch,
    key: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Update actor (and optionally alpha) once."""
    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def _single_update(train_state, key):
        actor_key, alpha_key = jax.random.split(key)
        train_state, actor_info = update_actor(
            train_state, config, batch, actor_key)
        if config.autotune and train_state.alpha is not None:
            train_state, alpha_info = update_alpha(
                train_state, config, batch, alpha_key)
        else:
            alpha_info = {
                "training/alpha_loss": jnp.array(0.0, dtype=jnp.float32),
                "training/alpha_value": jnp.array(config.alpha, dtype=jnp.float32),
            }
        return train_state, {**actor_info, **alpha_info}

    updated_train_state, infos = _single_update(
        train_state, jax.random.split(key, config.policy_frequency))
    info = jax.tree.map(jnp.mean, infos)
    return updated_train_state, info


def update_target_networks(train_state: TrainState, config: SACConfig) -> TrainState:
    """Soft-update target critic parameters and return the updated TrainState."""
    soft_update(train_state.critic, train_state.target_critic, config.tau)
    return train_state


def update_sac(train_state: TrainState, config: SACConfig, key: jax.Array, big_batch: Batch):
    """(multiple SGD steps per env step)."""

    if train_state.rms is not None and config.normalize_observation:
        rms = rms_update(train_state.rms, big_batch.observations)
        rms = rms_update(rms, big_batch.next_observations)
        obs_for_policy, _ = rms_normalize(big_batch.observations, rms, update=False)
        next_obs_for_policy, _ = rms_normalize(big_batch.next_observations, rms, update=False)
        train_state = train_state.replace(rms=rms)
        big_batch = big_batch._replace(
            observations=obs_for_policy,
            next_observations=next_obs_for_policy,
        )

    batches = jax.tree.map(
        lambda x: x.reshape(
            config.grad_step_per_env_step, config.batch_size, *x.shape[1:]),
        big_batch,
    )

    update_keys = jax.random.split(key, config.grad_step_per_env_step)

    @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
    def update_sac_minibatch(train_state, sub_batch: Batch, key: jax.Array):
        critic_key, policy_key = jax.random.split(key, 2)
        train_state, critic_info = update_critic(train_state, config, sub_batch, critic_key)
        train_state = train_state.replace(grad_updates=train_state.grad_updates + 1)
        # keep info structure fixed
        zero = jnp.array(0.0, dtype=jnp.float32)
        default_alpha_value = train_state.alpha() if config.autotune else jnp.array(config.alpha, dtype=jnp.float32)
        
        policy_info_template = {
            "training/actor_loss": zero,
            "training/alpha_loss": zero,
            "training/alpha_value": default_alpha_value,
        }

        def _do_policy_update(args):
            train_state, policy_keys = args
            train_state, policy_infos = update_policy(train_state, config, sub_batch, policy_keys)
            return train_state, policy_infos

        def _skip_policy_update(args):
            train_state, policy_keys = args
            return train_state, policy_info_template

        train_state, policy_info = nnx.cond(
            train_state.grad_updates % config.policy_frequency == 0,
            _do_policy_update,
            _skip_policy_update,
            (train_state, policy_key),
        )
        train_state = nnx.cond(
            train_state.grad_updates % config.target_frequency == 0,
            lambda ts: update_target_networks(ts, config),
            lambda ts: ts,
            train_state,
        )

        info = {**critic_info, **policy_info}
        return train_state, info

    updated_train_state, infos = update_sac_minibatch(
        train_state, batches, update_keys)
    avg_info = jax.tree.map(jnp.mean, infos)
    return updated_train_state, avg_info


def sample_and_update_sac(
    train_state: TrainState,
    config: SACConfig,
    key: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """(multiple SGD steps per env step)."""
    assert train_state.rb is not None, "Replay buffer state must be initialized"
    sample_key, update_key = jax.random.split(key, 2)

    # 1) sample one big batch, then reshape to (grad_step_per_env_step, batch_size, ...)
    big_batch = sample(
        train_state.rb,
        sample_key,
        config.batch_size * config.grad_step_per_env_step
    )
    ts, info = update_sac(train_state, config, update_key, big_batch)

    return ts, info

#####################################################################
################## JAX ENV ##########################################
#####################################################################




def make_train(envs: MjxEnv, config: SACConfig, train_log_fn: Callable):
    def train(train_state: TrainState):
        key = jax.random.PRNGKey(config.seed)
        env_key, train_key = jax.random.split(key, 2)
        state = envs.reset(jax.random.split(env_key, config.num_envs))
        num_env_step = config.total_timesteps // config.num_envs
        eval_indices = jnp.floor(
            jnp.linspace(1, num_env_step, config.num_evals)
        ).astype(jnp.int32)

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry))
        def step(
            carry, step_idx
        ):
            state, train_state = carry
            if config.normalize_observation and train_state.rms is not None:
                obs_for_policy, new_rms = rms_normalize(state.obs, train_state.rms)
                train_state = train_state.replace(rms=new_rms)
            else:
                obs_for_policy = state.obs
            action_key, update_key, eval_key = jax.random.split(jax.random.fold_in(train_key, step_idx), 3)
            actions = nnx.cond(
                step_idx * config.num_envs < config.learning_starts,
                lambda ts: jax.random.uniform(
                    action_key,
                    (config.num_envs, envs.action_size),
                    minval=ts.actor.action_low,
                    maxval=ts.actor.action_high,
                ),
                lambda ts: ts.actor.get_action(obs_for_policy, action_key)[0],
                train_state
            )
            
            next_state = envs.step(state, actions)
            terminated = jnp.logical_and(
                next_state.done.astype(bool),
                jnp.logical_not(next_state.info["truncation"].astype(bool)),
            )
            buffer_state = add(
                train_state.rb,
                state.obs,
                actions,
                next_state.reward,
                next_state.info["true_obs"],
                terminated
            )
            train_state = train_state.replace(rb=buffer_state)

            should_log = jnp.any(eval_indices == step_idx)

            def _train_sac(train_state, key):
                ts, info = sample_and_update_sac(train_state, config, key)
                nnx.cond(
                    should_log,
                    lambda: jax.debug.callback(
                        train_log_fn, info, step_idx * config.num_envs),
                    lambda: None
                )
                return ts

            train_state = nnx.cond(
                step_idx * config.num_envs >= config.learning_starts,
                lambda args: _train_sac(*args),          
                lambda args: args[0],        
                (train_state, update_key),
            )

            def _eval_and_log():
                eval_metric = evaluate_policy(envs, lambda obs, key: train_state.actor.get_action(obs, key)[0], eval_key, train_state.rms)
                jax.debug.callback(train_log_fn, eval_metric, step_idx * config.num_envs)
                return None


            nnx.cond(
                should_log,
                _eval_and_log,
                lambda : None
            )




            return next_state, train_state

        _, final_train_state = step((state, train_state), jnp.arange(1, num_env_step + 1))

        return final_train_state
    return train
