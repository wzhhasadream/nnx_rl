from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx, struct
from typing import Callable, NamedTuple, Protocol
from .network import ActorCritic
from mujoco_playground._src import mjx_env
from ...utils.normalization import RMSState, rms_normalize
from ...utils.evaluate import evaluate_policy

class PPOConfig(Protocol):
    # This protocol describes the minimal config interface required by PPO training code.
    # Any object with these attributes (e.g., a dataclass from a script) can be used.
    seed: int
    total_timesteps: int
    num_envs: int
    rollout_steps: int

    gamma: float
    gae_lambda: float
    reward_scale: float

    clip_coef: float
    entropy_coef: float
    value_coef: float
    lr: float
    max_grad_norm: float | None

    normalize_observation: bool
    normalize_advantage: bool
    clip_value: bool
    target_kl: float | None

    num_minibatches: int
    update_epochs: int
    num_evals: int
    eval_step: int


@struct.dataclass
class TrainState:
    agent: ActorCritic
    opt: nnx.Optimizer
    rms: RMSState | dict[str, RMSState] | None = None


class Trajectory(NamedTuple):
                                                    # (num_steps, num_envs, obs_dim)
    observations: jax.Array | dict[str, jax.Array]                
                                                    # (num_steps, num_envs, action_dim) or (num_steps, num_envs)
    actions: jax.Array
    log_probs: jax.Array                            # (num_steps, num_envs)
    values: jax.Array                               # (num_steps + 1, num_envs)
    dones: jax.Array                                # (num_steps + 1, num_envs)
    rewards: jax.Array                              # (num_steps, num_envs)


def ppo_loss(
    agent: ActorCritic,
    batch: dict[str, jax.Array],
    config: PPOConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    _, new_log_probs, values, entropy = agent.get_action_and_value(
        batch["observations"], actions=batch["actions"])

    log_ratio = new_log_probs - batch["log_probs"]
    ratio = jnp.exp(log_ratio)

    pg_loss_unclipped = ratio * batch["advantages"]
    pg_loss_clipped = jnp.clip(
        ratio, 1 - config.clip_coef, 1 + config.clip_coef
    ) * batch["advantages"]
    pg_loss = -jnp.mean(jnp.minimum(pg_loss_unclipped, pg_loss_clipped))

    
    if config.clip_value:
        value_pred_clipped = batch["old_values"] + (
            values - batch["old_values"]
                    ).clip(-config.clip_coef, config.clip_coef)
        value_loss_unclipped = (values - batch["returns"]) ** 2
        value_loss_clipped = (value_pred_clipped - batch["returns"]) ** 2

        value_loss = 0.5 * jnp.maximum(value_loss_clipped, value_loss_unclipped).mean()
    else:
        value_loss = 0.5 * jnp.mean((values - batch["returns"]) ** 2)
        
    entropy_loss = jnp.mean(entropy)

    loss = pg_loss + config.value_coef * \
        value_loss - config.entropy_coef * entropy_loss

    approx_kl = ((ratio - 1) - log_ratio).mean()  # Approximate KL divergence
    clipfracs = (jnp.abs(ratio - 1.0) > config.clip_coef).mean()

    return loss, {
        "training/loss": loss,
        "training/pg_loss": pg_loss,
        "training/value_loss": value_loss,
        "training/entropy_loss": entropy_loss,
        "training/approx_kl": approx_kl,
        "training/clipfracs": clipfracs
    }


def ppo_update_minibatch(
    train_state: TrainState,
    batch: dict[str, jax.Array],
    config: PPOConfig
) -> tuple[TrainState, dict[str, jax.Array]]:
    if config.normalize_advantage:
        batch["advantages"] = (batch["advantages"] - jnp.mean(batch["advantages"])) / (
            jnp.std(batch["advantages"]) + 1e-8
        )

    def loss_fn(agent: ActorCritic):
        return ppo_loss(agent, batch, config)
    (_loss, info), grads = nnx.value_and_grad(
        loss_fn, has_aux=True)(train_state.agent)
    
    train_state.opt.update(grads)
    return train_state, info


def compute_gae(
    rewards: jax.Array,     # (rollout_step, num_envs)
    values: jax.Array,      # (rollout_step + 1, num_envs)
    dones: jax.Array,       # (rollout_step + 1, num_envs)
    gamma: float,
    gae_lambda: float
) -> tuple[jax.Array, jax.Array]:
    """Compute GAE-Lambda advantages for a rollout.

    Time indexing convention (important):
      - Let s_t be the observation at time t, and a_t the action taken at time t.
      - The environment transition is: s_t --a_t--> s_{t+1}, producing reward r_t.

    Tensor alignment used here:
      - rewards[t, e]    = r_t for env e (reward observed after stepping from s_t to s_{t+1})
      - values[t, e]     = V(s_t) for env e
      - values[t+1, e]   = V(s_{t+1}) for env e
      - dones[t+1, e]    = done_{t+1} for env e (whether s_{t+1} is terminal)

    We use dones[t+1] (not dones[t]) because the bootstrap term V(s_{t+1})
    should be masked out when the *next* state is terminal:
      mask_t = 1 - done_{t+1}
      delta_t = r_t + gamma * V(s_{t+1}) * mask_t - V(s_t)
      gae_t   = delta_t + gamma * lambda * mask_t * gae_{t+1}
    """

    num_envs = rewards.shape[1]

    def gae_step(carry, inputs):
        gae = carry
        reward_t, value_t, value_tp1, done_tp1 = inputs

        mask = 1.0 - done_tp1
        delta = reward_t + gamma * value_tp1 * mask - value_t
        gae = delta + gamma * gae_lambda * mask * gae

        return gae, gae

    init_gae = jnp.zeros(num_envs)

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0), reverse=True)
    def scan_gae(carry, inputs):
        return gae_step(carry, inputs)

    _, advantages = scan_gae(
        init_gae, (rewards, values[:-1], values[1:], dones[1:]))

    returns = advantages + values[:-1]

    return advantages, returns


def ppo_update_rollout(
    train_state: TrainState,
    trajectory: Trajectory,
    advantages: jax.Array,
    returns: jax.Array,
    config: PPOConfig,
    rng: jax.Array
):
    batch_size = trajectory.actions.shape[0] * \
        trajectory.actions.shape[1]

    def _flatten_time_env(x: jax.Array) -> jax.Array:
        """Flattens (T, N, ...) -> (T*N, ...) without collapsing feature dimensions.
        """
        if x.ndim >= 3:
            return x.reshape((batch_size,) + x.shape[2:])
        return x.reshape((batch_size,))

    flat_data = jax.tree.map(
        _flatten_time_env,
        {
            "observations": trajectory.observations,
            "actions": trajectory.actions,
            "advantages": advantages,
            "returns": returns,
            "log_probs": trajectory.log_probs,
            "old_values": trajectory.values[:-1]
        },
    )

    minibatch_size = batch_size // config.num_minibatches

    # Ensure we can reshape into (num_minibatches, minibatch_size) without remainder.
    # If there is a remainder, drop the last few samples for this update.
    effective_batch_size = minibatch_size * config.num_minibatches
    if effective_batch_size != batch_size:
        flat_data = jax.tree.map(lambda x: x[:effective_batch_size], flat_data)
        batch_size = effective_batch_size

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def update_epoch(train_state, epoch_idx):
        epoch_key = jax.random.fold_in(rng, epoch_idx)
        shuff_idx = jax.random.permutation(epoch_key, batch_size)
        shuff_idx = shuff_idx.reshape(config.num_minibatches, minibatch_size)


        def update_minibatch(loop_state):
            ts, mb_idx, info = loop_state
            idx = shuff_idx[mb_idx]
            batch = jax.tree.map(lambda x: x[idx], flat_data)
            ts, info = ppo_update_minibatch(ts, batch, config)
            return (ts, mb_idx + 1, info)

        zero = jnp.array(0.0, dtype=jnp.float32)
        init_info = {
            "training/loss": zero,
            "training/pg_loss": zero,
            "training/value_loss": zero,
            "training/entropy_loss": zero,
            "training/approx_kl": zero,
            "training/clipfracs": zero,
        }

        def cond(loop_state):
            _ts, mb_idx, info = loop_state
            within_epoch = mb_idx < config.num_minibatches
            if config.target_kl is None:
                return within_epoch
            else:
                kl_ok = info["training/approx_kl"] <= config.target_kl
                return within_epoch & kl_ok

        ts_out, _mb_idx_out, info_out = nnx.while_loop(
            cond,
            update_minibatch,
            (train_state, jnp.array(0, dtype=jnp.int32), init_info),
        )

        return ts_out, info_out

    updated_train_state, all_infos = update_epoch(
        train_state, jnp.arange(config.update_epochs))

    metrics = jax.tree.map(jnp.mean, all_infos)

    return updated_train_state, metrics


def ppo_update_from_trajectory(
    train_state: TrainState,
    trajectory: Trajectory,
    config: PPOConfig,
    rng: jax.Array
) -> tuple[TrainState, dict[str, jax.Array]]:
    trajectory = trajectory._replace(rewards = config.reward_scale * trajectory.rewards)
    advantages, returns = compute_gae(
        trajectory.rewards,
        trajectory.values,
        trajectory.dones,
        config.gamma,
        config.gae_lambda
    )
    train_state, avg_metrics = ppo_update_rollout(
        train_state,
        trajectory,
        advantages,
        returns,
        config,
        rng
    )

    return train_state, avg_metrics


#####################################################################
################## JAX ENV ##########################################
#####################################################################


def collect_rollout(
    train_state: TrainState,
    envs: mjx_env.MjxEnv,
    init_state: mjx_env.State,
    key: jax.Array,
    config: PPOConfig,
):


    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def collect_step(carry, step_idx):
        state, train_state = carry
        observations = state.obs
        if config.normalize_observation and train_state.rms is not None:
            obs_for_policy, rms = rms_normalize(
                observations, train_state.rms
            )
            train_state = train_state.replace(rms=rms)
        else:
            obs_for_policy = observations

        actions, log_probs, values, _ = train_state.agent.get_action_and_value(
            obs_for_policy, key=jax.random.fold_in(key, step_idx)
        )

        next_state = envs.step(state, actions)

        return (next_state, train_state), (
            obs_for_policy,
            actions,
            log_probs.reshape(config.num_envs),
            values.reshape(config.num_envs),
            next_state.reward.reshape(config.num_envs),
            state.done.reshape(config.num_envs),
        )

    (final_state, train_state), (
        stacked_obs,
        stacked_actions,
        stacked_log_probs,
        stacked_values,
        stacked_rewards,
        stacked_dones,
    ) = collect_step((init_state, train_state), jnp.arange(config.rollout_steps))

    if config.normalize_observation and train_state.rms is not None:
        obs_for_value, _ = rms_normalize(
            final_state.obs, train_state.rms, update=False)
    else:
        obs_for_value = final_state.obs
    final_value = train_state.agent.get_value(obs_for_value).reshape(config.num_envs)   # (num_envs, )
    final_done = final_state.done.reshape(config.num_envs)
    stacked_values = jnp.concatenate(
        [stacked_values, final_value[None, :]], axis=0)
    stacked_dones = jnp.concatenate(
        [stacked_dones, final_done[None, :]], axis=0)



    return (final_state, train_state), Trajectory(
        stacked_obs,
        stacked_actions,
        stacked_log_probs,
        stacked_values,
        stacked_dones,
        stacked_rewards,
    )


def make_train(
        envs: mjx_env.MjxEnv,
        config: PPOConfig,
        train_log_fn: Callable):
    def train(init_train_state: TrainState) -> TrainState:
        key = jax.random.PRNGKey(config.seed)
        env_key, key = jax.random.split(key, 2)
        jit_reset = jax.jit(envs.reset)
        init_state = jit_reset(jax.random.split(env_key, config.num_envs))
        num_rollout = config.total_timesteps // (
            config.num_envs * config.rollout_steps)
        rollout_indices = jnp.arange(1, num_rollout + 1, dtype=jnp.int32)

        eval_indices = jnp.floor(
            jnp.linspace(1, num_rollout, config.num_evals)
        ).astype(jnp.int32)

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry))
        def rollout_and_update(carry, rollout_idx):
            rollout_key = jax.random.fold_in(key, rollout_idx)
            rollout_rng, update_rng = jax.random.split(rollout_key, 2)
            state, train_state = carry

            (next_state, train_state), trajectory = collect_rollout(

                train_state,
                envs,
                state,
                rollout_rng,
                config=config
            )

            train_state, train_metrics = ppo_update_from_trajectory(
                train_state,
                trajectory,
                config,
                update_rng
            )

            should_eval = jnp.any(eval_indices == rollout_idx)

            def _eval_and_log():
                eval_rng = jax.random.fold_in(rollout_rng, rollout_idx)
                eval_metrics = evaluate_policy(
                    envs,
                    lambda obs, key: train_state.agent(obs).sample(seed=key),
                    eval_rng,
                    train_state.rms,
                    num_steps=config.eval_step
                )
                merged = {**train_metrics, **eval_metrics}
                jax.debug.callback(
                    train_log_fn, merged, rollout_idx * config.num_envs * config.rollout_steps)
                return None

            nnx.cond(should_eval, _eval_and_log, lambda: None)
            return (next_state, train_state)

        final_state, final_train_state = rollout_and_update(
            (init_state, init_train_state), rollout_indices,
        )
        return final_train_state
    return train
