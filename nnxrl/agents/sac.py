import jax
import jax.numpy as jnp
from flax import nnx
import flax.struct as struct
from copy import deepcopy
from mujoco_playground import MjxEnv
from typing import Callable, Protocol
from ..utils.network import (
SquashedTanhGaussianActor, 
Alpha,
EnsembleCritic,
soft_update)
from ..utils.replaybuffer import Batch, JAXReplayBuffer, ReplayBuffer
from ..utils.normalization import RMS
from ..utils.evaluate import evaluate_policy
from ..utils.checkpoint import load_states, save_states


class SACConfig(Protocol):
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
    target_frequency: int

    normalize_observation: bool

    autotune: bool
    alpha: float
    target_entropy: float


@struct.dataclass
class TrainState:
    actor: SquashedTanhGaussianActor
    critic: EnsembleCritic
    actor_opt: nnx.Optimizer
    critic_opt: nnx.Optimizer
    target_critic: EnsembleCritic


    rms: RMS | None = None
    grad_updates: int = 0  
    alpha: Alpha | None = None
    alpha_opt: nnx.Optimizer | None = None

    @classmethod
    def create(cls,
    actor,
    critic,
    actor_opt,
    critic_opt,
    rms=None,
    alpha=None,
    alpha_opt=None):
        target_critic = deepcopy(critic)
        return cls(
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            rms=rms,
            alpha=alpha,
            alpha_opt=alpha_opt,
            grad_updates=0
        )


    def save(self, path: str):
        save_states(path, {
            "actor" : self.actor,
            "critic": self.critic,
            "target_critic": self.target_critic,
            "actor_opt": self.actor_opt,
            "critic_opt": self.critic_opt,
            "rms": self.rms,
            "alpha": self.alpha,
            "alpha_opt": self.alpha_opt,
            "grad_updates": self.grad_updates
        })

    def load(self, path: str):
        state_map = load_states(path, {
            "actor": self.actor,
            "critic": self.critic,
            "actor_opt": self.actor_opt,
            "target_critic": self.target_critic,
            "critic_opt": self.critic_opt,
            "rms": self.rms,
            "alpha": self.alpha,
            "alpha_opt": self.alpha_opt,
            "grad_updates": self.grad_updates
        })

        self.grad_updates = state_map["grad_updates"]



    

def update_critic(train_state: TrainState, config: SACConfig, batch: Batch, key: jax.Array):
    next_actions, next_log_pi = train_state.actor.get_action(
        batch.next_observations, key=key)
    next_q = train_state.target_critic(batch.next_observations, next_actions)
    min_next_q = jnp.min(next_q, axis=0)
    alpha_value = train_state.alpha() if config.autotune else config.alpha
    target_q = batch.rewards + (1.0 - batch.dones) * config.gamma * (
        min_next_q - alpha_value * next_log_pi
    )
    target_q = jax.lax.stop_gradient(target_q)

    def critic_loss_fn(critic_model: EnsembleCritic):
        q = critic_model(batch.observations, batch.actions)
        critic_loss = jnp.mean((q - target_q) ** 2)
        info = {
            "training/q_loss": critic_loss,
            "training/q_mean": jnp.mean(q),
        }
        return critic_loss, info

    (_loss, info), grads = nnx.value_and_grad(
        critic_loss_fn, has_aux=True)(train_state.critic)
    train_state.critic_opt.update(grads)
    return train_state, info



def update_actor(
    train_state: TrainState,
    config: SACConfig,
    batch: Batch,
    key: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Update actor parameters and return the updated TrainState."""
    alpha_value = train_state.alpha() if config.autotune else config.alpha
    alpha_value = jax.lax.stop_gradient(alpha_value)

    def actor_loss_fn(actor_model: SquashedTanhGaussianActor, critic_model: EnsembleCritic):
        actions, log_pi = actor_model.get_action(batch.observations, key=key)
        q = critic_model(batch.observations, actions)
        min_q = jnp.min(q, axis=0)
        actor_loss = -jnp.mean(min_q - alpha_value * log_pi)
        return actor_loss, {"training/actor_loss": actor_loss}

    (_loss, info), grads = nnx.value_and_grad(
        actor_loss_fn, argnums=0, has_aux=True
    )(train_state.actor, train_state.critic)
    train_state.actor_opt.update(grads)
    return train_state, info


def update_alpha(
    train_state: TrainState,
    config: SACConfig,
    batch: Batch,
    key: jax.Array,
) -> tuple[TrainState, dict[str, jax.Array]]:
    """Update entropy temperature (alpha) and return the updated TrainState."""
    log_pi = train_state.actor.get_action(batch.observations, key=key)[1]
    log_pi = jax.lax.stop_gradient(log_pi)

    def alpha_loss_fn(alpha_model: Alpha):
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



def update_sac(train_state: TrainState, config: SACConfig, key: jax.Array, big_batch: Batch):
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

    update_keys = jax.random.split(key, config.grad_step_per_env_step)

    @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
    def update_sac_minibatch(train_state, sub_batch: Batch, key: jax.Array):
        critic_key, policy_key = jax.random.split(key, 2)
        train_state, critic_info = update_critic(train_state, config, sub_batch, critic_key)
        train_state = train_state.replace(grad_updates=train_state.grad_updates + 1)
        alpha_value = train_state.alpha() if config.autotune else jnp.array(config.alpha, dtype=jnp.float32)

        train_state, policy_info = nnx.cond(
            train_state.grad_updates % config.policy_frequency == 0,
            lambda ts: update_policy(ts, config, sub_batch, policy_key),
            lambda ts: (ts, {
            "training/actor_loss": jnp.array(0.0),
            "training/alpha_loss": jnp.array(0.0),
            "training/alpha_value": alpha_value,
            }),
            train_state,
        )
        nnx.cond(
            train_state.grad_updates % config.target_frequency == 0,
            lambda ts: soft_update(ts.critic, ts.target_critic, config.tau),
            lambda ts: None,
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
    rb: JAXReplayBuffer
) -> tuple[TrainState, dict[str, jax.Array]]:
    """(multiple SGD steps per env step)."""
    assert train_state.rb is not None, "Replay buffer state must be initialized"
    sample_key, update_key = jax.random.split(key, 2)

    # 1) sample one big batch, then reshape to (grad_step_per_env_step, batch_size, ...)
    big_batch = rb.sample(
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
