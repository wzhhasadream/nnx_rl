import jax
import jax.numpy as jnp
from flax import nnx
import flax.struct as struct
from copy import deepcopy
from typing import Protocol

from nnxrl.utils.policy import diagonal_gaussian_kl
from ..utils.network import (
GaussianActor,
Alpha,
EnsembleCritic,
soft_update,
SquashedAlpha)
from ..utils.replaybuffer import Batch
from ..utils.normalization import RMS
from ..utils.checkpoint import load_states, save_states


class BROConfig(Protocol):
    seed: int
    total_timesteps: int
    num_envs: int
    learning_starts: int
    num_evals: int
    pessimism: float
    num_quantile: int
    kappa: float
    dist_q: bool

    gamma: float
    tau: float

    batch_size: int
    grad_step_per_env_step: int
    policy_frequency: int
    target_frequency: int

    normalize_observation: bool
    target_entropy: float
    target_kl: float


@struct.dataclass
class TrainState:
    actor: GaussianActor
    optimism_actor: GaussianActor
    critic: EnsembleCritic
    actor_opt: nnx.Optimizer
    critic_opt: nnx.Optimizer
    target_critic: EnsembleCritic
    optimism_actor_opt: nnx.Optimizer

    
    entropy_coef: Alpha 
    optimism_coef: SquashedAlpha 
    kl_coef: SquashedAlpha 
    entropy_coef_opt: nnx.Optimizer 
    optimism_coef_opt: nnx.Optimizer 
    kl_coef_opt: nnx.Optimizer 

    rms: RMS | None = None

    @classmethod
    def create(cls,
    actor,
    optimism_actor,
    critic,
    actor_opt,
    critic_opt,
    optimism_actor_opt,
    entropy_coef,
    optimism_coef,
    kl_coef,
    entropy_coef_opt,
    optimism_coef_opt,
    kl_coef_opt,
    rms=None):
        target_critic = deepcopy(critic)
        return cls(
            actor=actor,
            optimism_actor=optimism_actor,
            critic=critic,
            target_critic=target_critic,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            optimism_actor_opt=optimism_actor_opt,
            rms=rms,
            entropy_coef=entropy_coef,
            optimism_coef=optimism_coef,
            kl_coef=kl_coef,
            entropy_coef_opt=entropy_coef_opt,
            optimism_coef_opt=optimism_coef_opt,
            kl_coef_opt=kl_coef_opt
        )


    def save(self, path: str):
        save_states(path, {
            "actor" : self.actor,
            "optimism_actor": self.optimism_actor,
            "critic": self.critic,
            "target_critic": self.target_critic,
            "actor_opt": self.actor_opt,
            "critic_opt": self.critic_opt,
            "rms": self.rms,
            "optimism_actor_opt": self.optimism_actor_opt,
            "entropy_coef": self.entropy_coef,
            "optimism_coef": self.optimism_coef,
            "kl_coef": self.kl_coef,
            "entropy_coef_opt": self.entropy_coef_opt,
            "optimism_coef_opt": self.optimism_coef_opt,
            "kl_coef_opt": self.kl_coef_opt,
        })

    def load(self, path: str):
        load_states(path, {
            "actor": self.actor,
            "optimism_actor": self.optimism_actor,
            "critic": self.critic,
            "actor_opt": self.actor_opt,
            "target_critic": self.target_critic,
            "critic_opt": self.critic_opt,
            "rms": self.rms,
            "optimism_actor_opt": self.optimism_actor_opt,
            "entropy_coef": self.entropy_coef,
            "optimism_coef": self.optimism_coef,
            "kl_coef": self.kl_coef,
            "entropy_coef_opt": self.entropy_coef_opt,
            "optimism_coef_opt": self.optimism_coef_opt,
            "kl_coef_opt": self.kl_coef_opt,
        })




def huber_replace(diff, kappa: float = 1.0):
    return jnp.where(
        jnp.abs(diff) <= kappa,
        0.5 * diff ** 2,
        kappa * (jnp.abs(diff) - 0.5 * kappa),
    )


def quantile_huber_loss(diff, bro_taus, kappa: float = 1.0):
    # diff: [B, num_quantile, num_quantile]
    # taus: [num_quantile] or [1, num_quantile]
    weight = jnp.abs(bro_taus[..., None] - (diff < 0).astype(diff.dtype))
    huber_loss = huber_replace(diff, kappa)
    return (weight * huber_loss / kappa).sum(axis=1).mean()


def quantile_loss(q_distributional, target_q_distributional, kappa: float = 1.0):
    # q_distributional: [B, num_quantile]
    # target_q_distributional: [B, num_quantile]
    assert q_distributional.shape == target_q_distributional.shape, "The shape of q_distributional and target_q_distributional should be the same"
    num_quantiles = q_distributional.shape[1]
    bro_taus = jnp.arange(0, num_quantiles + 1) / num_quantiles
    bro_taus = (bro_taus[1:] + bro_taus[:-1]) / 2
    diff = target_q_distributional[:, None, :] - q_distributional[:, :, None]
    return quantile_huber_loss(diff, bro_taus, kappa)


def update_critic(ts: TrainState, batch: Batch, config: BROConfig, key: jax.Array):
    entropy_coef = ts.entropy_coef()
    def dist_critic_loss(critic, target_critic, actor):
            next_actions, next_log_pi, _ = actor.get_action(batch.next_observations, key=key)
            next_q_dist = target_critic(batch.next_observations, next_actions)  # (2, B, num_quantile)
            q_uncertainty = jnp.abs(next_q_dist[0] - next_q_dist[1]) / 2      # (B, num_quantile)
            next_q_dist = next_q_dist.mean(0) - config.pessimism * q_uncertainty # (B, num_quantile)
            target_q_dist = batch.rewards + config.gamma * (1 - batch.dones) * (next_q_dist - entropy_coef * next_log_pi)  # (B, num_quantile)
            q_dist = critic(batch.observations, batch.actions)  # (2, B, num_quantile)

            q_loss = quantile_loss(q_dist[0], target_q_dist, config.kappa) + quantile_loss(q_dist[1], target_q_dist, config.kappa)

            return q_loss, {
            "training/q_loss": q_loss,
            "training/q_mean": q_dist.mean(),
            "training/q_uncertainty": q_uncertainty.mean()
            }

    def critic_loss(critic, target_critic, actor):
        next_actions, next_log_pi, _ = actor.get_action(
            batch.next_observations, key=key
        )
        next_q = target_critic(batch.next_observations, next_actions)    # (2, B, 1)
        q_uncertainty = jnp.abs(next_q[0] - next_q[1]) / 2      # (B, 1)
        next_q = next_q.mean(0) - config.pessimism * q_uncertainty          # (B, 1)
        target_q = batch.rewards + config.gamma * \
            (1 - batch.dones) * (next_q - entropy_coef * next_log_pi)

        q = critic(batch.observations, batch.actions)
        q_loss = jnp.mean((q - target_q)**2)

        return q_loss, {
            "training/q_loss": q_loss,
            "training/q_mean": q.mean(),
            "training/q_uncertainty": q_uncertainty.mean()
            }
    loss_fn = dist_critic_loss if config.dist_q else critic_loss

    (_, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(ts.critic, ts.target_critic, ts.actor)
    ts.critic_opt.update(grads)

    soft_update(ts.critic, ts.target_critic, config.tau)

    return ts, info


def update_entropy_coef(
    entropy_coef: Alpha,
    entropy_coef_opt: nnx.Optimizer,
    config: BROConfig,
    log_pi: jax.Array,
) -> dict[str, jax.Array]:
    def entropy_coef_loss(entropy_coef):
        loss = (-entropy_coef() * (log_pi + config.target_entropy)).mean()
        return loss, {
            "training/entropy_coef_loss": loss,
            "training/entropy_coef": entropy_coef(),
        }

    (_loss, info), grads = nnx.value_and_grad(entropy_coef_loss, has_aux=True)(entropy_coef)
    entropy_coef_opt.update(grads)
    return info


def update_optimism_coef(
    optimism_coef: SquashedAlpha,
    optimism_coef_opt: nnx.Optimizer,
    config: BROConfig,
    empirical_kl: jax.Array,
) -> dict[str, jax.Array]:
    def optimism_coef_loss(optimism_coef):
        loss = (optimism_coef() * (empirical_kl - config.target_kl)).mean()
        return loss, {
            "training/optimism_coef_loss": loss,
            "training/optimism_coef": optimism_coef(),
        }
    (_loss, info), grads = nnx.value_and_grad(optimism_coef_loss, has_aux=True)(optimism_coef)
    optimism_coef_opt.update(grads)
    return info


def update_kl_coef(
    kl_coef: SquashedAlpha,
    kl_coef_opt: nnx.Optimizer,
    config: BROConfig,
    empirical_kl: jax.Array,
) -> dict[str, jax.Array]:
    def kl_coef_loss(kl_coef):
        loss = (- kl_coef() * (empirical_kl - config.target_kl)).mean()
        return loss, {
            "training/kl_coef_loss": loss,
            "training/kl_coef": kl_coef(),
        }
    (_loss, info), grads = nnx.value_and_grad(kl_coef_loss, has_aux=True)(kl_coef)
    kl_coef_opt.update(grads)
    return info

def update_actor(ts: TrainState, batch: Batch, config: BROConfig, key: jax.Array):
    def actor_loss(actor, critic, entropy_coef, entropy_coef_opt):
        actions, log_pi, _ = actor.get_action(batch.observations, key=key)
        q_dist = critic(batch.observations, actions)    # (2, B, num_quantile)
        q_dist = q_dist.mean(0) - config.pessimism * jnp.abs(q_dist[0] - q_dist[1]) / 2
        q_value = q_dist.mean(-1, keepdims=True)    # (B, 1)

        loss = - (q_value - entropy_coef() * log_pi).mean()

        alpha_info = update_entropy_coef(entropy_coef, entropy_coef_opt, config, log_pi)

        return loss, {"training/actor_loss": loss, **alpha_info}

    (_, info), grads = nnx.value_and_grad(actor_loss, has_aux=True)(
        ts.actor,
        ts.critic,
        ts.entropy_coef,
        ts.entropy_coef_opt,
    )

    ts.actor_opt.update(grads)

    return ts, info


def update_optimism_actor(ts: TrainState, batch: Batch, config: BROConfig, key: jax.Array):
    def optimism_actor_loss(optimism_actor, critic, actor, optimism_coef, kl_coef, optimism_coef_opt, kl_coef_opt):
        actor_dist = actor(batch.observations)

        optimism_dist = optimism_actor(batch.observations)

        actions = optimism_dist.sample(seed=key)
        q_dist = critic(batch.observations, actions)
        kl = diagonal_gaussian_kl(
            actor_dist.mean(),
            actor_dist.stddev(),
            optimism_dist.mean(),
            optimism_dist.stddev(),
        )
        q_ub = q_dist.mean(0) + optimism_coef() * jnp.abs(q_dist[0] - q_dist[1]) / 2
        q_ub = q_ub.mean(-1, keepdims=True)
        actor_e_loss = (-q_ub).mean() + kl_coef() * kl.mean()
        optimism_info = update_optimism_coef(optimism_coef, optimism_coef_opt, config, kl)
        kl_info = update_kl_coef(kl_coef, kl_coef_opt, config, kl)
        return actor_e_loss, {"training/optimism_actor_loss": actor_e_loss, "training/kl": kl.mean(), "training/Q_mean": q_ub.mean(), **optimism_info, **kl_info}
    (_, info), grads = nnx.value_and_grad(optimism_actor_loss, has_aux=True)(ts.optimism_actor, ts.critic, ts.actor, ts.optimism_coef, ts.kl_coef, ts.optimism_coef_opt, ts.kl_coef_opt)
    ts.optimism_actor_opt.update(grads)
    return ts, {**info}




def update_bro(ts: TrainState, config: BROConfig, big_batch: Batch, key: jax.Array):
    if ts.rms is not None and config.normalize_observation:
        stacked_obs = jnp.concatenate(
            [big_batch.observations, big_batch.next_observations],
            axis=0,
        )
        normalized_obs, rms = ts.rms.normalize(
            stacked_obs, update=True)
        ts = ts.replace(rms=rms)

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
    def single_update(ts, sub_batch, key):
        keys = jax.random.split(key, 3)
        ts, critic_info = update_critic(ts, sub_batch, config, keys[0])
        ts, actor_info = update_actor(ts, sub_batch, config, keys[1])
        ts, optimism_actor_info = update_optimism_actor(ts, sub_batch, config, keys[2])

        return ts, {**critic_info, **actor_info, **optimism_actor_info}

    ts, infos = single_update(ts, batches, update_keys)

    info = jax.tree.map(jnp.mean, infos)

    return ts, info

    
