from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class RMSState:
    """Running mean/variance state for normalization."""

    mean: jax.Array
    var: jax.Array
    count: jax.Array

    @classmethod
    def create(cls, obs_shape: int | tuple[int, ...], epsilon: float = 1e-4) -> "RMSState":
        """Create an RMSState for a single observation tensor (non-dict).

        Notes:
          - `flax.struct.dataclass` generates a constructor automatically:
            `RMSState(mean=..., var=..., count=...)`.
          - This classmethod is a convenience initializer.
        """
        if isinstance(obs_shape, int):
            obs_shape = (obs_shape,)
        return cls(
            mean=jnp.zeros(obs_shape, dtype=jnp.float32),
            var=jnp.ones(obs_shape, dtype=jnp.float32),
            count=jnp.array(epsilon, dtype=jnp.float32),
        )


def rms_init(
    obs_dim: int | dict[str, tuple[int, ...]] | tuple[int, ...],
    epsilon: float = 1e-4,
) -> RMSState | dict[str, RMSState]:
    """Initialize RMS statistics for a single tensor or a dict of tensors."""
    if isinstance(obs_dim, dict):
        return {k: RMSState.create(v, epsilon=epsilon) for k, v in obs_dim.items()}
    if isinstance(obs_dim, int):
        return RMSState.create(obs_dim, epsilon=epsilon)
    if isinstance(obs_dim, tuple):
        return RMSState.create(obs_dim, epsilon=epsilon)
    raise TypeError(f"Unsupported obs_dim type: {type(obs_dim)}")


def _reshape_to_samples(batch: jax.Array, obs_shape: tuple[int, ...]) -> jax.Array:
    """Flatten leading dims into a single sample dimension."""
    batch = jnp.asarray(batch, dtype=jnp.float32)
    return batch.reshape((-1,) + obs_shape)


def rms_update(
    rms: RMSState | dict[str, RMSState],
    batch: jax.Array | dict[str, jax.Array],
) -> RMSState | dict[str, RMSState]:
    """Update RMS statistics with a new batch using Welford's algorithm."""
    if isinstance(batch, dict):
        assert isinstance(
            rms, dict
        ), "When batch is a dict, rms must also be a dict with matching keys."
        return {k: rms_update(rms[k], v) for k, v in batch.items()}

    if rms is None:
        return None

    assert isinstance(rms, RMSState), "rms must be an RMSState for tensor batches."
    batch = _reshape_to_samples(batch, tuple(rms.mean.shape))

    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = batch.shape[0]

    # Welford's algorithm for combining statistics.
    delta = batch_mean - rms.mean
    tot_count = rms.count + batch_count

    new_mean = rms.mean + delta * batch_count / tot_count
    m_a = rms.var * rms.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + jnp.square(delta) * rms.count * batch_count / tot_count
    new_var = m2 / tot_count

    return rms.replace(mean=new_mean, var=new_var, count=tot_count)


def rms_normalize(
    obs: jax.Array | dict[str, jax.Array],
    rms: RMSState | dict[str, RMSState] | None,
    epsilon: float = 1e-4,
    update: bool = True,
) -> tuple[jax.Array | dict[str, jax.Array], RMSState | dict[str, RMSState] | None]:
    """Normalize observations using RMS statistics.

    Supports both single tensors and dicts of tensors.
    """
    if rms is None:
        return obs, None

    if isinstance(obs, dict):
        assert isinstance(
            rms, dict
        ), "When obs is a dict, rms must also be a dict with matching keys."
        if update:
            rms = rms_update(rms, obs)
        norm_obs = {k: (jnp.asarray(v, dtype=jnp.float32) - rms[k].mean) / jnp.sqrt(rms[k].var + epsilon) for k, v in obs.items()}
        return norm_obs, rms

    assert isinstance(rms, RMSState), "When obs is a tensor, rms must be an RMSState."
    if update:
        rms = rms_update(rms, obs)

    obs_arr = jnp.asarray(obs, dtype=jnp.float32)
    norm_obs = (obs_arr - rms.mean) / jnp.sqrt(rms.var + epsilon)
    return norm_obs, rms

