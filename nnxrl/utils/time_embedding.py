import math

import jax
import jax.numpy as jnp
from flax import nnx
import jax.scipy as jsp

def fourier_time_embedding(
    t: jax.Array,
    dim: int,
    max_period: float = 10000.0,
    dtype=jnp.float32,
) -> jax.Array:
    """
    Standard sinusoidal / Fourier time embedding.

    Args:
        t:
            Shape (batch,) or (batch, 1). Usually values in [0, 1] for flow matching
            or diffusion time, but any real-valued time is allowed.
        dim:
            Output embedding dimension.
        max_period:
            Controls the minimum frequency. Same role as in common diffusion
            sinusoidal embeddings.
        dtype:
            Output dtype.

    Returns:
        emb: shape (batch, dim)
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    t = jnp.asarray(t, dtype=dtype)
    if t.ndim == 1:
        t = t[:, None]
    elif t.ndim != 2 or t.shape[-1] != 1:
        raise ValueError(
            f"t must have shape (batch,) or (batch, 1), got {t.shape}"
        )

    half_dim = dim // 2
    if half_dim == 0:
        raise ValueError(f"dim must be at least 2, got {dim}")

    # Exponentially spaced frequencies.
    # Shape: (half_dim,)
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(half_dim, dtype=dtype) / half_dim
    )

    # Shape: (batch, half_dim)
    args = t * freqs[None, :]

    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)

    # If dim is odd, pad one zero column so output stays exactly (batch, dim).
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))

    return emb.astype(dtype)


class TimeEmbedding(nnx.Module):
    """
    Fourier time embedding + 2-layer MLP projection.

    Typical usage in flow matching / diffusion:
        t_emb = time_embed(t)
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int | None = None,
        *,
        max_period: float = 10000.0,
        rngs: nnx.Rngs,
    ):
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim or emb_dim * 4
        self.max_period = max_period

        self.fc1 = nnx.Linear(emb_dim, self.hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(self.hidden_dim, emb_dim, rngs=rngs)

    def __call__(self, t: jax.Array) -> jax.Array:
        x = fourier_time_embedding(
            t,
            dim=self.emb_dim,
            max_period=self.max_period,
        )
        x = self.fc1(x)
        x = nnx.silu(x)
        x = self.fc2(x)
        return x


class HLGaussEmbedding(nnx.Module):
    def __init__(
        self,
        num_bins: int = 51,
        q_min: float = -100.0,
        q_max: float = 100.0,
        sigma: float = 16.0,
        eps: float = 1e-8,
    ):
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2")
        if q_max <= q_min:
            raise ValueError("q_max must be > q_min")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")

        self.num_bins = num_bins
        self.q_min = q_min
        self.q_max = q_max
        self.sigma = sigma
        self.eps = eps

        self.support = jnp.linspace(q_min, q_max, num_bins, dtype=jnp.float32)

    def to_probs(self, target: jax.Array) -> jax.Array:
        # target: [B] or [B, 1]
        target = jnp.asarray(target, dtype=jnp.float32)
        if target.ndim == 2 and target.shape[-1] == 1:
            target = target.squeeze(-1)

        cdf_evals = jsp.special.erf(
            (self.support[None, :] - target[:, None]) /
            (jnp.sqrt(2.0) * self.sigma)
        )
        z = cdf_evals[:, -1:] - cdf_evals[:, :1]
        bin_probs = cdf_evals[:, 1:] - cdf_evals[:, :-1]
        probs = bin_probs / (z + self.eps)
        return probs  # [B, num_bins - 1]

    def __call__(self, target: jax.Array) -> jax.Array:
        return self.to_probs(target)
