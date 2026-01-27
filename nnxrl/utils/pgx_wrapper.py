from __future__ import annotations
from typing import Any

import jax
import jax.numpy as jnp
import pgx
from pgx.experimental.wrappers import auto_reset
from flax import struct


class PgxWrapperBase:
    def __init__(self, env):
        self.env = env
        self.step_fn = auto_reset(env.step, env.init)

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return tuple(self.env.observation_shape)

    @property
    def num_actions(self) -> int:
        return int(self.env.num_actions)

    # Compatibility properties for playground and Brax environments
    @property
    def observation_size(self) ->  tuple[int, ...]:
        """(compatible with playground/Brax)."""
        return tuple(self.env.observation_shape)

    @property
    def action_size(self) -> int:
        """Number of actions (compatible with playground/Brax)."""
        return int(self.env.num_actions)


def _split_key(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Split a PRNG key (supports both scalar and batched keys)."""
    if key.ndim == 1:
        k0, k1 = jax.random.split(key, 2)
        return k0, k1
    keys = jax.vmap(jax.random.split, in_axes=(0, None))(key, 2)  # (B, 2, 2)
    return keys[:, 0], keys[:, 1]


@struct.dataclass
class PgxState:
    """A minimal RL-friendly state wrapper for PGX environments."""

    pgx_state: pgx.State
    key: jax.Array

    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    legal_action_mask: jax.Array
    current_player: jax.Array

    info: dict[str, Any] = struct.field(default_factory=dict)


class VmapWrapper(PgxWrapperBase):
    """Vectorizes a PGX env and maintains per-env RNG keys in-state.
    """

    def __init__(self, env: pgx.Env, batch_size: int | None = None):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: jax.Array) -> PgxState:
        if self.batch_size is not None:
            keys = jax.random.split(rng, self.batch_size)
            init_key, keys = _split_key(keys)

        else:
            init_key, keys = _split_key(rng)
        pgx_state = jax.vmap(self.env.init)(init_key)

        reward = pgx_state.rewards
        terminated = pgx_state.terminated
        truncated = pgx_state.truncated
        legal_action_mask = pgx_state.legal_action_mask
        current_player = pgx_state.current_player


        # truncation does *NOT* actually happens in PGX
        done = jnp.logical_or(terminated, truncated)[:, None]   # (num_envs, 1)

        info = {}
        return PgxState(
            pgx_state=pgx_state,
            key=keys,
            obs=pgx_state.observation,
            reward=reward,
            legal_action_mask=legal_action_mask,
            current_player=current_player,
            done=done,
            info=info,
        )

    def step(self, state: PgxState, action: jax.Array) -> PgxState:
        key, step_key = _split_key(state.key)
        pgx_state = jax.vmap(self.step_fn)(state.pgx_state, action, step_key)

        reward = pgx_state.rewards
        terminated = pgx_state.terminated
        truncated = pgx_state.truncated
        legal_action_mask = pgx_state.legal_action_mask
        current_player = pgx_state.current_player

        done = jnp.logical_or(terminated, truncated)[:, None]
        # Keep bookkeeping fields untouched here; EpisodeWrapper will update them.
        return state.replace(
            pgx_state=pgx_state,
            key=key,
            obs=pgx_state.observation,
            legal_action_mask=legal_action_mask,
            reward=reward,
            current_player=current_player,
            done=done
        )


class RecordEpisodeStatistics(PgxWrapperBase):
    """Adds episode return/length bookkeeping for PGX.
    """

    def __init__(self, env):
        self.env = env

    def reset(self, rng: jax.Array) -> PgxState:
        state = self.env.reset(rng)
        done = state.done  # (num_env,)
        reward = state.reward       # (num_env, n_player)

        episode_return = jnp.zeros_like(
            reward, dtype=jnp.float32)  # (num_env, n_player)
        episode_length = jnp.zeros_like(done, dtype=jnp.float32)   # (num_env, 1)

        info = {
            "episode": {
                "r": episode_return,
                "l": episode_length,
            }
        }
        return state.replace(
            info=info
        )

    def step(self, state: PgxState, action: jax.Array) -> PgxState:
        nstate = self.env.step(state, action)

        prev_done = state.done
        episode_return = (
            state.info["episode"]["r"] + nstate.reward) * (1.0 - prev_done.astype(jnp.float32))
        episode_length = (state.info["episode"]["l"] + 1.0) * \
            (1.0 - prev_done.astype(jnp.float32))

        nstate.info["episode"]["r"] = episode_return
        nstate.info["episode"]["l"] = episode_length

        return nstate


def make_pgx_env(
    pgx_id: str,
    **kwargs: Any,
) -> Any:
    """Convenience constructor for a vectorized PGX env with optional episode wrapper."""
    env = pgx.make(pgx_id, **kwargs)
    env = VmapWrapper(env)
    env = RecordEpisodeStatistics(env)
    return env
