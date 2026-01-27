

import jax
import jax.numpy as jnp
from typing import Callable
from flax import nnx
from .normalization import RMSState, rms_normalize


def evaluate_policy(
    envs,
    policy: Callable[jax.Array, jax.Array],
    rng: jax.Array,
    rms: RMSState = None,
    *,
    num_envs: int = 100,
    num_steps: int = 1000,
) -> dict[str, jax.Array]:
    """Evaluate a policy by running it in parallel environments.

    This function runs the given policy for a fixed number of steps across multiple
    parallel environments, collecting episode statistics including returns, lengths,
    and their standard deviations.

    Args:
        envs: Vectorized environment with reset() and step() methods
        policy: Policy function that takes (observations, rng_key) and returns actions.
               Should be JIT-compiled for best performance.
        rng: JAX random key for seeding environment resets and policy sampling
        rms: Optional RMSState for observation normalization during evaluation
        num_envs: Number of parallel environments to evaluate on (default: 100)
        num_steps: Number of environment steps to run (default: 1000)

    Returns:
        Dictionary containing evaluation metrics:
        - "eval/episode_return": Mean episode return across completed episodes
        - "eval/episode_return_std": Standard deviation of episode returns
        - "eval/episode_length": Mean episode length across completed episodes
        - "eval/episode_length_std": Standard deviation of episode lengths
        - "eval/num_episodes": Total number of completed episodes

    Note:
        Episodes are detected when env.done becomes True. The function accumulates
        statistics only for completed episodes during the evaluation period.
    """
    env_key, policy_key = jax.random.split(rng, 2)
    state = envs.reset(jax.random.split(env_key, num_envs))

    init = (
        state,
        jnp.array(0.0, dtype=jnp.float32),  # sum_return
        jnp.array(0.0, dtype=jnp.float32),  # sum_return_sq
        jnp.array(0.0, dtype=jnp.float32),  # sum_length
        jnp.array(0.0, dtype=jnp.float32),  # sum_length_sq
        jnp.array(0.0, dtype=jnp.float32),  # num_episodes
    )

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def eval_step(carry, idx):
        state, sum_r, sum_r2, sum_l, sum_l2, n_ep = carry
        if rms is not None:
            obs_for_policy, _ = rms_normalize(state.obs, rms, update=False)
        else:
            obs_for_policy = state.obs
        actions = policy(obs_for_policy, jax.random.fold_in(policy_key, idx))
        next_state = envs.step(state, actions)

        done = next_state.done.astype(jnp.float32)
        ep_ret = next_state.info.get("episode", {}).get(
            "r", jnp.zeros_like(next_state.reward)
        )
        ep_len = next_state.info.get("episode", {}).get(
            "l", jnp.zeros_like(next_state.reward)
        )
        assert done.ndim == ep_ret.ndim == ep_len.ndim, "All arrays must have the same number of dimensions for broadcasting"
        sum_r = sum_r + jnp.sum(ep_ret * done)
        sum_r2 = sum_r2 + jnp.sum((ep_ret * ep_ret) * done)
        sum_l = sum_l + jnp.sum(ep_len * done)
        sum_l2 = sum_l2 + jnp.sum((ep_len * ep_len) * done)
        n_ep = n_ep + jnp.sum(done)
        return (next_state, sum_r, sum_r2, sum_l, sum_l2, n_ep), ()

    (final_state, sum_r, sum_r2, sum_l, sum_l2, n_ep), _ = eval_step(
        init, jnp.arange(num_steps, dtype=jnp.int32)
    )

    denom = jnp.maximum(n_ep, 1.0)
    mean_r = sum_r / denom
    mean_l = sum_l / denom
    var_r = jnp.maximum(sum_r2 / denom - mean_r * mean_r, 0.0)
    var_l = jnp.maximum(sum_l2 / denom - mean_l * mean_l, 0.0)
    std_r = jnp.sqrt(var_r)
    std_l = jnp.sqrt(var_l)

    _ = final_state

    return {
        "eval/episode_return": mean_r,
        "eval/episode_return_std": std_r,
        "eval/episode_length": mean_l,
        "eval/episode_length_std": std_l,
        "eval/num_episodes": n_ep,
    }
