
from mujoco_playground import registry
from brax import envs

from typing import Any, Optional

import jax
import jax.numpy as jnp

from ..utils.normalization import RMSState, rms_init, rms_normalize
from brax.envs.base import Env, State, Wrapper


class VmapWrapper(Wrapper):
  """Vectorizes Brax env."""

  def __init__(self, env: Env, batch_size: Optional[int] = None):
    super().__init__(env)
    self.batch_size = batch_size

  def reset(self, rng: jax.Array) -> State:
    if self.batch_size is not None:
      rng = jax.random.split(rng, self.batch_size)
    return jax.vmap(self.env.reset)(rng)

  def step(self, state: State, action: jax.Array) -> State:
    return jax.vmap(self.env.step)(state, action)


class EpisodeWrapper(Wrapper):
  """Maintains episode step count and sets done at episode end."""

  def __init__(self, env: Env, episode_length: int, action_repeat: int):
    super().__init__(env)
    self.episode_length = episode_length
    self.action_repeat = action_repeat

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    state.info['steps'] = jnp.zeros(rng.shape[:-1])
    state.info['truncation'] = jnp.zeros(rng.shape[:-1])
    # Keep separate record of episode done as state.info['done'] can be erased
    # by AutoResetWrapper
    state.info['episode_done'] = jnp.zeros(rng.shape[:-1])
    episode_metrics = dict()
    episode_metrics['r'] = jnp.zeros(rng.shape[:-1])
    episode_metrics['l'] = jnp.zeros(rng.shape[:-1])
    for metric_name in state.metrics.keys():
      episode_metrics[metric_name] = jnp.zeros(rng.shape[:-1])
    state.info['episode'] = episode_metrics
    return state

  def step(self, state: State, action: jax.Array) -> State:
    def f(state, _):
      nstate = self.env.step(state, action)
      return nstate, nstate.reward

    state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
    state = state.replace(reward=jnp.sum(rewards, axis=0))
    steps = state.info['steps'] + self.action_repeat
    one = jnp.ones_like(state.done)
    zero = jnp.zeros_like(state.done)
    episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
    done = jnp.where(steps >= episode_length, one, state.done)
    state.info['truncation'] = jnp.where(
        steps >= episode_length, 1 - state.done, zero
    )
    state.info['steps'] = steps

    # Aggregate state metrics into episode metrics
    prev_done = state.info['episode_done']
    state.info['episode']['r'] += jnp.sum(rewards, axis=0)
    state.info['episode']['r'] *= (1 - prev_done)
    state.info['episode']['l'] += self.action_repeat
    state.info['episode']['l'] *= (1 - prev_done)
    for metric_name in state.metrics.keys():
      if metric_name != 'reward':
        state.info['episode'][metric_name] += state.metrics[metric_name]
        state.info['episode'][metric_name] *= (1 - prev_done)
    state.info['episode_done'] = done
    return state.replace(done=done)


class AutoResetWrapper(Wrapper):
  """Automatically resets Brax envs that are done."""

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    # Store the initial state/obs for in-graph auto-reset. We support both:
    # - Brax: `pipeline_state`
    # - MJX:  `data`
    if hasattr(state, "pipeline_state"):
      state.info['first_pipeline_state'] = state.pipeline_state
    elif hasattr(state, "data"):
      state.info['first_state'] = state.data
    else:
      raise AttributeError(
          "Unsupported env state: expected `pipeline_state` (Brax) or `data` (MJX)."
      )
    state.info['first_obs'] = state.obs
    state.info['true_obs'] = state.obs
    return state

  def step(self, state: State, action: jax.Array) -> State:
    if 'steps' in state.info:
      steps = state.info['steps']
      steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
      state.info.update(steps=steps)
    state = state.replace(done=jnp.zeros_like(state.done))
    state = self.env.step(state, action)
    # Preserve terminal observation from the base env step (pre-reset).
    state.info['true_obs'] = state.obs

    def where_done(x, y):
      done = state.done
      if done.shape and done.shape[0] != x.shape[0]:
        return y
      if done.shape:
        done = jnp.reshape(done, [x.shape[0]] + [1] *
                           (len(x.shape) - 1))  # type: ignore
      return jnp.where(done, x, y)

    obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
    if hasattr(state, "pipeline_state"):
      pipeline_state = jax.tree.map(
          where_done, state.info['first_pipeline_state'], state.pipeline_state
      )
      return state.replace(pipeline_state=pipeline_state, obs=obs)
    data = jax.tree.map(where_done, state.info['first_state'], state.data)
    return state.replace(data=data, obs=obs)






def _replicate_rms(rms: RMSState, batch_size: int) -> RMSState:
    """Replicates a scalar/vector RMSState across the batch dimension.

    This is required because Brax's `VmapWrapper` vmaps over the entire `State`
    pytree. Any leaf stored in `state.info` must have a leading batch axis.
    """
    mean = jnp.broadcast_to(rms.mean, (batch_size,) + rms.mean.shape)
    var = jnp.broadcast_to(rms.var, (batch_size,) + rms.var.shape)
    count = jnp.broadcast_to(rms.count, (batch_size,))
    return RMSState(mean=mean, var=var, count=count)


def _unreplicate_rms(rms: RMSState) -> RMSState:
    """Extracts a single RMSState from a replicated (batched) RMSState."""
    return RMSState(mean=rms.mean[0], var=rms.var[0], count=rms.count[0])











class RewardNormWrapper(Wrapper):
    """

    This is similar to VecNormalize-style reward normalization:
      returns_t = gamma * returns_{t-1} * (1 - done) + reward_t
      norm_reward = reward / sqrt(var(returns) + eps)

    Statistics are stored in:
      - `state.info['reward_rms']`
      - `state.info['reward_returns']`
    """

    def __init__(
        self,
        env: Any,
        *,
        gamma: float = 0.99,
        epsilon: float = 1e-4,
        update_stats: bool = True,
    ):
        super().__init__(env)
        self._gamma = float(gamma)
        self._epsilon = float(epsilon)
        self._update_stats = bool(update_stats)

    def reset(self, rng: jax.Array):
        state = self.env.reset(rng)
        batch_shape = state.reward.shape
        state.info["reward_returns"] = jnp.zeros(batch_shape, dtype=jnp.float32)
        reward_rms = rms_init((), epsilon=self._epsilon)
        state.info["reward_rms"] = _replicate_rms(reward_rms, batch_shape[0])
        return state

    def step(self, state, action: jax.Array):
        nstate = self.env.step(state, action)
        reward_returns = state.info.get("reward_returns", jnp.zeros_like(nstate.reward))
        reward_rms = state.info.get("reward_rms", None)
        if reward_rms is None:
            reward_rms = rms_init((), epsilon=self._epsilon)
        else:
            reward_rms = _unreplicate_rms(reward_rms)

        reward_returns = reward_returns * self._gamma * (1.0 - nstate.done) + nstate.reward
        _, reward_rms = rms_normalize(
            reward_returns, reward_rms, epsilon=self._epsilon, update=self._update_stats
        )

        norm_reward = nstate.reward / jnp.sqrt(reward_rms.var + self._epsilon)
        nstate = nstate.replace(reward=norm_reward)
        nstate.info["reward_returns"] = reward_returns
        nstate.info["reward_rms"] = _replicate_rms(reward_rms, nstate.reward.shape[0])
        return nstate


class ActionClip(Wrapper):
    def __init__(self, env: Env):
       super().__init__(env)

    def step(self, state: State, action: jax.Array) -> State:
       return self.env.step(state, jnp.clip(action, -1, 1))







def load_env(env_id: str):
    try:
        # Brax env id examples: "ant", "humanoid", ...
        env = envs.get_environment(env_id)
    except Exception:
        # MuJoCo Playground env id examples: "CartpoleBalance", ...
        env = registry.load(env_id)
    return env



def wrap_for_training(
    env_id: Any,
    *,
    episode_length: int = 1000,
    action_repeat: int = 1,
    normalize_reward: bool = False,
    reward_epsilon: float = 1e-4,
    reward_gamma: float = 0.99,
    reward_update_stats: bool = True,
) -> Any:
    """Composes common wrappers for MuJoCo Playground-style environments."""
    envs = load_env(env_id)
    wrapped = VmapWrapper(envs)
    wrapped = EpisodeWrapper(
        wrapped, episode_length=episode_length, action_repeat=action_repeat
    )
    wrapped = AutoResetWrapper(wrapped)
    if normalize_reward:
        wrapped = RewardNormWrapper(
            wrapped,
            gamma=reward_gamma,
            epsilon=reward_epsilon,
            update_stats=reward_update_stats,
        )
    wrapped = ActionClip(wrapped)

    return wrapped

