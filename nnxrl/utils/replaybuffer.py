from typing import NamedTuple, Any
import numpy as np
import jax.numpy as jnp
import jax
from gymnasium import spaces
import gymnasium as gym
from flax import struct

def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


ObsTree = jax.Array | dict[str, jax.Array]

class Batch(NamedTuple):
    observations: ObsTree
    actions: jax.Array
    rewards: jax.Array
    dones: jax.Array
    next_observations: ObsTree


def create_batch(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    next_observations: np.ndarray
) -> Batch:
    """
    Create a batch dictionary for JAX.


    Args:
        observations: shape (batch_size, obs_dim)
        actions: shape (batch_size, action_dim)
        rewards: shape (batch_size, 1)
        dones: shape (batch_size, 1)
        next_observations: shape (batch_size, obs_dim)

    Returns:
        Batch dictionary with JAX arrays
    """
    return Batch(
        jnp.array(observations), 
        jnp.array(actions), 
        jnp.array(rewards.reshape(-1, 1)), 
        jnp.array(dones.reshape(-1, 1)), 
        jnp.array(next_observations)
        )




class ReplayBuffer:
    """
    Replay buffer for online RL with multi-env support and optional linear bias sampling.

    Args:
        obs_shape_space: Observation space from environment
        action_shape_space: Action space from environment
        max_size: Maximum buffer size
        n_envs: Number of parallel environments
        linear_decay_steps: Controls sampling bias direction:
            - 0: uniform sampling (no bias)
            - >0: newer-biased (prefer recent experiences)
            - <0: older-biased (prefer older experiences)
        min_weight: Minimum weight for biased experiences (0.1 = 10% of maximum weight)
    """

    def __init__(
        self,
        obs_shape_space: spaces.Space,
        action_shape_space: spaces.Space,
        max_size: int = int(1e6),
        n_envs: int = 1,
        linear_decay_steps: int = 0,
        min_weight: float = 0.1,
        num_buckets: int = 2000,
        use_approximate_sampling: bool = False,
        optimize_memory_usage: bool = False
    ):
        self.buffer_size = max(max_size // n_envs, 1)
        self.n_envs = n_envs
        self.size = 0
        self.ptr = 0
        self.full = False
        self.optimize_memory_usage = optimize_memory_usage

        # Linear bias parameters
        self._raw_linear_decay_steps = linear_decay_steps  # Keep original sign
        self.linear_decay_steps = abs(linear_decay_steps)  # Use absolute value for calculations
        if use_approximate_sampling:
            self.num_buckets = num_buckets
        self.use_approximate_sampling = use_approximate_sampling
        self.min_weight = min_weight

        # Validate parameters
        assert 0 <= min_weight <= 1, f"min_weight must be in [0, 1], got {min_weight}"

        # Extract shapes from spaces
        obs_shape = get_obs_shape(obs_shape_space)
        action_dim = get_action_dim(action_shape_space)

        # Handle both int and tuple for obs_shape
        if isinstance(obs_shape, int):
            self.obs_shape = (obs_shape,)
        else:
            self.obs_shape = obs_shape

        self.action_shape = (action_dim,)

        # Initialize buffers with proper shapes
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=obs_shape_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, *self.action_shape), dtype=action_shape_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        if linear_decay_steps != 0:
            self.timestamps = np.zeros(self.buffer_size, dtype=np.int64)  # Track when each sample was added
            self.current_time = 0
        if not optimize_memory_usage:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=obs_shape_space.dtype)    
        # Initialize dictionary for truncated observations
        # Key: buffer_index (int), Value: {env_index (int): observation (np.ndarray)}
        if optimize_memory_usage:
            self.truncated_next_obs = {}

    @classmethod
    def from_env(
        cls,
        env: gym.vector.VectorEnv,  
        max_size: int = int(1e6),
        linear_decay_steps: int = 0,
        min_weight: float = 0.1,
        num_buckets: int = 2000,
        use_approximate_sampling: bool = False,
        optimize_memory_usage: bool = False
    ) -> 'ReplayBuffer':
        """Create ReplayBuffer from environment - convenience method."""
        obs_shape_space = env.single_observation_space
        action_shape_space = env.single_action_space
        n_envs = getattr(env, 'num_envs', 1)
        return cls(obs_shape_space, action_shape_space, max_size, n_envs, linear_decay_steps, min_weight, num_buckets, use_approximate_sampling, optimize_memory_usage)

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float | np.ndarray,
            next_obs: np.ndarray, done: bool | np.ndarray, truncations: None | np.ndarray = None):
        """Add transition(s) to the buffer. Supports both single and multi-env."""
        if self.optimize_memory_usage:
            assert truncations is not None, "truncations must be provided when optimize_memory_usage is True"

        self.observations[self.ptr] = obs.reshape(self.n_envs, *self.obs_shape)
        self.actions[self.ptr] = action.reshape(self.n_envs, *self.action_shape)
        if not self.optimize_memory_usage:
            self.next_observations[self.ptr] = next_obs.reshape(self.n_envs, *self.obs_shape)
        else:
            self.observations[(self.ptr + 1) % self.buffer_size] = next_obs.reshape(self.n_envs, *self.obs_shape)
            if self.ptr in self.truncated_next_obs:
                del self.truncated_next_obs[self.ptr]
            trunc_indices = np.where(np.atleast_1d(truncations))[0]
            if len(trunc_indices) > 0:
                reshaped_next_obs = next_obs.reshape(self.n_envs, *self.obs_shape)
                self.truncated_next_obs[self.ptr] = {}
                for env_idx in trunc_indices:
                    self.truncated_next_obs[self.ptr][env_idx] = reshaped_next_obs[env_idx].copy()
                    
        self.rewards[self.ptr] = np.asarray(reward, dtype=np.float32).reshape(self.n_envs)
        self.dones[self.ptr] = np.asarray(done, dtype=np.float32).reshape(self.n_envs)
        if self.linear_decay_steps != 0:
            self.timestamps[self.ptr] = self.current_time
            self.current_time += 1
            
        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)

        if self.ptr == self.buffer_size:
            self.full = True
            self.ptr = 0


    def sample(self, batch_size: int) -> Batch:
        """
        Args:
            batch_size: Number of samples to draw

        Returns:
            Batch, each field shape (batch_size, ...) with:
                'observations': shape: (batch_size, obs_dim)
                'actions': shape: (batch_size, action_dim)
                'rewards': shape: (batch_size, 1)
                'dones': shape: (batch_size, 1)
                'next_observations': shape: (batch_size, obs_dim)
        """

        if self._raw_linear_decay_steps == 0:
            if self.optimize_memory_usage:
                if not self.full:  
                    batch_index = np.random.randint(0, self.ptr, size=batch_size)
                else:  
                    offsets = np.random.randint(0, self.buffer_size - 1, size=batch_size) # avoid samplering from the current pointer location
                    batch_index = (self.ptr + 1 + offsets) % self.buffer_size
            else:
                batch_index = np.random.randint(0, self.size, size=batch_size)
        else:
            if self.use_approximate_sampling:
                batch_index = self._sample_with_approximate_bias(batch_size)
            else:
                batch_index = self._sample_with_bias(batch_size)

        env_index = np.random.randint(0, self.n_envs, size=batch_size)

        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_index + 1) % self.buffer_size, env_index].copy()
            for i in range(batch_size):
                idx = batch_index[i]
                env_idx = env_index[i]
                if idx in self.truncated_next_obs and env_idx in self.truncated_next_obs[idx]:
                    next_obs[i] = self.truncated_next_obs[idx][env_idx]
        else:
            next_obs = self.next_observations[batch_index, env_index]

        return create_batch(
            observations=self.observations[batch_index, env_index], 
            actions=self.actions[batch_index, env_index],
            rewards=self.rewards[batch_index, env_index],
            dones=self.dones[batch_index, env_index],
            next_observations=next_obs 
        )

    def _sample_with_bias(self, batch_size: int) -> np.ndarray:
        """
        Sample indices with linear bias weighting.
        Returns batch_index.
        """
        valid_timestamps = self.timestamps[:self.size] 
        age = self.current_time - valid_timestamps

        if self._raw_linear_decay_steps > 0:
            weights = np.maximum(self.min_weight, 1.0 - age / self.linear_decay_steps)
        else:
            weights = np.minimum(1.0, self.min_weight + age / self.linear_decay_steps)

        # If replaybuffer is full, set the weight of the oldest sample to 0
        if self.optimize_memory_usage:
            if self.full:
                weights[self.ptr] = 0

        probabilities = weights / weights.sum()

        batch_index = np.random.choice(self.size, size=batch_size, p=probabilities)
    
        return batch_index

    def _sample_with_approximate_bias(self, batch_size: int) -> np.ndarray:
        """
        Sample indices with approximate linear bias weighting using bucketing.
        Returns batch_index.
        """
        # 1. Determine bucket size
        bucket_size = self.size // self.num_buckets
        if bucket_size == 0:
            return self._sample_with_bias(batch_size)

        # 2. Calculate approximate weight for each bucket
        mid_point_indices = np.arange(bucket_size // 2, self.size, bucket_size)
        bucket_timestamps = self.timestamps[mid_point_indices]
        bucket_ages = self.current_time - bucket_timestamps

        if self._raw_linear_decay_steps > 0:
            bucket_weights = np.maximum(self.min_weight, 1.0 - bucket_ages / self.linear_decay_steps)
        else:
            bucket_weights = np.minimum(1.0, self.min_weight + bucket_ages / self.linear_decay_steps)

        if bucket_weights.sum() == 0:
            bucket_weights = np.ones_like(bucket_weights)

        bucket_probabilities = bucket_weights / bucket_weights.sum()

        # 3. Sample bucket indices
        sampled_bucket_indices = np.random.choice(
            len(bucket_probabilities),
            size=batch_size,
            p=bucket_probabilities
        )

        # 4. Sample uniformly within each chosen bucket
        bucket_starts = sampled_bucket_indices * bucket_size
        random_offsets = np.random.randint(0, bucket_size, size=batch_size)
        batch_index = bucket_starts + random_offsets

        # Ensure indices are within range
        batch_index = np.minimum(batch_index, self.size - 1)
        if self.optimize_memory_usage:
            if self.full:
                mask = (batch_index == self.ptr)
                if np.any(mask):
                    batch_index[mask] = (batch_index[mask] + 1) % self.buffer_size

        return batch_index

    def ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= batch_size

    def reset(self):
        """Reset the buffer."""
        self.ptr = 0
        self.size = 0
        self.current_time = 0
        if self.linear_decay_steps != 0:
            self.timestamps.fill(0)


    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        decay_info = f", decay_steps={self.linear_decay_steps}, min_weight={self.min_weight}" if self.linear_decay_steps > 0 else ""
        return f"ReplayBuffer(size={self.size}/{self.buffer_size}, obs_shape={self.obs_shape}, n_envs={self.n_envs}{decay_info})"




#######################################################
############ JAX ReplayBuffer #########################
#######################################################


@struct.dataclass
class ReplayBufferState:
    observations: ObsTree
    actions: jax.Array
    rewards: jax.Array
    dones: jax.Array
    next_observations: ObsTree
    ptr: jax.Array
    size: jax.Array
    timestamps: jax.Array
    current_time: jax.Array

    buffer_size: int = struct.field(pytree_node=False)
    n_envs: int = struct.field(pytree_node=False)
    obs_shape: tuple[int, ...] | dict[str, tuple[int, ...]
                                      ] = struct.field(pytree_node=False)
    action_shape: tuple[int, ...] = struct.field(pytree_node=False)
    obs_dtype: jnp.dtype = struct.field(pytree_node=False)
    action_dtype: jnp.dtype = struct.field(pytree_node=False)
    raw_linear_decay_steps: int = struct.field(pytree_node=False, default=0)
    linear_decay_steps: int = struct.field(pytree_node=False, default=0)
    min_weight: float = struct.field(pytree_node=False, default=0.1)


def init_replay_buffer(
    observation_shape: tuple[int, ...] | dict[str, tuple[int, ...]] | int,
    action_shape: tuple[int, ...],
    capacity: int,
    n_envs: int,
    *,
    obs_dtype: jnp.dtype = jnp.float32,
    action_dtype: jnp.dtype = jnp.float32,
    linear_decay_steps: int = 0,
    min_weight: float = 0.1,
) -> ReplayBufferState:
    """Initialize a JIT-friendly replay buffer.

    Notes:
      - `observation_shape` may be a dict, e.g. {"state": ..., "privileged_state": ...}.
      - If `linear_decay_steps != 0`, sampling uses an exact time-biased distribution
        (matching the CPU ReplayBuffer implementation in this repo):
          * `> 0`: newer-biased (prefer recent experiences)
          * `< 0`: older-biased (prefer old experiences)
    """

    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")

    buffer_size = max(int(capacity) // int(n_envs), 1)

    def zeros_obs(shape_spec: tuple[int, ...]) -> jax.Array:
        return jnp.zeros((buffer_size, n_envs, *shape_spec), dtype=obs_dtype)

    if isinstance(observation_shape, dict):
        observations = {k: zeros_obs(v) for k, v in observation_shape.items()}
        next_observations = {k: zeros_obs(v)
                             for k, v in observation_shape.items()}
    elif isinstance(observation_shape, tuple):
        observations = zeros_obs(observation_shape)
        next_observations = zeros_obs(observation_shape)
    elif isinstance(observation_shape, int):
        observation_shape = (observation_shape,)
        observations = zeros_obs(observation_shape)
        next_observations = zeros_obs(observation_shape)
    else:
        raise TypeError(
            f"expected obs_dim to be int|tuple|dict, got {type(observation_shape)}")

    actions = jnp.zeros(
        (buffer_size, n_envs, *action_shape), dtype=action_dtype)
    rewards = jnp.zeros((buffer_size, n_envs), dtype=jnp.float32)
    dones = jnp.zeros((buffer_size, n_envs), dtype=jnp.float32)

    ptr = jnp.array(0, dtype=jnp.int32)
    size = jnp.array(0, dtype=jnp.int32)
    timestamps = jnp.zeros((buffer_size,), dtype=jnp.int32)
    current_time = jnp.array(0, dtype=jnp.int32)

    return ReplayBufferState(
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        next_observations=next_observations,
        ptr=ptr,
        size=size,
        timestamps=timestamps,
        current_time=current_time,
        buffer_size=buffer_size,
        n_envs=n_envs,
        obs_shape=observation_shape,
        action_shape=action_shape,
        obs_dtype=obs_dtype,
        action_dtype=action_dtype,
        raw_linear_decay_steps=int(linear_decay_steps),
        linear_decay_steps=abs(int(linear_decay_steps)),
        min_weight=float(min_weight),
    )


def _reshape_obs_leaf(x: Any, shape_spec: tuple[int, ...], n_envs: int, dtype: jnp.dtype) -> jax.Array:
    x = jnp.asarray(x, dtype=dtype)
    return x.reshape((n_envs, *shape_spec))


def _reshape_action(x: Any, shape_spec: tuple[int, ...], n_envs: int, dtype: jnp.dtype) -> jax.Array:
    x = jnp.asarray(x, dtype=dtype)
    return x.reshape((n_envs, *shape_spec))


def _reshape_scalar_vec(x: Any, n_envs: int, dtype: jnp.dtype) -> jax.Array:
    x = jnp.asarray(x, dtype=dtype)
    return x.reshape((n_envs,))


def add(
    state: ReplayBufferState,
    obs: ObsTree | np.ndarray,
    action: jax.Array | np.ndarray,
    reward: jax.Array | np.ndarray,
    next_obs: ObsTree | np.ndarray,
    done: jax.Array | np.ndarray,
) -> ReplayBufferState:
    """Add one transition for each env (vectorized) in a JIT-compatible way."""
    if isinstance(state.obs_shape, dict):
        assert isinstance(obs, dict) and isinstance(next_obs, dict), (
            "When observation_shape is a dict, obs and next_obs must be dicts with matching keys."
        )
        obs_arr = {
            k: _reshape_obs_leaf(obs[k], shape_spec,
                                 state.n_envs, state.obs_dtype)
            for k, shape_spec in state.obs_shape.items()
        }
        next_obs_arr = {
            k: _reshape_obs_leaf(
                next_obs[k], shape_spec, state.n_envs, state.obs_dtype)
            for k, shape_spec in state.obs_shape.items()
        }
        new_observations = {k: state.observations[k].at[state.ptr].set(
            v) for k, v in obs_arr.items()}
        new_next_observations = {
            k: state.next_observations[k].at[state.ptr].set(v) for k, v in next_obs_arr.items()
        }
    else:
        assert not isinstance(obs, dict) and not isinstance(next_obs, dict), (
            "When observation_shape is a tuple, obs and next_obs must be arrays."
        )
        obs_arr = _reshape_obs_leaf(
            obs, state.obs_shape, state.n_envs, state.obs_dtype)
        next_obs_arr = _reshape_obs_leaf(
            next_obs, state.obs_shape, state.n_envs, state.obs_dtype)
        new_observations = state.observations.at[state.ptr].set(obs_arr)
        new_next_observations = state.next_observations.at[state.ptr].set(
            next_obs_arr)

    action_arr = _reshape_action(
        action, state.action_shape, state.n_envs, state.action_dtype)
    reward_arr = _reshape_scalar_vec(reward, state.n_envs, jnp.float32)
    done_arr = _reshape_scalar_vec(done, state.n_envs, jnp.float32)

    new_actions = state.actions.at[state.ptr].set(action_arr)
    new_rewards = state.rewards.at[state.ptr].set(reward_arr)
    new_dones = state.dones.at[state.ptr].set(done_arr)

    bias_enabled = state.raw_linear_decay_steps != 0
    new_timestamps = jax.lax.cond(
        bias_enabled,
        lambda ts: ts.timestamps.at[ts.ptr].set(ts.current_time),
        lambda ts: ts.timestamps,
        state,
    )
    new_current_time = jax.lax.cond(
        bias_enabled,
        lambda ts: ts.current_time + jnp.array(1, dtype=ts.current_time.dtype),
        lambda ts: ts.current_time,
        state,
    )

    new_ptr = (state.ptr + 1) % state.buffer_size
    new_size = jnp.minimum(state.size + 1, state.buffer_size)

    return state.replace(
        observations=new_observations,
        actions=new_actions,
        rewards=new_rewards,
        dones=new_dones,
        next_observations=new_next_observations,
        ptr=new_ptr,
        size=new_size,
        timestamps=new_timestamps,
        current_time=new_current_time,
    )


def _gather_time_env(x: jax.Array, time_idx: jax.Array, env_idx: jax.Array) -> jax.Array:
    return x[time_idx, env_idx]


def sample(
    state: ReplayBufferState,
    key: jax.Array,
    batch_size: int
) -> Batch:
    """Sample a batch of 1-step transitions.

    Args:
      state: replay buffer state
      key: PRNGKey
      batch_size: number of transitions
    """

    max_t = jnp.maximum(state.size, 1)
    key_t, key_e = jax.random.split(key, 2)

    def _uniform_time_idx():
        return jax.random.randint(key_t, (batch_size,), 0, max_t)

    def _biased_time_idx():
        def _do_biased():
            # Compute weights over [0, size). Mask invalid entries to 0 probability.
            idx = jnp.arange(state.buffer_size, dtype=jnp.int32)
            valid = idx < state.size
            age = (state.current_time - state.timestamps).astype(jnp.float32)

            # - newer-biased (raw > 0): w = max(min_weight, 1 - age/decay)
            # - older-biased (raw < 0): w = min(1, min_weight + age/decay)
            decay = jnp.array(state.linear_decay_steps, dtype=jnp.float32)
            decay = jnp.maximum(decay, 1.0)
            min_w = jnp.array(state.min_weight, dtype=jnp.float32)

            newer = jnp.maximum(min_w, 1.0 - age / decay)
            older = jnp.minimum(1.0, min_w + age / decay)
            w = jnp.where(state.raw_linear_decay_steps > 0, newer, older)
            w = jnp.where(valid, w, 0.0)

            wsum = jnp.sum(w)
            # If weights sum to zero (shouldn't happen with sane configs), fall back to uniform.
            denom = jnp.maximum(jnp.sum(valid), 1)
            p = jnp.where(wsum > 0.0, w / wsum, valid.astype(jnp.float32) / denom)
            return jax.random.choice(key_t, state.buffer_size, shape=(batch_size,), p=p)

        # If the buffer is empty, return zeros (caller should normally avoid sampling then).
        return jax.lax.cond(
            state.size > 0,
            lambda _: _do_biased(),
            lambda _: jnp.zeros((batch_size,), dtype=jnp.int32),
            operand=jnp.int32(0),
        )

    time_idx = jax.lax.cond(
        state.raw_linear_decay_steps == 0,
        lambda _: _uniform_time_idx(),
        lambda _: _biased_time_idx(),
        operand=jnp.int32(0),
    )
    env_idx = jax.random.randint(key_e, (batch_size,), 0, state.n_envs)

    if isinstance(state.obs_shape, dict):
        observations = {k: _gather_time_env(
            v, time_idx, env_idx) for k, v in state.observations.items()}
        next_observations = {
            k: _gather_time_env(v, time_idx, env_idx) for k, v in state.next_observations.items()
        }
    else:
        observations = _gather_time_env(state.observations, time_idx, env_idx)
        next_observations = _gather_time_env(
            state.next_observations, time_idx, env_idx)

    actions = _gather_time_env(state.actions, time_idx, env_idx)
    rewards = _gather_time_env(state.rewards, time_idx, env_idx)
    dones = _gather_time_env(state.dones, time_idx, env_idx)

    return Batch(
        observations=observations,
        actions=actions,
        rewards=rewards[:, None],
        dones=dones[:, None],
        next_observations=next_observations,
    )
