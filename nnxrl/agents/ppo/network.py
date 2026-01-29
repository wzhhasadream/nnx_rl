from typing import Any, Sequence, Callable

import jax
import jax.numpy as jnp
from flax import nnx
from ...utils.network import MLP, orthogonal
from ...utils.policy import (
    GaussianPolicy,
    MaskedCategoricalPolicy,
    split_observation,
    flattened_dim,
)




class ActorCritic(nnx.Module):
    """Actor-critic model for PPO.

    Supports both:
      - plain observation arrays: `obs: (B, obs_dim)`
      - dict observations with privileged critic inputs:
          {
            "state": (B, actor_obs_dim),
            "privileged_state": (B, critic_obs_dim),
          }

    The actor always uses the "state" part.
    The critic prefers "privileged_state" when available.
    """

    def __init__(
        self,
        obs_dim: int | dict[str, tuple],
        action_dim: int,
        rngs: nnx.Rngs,
        action_high: jax.Array = 1,
        action_low: jax.Array = -1,
        hidden_dim: Sequence[int] = (256, 256),
        activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.relu,
    ):
        self.action_high = action_high
        self.action_low = action_low
        self.hidden_dim = tuple(hidden_dim)

        if isinstance(obs_dim, dict):
            if "state" in obs_dim:
                self.actor_obs_dim = flattened_dim(obs_dim["state"])
            else:
                raise KeyError("obs_dim dict must contain 'state' ")

            if "privileged_state" in obs_dim:
                self.critic_obs_dim = flattened_dim(obs_dim["privileged_state"])
            else:
                self.critic_obs_dim = self.actor_obs_dim
        else:
            self.actor_obs_dim = int(obs_dim)
            self.critic_obs_dim = int(obs_dim)

        # Critic network.
        self.critic_encoder = MLP(self.critic_obs_dim, list(self.hidden_dim), rngs=rngs, activation_fn=activation_fn)
        self.critic_head = nnx.Linear(
            self.hidden_dim[-1], 1, rngs=rngs, kernel_init=orthogonal())

        # Actor network.
        self.actor_encoder = MLP(self.actor_obs_dim, list(self.hidden_dim), rngs=rngs, activation_fn=activation_fn)
        self.mean_head = nnx.Linear(self.hidden_dim[-1], action_dim, rngs=rngs, kernel_init=orthogonal())
        self.log_std = nnx.Param(jnp.zeros((action_dim,)))


    def __call__(self, observations: Any):
        """Returns action distribution for actor observations."""
        actor_obs, _ = split_observation(observations)
        actor_features = self.actor_encoder(actor_obs)
        mean = self.mean_head(actor_features)
        log_std = jnp.broadcast_to(self.log_std.value, mean.shape)
        return GaussianPolicy(squash_log_std=False).dist(mean, log_std)
    
    def get_value(self, observations: Any) -> jax.Array:
        """Return critic value estimate (uses privileged observations when provided)."""
        _, critic_obs = split_observation(observations)
        critic_features = self.critic_encoder(critic_obs)
        value = self.critic_head(critic_features)
        return value.squeeze(-1)  # (batch_size, )

    def get_action_and_value(
        self,
        observations: Any,
        *,
        actions: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return (actions, log_probs, values, entropy) for PPO updates."""
        action_distribution = self(observations)
        if actions is None:
            actions = action_distribution.sample(seed=key)
        log_probs = action_distribution.log_prob(actions)
        values = self.get_value(observations)
        entropy = action_distribution.entropy()
        # (batch_size, action_dim), (batch_size, ), (batch_size, ), (batch_size, )
        return actions, log_probs, values, entropy


########################################################################
################### CategoricalPolicy ##################################
########################################################################

class CategoricalActorCritic(nnx.Module):
    """Actor-critic model for PPO with categorical policy.
    
    This class implements the actor-critic architecture specifically
    designed for discrete action spaces using categorical distributions.

    """

    def __init__(
        self,
        obs_shape: Sequence[int],
        num_action: int,
        rngs: nnx.Rngs,
        *,
        hidden_dim: Sequence[int] = (256, 256)
    ):
        if len(obs_shape) != 3:
            raise ValueError(f"obs_shape must be (H, W, C), got {obs_shape}")

        self.num_action = int(num_action)

        h, w, c = int(obs_shape[0]), int(obs_shape[1]), int(obs_shape[2])



        # Shared trunk.
        self.encoder = MLP(h * w * c, hidden_dim, rngs, orthogonal_init=True)

        # Actor head.
        self.actor_head = nnx.Linear(hidden_dim[-1], self.num_action, rngs=rngs, kernel_init=orthogonal())

        # Critic head.
        self.critic_head = nnx.Linear(hidden_dim[-1], 1, rngs=rngs, kernel_init=orthogonal())


    def __call__(self, observations: jax.Array, legal_action_mask: jax.Array | None = None):
        """Returns action distribution for actor observations."""
        x = observations.astype(jnp.float32)
        x = x.reshape(x.shape[0], -1)
        x = self.encoder(x)
        logits = self.actor_head(x)
        return MaskedCategoricalPolicy().dist(logits, legal_action_mask)

    def get_value(self, observations: Any) -> jax.Array:
        """Return critic value estimate (uses privileged observations when provided)."""
        x = observations.astype(jnp.float32)
        x = x.reshape(x.shape[0], -1)
        x = self.encoder(x)
        value = self.critic_head(x)
        return value.squeeze(-1)

    def get_action_and_value(
        self,
        observations: Any,
        *,
        actions: jax.Array | None = None,
        key: jax.Array | None = None,
        legal_action_mask: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return (actions, log_probs, values, entropy) for PPO updates."""
        action_distribution = self(observations, legal_action_mask)
        if actions is None:
            actions = action_distribution.sample(seed=key).astype(jnp.int32)
        else:
            actions = actions.astype(jnp.int32)
        log_probs = action_distribution.log_prob(actions)
        values = self.get_value(observations)
        entropy = action_distribution.entropy()
        # (batch_size, ), (batch_size, ), (batch_size, ), (batch_size, )
        return actions, log_probs, values, entropy
