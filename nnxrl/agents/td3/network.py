import jax
import jax.numpy as jnp
from flax import nnx
from typing import Callable
from ...utils.network import MLP, flattened_dim, SimBaEncoder
from ...utils.policy import TanhDeterministicPolicy, split_observation
########################
# Actor Network
########################


class Actor(nnx.Module):
    """
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
        obs_dim: int | dict[str, tuple[int,...]],
        action_dim: int,
        rngs: nnx.Rngs,
        hidden_dim: tuple = (256, 256),
        action_high: jax.Array = 1,
        action_low: jax.Array = -1,
        activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        layer_norm: bool = False,
        simba_encoder: bool = False
    ):
        self.action_high = action_high
        self.action_low = action_low
        self.action_scale = (action_high - action_low) / 2
        self.action_dim = action_dim

        if isinstance(obs_dim, dict):
            if 'state' in obs_dim:
                self.actor_obs_dim = flattened_dim(obs_dim["state"])
            else:
                raise KeyError("actor obs dim dict must contain 'state' ")
        else:
            self.actor_obs_dim = flattened_dim(obs_dim)
        if simba_encoder:
            self.encoder = SimBaEncoder(self.actor_obs_dim, 128, 1, rngs)
            out_dim = 128
        else:
            self.encoder = MLP(self.actor_obs_dim,
                           hidden_dim, rngs, layer_norm, activation_fn=activation_fn)
            out_dim = hidden_dim[-1]
        self.actor_head = nnx.Linear(
            out_dim, action_dim, rngs=rngs)



    def get_action(self, x: jax.Array) -> jax.Array:
        """Returns an action in [action_low, action_high].
        """
        actor_obs, _ = split_observation(x)

        pre_tanh = self.actor_head(self.encoder(actor_obs))
        return TanhDeterministicPolicy(self.action_low, self.action_high).action(pre_tanh)







