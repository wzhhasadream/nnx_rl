import jax
import jax.numpy as jnp
from flax import nnx
from typing import Any, Callable
from ...utils.network import MLP, SimBaEncoder
from ...utils.policy import SquashedTanhGaussianPolicy, flattened_dim, split_observation



    
########################################################
# Actor Network
########################################################

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
        hidden_dim=(256, 256),
        action_high: jax.Array = 1,
        action_low: jax.Array = -1,
        activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        layer_norm: bool = False,
        simba_encoder: bool = False
    ):

        self.action_high = action_high
        self.action_low = action_low
        self.action_scale = (action_high - action_low) / 2

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
            self.encoder = MLP(self.actor_obs_dim, hidden_dim, rngs, layer_norm,
                           activation_fn=activation_fn)
            out_dim = hidden_dim[-1]
        self.fc_mean = nnx.Linear(out_dim, action_dim, rngs=rngs)
        self.fc_logstd = nnx.Linear(
            out_dim, action_dim, rngs=rngs)


    def __call__(self, x: Any) -> Any:
        actor_obs, _ = split_observation(x)
        x = self.encoder(actor_obs)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        return SquashedTanhGaussianPolicy(self.action_low, self.action_high).dist(mean, log_std)

    

    def get_action(self, x: Any, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_distribution = self(x)
        action = action_distribution.sample(seed=key)
        log_prob = action_distribution.log_prob(action)


        return action, log_prob[:, None]    # (batch_size, action_dim), (batch_size, 1)

                
class Alpha(nnx.Module):
    def __init__(self, init_value: float = 0.0):
        self.log_alpha = nnx.Param(jnp.array(init_value))

    def __call__(self) -> jax.Array:
        return jnp.exp(self.log_alpha.value)
