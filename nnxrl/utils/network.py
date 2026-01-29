from typing import Any, Callable
import jax
import jax.numpy as jnp
from flax import nnx
from .policy import split_observation, flattened_dim


def orthogonal(scale: jax.Array = jnp.sqrt(2)):
    return nnx.initializers.orthogonal(scale)


class MLP(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        rngs: nnx.Rngs,
        layer_norm: bool = False,
        activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        orthogonal_init: bool = True
    ):
        dims = [in_dim] + list(hidden_dims)

        self.layers = [
            nnx.Linear(
                dims[i], dims[i + 1],
                rngs=rngs,
                kernel_init=orthogonal() if orthogonal_init else nnx.initializers.lecun_normal()
            )
            for i in range(len(hidden_dims))
        ]

        self.layer_norm = layer_norm
        self.activation_fn = activation_fn
        if layer_norm:
            self.norms = [
                nnx.LayerNorm(num_features=dims[i + 1], rngs=rngs)
                for i in range(len(hidden_dims))
            ]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.layer_norm:
                x = self.norms[i](x)
            x = self.activation_fn(x)

        return x


class QNetwork(nnx.Module):
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

    def __init__(self, obs_dim: int | dict[str, tuple],
                 action_dim: int,
                 rngs: nnx.Rngs,
                 hidden_dim=(256, 256),
                 activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.relu,
                 layer_norm: bool = False,
                 simba_encoder: bool = False
                 ):
        if isinstance(obs_dim, dict):
            if "privileged_state" in obs_dim:
                self.critic_obs_dim = flattened_dim(
                    obs_dim["privileged_state"])
            else:
                raise KeyError(
                    "critic obs dim dict must contain 'privileged_state' ")
        else:
            self.critic_obs_dim = flattened_dim(obs_dim)
        if simba_encoder:
            self.mlp = SimBaEncoder(
                self.critic_obs_dim + action_dim, 512, 2, rngs)
            out_dim = 512
        else:
            self.mlp = MLP(self.critic_obs_dim + action_dim, list(hidden_dim),
                           rngs=rngs, layer_norm=layer_norm, activation_fn=activation_fn)
            out_dim = hidden_dim[-1]
        self.out = nnx.Linear(
            out_dim, 1, rngs=rngs, kernel_init=orthogonal())

    def __call__(self, x, a):
        _, obs_for_critic = split_observation(x)
        h = jnp.concatenate([obs_for_critic, a], axis=1)
        h = self.mlp(h)
        return self.out(h)


class DoubleCritic(nnx.Module):
    """Double Q network using NNX API."""

    def __init__(
        self, obs_dim,
        action_dim, rngs,
        hidden_dim=(256, 256),
        activation_fn: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        layer_norm: bool = False,
        simba_encoder: bool = False
    ):
        critic1_rngs = rngs.fork()
        critic2_rngs = rngs.fork()

        self.critic1 = QNetwork(
            obs_dim, action_dim, critic1_rngs, hidden_dim, activation_fn, layer_norm, simba_encoder)
        self.critic2 = QNetwork(
            obs_dim, action_dim, critic2_rngs, hidden_dim, activation_fn, layer_norm, simba_encoder)

    def __call__(self, observations: Any, actions: jax.Array) -> jax.Array:
        q1 = self.critic1(observations, actions)
        q2 = self.critic2(observations, actions)
        return jnp.minimum(q1, q2)


def copy_model(copied_model: nnx.Module) -> nnx.Module:
    state = nnx.state(copied_model)
    model = nnx.clone(copied_model)
    copied_state = jax.tree.map(lambda x: jnp.asarray(x, copy=True), state)
    nnx.update(model, copied_state)
    return model


def soft_update(online: nnx.Module, target_net: nnx.Module, tau: float = 0.05):
    """Soft-update network parameters."""
    online_params = nnx.state(online, nnx.Param)
    target_params = nnx.state(target_net, nnx.Param)
    new_params = jax.tree.map(
        lambda online_p, target_p: tau *
        online_p + (1 - tau) * target_p,
        online_params,
        target_params,
    )
    nnx.update(target_net, new_params)
    return target_net


# Adapted from SimBa: https://github.com/SonyResearch/simba/blob/master/scale_rl/networks/layers.py
class ResidualBlock(nnx.Module):
    """
    Residual block used in SimBa architecture.

    Architecture:
    - LayerNorm
    - Linear(hidden_dim -> hidden_dim * 4) + ReLU
    - Linear(hidden_dim * 4 -> hidden_dim)
    - Residual connection
    """

    def __init__(
        self,
        hidden_dim: int,
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        self.hidden_dim = hidden_dim

        # Layer normalization
        self.layer_norm = nnx.LayerNorm(
            num_features=hidden_dim,
            rngs=rngs
        )

        # Feedforward network with 4x expansion
        self.dense1 = nnx.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim * 4,
            kernel_init=nnx.initializers.he_normal(),
            rngs=rngs
        )

        self.dense2 = nnx.Linear(
            in_features=hidden_dim * 4,
            out_features=hidden_dim,
            kernel_init=nnx.initializers.he_normal(),  
            rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with residual connection."""
        # Store residual connection
        residual = x

        # Pre-norm residual block
        x = self.layer_norm(x)
        x = self.dense1(x)
        x = nnx.relu(x)
        x = self.dense2(x)

        # Add residual connection
        return residual + x


# Adapted from SimBa:https://github.com/SonyResearch/simba/blob/master/scale_rl/agents/sac/sac_network.py#L33
class SimBaEncoder(nnx.Module):
    """
    SimBa encoder residual block architectures.

    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layers
        num_blocks: Number of residual blocks (default: 1)
        rngs: Random number generators for initialization

    Returns:
        jnp.ndarray: Encoded features
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int = 1,
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.input_projection = nnx.Linear(
            in_features=input_dim,
            out_features=hidden_dim,
            kernel_init=orthogonal(1),
            rngs=rngs
        )

        # Stack residual blocks
        self.residual_blocks = [
            ResidualBlock(hidden_dim, rngs=rngs)
            for _ in range(num_blocks)
        ]

        # Final layer norm
        self.final_layer_norm = nnx.LayerNorm(
            num_features=hidden_dim,
            rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            encoded: Encoded features of shape (batch_size, hidden_dim)
        """
        # Initial projection
        x = self.input_projection(x)

        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Final layer normalization
        x = self.final_layer_norm(x)

        return x




        


# Adapted from ProcGen starter kit: https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class CNNResidualBlock(nnx.Module):
    """A simple residual block with two 3x3 convolutions.

    This block applies:
      ReLU -> Conv(3x3) -> ReLU -> Conv(3x3) -> residual add

    """

    def __init__(self, channels: int, rngs: nnx.Rngs):
        super().__init__()
        self.conv0 = nnx.Conv(in_features=channels, out_features=channels,
                              kernel_size=3, padding=1, kernel_init=orthogonal(), rngs=rngs)
        self.conv1 = nnx.Conv(in_features=channels, out_features=channels,
                              kernel_size=3, padding=1, kernel_init=orthogonal(), rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        inputs = x
        x = jax.nn.relu(x)
        x = self.conv0(x)
        x = jax.nn.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nnx.Module):
    """A convolutional feature extractor stage: Conv + downsampling + residual blocks.

    Structure:
      - 3x3 convolution to `out_channels`
      - 3x3 max-pooling with stride 2 (spatial downsample by ~2)
      - Two residual blocks

    Notes:
        - Input shape should be (H, W, C) for NHWC tensors.
        - Output shape is (ceil(H/2), ceil(W/2), out_channels) with "SAME" pooling.
    """

    def __init__(self, input_shape: tuple[int, int, int], out_channels: int, rngs: nnx.Rngs):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        # input_shape is expected to be (H, W, C) for NHWC inputs.
        self.conv = nnx.Conv(
            in_features=self._input_shape[2],
            out_features=self._out_channels,
            kernel_size=3,
            padding=1,
            rngs=rngs
        )
        self.res_block0 = CNNResidualBlock(self._out_channels, rngs=rngs)
        self.res_block1 = CNNResidualBlock(self._out_channels, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv(x)

        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = self.res_block0(x)
        x = self.res_block1(x)

        return x

    def get_output_shape(self):
        h,w,c = self._input_shape
        return ((h + 1) // 2, (w + 1) // 2, self._out_channels)
