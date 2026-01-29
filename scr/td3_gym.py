import nnxrl.utils.logger as wandb
from flax import nnx
from typing import Literal, Sequence
from nnxrl.utils.replaybuffer import init_replay_buffer, add
from nnxrl.agents.td3.train import sample_and_update_td3, TrainState
from nnxrl.utils.normalization import rms_init, rms_normalize
from nnxrl.agents.td3.network import Actor
from nnxrl.utils.network import DoubleCritic, copy_model
from nnxrl.utils.dmc_wrapper import make_env_dmc
import time
import numpy as np
import jax.numpy as jnp
import jax
import optax
import tyro
import gymnasium as gym
import dataclasses


@dataclasses.dataclass
class Args:
    env_id: str = "humanoid-run"
    seed: int = 0
    num_envs: int = 1
    total_timesteps: int = int(1e6)
    buffer_size: int = int(1e6)
    policy_frequency: int = 2
    learning_starts: int = int(25e3)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    policy_noise: float = 0.2
    actor_noise: float | None = None
    noise_clip: float = 0.5
    exploration_noise: float = 0.1
    hidden_dim: Sequence[int] = (256, 256)

    # Automatically set for DMC environments, can be overridden manually
    action_repeat: int = 2
    normalize_observation: Literal[True, False] = False
    simba: Literal[True, False] = False
    decay_step: int = 0  # 0 means uniformal sampling
    grad_step_per_env_step: int = 1


def make_env(env_id: str, seed: int, action_repeat: int = 1):
    """Create environment."""
    def load_env(env_id):
        try:
            # MUJOCO env id examples: "Ant-v5", "Humanoid-v5", ...
            env = gym.make(env_id)
        except Exception:
            # DMC env id examples: "humanoid-run", "humanoid-stand", ...
            env = make_env_dmc(env_id, action_repeat)
        return env

    def thunk():
        env = load_env(env_id)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        if hasattr(env, 'seed'):
            env.seed(seed)
        return env
    return thunk


def is_dmc_env(env_id: str) -> bool:
    try:
        make_env_dmc(env_id, action_repeat=1)
        return True
    except Exception:
        return False


def main():
    print("🚀 td3 training")
    print("=" * 60)

    activation_fn = jax.nn.mish

    args = tyro.cli(Args)

    if is_dmc_env(args.env_id):
        args.normalize_observation = True
        args.simba = True
        args.policy_lr = 1e-4
        args.q_lr = 1e-4
        #args.decay_step = 80_000

    np.random.seed(args.seed)
    env = [make_env(args.env_id, args.seed + i, args.action_repeat)
           for i in range(args.num_envs)]
    envs = gym.vector.SyncVectorEnv(env, autoreset_mode='SameStep')
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    action_dim = int(np.prod(np.asarray(envs.single_action_space.shape)))
    obs_dim = int(np.prod(np.asarray(envs.single_observation_space.shape)))

    wandb.init(project='td3', config=vars(args),
               name=f'{args.env_id}//decay_step_{args.decay_step}')

    rngs = nnx.Rngs(args.seed)
    actor = Actor(obs_dim, action_dim,
                  rngs.fork(), hidden_dim=args.hidden_dim, action_high=envs.single_action_space.high, action_low=envs.single_action_space.low, activation_fn=activation_fn, simba_encoder=args.simba)
    target_actor = copy_model(actor)

    critic = DoubleCritic(
        obs_dim, action_dim, rngs.fork(), hidden_dim=args.hidden_dim, activation_fn=activation_fn, simba_encoder=args.simba)
    target_critic = copy_model(critic)

    actor_opt = nnx.Optimizer(actor, optax.adamw(
        args.policy_lr, weight_decay=1e-2)) if args.simba else nnx.Optimizer(actor, optax.adam(args.policy_lr))
    critic_opt = nnx.Optimizer(critic, optax.adamw(
        args.q_lr, weight_decay=1e-2)) if args.simba else nnx.Optimizer(critic, optax.adam(args.q_lr))

    buffer_state = init_replay_buffer(
        envs.single_observation_space.shape,
        envs.single_action_space.shape,
        args.buffer_size,
        n_envs=args.num_envs,
        linear_decay_steps=args.decay_step
    )
    if args.normalize_observation:
        rms = rms_init(envs.single_observation_space.shape)
    else:
        rms = None
    train_state = TrainState(actor, critic, target_actor, target_critic, actor_opt, critic_opt,
                             buffer_state, rms)
    start_time = time.time()
    jit_add = nnx.jit(add, donate_argnums=0)

    def get_action_with_exploration_noise(actor, rms, obs, key):
        if args.normalize_observation:
            obs_for_policy, rms = rms_normalize(obs, rms)
        else:
            obs_for_policy = obs
        actions = actor.get_action(obs_for_policy)
        noise = jax.random.normal(key, shape=actions.shape) * train_state.actor.action_scale * args.exploration_noise
        actions = jnp.clip(
            noise + actions, envs.single_action_space.low,  envs.single_action_space.high)
        return rms, actions

    jit_get_action = nnx.jit(
        get_action_with_exploration_noise, donate_argnums=(0, 1))

    jit_update = nnx.jit(lambda ts, key: sample_and_update_td3(
        ts, args, key), donate_argnums=0)
    obs, _ = envs.reset(seed=args.seed)
    step_key, update_key = jax.random.split(jax.random.PRNGKey(args.seed), 2)

    for global_step in range(1, args.total_timesteps + 1):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample()
                               for _ in range(args.num_envs)])
        else:
            rms, actions = jit_get_action(
                train_state.actor, train_state.rms, obs, jax.random.fold_in(step_key, global_step))
            train_state = train_state.replace(rms=rms)
            actions = np.asarray(actions)

        next_obs, rewards, terminations, truncations, infos = envs.step(
            actions)

        if terminations.any() or truncations.any():
            done = np.logical_or(terminations, truncations)

            episode_return = infos["episode"]['r'][done].mean()
            episode_length = infos["episode"]['l'][done].mean()
            current_time = time.time()
            total_time = current_time - start_time
            avg_speed = (global_step) / \
                total_time if total_time > 0 else 0

            print(f"🎉 Step {global_step:,}: Episode return {episode_return:.1f} (length: {episode_length}) "
                    f"⚡ {avg_speed:.1f} steps/s")

            wandb.log({
                "episode_return": episode_return,
                "episode_length": episode_length,
                "wall_time": total_time
            }, global_step, commit=False)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_obs"][idx]

        new_rb = jit_add(
            train_state.rb,
            obs,
            actions,
            rewards,
            real_next_obs,
            terminations,
        )
        train_state = train_state.replace(rb=new_rb)

        if global_step >= args.learning_starts:
            train_state, info = jit_update(
                train_state, jax.random.fold_in(
                    update_key, global_step)
            )
            if global_step % 999 == 0:
                wandb.log(info, global_step)

        obs = next_obs

    envs.close()
    wandb.finish()


if __name__ == "__main__":
    main()
