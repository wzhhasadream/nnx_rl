
import jax.numpy as jnp
from nnxrl.agents.ppo.train import TrainState, ppo_update_from_trajectory, Trajectory
from nnxrl.agents.ppo.network import ActorCritic
from flax import nnx
import jax
from nnxrl.utils.dmc_wrapper import make_env_dmc
import optax
import numpy as np
import dataclasses
import tyro
import nnxrl.utils.logger as wandb
from typing import Sequence, Literal
import time
import gymnasium as gym


@dataclasses.dataclass
class Args:
    env_id: str = 'HalfCheetah-v4'
    seed: int = 1
    num_envs: int = 16
    total_timesteps: int = int(1e6)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float | None = 0.5
    normalize_observation: Literal[True, False] = True
    normalize_advantage: Literal[True, False] = True
    normalize_reward: Literal[True, False] = True
    clip_value: Literal[True, False] = False
    reward_scale: float = 1
    target_kl: float | None = None
    rollout_steps: int = 256  # steps per rollout before update
    num_minibatches: int = 32
    update_epochs: int = 16
    hidden_dim: Sequence[int] = (256, 256)
    action_repeat: int = 1


def make_env(env_id: str, seed: int, action_repeat: int = 2):
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
        env = gym.wrappers.ClipAction(env)
        if hasattr(env, 'seed'):
            env.seed(seed)
        return env
    return thunk





def main():
    args = tyro.cli(Args)
    np.random.seed(args.seed)
    num_rollout = int(args.total_timesteps /
                      (args.rollout_steps * args.num_envs))
    steps_per_rollout = args.num_minibatches * args.update_epochs
    env = [make_env(args.env_id, args.seed + i, args.action_repeat)
           for i in range(args.num_envs)]
    envs = gym.vector.SyncVectorEnv(env)
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    if args.normalize_observation:
        envs = gym.wrappers.vector.NormalizeObservation(envs)

    if args.normalize_reward:
        envs = gym.wrappers.vector.NormalizeReward(envs, args.gamma)
              

    action_dim = int(np.prod(np.asarray(envs.single_action_space.shape)))
    obs_dim = int(np.prod(np.asarray(envs.single_observation_space.shape)))

    wandb.init(project='ppo', config=vars(args), name=f'{args.env_id}')
    rngs = nnx.Rngs(args.seed)
    agent = ActorCritic(obs_dim, action_dim, rngs, envs.single_observation_space.high, envs.single_observation_space.low, args.hidden_dim)
    jit_get_action = nnx.jit(lambda agent, obs, key: agent.get_action_and_value(obs, key=key), donate_argnums=0)
    jit_update = nnx.jit(
        lambda ts, traj, key: ppo_update_from_trajectory(ts, traj, args, key), donate_argnums=0)

    start_time = time.time()

    def rollout(
        agent: ActorCritic,
        envs: gym.vector.VectorEnv,
        rollout_steps: int,
        init_obs: jnp.ndarray,
        init_dones: jnp.ndarray,
        key: jax.Array,
        init_global_step: int
    ):
        """
        Rollout agent in environment for num_steps using Python loop.
        Used for Gymnasium environments that cannot be JIT-compiled.

        Returns:
            tuple: (final_obs, final_dones, final_global_step)
            Trajectory: NamedTuple containing rollout data
                - stacked_obs: (num_steps, num_envs, obs_dim)
                - stacked_actions: (num_steps, num_envs, action_dim)
                - stacked_log_probs: (num_steps, num_envs)
                - stacked_values: (num_steps + 1, num_envs)
                - stacked_rewards: (num_steps, num_envs)
                - stacked_dones: (num_steps + 1, num_envs)
        """
        # Preallocate storage
        obs_list = []
        actions_list = []
        log_probs_list = []
        values_list = []
        rewards_list = []
        dones_list = [init_dones]

        cur_obs = init_obs
        cur_dones = init_dones
        global_step = init_global_step

        keys = jax.random.split(key, rollout_steps)


        # Python loop for environment interaction
        for step_key in keys:
            actions, log_probs, values, _ = jit_get_action(agent, cur_obs, key=step_key)

            # Store current step data
            obs_list.append(cur_obs)
            actions_list.append(actions)
            log_probs_list.append(log_probs)
            values_list.append(values)

            # Step environment
            next_obs, rewards, next_terminations, next_truncations, infos = envs.step(
                actions)
            next_dones = next_terminations | next_truncations
            global_step += actions.shape[0]

            if next_dones.any():
                episode_return = infos["episode"]['r'][next_dones].mean()
                episode_length = infos["episode"]['l'][next_dones].mean()
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

            # Store rewards and dones
            rewards_list.append(rewards)
            dones_list.append(next_dones)

            # Update state
            cur_obs = next_obs
            cur_dones = next_dones

        # Get final value for bootstrap
        final_value = agent.get_value(cur_obs)
        values_list.append(final_value)

        # Stack all data
        stacked_obs = jnp.stack(obs_list, axis=0)
        stacked_actions = jnp.stack(actions_list, axis=0)
        stacked_log_probs = jnp.stack(log_probs_list, axis=0)
        stacked_values = jnp.stack(values_list, axis=0)
        stacked_rewards = jnp.stack(rewards_list, axis=0)
        stacked_dones = jnp.stack(dones_list, axis=0)

        traj = Trajectory(stacked_obs, stacked_actions, stacked_log_probs, stacked_values, stacked_dones, stacked_rewards)

        return (cur_obs, cur_dones, global_step), traj

    def lr_schedule(step):
        progress = (step // steps_per_rollout) / max(num_rollout, 1)
        return args.lr * jnp.maximum(0.0, 1.0 - progress)

    if args.max_grad_norm is not None and args.max_grad_norm > 0:
        optimizer_chain = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(lr_schedule)
        )
    else:
        optimizer_chain = optax.chain(
            optax.adam(lr_schedule)
        )
    optimizer = nnx.Optimizer(agent, optimizer_chain)
    train_state = TrainState(agent, optimizer)

    obs, _ = envs.reset(seed=args.seed)
    dones = jnp.zeros((args.num_envs, ))
    global_step = 0
    rollout_key, update_key = jax.random.split(jax.random.PRNGKey(args.seed))

    for rollout_idx in range(1, num_rollout):
        (next_obs, next_dones, global_step), traj = rollout(
            train_state.agent, envs, args.rollout_steps, obs, dones, jax.random.fold_in(rollout_key, rollout_idx), global_step)
        
        train_state, info = jit_update(train_state, traj, jax.random.fold_in(update_key, rollout_idx))
        if rollout_idx % 10 == 0:
            wandb.log(info, global_step)
        obs = next_obs
        dones = next_dones

    envs.close()
    wandb.finish()


if __name__ == "__main__":
    main()

