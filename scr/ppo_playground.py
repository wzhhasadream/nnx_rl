import jax.numpy as jnp
import jax
from nnxrl.agents.ppo import TrainState, rollout, update_ppo, ActorCritic
from nnxrl.utils import RMS, evaluate_playground_policy
from nnxrl.model import GaussianActor, VNetwork
from nnxrl.env import make_playground_env
from flax import nnx
import optax
import dataclasses
import tyro
import nnxrl.utils.logger as wandb
from typing import Literal, Sequence


@dataclasses.dataclass
class Args:
    env_id: str = 'BallInCup'
    env_type: Literal["brax", "playground"] = "playground"
    seed: int = 0
    num_envs: int = 2048
    total_timesteps: int = int(6e7)
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float | None = 0.5
    normalize_observation: Literal[True, False] = True
    normalize_advantage: Literal[True, False] = True
    normalize_reward: Literal[True, False] = False
    clip_value: Literal[True, False] = False
    target_kl: float | None = None
    rollout_steps: int = 30  # steps per rollout before update
    num_minibatches: int = 32
    update_epochs: int = 4
    reward_scale: float = 10
    actor_hidden_dim: Sequence[int] = (512, 256, 128)
    critic_hidden_dim: Sequence[int] = (512, 256, 128)
    actor_ln: Literal[True, False] = False
    critic_ln: Literal[True, False] = False
    simba: Literal[True, False] = False
    action_repeat: int = 1
    log_frequency: int = 50
    eval_episode: int = 100


def main():
    args = tyro.cli(Args)
    if args.env_id.startswith("AcrobotSwingup"):
        args.total_timesteps = 100_000_000
    if args.env_id == "BallInCup":
        args.gamma = 0.95
    elif args.env_id.startswith("Swimmer"):
        args.total_timesteps = 100_000_000
    elif args.env_id == "WalkerRun":
        args.total_timesteps = 100_000_000
    elif args.env_id == "FingerSpin":
        args.gamma = 0.95
    elif args.env_id == "PendulumSwingUp":
        args.action_repeat = 4
    num_rollout = int(args.total_timesteps /
                      (args.rollout_steps * args.num_envs))


    steps_per_rollout = args.num_minibatches * args.update_epochs
    wandb.init(project='ppo', config=vars(args), name=f'{args.env_id}')
    rngs = nnx.Rngs(args.seed)

    env = make_playground_env(
        args.env_id,
        args.env_type,
        action_repeat=args.action_repeat
    )
    obs_dim = env.observation_size
    action_dim = env.action_size
    actor = GaussianActor(obs_dim, action_dim, rngs.fork(), args.actor_hidden_dim, layer_norm=args.actor_ln, simba_encoder=args.simba, squash_log_std=False, shared_std=True, activation_fn=jax.nn.relu)
    v = VNetwork(obs_dim, rngs.fork(), args.critic_hidden_dim, layer_norm=args.critic_ln, simba_encoder=args.simba, activation_fn=jax.nn.relu)
    agent = ActorCritic(actor, v)

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
    if args.normalize_observation:
        rms = RMS.create(env.observation_size)
    else:
        rms = None
    ts = TrainState(agent=agent, opt=optimizer,
                             rms=rms) 

    jit_update = nnx.jit(lambda ts, traj, key: update_ppo(ts, traj, args, key), donate_argnums=0)
    jit_rollout = nnx.jit(lambda ts, state, key: rollout(ts, env, state, key, args))

    def eval(ts):
        def policy(obs):
            if args.normalize_observation:
                obs_for_policy, _ = ts.rms.normalize(obs, update=False)
            else:
                obs_for_policy = obs
            actions = ts.agent.actor.get_mean_action(obs_for_policy)
            return actions
        return evaluate_playground_policy(env, policy, args.eval_episode, seed=args.seed + 1)

    jit_eval = nnx.jit(eval)

    state = env.reset(jax.random.split(jax.random.PRNGKey(args.seed), args.num_envs))
    rollout_key, update_key = jax.random.split(jax.random.PRNGKey(args.seed))

    for rollout_idx in range(1, num_rollout + 1):
        state, ts, traj = jit_rollout(ts, state, jax.random.fold_in(rollout_key, rollout_idx))
        
        ts, info = jit_update(ts, traj, jax.random.fold_in(update_key, rollout_idx))
        if rollout_idx % args.log_frequency == 0:
            score = jit_eval(ts)
            print(score)
            wandb.log({**info, "episode_return": score}, rollout_idx * args.num_envs * args.rollout_steps)


    score = jit_eval(ts)
    wandb.log({"episode_return": score}, args.total_timesteps)
    wandb.finish()


if __name__ == "__main__":
    main()
