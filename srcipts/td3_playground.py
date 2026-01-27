import dataclasses
import tyro
import optax
import time
from nnxrl.agents.td3.network import Actor
from nnxrl.utils.network import copy_model, DoubleCritic
from nnxrl.agents.td3.train import make_train, TrainState
from nnxrl.utils.replaybuffer import init_replay_buffer
from nnxrl.utils.normalization import rms_init
from flax import nnx
from nnxrl.utils.brax_wrapper import wrap_for_training
import nnxrl.utils.logger as wandb
from typing import Sequence, Literal

@dataclasses.dataclass
class Args:
    env_id: str = "BallInCup"
    seed: int = 0
    num_envs: int = 1024
    total_timesteps: int = int(0.5e7)
    buffer_size: int = int(1e6)
    policy_frequency: int = 8
    learning_starts: int = int(1e5)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 512
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    normalize_observation: Literal[True, False] = True
    grad_step_per_env_step: int = 32
    num_evals: int = 10
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.5
    hidden_dim: Sequence[int] = (512, 256, 128)



def main():

    print("🚀 td3 training")
    print("=" * 60)

    args = tyro.cli(Args)
    env = wrap_for_training(
        args.env_id,
        reward_gamma=args.gamma
    )
    wandb.init(project='td3', config=vars(args), name=f'{args.env_id}')
    start_time = time.time()

    def train_log_fn(avg_metrics, global_step):
        avg_metrics["time"] = time.time() - start_time
        wandb.log(avg_metrics, int(global_step), commit=False)
        if "eval/episode_return" in avg_metrics:
            print(
                f"eval/episode_return:{float(avg_metrics['eval/episode_return']):.3f},"
                f"eval/episode_length:{float(avg_metrics.get('eval/episode_length', 0.0)):.3f},"
                f"time:{time.time() - start_time:.3f},"
                f"global_step:{global_step}"
            )

    rngs = nnx.Rngs(args.seed)
    actor = Actor(env.observation_size, env.action_size, rngs.fork())
    critic = DoubleCritic(env.observation_size, env.action_size, rngs.fork(), hidden_dim=args.hidden_dim, layer_norm=True)
    target_actor = copy_model(actor)
    target_critic = copy_model(critic)


    actor_opt = nnx.Optimizer(actor, optax.adam(args.policy_lr))

    critic_opt = nnx.Optimizer(critic, optax.adam(args.q_lr))

    buffer_state = init_replay_buffer(
        env.observation_size,
        (env.action_size,),
        args.buffer_size,
        n_envs=args.num_envs
    )
    if args.normalize_observation:
        rms = rms_init(env.observation_size)
    else:
        rms = None

    train_state = TrainState(actor, critic, target_actor, target_critic, actor_opt, critic_opt, buffer_state, rms)

    train = nnx.jit(make_train(env, args, train_log_fn), donate_argnums=0)
    train(train_state)
    wandb.finish()


if __name__ == "__main__":
    main()
