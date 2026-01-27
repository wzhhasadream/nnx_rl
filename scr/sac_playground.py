import dataclasses
from typing import Sequence, Literal
import tyro
import optax
import time
from nnxrl.agents.sac.network import Actor, Alpha
from nnxrl.agents.sac.train import make_train, TrainState
from nnxrl.utils.replaybuffer import init_replay_buffer
from nnxrl.utils.normalization import rms_init
from flax import nnx
from nnxrl.utils.brax_wrapper import wrap_for_training
import nnxrl.utils.logger as wandb
from nnxrl.utils.network import DoubleCritic, copy_model


@dataclasses.dataclass
class Args:
    env_id: str = "BallInCup"
    seed: int = 0
    num_envs: int = 128
    total_timesteps: int = int(0.5e7)
    buffer_size: int = 1048576 * 4
    policy_frequency: int = 1
    target_frequency: int = 1
    learning_starts: int = 8192
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 512
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    alpha: float = 0.2  
    autotune: Literal[True, False] = True
    normalize_observation: Literal[True, False] = True
    grad_step_per_env_step: int = 8
    num_evals: int = 10
    target_entropy: float = 0  # will be set automatically
    hidden_dim: Sequence = (256, 256)



def main():
    
    print("🚀 SAC training")
    print("=" * 60)

    args = tyro.cli(Args)
    env = wrap_for_training(
        args.env_id,
        reward_gamma=args.gamma
    )
    args.target_entropy = -env.action_size / 2
    wandb.init(project='sac', config=vars(args), name=f'{args.env_id}')
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
    actor = Actor(env.observation_size, env.action_size, rngs.fork(), simba_encoder=True)
    critic = DoubleCritic(env.observation_size, env.action_size, rngs.fork(
    ), hidden_dim=args.hidden_dim, simba_encoder=True)
    target_critic = copy_model(critic)
    alpha = Alpha() if args.autotune else None
    actor_opt = nnx.Optimizer(actor, optax.adam(args.policy_lr))
    alpha_opt = nnx.Optimizer(alpha, optax.adam(args.policy_lr)) if args.autotune else None
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

    train_state = TrainState(actor, critic,
                             actor_opt, critic_opt, target_critic, buffer_state, rms, alpha=alpha, alpha_opt=alpha_opt)

    train = nnx.jit(make_train(env, args, train_log_fn), donate_argnums=0)
    train(train_state)
    wandb.finish()




if __name__ == "__main__":
    main()
