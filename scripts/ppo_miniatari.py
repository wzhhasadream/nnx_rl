import jax.numpy as jnp
from mujoco import Sequence
from nnxrl.agents.ppo.train import TrainState, make_train
from nnxrl.utils.normalization import rms_init
from nnxrl.utils.pgx_wrapper import make_pgx_env
from nnxrl.agents.ppo.network import CategoricalActorCritic
from flax import nnx
import optax
import dataclasses
import tyro
import nnxrl.utils.logger as wandb
import time
from typing import Literal




@dataclasses.dataclass
class Args:
    env_id: Literal[
        "minatar-breakout",
        "minatar-freeway",
        "minatar-space_invaders",
        "minatar-asterix",
        "minatar-seaquest",
    ] = "minatar-breakout"
    seed: int = 0
    num_envs: int =  2048
    total_timesteps: int = int(2e7)
    gamma: float = 0.99
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
    reward_scale: float = 1
    hidden_dim: Sequence = (512, 256, 128)
    num_evals: int = 10
    eval_step: int = 3000   # MinAtar-Freeway takes ~2500 steps per episode, so we need a large eval step count





def main():
    args = tyro.cli(Args)
    num_rollout = int(args.total_timesteps / (args.rollout_steps * args.num_envs))
    if num_rollout < args.num_evals:
        raise ValueError(
            f"num_rollout ({num_rollout}) must be >= num_evals ({args.num_evals}) "
        )
    steps_per_rollout = args.num_minibatches * args.update_epochs
    wandb.init(project='ppo', config=vars(args), name=f'{args.env_id}')
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

    env = make_pgx_env(
        args.env_id
    )
    agent = CategoricalActorCritic(env.observation_size, env.action_size, nnx.Rngs(args.seed), hidden_dim=args.hidden_dim)

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
        rms = rms_init(env.observation_size)
    else:
        rms = None
    train_state = TrainState(agent=agent, opt=optimizer, rms=rms)  # Init trainstate

    train = nnx.jit(make_train(env, args, train_log_fn), donate_argnums = 0)
    final_ts = train(train_state)   # return the final trainstate
    wandb.finish()
    


if __name__ == "__main__":
    main()
