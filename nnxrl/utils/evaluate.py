from typing import Callable
import gymnasium
import numpy as np




def evaluate_policy(
    env_creater: Callable,
    policy: Callable[[np.ndarray], np.ndarray],
    eval_episodes: int = 100
) -> dict[str, np.ndarray]:

    envs = gymnasium.vector.SyncVectorEnv(
                [env_creater] * 10
            )
    envs = gymnasium.wrappers.vector.RecordEpisodeStatistics(envs)

    obs, _ = envs.reset()

    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions = policy(obs)

        next_obs, _,  terminated, truncated, infos = envs.step(actions)
        dones = np.logical_or(terminated, truncated)
        if dones.any():
            episodic_returns.extend(infos["episode"]['r'][dones].tolist())



        obs = next_obs

    return np.mean(episodic_returns)

    





