import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from model.smolpi import SmolPI, Observation, SmolPIConfig
from environment import MujocoEnvironment
from contextlib import nullcontext
from objectives import *
from config import Config

OBJECTIVE = MoveForwardRewardModel

def batch_test(baseline=False):
    cfg = Config(
        smolpi=SmolPIConfig(action_dim=2, action_horizon=5, precision=torch.float16),
        cam_sim_output_path="vids/front_cam_vla_batch_test.mp4",
        cam_omni_output_path="vids/omniscient_vla_batch_test.mp4",
    )
    env = MujocoEnvironment(cfg)

    policy = SmolPI(cfg.smolpi).to(cfg.device)
    if not baseline:
        checkpoint = torch.load("./ddpo_training.pt", map_location=cfg.device)
        policy.load_state_dict(checkpoint["policy_state_dict"])

    policy.eval()

    for objective in OBJECTIVE_CLASSES[:5]:
        all_samples = env.rollout(
            policy=policy,
            reward_models=[objective(cfg) for _ in range(cfg.num_parallel_rollouts)],
            write_to_video=True,
        )

        episode_returns = [
            sum(s["reward"] for s in all_samples if s["env_idx"] == env_idx)
            for env_idx in range(cfg.num_parallel_rollouts)
        ]
        mean_episode_return = float(np.mean(episode_returns))
        print(f"Objective: {objective.__name__}, Mean Episode Return: {mean_episode_return:.2f}")

def single_test():
    cfg = Config(
        smolpi=SmolPIConfig(action_dim=2, action_horizon=5, precision=torch.float16),
        num_parallel_rollouts=1,
        cam_sim_output_path="vids/front_cam_vla_test_rollout.mp4",
        cam_omni_output_path="vids/omniscient_vla_test_rollout.mp4",
        sim_duration_sec=12,
    )
    env = MujocoEnvironment(cfg)

    policy = SmolPI(cfg.smolpi).to(cfg.device)
    checkpoint = torch.load("./ddpo_training.pt", map_location=cfg.device)
    policy.load_state_dict(checkpoint["policy_state_dict"])

    policy.eval()


    all_samples = env.rollout(
        policy=policy,
        reward_models=[OBJECTIVE(cfg)],
        write_to_video=True,
    )
    episode_returns = [
        sum(s["reward"] for s in all_samples if s["env_idx"] == env_idx)
        for env_idx in range(cfg.num_parallel_rollouts)
    ]

    mean_episode_return = float(np.mean(episode_returns))
    print(f"Mean Episode Return: {mean_episode_return:.2f}")

    env.close()


if __name__ == "__main__":
    # print("-- Running All Objectives Test --")
    # batch_test(baseline=False)
    # print("\n-- Running Baseline Test --")
    # batch_test(baseline=True)
    print("-- Running Single Objective Test --")
    single_test()