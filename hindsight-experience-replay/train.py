# A script to train a Soft Actor-Critic (SAC) agent with Hindsight Experience Replay (HER) on a specified gym environment.
# By default it uses the PandaReach-v3 environment from the panda_gym package.
# But it can be easily modified to use any other gym environment, including those from the gymnasium_robotics package.
# Just uncomment the import statement for gymnasium_robotics and register the environments and use for example the FetchReach-v3 environment.

# Also give it a try on push and pick and place tasks: PandaPush-v3, PandaPickAndPlace-v3, FetchPush-v3, FetchPickAndPlace-v3

import argparse

import gymnasium as gym
# Uncomment the following line to use gymnasium_robotics environments
# import gymnasium_robotics
import panda_gym
import torch
from gymnasium.wrappers import RecordVideo
import os

# Uncomment the following lines to register gymnasium_robotics environments
# gym.register_envs(gymnasium_robotics)

from asdf.algos import SAC
from asdf.buffers import HerReplayBuffer
from asdf.extractors import DictExtractor
from asdf.loggers import TensorboardLogger
from asdf.policies import MlpPolicy

def dir_exists_and_not_empty(path: str) -> bool:
    return os.path.exists(path) and os.path.isdir(path) and bool(os.listdir(path))

# There are two challenges in this exercise:
# 1. Implement the Hindsight Experience Replay (HER) algorithm.
#    This is done in the HerReplayBuffer class.
# 2. Improve the SAC algorithm with an automatically adjusted temperature (alpha) parameter.
#    This is done in the SAC class.
def main(env_id: str) -> None:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU")
    else:
        print("Using CPU")
        device = "cpu"

    # create folders for recordings if they don't exist
    experiment_name = "PandaPush_alpha"

    base_path = os.path.join("outputs", experiment_name)
    videos_path = os.path.join(base_path, "videos")
    videos_train_path = os.path.join(videos_path, "train")
    videos_test_path = os.path.join(videos_path, "test")

    os.makedirs(videos_path, exist_ok=True)
    os.makedirs(videos_train_path, exist_ok=True)
    os.makedirs(videos_test_path, exist_ok=True)

    if dir_exists_and_not_empty(videos_train_path):
        raise RuntimeError(f"Directory '{videos_train_path}' already exists and is not empty.")

    env = gym.make(env_id, render_mode="rgb_array")

    # add recording every frequency episodes - doesn't work xd 
    env = RecordVideo(
        env,
        video_folder=videos_train_path,
        episode_trigger=lambda i: i % 1000 == 0,
        name_prefix=f"{experiment_name}_train"
    )

    # policy = MlpPolicy(
    # env.observation_space,
    # env.action_space,
    # hidden_sizes=[64, 64],
    # extractor_type=DictExtractor,
    # )

    # PandaPush
    policy = MlpPolicy(
        env.observation_space,
        env.action_space,
        hidden_sizes=[512, 512, 512],
        extractor_type=DictExtractor,
    )
    policy.to(device)   # tells PyTorch whether to move to gpu or cpu for matmul during training

    buffer = HerReplayBuffer(
        env=env,
        size=1_000_000,
        n_sampled_goal=3,
        goal_selection_strategy="future",
        device=device,
    )
    logger = TensorboardLogger()
    logger.open()
    
    # algo = SAC(
    #     env,
    #     policy=policy,
    #     buffer=buffer,
    #     update_every=1, # how often to run gradient updates
    #     update_after=1000,  # Start gradient updates only after this many environment steps have been taken
    #     batch_size=64, # how many transitions to sample from the buffer per update
    #     # alpha="auto", # use automatic alpha adjustment (uncoment when implemented)
    #     alpha=0.05, # use fixed alpha (comment out when implementing automatic alpha adjustment)
    #     gamma=0.9,
    #     # polyak=0.95,
    #     lr=1e-4,
    #     logger=logger,
    #     max_episode_len=100,
    #     start_steps=1_000,  # when to start updating the gradients - so how many steps are done not using the actor's predictions but just random action selection
    #     n_updates=None  # number of training (gradient) updates to run per environment step; defaults to update_every if None
    # )

    algo = SAC(
        env,
        policy=policy,
        buffer=buffer,
        update_every=1, # how often to run gradient updates
        update_after=1000,  # Start gradient updates only after this many environment steps have been taken
        batch_size=64, # how many transitions to sample from the buffer per update
        # alpha="auto", # use automatic alpha adjustment (uncoment when implemented)
        alpha="auto", # use fixed alpha (comment out when implementing automatic alpha adjustment)
        gamma=0.9,
        # polyak=0.95,
        lr=1e-4,
        logger=logger,
        max_episode_len=100,
        start_steps=1_000,  # when to start updating the gradients - so how many steps are done not using the actor's predictions but just random action selection
        n_updates=None  # number of training (gradient) updates to run per environment step; defaults to update_every if None
    )

    # algo.train(n_steps=100_000, log_interval=1000, path_to_vid=videos_train_path) # PandaReach
    algo.train(n_steps=500_000, log_interval=1000, path_to_vid=videos_train_path) # PandaPush

    # algo.train(n_steps=100_000, log_interval=100, path_to_vid=videos_train_path)
    env.close()
    logger.close()

    # testing phase - move to cpu
    policy.cpu()
    env = gym.make(env_id, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=videos_test_path,
        episode_trigger=lambda i: i % 5 == 0,
        name_prefix=f"{experiment_name}_test"
    )

    results = algo.test(env, n_episodes=50, sleep=1 / 30)
    env.close()
    print(f"Test reward {results['mean_ep_ret']}, Test episode length: {results['mean_ep_len']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="PandaPush-v3", help="Gym environment ID"
    )

    args = parser.parse_args()

    main(args.env)
