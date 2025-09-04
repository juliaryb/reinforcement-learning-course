from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .utils import combined_shape


class BaseBuffer(ABC):
    @abstractmethod
    def __init__(
        self, env: gym.Env, size: int = 100000, device: Optional[torch.device] = None
    ) -> None:
        self.device = device

        self.actions = torch.zeros(
            combined_shape(size, env.action_space.shape),
            dtype=torch.float32,
            device=device,
        )
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.terminations = torch.zeros(size, dtype=torch.float32, device=device)
        self.truncations = torch.zeros(size, dtype=torch.float32, device=device)
        self.infos = np.empty((size,), dtype=object)
        self._ptr, self.size, self.max_size = 0, 0, size # the pointer to where the next transition will be stored

    def store(
        self,
        observation: Union[NDArray, dict[str, NDArray]],
        action: NDArray,
        reward: float,
        next_observation: Union[NDArray, dict[str, NDArray]],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        self._store_observations(observation, next_observation)
        self.actions[self._ptr] = torch.as_tensor(action, dtype=torch.float32)
        self.rewards[self._ptr] = torch.as_tensor(reward, dtype=torch.float32)
        self.terminations[self._ptr] = torch.as_tensor(terminated, dtype=torch.float32)
        self.truncations[self._ptr] = torch.as_tensor(truncated, dtype=torch.float32)
        self.infos[self._ptr] = info
        self._ptr = (self._ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    @abstractmethod
    def _store_observations(
        self,
        observation: Union[NDArray, dict[str, NDArray]],
        next_observation: Union[NDArray, dict[str, NDArray]],
    ) -> None: ...

    def sample_batch(
        self, batch_size: int = 32
    ) -> dict[str, Union[Tensor, dict[str, Tensor]]]:
        idxs = torch.randint(0, self.size, size=(batch_size,))
        # idxs = np.random.randint(0, self.size, size=batch_size)
        return self.batch(idxs)

    def batch(self, idxs: Tensor) -> dict[str, Union[Tensor, dict[str, Tensor]]]:
        data = dict(
            action=self.actions[idxs],
            reward=self.rewards[idxs],
            terminated=self.terminations[idxs],
            truncated=self.truncations[idxs],
            info=self.infos[idxs],
        )
        observations = self._observations_batch(idxs)
        data.update(observations)

        return data

    @abstractmethod
    def _observations_batch(
        self, idxs: Tensor
    ) -> dict[str, Union[Tensor, dict[str, Tensor]]]: ...

    def start_episode(self):
        pass

    def end_episode(self):
        pass

    def clear(self):
        self.actions.zero_()
        self.rewards.zero_()
        self.terminations.zero_()
        self.truncations.zero_()
        self.infos.fill(None)
        self._ptr, self.size = 0, 0


class DictReplayBuffer(BaseBuffer):
    """
    A dictionary experience replay buffer for off-policy agents.
    """

    def __init__(
        self, env: gym.Env, size: int = 100000, device: Optional[torch.device] = None
    ):
        assert isinstance(env.observation_space, gym.spaces.Dict)
        super().__init__(env=env, size=size, device=device)

        obs_space = {
            k: combined_shape(size, v.shape) for k, v in env.observation_space.items()
        }

        self.observations: dict[str, Tensor] = {
            k: torch.zeros(obs_space[k], dtype=torch.float32, device=device)
            for k, v in env.observation_space.items()
        }
        self.next_observations: dict[str, Tensor] = {
            k: torch.zeros(obs_space[k], dtype=torch.float32, device=device)
            for k, v in env.observation_space.items()
        }

    def _store_observations(
        self,
        observation: dict[str, NDArray],
        next_observation: dict[str, NDArray],
    ) -> None:
        for k in observation.keys():
            self.observations[k][self._ptr] = torch.as_tensor(
                observation[k], dtype=torch.float32
            )
        for k in next_observation.keys():
            self.next_observations[k][self._ptr] = torch.as_tensor(
                next_observation[k], dtype=torch.float32
            )

    def _observations_batch(self, idxs: Tensor) -> dict[str, dict[str, Tensor]]:
        return dict(
            observation={k: v[idxs] for k, v in self.observations.items()},
            next_observation={k: v[idxs] for k, v in self.next_observations.items()},
        )




class HerReplayBuffer(DictReplayBuffer):
    def __init__(
        self,
        env: gym.Env,
        size: int = 100000,
        device: Optional[torch.device] = None,
        n_sampled_goal: int = 1,
        goal_selection_strategy: str = "final",
    ):
        super().__init__(env=env, size=size, device=device)
        self.env = env
        self.n_sampled_goal = n_sampled_goal
        self.selection_strategy = goal_selection_strategy
        # TODO: fill this in
        # You can put additional attributes here if needed.
        # Also: There is a number of methods in the base class that could be useful to override.
        
        self.episodes = []  # list of completed episodes
        self.current_episode = None  # will be filled in by start_episode()
   
    def store(
        self,
        observation: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_observation: dict[str, torch.Tensor],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ):
        # TODO: fill this in
        # Just a suggestion: it may make sense to modify this method
        
        # IMPORTANT: Store the transition - NO because we don't want to store every transition every step - just full episodes?
        # super().store(
        #     observation=observation,
        #     action=action,
        #     reward=reward,
        #     next_observation=next_observation,
        #     terminated=terminated,
        #     truncated=truncated,
        #     info=info,
        # )
        done = terminated or truncated
        self.current_episode["observations"].append(observation)
        self.current_episode["actions"].append(action)
        self.current_episode["rewards"].append(reward)
        self.current_episode["next_observations"].append(next_observation)
        self.current_episode["terminations"].append(terminated)
        self.current_episode["truncations"].append(truncated)
        self.current_episode["infos"].append(info)
        # TODO: fill this in
        # Or maybe here?

    def start_episode(self):    # this is run when an episode starts in a training loop 
        self.current_episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "next_observations": [],
            "terminations": [],
            "truncations": [],
            "infos": [],
        }

    def end_episode(self):
        if self.selection_strategy == "future":
            episode = self.current_episode
            ep_len = len(episode["actions"])
            
            # # Sanity check
            # if ep_len == 0:
            #     return

            # For each transition, create n_sampled_goal augmented transitions where the goal is replaced with a future achieved goal
            
            for t in range(ep_len):
                # Store original transition
                self._store_transition_at_index(t, episode)
                
                future_idxs = []
                # Only sample future goals if there are future steps available
                if t + 1 < ep_len:
                    # Define the range of future timesteps
                    possible_future_steps = list(range(t + 1, ep_len))

                    # Sample n_sampled_goal future timesteps (with replacement)
                    future_idxs = np.random.choice(
                        possible_future_steps, 
                        size=self.n_sampled_goal, 
                        replace=True  # allow duplicates
                    )

                # for each sampled future index, relabel the goal
                for future_t in future_idxs:
                    new_goal = episode["next_observations"][future_t]["achieved_goal"]

                    # Copy transition
                    obs = episode["observations"][t].copy()
                    next_obs = episode["next_observations"][t].copy()

                    # Relabel goal
                    obs["desired_goal"] = new_goal
                    next_obs["desired_goal"] = new_goal

                    # Recompute reward
                    reward = self.env.unwrapped.compute_reward(
                        next_obs["achieved_goal"],  # checking if the result is what we wanted to do; this reflects the effect of the action
                        new_goal,
                        info={}
                    )

                    # Store relabeled transition
                    super().store(
                        observation=obs,
                        action=episode["actions"][t],
                        reward=reward,
                        next_observation=next_obs,
                        terminated=episode["terminations"][t],
                        truncated=episode["truncations"][t],
                        info=episode["infos"][t],
                    )
        # reset for the next episode
        # self.current_episode = None

    def _store_transition_at_index(self, t, episode):
        super().store(
            observation=episode["observations"][t],
            action=episode["actions"][t],
            reward=episode["rewards"][t],
            next_observation=episode["next_observations"][t],
            terminated=episode["terminations"][t],
            truncated=episode["truncations"][t],
            info=episode["infos"][t],
        )





