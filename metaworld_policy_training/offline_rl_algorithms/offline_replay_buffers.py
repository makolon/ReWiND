from stable_baselines3.common.buffers import ReplayBuffer, BaseBuffer
import h5py
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, NamedTuple

import numpy as np
import torch as th
from gym import spaces
import os

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

from reward_model.base_reward_model import BaseRewardModel

from copy import deepcopy

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class CombinedBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    mc_returns: th.Tensor
    offline_data_mask: th.Tensor
    valid_length: th.Tensor  # for chunked actions


def compute_debug_reward(state):
    # In debug mode, we apply a manual reward function based on the current state
    state = state
    # # Let us set the task to be to approach a specific goal position
    goal_position = [
        90,
        0,
        0,
        0,
        0,
        -180,
        90,
        0,
        0,
        0,
        0,
        -180,
    ]

    goal_position = np.array(goal_position)

    # Reward is L2 distance to the goal position from state
    # reward = -torch.norm(state - goal_position)

    # The positions are rotations of motors, so we want the average degree difference
    difference = np.abs(state - goal_position)
    # Bound the difference to 180 degrees
    difference = np.minimum(difference, 180 - difference)

    reward = -np.sum(difference) / 12
    return reward


class H5ReplayBuffer(ReplayBuffer):
    """
    Replay buffer that can create an HDF5 dataset to store the transitions.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    mc_returns: np.ndarray
    offline_data_mask: np.ndarray

    def __init__(
        self,
        h5_path: str,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        success_bonus: float = 0.0,
        add_timestep: bool = False,
        use_language_embeddings: bool = True,
        calculate_mc_returns: bool = False,
        mc_return_gamma: float = 0.99,
        clip_actions: bool = False,
        sparsify_rewards: bool = False,
        dense_rewards_at_end: bool = False,
        filter_instructions: List[str] = None,
        reward_model: BaseRewardModel = None,
        is_state_based: bool = False,
        use_proprio: bool = False,
        reward_divisor: float = 1.0,
        is_metaworld: bool = False,
        normalize_actions_koch: bool = False,
        action_chunk_size: int = 1,
        pad_action_chunk_with_last_action: bool = True,
        debug_koch: bool = False,
    ):
        """
        Initialize the replay buffer.

        :param h5_path: Path to the HDF5 file that stores the transitions
        :param device: PyTorch device to store the transitions
        :param n_envs: Number of parallel environments
        :param success_bonus: Success bonus added to the rewards
        :param add_timestep: Add a column with the timesteps to the transitions
        :param use_language_embeddings: Whether to specifically incorporate language embeddings into the observations
        :param calculate_mc_returns: Whether to calculate the Monte-Carlo returns
        :param mc_return_gamma: The discount factor for the Monte-Carlo returns
        :param clip_actions: Whether to clip the actions to the action space to [-1, 1]
        :param sparsify_rewards: Converts reward to done
        :param dense_rewards_at_end: Whether to use the reward sum at the end of the episode instead.
        :param action_chunk_size: The size of the action chunk to use
        """
        assert not (dense_rewards_at_end and sparsify_rewards), (
            "Cannot use both dense rewards at end and sparsify as a precaution"
        )

        print(f"Loading transitions from {h5_path}")
        images = None
        with h5py.File(h5_path, "r") as f:
            observations = f["img_embedding"][()]
            lang_embeddings = f["policy_lang_embedding"][()]
            next_observations = observations
            actions = f["action"][()]

            # if 'img' in f.keys() and image_encoder is not None:
            # images = f["img"][()]

            # if normalize_actions_koch:
            # actions = actions.astype(np.float32) / 3.0  # Normalize the actions

            # if clip_actions:
            # actions = np.clip(actions, -3, 3)
            # actions /= 3.0

            if normalize_actions_koch:
                # actions /= 180  # normalize between -1 and 1
                # actions = np.clip(actions, -1, 1)
                low = -270
                high = 270

                actions = 2.0 * ((actions - low) / (high - low)) - 1.0

            # actions = -actions

            if sparsify_rewards:
                rewards = f["done"][()]

                # for each 1 in rewards, make the 3 previous frames also 1
                for i in range(len(rewards)):
                    if rewards[i] == 1:
                        for j in range(3):
                            if i - j >= 0:
                                rewards[i - j] = 1

                rewards = rewards.astype(np.float32)
            else:
                rewards = f["rewards"][()]

            if debug_koch:
                rewards = []
                for i in range(len(observations)):
                    rewards.append(compute_debug_reward(observations[i]))
                rewards = np.array(rewards)

            dones = f["done"][()]

            # timesteps = f["timesteps"][()]

            self.is_state_based = is_state_based
            # Process and save images if they are going to be used
            # if not self.is_state_based:
            # image_encoder_preprocessed_path = h5_path.replace(
            #     ".h5", f"_{image_encoder.name}_preprocessed.h5"
            # )
            # # replace "updated_trajs" with "image_encoder_preprocessed"
            # image_encoder_preprocessed_path = (
            #     image_encoder_preprocessed_path.replace(
            #         "updated_trajs", "image_encoder_preprocessed"
            #     )
            # )

            # # Check if the preprocessed file exists
            # try:
            #     with h5py.File(image_encoder_preprocessed_path, "r") as image_f:
            #         encoded = image_f["encoded"][()]

            #     print(
            #         f"Found preprocessed images for {image_encoder.name} in {image_encoder_preprocessed_path}"
            #     )
            # except:
            #     # If not, pre-process the images and save them
            #     images = f["img"]  # Lazy loading with h5py
            #     encoded = image_encoder.encode_images(images)
            #     # create folder if it doesn't exist
            #     os.makedirs(
            #         os.path.dirname(image_encoder_preprocessed_path), exist_ok=True
            #     )
            #     with h5py.File(image_encoder_preprocessed_path, "w") as image_f:
            #         image_f.create_dataset("encoded", data=encoded)

            #     print(
            #         f"Saved preprocessed images for {image_encoder.name} in {image_encoder_preprocessed_path}"
            #     )

            if not self.is_state_based:
                image_encodings_dict = {}
                # img_embedding_{i} is the key for the image encoding
                for key in f.keys():
                    if "img_embedding" in key:
                        image_encodings_dict[key] = f[key][()]

                # Append them together in order
                image_encodings = []
                # Always sort to make sure they are in the same order

                print("Loading image keys in this order:")
                for key in sorted(image_encodings_dict.keys()):  # sort by key
                    print("Loading key:", key)
                    image_encodings.append(image_encodings_dict[key])
                image_encodings = np.concatenate(image_encodings, axis=1)

                # If we're using images, let's replace the observations with the encoded
                # proprio is the first 4 observations
                if is_metaworld:
                    proprio = observations[:, :4]
                else:
                    proprio = observations

                img_obs = image_encodings
                if use_proprio:
                    img_obs = np.concatenate((image_encodings, proprio), axis=1)

                observations = img_obs
                next_observations = img_obs

            if filter_instructions is not None and len(filter_instructions) > 0:
                instructions = f["env_id"][()]
                indices_to_keep = []
                for i in range(len(instructions)):
                    if instructions[i].decode("utf-8") in filter_instructions:
                        indices_to_keep.append(i)
                observations = observations[indices_to_keep]
                lang_embeddings = lang_embeddings[indices_to_keep]
                next_observations = next_observations[indices_to_keep]
                actions = actions[indices_to_keep]
                rewards = rewards[indices_to_keep]
                dones = dones[indices_to_keep]
            else:
                indices_to_keep = np.arange(observations.shape[0])

            self.indices_to_keep = np.array(indices_to_keep, dtype=int)

        # Use the reward divisor
        rewards /= reward_divisor

        if dense_rewards_at_end:
            new_rewards = np.zeros_like(rewards)
            prev_start = 0
            for i in range(len(rewards)):
                if dones[i] == 1:
                    new_rewards[i] = np.sum(rewards[prev_start:i])
                    prev_start = i

            rewards = new_rewards

        # calculate monte-carlo returns
        self.mc_returns = None
        if calculate_mc_returns:
            # calculate discounted return-to-go for each timestep by using rewards and done
            mc_returns = np.zeros_like(rewards)
            prev_return = 0
            for i in range(len(rewards)):
                mc_returns[-i - 1] = rewards[-i - 1] + mc_return_gamma * prev_return * (
                    1 - dones[-i - 1]
                )
                prev_return = mc_returns[-i - 1]
            self.mc_returns = mc_returns

        # TODO: Temporary, but set timesteps to be going from 0-n until it hits a done of 1
        timesteps = np.zeros_like(rewards)
        current_timestep = 0
        for i in range(len(rewards)):
            if dones[i] == 1:
                timesteps[i] = current_timestep
                current_timestep = 0
            else:
                timesteps[i] = current_timestep
                current_timestep += 1

        # Because of noise in the reward function, we will iterate
        # through the obs/actions/rewards/dones/etc per-episode (based on the dones)
        # and then shorten them so that the highest reward is where the done occurs
        # # --- Begin: Truncate episodes at highest reward ---
        # episode_start = 0
        # new_obs = []
        # new_next_obs = []
        # new_actions = []
        # new_rewards = []
        # new_dones = []
        # new_timesteps = []
        # new_lang_embeddings = []
        # N = len(rewards)
        # num_episodes = 0

        # episode_lengths = []
        # for i in range(N):
        #     if dones[i] == 1:
        #         episode_end = i + 1
        #         ep_rewards = rewards[episode_start:episode_end]
        #         # hacky solution for now, but replace the last item of ep_rewards with the second to last one
        #         ep_rewards[-1] = ep_rewards[-2]
        #         max_idx = int(np.argmax(ep_rewards))
        #         print(
        #             "Max reward index:",
        #             max_idx,
        #             "Episode length:",
        #             episode_end - episode_start,
        #             "Max reward:",
        #             ep_rewards[max_idx],
        #         )
        #         trunc_end = episode_start + max_idx + 1  # include max reward index
        #         # NOTE: uncomment line above to truncate. Below will not truncate
        #         # trunc_end = episode_end
        #         # Copy truncated episode data
        #         episode_lengths.append(trunc_end - episode_start)
        #         new_obs.append(observations[episode_start:trunc_end])
        #         new_next_obs.append(next_observations[episode_start:trunc_end])
        #         new_actions.append(actions[episode_start:trunc_end])
        #         new_rewards.append(rewards[episode_start:trunc_end])
        #         # Set done to 0 except last one
        #         ep_dones = np.zeros(trunc_end - episode_start, dtype=dones.dtype)
        #         ep_dones[-1] = 1.0
        #         new_dones.append(ep_dones)
        #         new_timesteps.append(timesteps[episode_start:trunc_end])
        #         new_lang_embeddings.append(lang_embeddings[episode_start:trunc_end])
        #         num_episodes += 1
        #         episode_start = episode_end
        # print("Number of episodes:", num_episodes)
        # print("Mean episode length:", np.mean(episode_lengths))
        # # Concatenate all episodes
        # if new_obs:
        #     observations = np.concatenate(new_obs, axis=0)
        #     next_observations = np.concatenate(new_next_obs, axis=0)
        #     actions = np.concatenate(new_actions, axis=0)
        #     rewards = np.concatenate(new_rewards, axis=0)
        #     dones = np.concatenate(new_dones, axis=0)
        #     timesteps = np.concatenate(new_timesteps, axis=0)
        #     lang_embeddings = np.concatenate(new_lang_embeddings, axis=0)
        # # --- End: Truncate episodes at highest reward ---

        self.optimize_memory_usage = False

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions.astype(np.float32)
        self.rewards = rewards.astype(np.float32)
        self.dones = dones.astype(np.float32)
        self.timesteps = timesteps
        self.lang_embeddings = np.squeeze(lang_embeddings)

        self.buffer_size = self.rewards.shape[0]
        # self.buffer_size = len(self.indices_to_keep)
        self.success_bonus = success_bonus

        self.pos = self.buffer_size
        self.full = True
        self.device = get_device(device)

        self.add_timestep = add_timestep
        self.use_language_embeddings = use_language_embeddings
        self.calculate_mc_returns = calculate_mc_returns
        self.action_chunk_size = action_chunk_size
        self.pad_action_chunk_with_last_action = pad_action_chunk_with_last_action

    def add(
        self,
        *args,
        **kwargs,
    ) -> None:
        raise (NotImplementedError, "We cannot add transitions to an H5ReplayBuffer")

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        return self.buffer_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> CombinedBufferSamples:
        # Batch inds are in sampling indices_to_sample. Get the actual indices
        # batch_inds = np.array([self.indices_to_keep[i] for i in batch_inds])

        # Sample randomly the env idx
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[
                    (batch_inds + self.action_chunk_size) % self.buffer_size, :
                ],
                env=None,
            )
            # add timestep into the observation
            if self.add_timestep:
                timesteps = (
                    self.timesteps[
                        (batch_inds + self.action_chunk_size) % self.buffer_size
                    ]
                    / 500
                )  # 500 is the max episode length
                next_obs = np.concatenate((next_obs, timesteps.reshape(-1, 1)), axis=1)

            if self.use_language_embeddings:
                next_obs = np.concatenate(
                    (
                        self.lang_embeddings[
                            (batch_inds + self.action_chunk_size) % self.buffer_size, :
                        ],
                        next_obs,
                    ),
                    axis=1,
                )

        else:
            next_obs = self._normalize_obs(
                self.next_observations[
                    (batch_inds + self.action_chunk_size - 1) % self.buffer_size, :
                ],
                env=None,
            )
            if self.add_timestep:
                timesteps = (
                    self.timesteps[
                        (batch_inds + self.action_chunk_size) % self.buffer_size
                    ]
                    / 500
                )  # 500 is the max episode length
                next_obs = np.concatenate((next_obs, timesteps.reshape(-1, 1)), axis=1)

            if self.use_language_embeddings:
                # assumes language is the same throughout the entire epipsode
                next_obs = np.concatenate(
                    (self.lang_embeddings[batch_inds, :], next_obs), axis=1
                )

        observation = self._normalize_obs(self.observations[batch_inds, :], env=None)

        # add the timestep into the observation
        if self.add_timestep:
            assert not self.action_chunk_size > 1, "not supported with action chunking"
            timesteps = (
                self.timesteps[batch_inds] / 500
            )  # 500 is the max episode length
            observation = np.concatenate(
                (observation, timesteps.reshape(-1, 1)), axis=1
            )

        if self.use_language_embeddings:
            observation = np.concatenate(
                (self.lang_embeddings[batch_inds, :], observation), axis=1
            )

        # set dtype of observations to float32
        observation = observation.astype(np.float32)
        next_obs = next_obs.astype(np.float32)

        window_sizes = np.ones((len(batch_inds),))

        valid_lengths = np.ones((len(batch_inds),))
        if self.action_chunk_size > 1:
            # Create sliding window views for actions, rewards, and dones
            max_len = len(
                self.rewards
            )  # Assuming rewards, actions, and dones are of the same length
            window_size = self.action_chunk_size

            # Batch indices
            batch_inds = np.array(batch_inds)

            # Compute sliding window indices for each batch index
            start_indices = batch_inds[:, None] + np.arange(window_size)

            # Mask indices that go out of bounds
            valid_mask = (start_indices >= 0) & (start_indices < max_len)

            # Fetch the data using advanced indexing
            actions_chunked = np.zeros(
                (len(batch_inds), window_size, self.actions.shape[-1])
            )
            rewards_chunked = np.zeros((len(batch_inds), window_size))
            dones_chunked = np.zeros((len(batch_inds), window_size), dtype=bool)

            valid_indices = np.where(valid_mask, start_indices, 0)
            actions_chunked[:] = self.actions[valid_indices]
            rewards_chunked[:] = self.rewards[valid_indices]
            dones_chunked[:] = self.dones[valid_indices]

            # Find the valid length for each chunk based on dones
            # Calculate the index of the first done=True in each chunk
            first_done_index = np.argmax(dones_chunked, axis=1)
            # Check if any done=True exists in each chunk
            any_done_in_chunk = np.any(dones_chunked, axis=1)
            # If a done exists, the length is index + 1. Otherwise, it's the full window size.
            valid_lengths = np.where(
                any_done_in_chunk, first_done_index + 1, window_size
            )
            window_sizes = np.ones(len(batch_inds)) * window_size

            # Create masks for valid actions, rewards, and dones (mask is True up to *before* the valid_lengths index)
            valid_masks = np.arange(window_size)[None, :] < valid_lengths[:, None]

            # Apply masks to compute padded actions, rewards, and dones
            # Rewards up to and including the step with done=True are summed
            # summed_rewards = np.sum(np.where(valid_masks, rewards_chunked, 0), axis=1)
            rewards_chunked = np.where(valid_masks, rewards_chunked, 0)
            # Done is True if *any* done occurred within the valid length
            any_dones = np.any(np.where(valid_masks, dones_chunked, 0), axis=1)
            # Apply mask for padding actions (actions are padded *after* the valid length)
            padded_actions = np.where(valid_masks[:, :, None], actions_chunked, 0)

            # Handle padding for actions
            last_valid_indices = np.maximum(0, valid_lengths - 1)
            if self.pad_action_chunk_with_last_action:
                # Get the index of the *last valid action* for each chunk
                last_valid_actions = actions_chunked[
                    np.arange(len(valid_lengths)), last_valid_indices
                ]
                # Pad the actions *after* the valid length with the last valid action
                # should_mask = np.any(pad_mask, axis=1)
                # padded_actions[pad_mask] = last_valid_actions[
                # should_mask
                # ]  # Use broadcasting for efficiency

                # we need to expand last_valid_actions to match the shape of padded_actions
                last_valid_actions = last_valid_actions[:, None, :]
                # repeat last_valid_actions to match the shape of padded_actions
                last_valid_actions = np.repeat(last_valid_actions, window_size, axis=1)
                padded_actions = np.where(
                    valid_masks[:, :, None].repeat(padded_actions.shape[-1], axis=2),
                    padded_actions,
                    last_valid_actions,
                )

            # in this case, the last valid reward should also be repeated, without the success bonus since that's already there.
            last_valid_rewards = rewards_chunked[
                np.arange(len(valid_lengths)), last_valid_indices
            ]
            padded_rewards = np.where(
                valid_masks,
                rewards_chunked,
                np.repeat(last_valid_rewards[:, np.newaxis], window_size, axis=1),
            )

            summed_rewards = np.sum(padded_rewards, axis=1)

            # now we add the success bonus to anywhere with dones
            summed_rewards[any_dones] += self.success_bonus

            actions = padded_actions.astype(np.float32)
            rewards = summed_rewards.reshape(-1, 1).astype(np.float32)
            dones = any_dones.astype(np.float32).reshape(-1, 1)

        else:
            rewards = self.rewards[batch_inds].reshape(-1, 1).astype(np.float32)
            dones = self.dones[batch_inds].reshape(-1, 1).astype(np.float32)
            actions = self.actions[batch_inds, :].astype(np.float32)
            window_sizes = np.ones(len(batch_inds)) * 1

            rewards[dones] += self.success_bonus

        if self.calculate_mc_returns:
            mc_returns = self.mc_returns[batch_inds].reshape(-1, 1)
        else:
            mc_returns = rewards
        # # set rewards to have all zeros
        # rewards = np.zeros_like(rewards)

        data = (
            observation,
            actions,
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones,
            rewards,
            mc_returns,
            np.ones_like(rewards),  # offline_data_mask is 1 for all offline data,
            window_sizes,
        )

        return CombinedBufferSamples(*tuple(map(self.to_torch, data)))

    def clone(self):
        # we should clone this class but we have to make sure not to
        # deep copy ['action_space', and 'observation_space']

        new_buffer = type(self).__new__(type(self))
        output_dict = {}
        for key, value in self.__dict__.items():
            if key in ["action_space", "observation_space"]:
                output_dict[key] = value
            else:
                output_dict[key] = deepcopy(value)
        new_buffer.__dict__ = output_dict
        return new_buffer


class CombinedBuffer(ReplayBuffer):
    def __init__(
        self, old_buffer: ReplayBuffer, new_buffer: ReplayBuffer, ratio: float = 0.5
    ):
        self.old_buffer = old_buffer
        self.new_buffer = new_buffer
        self.ratio = ratio

    def _get_samples(
        self,
        batch_inds: np.ndarray,
    ) -> ReplayBufferSamples:
        return

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Add to new buffer
        self.new_buffer.add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        # print(
        #     "Sampling from CombinedBuffer",
        #     self.old_buffer,
        #     self.old_buffer.size(),
        #     self.new_buffer,
        #     self.new_buffer.size(),
        # )
        old_batch_size = int(batch_size * self.ratio)
        new_batch_size = batch_size - old_batch_size

        old_size = self.old_buffer.size()
        new_size = self.new_buffer.size()

        if old_size == 0 and new_size == 0:
            return CombinedBufferSamples(
                observations=th.empty(0),
                actions=th.empty(0),
                next_observations=th.empty(0),
                dones=th.empty(0),
                rewards=th.empty(0),
                mc_returns=th.empty(0),
                offline_data_mask=th.empty(0),
                valid_length=th.empty(0),
            )

        if old_size == 0:
            new_samples = self.new_buffer.sample(batch_size, env=env)
            old_samples = None
        elif new_size == 0:
            old_samples = self.old_buffer.sample(batch_size, env=env)
            new_samples = None
        else:
            old_batch_size = int(batch_size * self.ratio)
            new_batch_size = batch_size - old_batch_size
            old_samples = self.old_buffer.sample(old_batch_size, env=env)
            new_samples = self.new_buffer.sample(new_batch_size, env=env)
        # Concatenate the samples into old_samples
        cat_names = [
            "observations",
            "actions",
            "next_observations",
            "dones",
            "rewards",
            "mc_returns",
            "offline_data_mask",
            "valid_length",
        ]
        attributes = {}
        for name in cat_names:
            # NOTE: This is incorrect when using this as a success/fail buffer
            if name == "offline_data_mask":
                # 1 for the old data, 0 for the new data
                old_data = th.ones(old_batch_size, 1)
                new_data = th.zeros(new_batch_size, 1)

            # TODO: This is incorrect when using this as a success/fail buffer
            elif name == "mc_returns":
                if old_samples is not None:
                    old_data = getattr(old_samples, name)
                if new_samples is not None:
                    new_data = th.zeros_like(
                        old_data
                    )  # set all mc_returns to 0 for new data as it's currently not supported
            else:
                if old_samples is None:
                    old_data = th.empty(0)
                else:
                    try:
                        old_data = getattr(old_samples, name)
                    except AttributeError:
                        # print("old_samples failed on " + name)
                        old_data = th.ones(old_batch_size).to(old_data.device)
                if new_samples is None:
                    new_data = th.empty(0)
                else:
                    try:
                        new_data = getattr(new_samples, name)
                    except AttributeError:
                        # print("new_samples failed on " + name)
                        new_data = th.ones(new_batch_size).to(old_data.device)

            try:
                if old_samples is not None and new_samples is not None:
                    attributes[name] = th.cat((old_data, new_data), dim=0)
                elif old_samples is not None:
                    attributes[name] = old_data
                elif new_samples is not None:
                    attributes[name] = new_data
            except:
                breakpoint()

        old_samples = CombinedBufferSamples(**attributes)
        return old_samples

    def size(self) -> int:
        """
        :return: The total size of the buffer
        """
        return self.new_buffer.size() + self.old_buffer.size()

    def clone(self):
        return CombinedBuffer(
            old_buffer=self.old_buffer.clone(),
            new_buffer=self.new_buffer.clone(),
            ratio=self.ratio,
        )


class ActionChunkedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        action_chunk_size,
        pad_action_chunk_with_last_action,  # if True, pad the action chunk with the last action otherwise, pad with zeros. zeros is for delta control, last action is for absolute control
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        success_bonus: float = 0.0,
    ):
        super(ActionChunkedReplayBuffer, self).__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
        self.action_chunk_size = action_chunk_size
        self.pad_action_chunk_with_last_action = pad_action_chunk_with_last_action
        self.success_bonus = success_bonus

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[
                    (batch_inds + self.action_chunk_size) % self.buffer_size,
                    env_indices,
                    :,
                ],
                env,
            )
        else:
            # - 1 offset here because the next_observations are the observations after the actions
            next_obs = self._normalize_obs(
                self.next_observations[
                    (batch_inds + self.action_chunk_size - 1) % self.buffer_size,
                    env_indices,
                    :,
                ],
                env,
            )

        valid_lengths = np.ones(len(batch_inds))
        window_sizes = np.ones(len(batch_inds)) * self.action_chunk_size
        if self.action_chunk_size > 1:
            # Create sliding window views for actions, rewards, and dones
            max_len = len(
                self.rewards
            )  # Assuming rewards, actions, and dones are of the same length
            window_size = self.action_chunk_size

            # Batch indices
            batch_inds = np.array(batch_inds)

            # Compute sliding window indices for each batch index
            start_indices = batch_inds[:, None] + np.arange(window_size)

            # Mask indices that go out of bounds
            valid_mask = (start_indices >= 0) & (start_indices < max_len)

            # Fetch the data using advanced indexing
            actions_chunked = np.zeros(
                (len(batch_inds), window_size, self.actions.shape[-1])
            )
            rewards_chunked = np.zeros((len(batch_inds), window_size))
            episode_ends_chunked = np.zeros((len(batch_inds), window_size), dtype=bool)
            successes_chunked = np.zeros((len(batch_inds), window_size), dtype=bool)

            valid_indices = np.where(valid_mask, start_indices, 0)

            # now select based on env_indices
            # Reshape valid_indices to use with self.actions (N, n_envs, action_dim)
            # This will select window elements for each batch item
            reshaped_valid_indices = valid_indices.reshape(-1)

            # Get actions, rewards, and dones for all environments at selected time indices
            temp_actions = self.actions[
                reshaped_valid_indices
            ]  # Shape: (batch_size*window_size, n_envs, action_dim)
            temp_rewards = self.rewards[
                reshaped_valid_indices
            ]  # Shape: (batch_size*window_size, n_envs)
            episode_ends = self.dones[
                reshaped_valid_indices
            ]  # Shape: (batch_size*window_size, n_envs)
            successes = self.dones[reshaped_valid_indices] * (
                1 - self.timeouts[reshaped_valid_indices]
            )

            # Reshape to separate batch and window dimensions
            temp_actions = temp_actions.reshape(
                len(batch_inds), window_size, self.n_envs, -1
            )
            temp_rewards = temp_rewards.reshape(
                len(batch_inds), window_size, self.n_envs
            )
            episode_ends = episode_ends.reshape(
                len(batch_inds), window_size, self.n_envs
            )
            # done only on success
            successes = successes.reshape(len(batch_inds), window_size, self.n_envs)

            # Select specific environment for each batch item
            for i, env_idx in enumerate(env_indices):
                actions_chunked[i] = temp_actions[i, :, env_idx]
                rewards_chunked[i] = temp_rewards[i, :, env_idx]
                episode_ends_chunked[i] = episode_ends[i, :, env_idx]
                successes_chunked[i] = successes[i, :, env_idx]

            # Find the valid length for each chunk based on dones
            # Calculate the index of the first episode_end=True in each chunk (comes from either success or timeout/truncation)
            first_end_index = np.argmax(episode_ends_chunked, axis=1)
            # Check if any episode_end=True exists in each chunk
            any_end_in_chunk = np.any(episode_ends_chunked, axis=1)
            # If a end exists, the length is index + 1. Otherwise, it's the full window size.
            valid_lengths = np.where(any_end_in_chunk, first_end_index + 1, window_size)
            # Create masks for valid actions, rewards, and dones (mask is True up to *before* the valid_lengths index)
            valid_masks = np.arange(window_size)[None, :] < valid_lengths[:, None]

            # Apply masks to compute padded actions, rewards, and dones
            # Rewards up to and including the step with done=True are summed
            # summed_rewards = np.sum(np.where(valid_masks, rewards_chunked, 0), axis=1)
            rewards_chunked = np.where(valid_masks, rewards_chunked, 0)
            # Success is True if *any* success occurred within the valid length
            any_success = np.any(np.where(valid_masks, successes_chunked, 0), axis=1)
            # Apply mask for padding actions (actions are padded *after* the valid length)
            padded_actions = np.where(valid_masks[:, :, None], actions_chunked, 0)

            # Handle padding for actions
            last_valid_indices = np.maximum(0, valid_lengths - 1)
            if self.pad_action_chunk_with_last_action:
                # Get the index of the *last valid action* for each chunk
                last_valid_actions = actions_chunked[
                    np.arange(len(valid_lengths)), last_valid_indices
                ]
                # Pad the actions *after* the valid length with the last valid action
                # should_mask = np.any(pad_mask, axis=1)
                # padded_actions[pad_mask] = last_valid_actions[
                # should_mask
                # ]  # Use broadcasting for efficiency

                # we need to expand last_valid_actions to match the shape of padded_actions
                last_valid_actions = last_valid_actions[:, None, :]
                # repeat last_valid_actions to match the shape of padded_actions
                last_valid_actions = np.repeat(last_valid_actions, window_size, axis=1)
                padded_actions = np.where(
                    valid_masks[:, :, None].repeat(padded_actions.shape[-1], axis=2),
                    padded_actions,
                    last_valid_actions,
                )

            # in this case, the last valid reward should also be repeated WITHOUT success bonus since success = Termination
            last_valid_rewards = (
                rewards_chunked[np.arange(len(valid_lengths)), last_valid_indices]
                - self.success_bonus * any_success
            )
            padded_rewards = np.where(
                valid_masks,
                rewards_chunked,
                np.repeat(last_valid_rewards[:, np.newaxis], window_size, axis=1),
            )

            summed_rewards = np.sum(padded_rewards, axis=1)

            actions = padded_actions.astype(np.float32)
            rewards = summed_rewards.reshape(-1, 1).astype(np.float32)
            dones = any_success.astype(np.float32).reshape(-1, 1)

        else:
            rewards = self.rewards[batch_inds].reshape(-1, 1).astype(np.float32)
            dones = self.dones[batch_inds].reshape(-1, 1).astype(np.float32)
            actions = self.actions[batch_inds, :].astype(np.float32)

        # Compute final results
        all_actions = actions
        all_rewards = rewards
        all_dones = dones
        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            all_actions,
            next_obs,
            all_dones,
            self._normalize_reward(all_rewards.reshape(-1, 1), env),
            rewards,  # set mc_returns to rewards
            np.zeros_like(rewards),  # offline_data_mask is 0 for online data
            window_sizes,  # valid_lengths is the number of valid actions
        )

        # shuffle the data before returning
        data = list(data)
        idx = np.random.permutation(len(data[0]))
        for i in range(len(data)):
            data[i] = data[i][idx]
        data = tuple(data)

        return CombinedBufferSamples(*tuple(map(self.to_torch, data)))

    def clone(self):
        # we should clone this class but we have to make sure not to
        # deep copy ['action_space', and 'observation_space']
        new_buffer = type(self).__new__(type(self))
        output_dict = {}
        for key, value in self.__dict__.items():
            if key in ["action_space", "observation_space"]:
                output_dict[key] = value
            else:
                output_dict[key] = deepcopy(value)
        new_buffer.__dict__ = output_dict
        return new_buffer


class SuccessFailSplitBuffer(CombinedBuffer):
    # A buffer that modifies the "add" function and maintains two buffers
    # One for successful trajs and one for failed trajs
    # That way, we can 50/50 sample from both
    def __init__(
        self,
        original_buffer: ReplayBuffer,
        ratio: float = 0.5,
    ):
        # maintain two new buffers that are a copy of the original
        # one for success and one for failure
        self.success_buffer = original_buffer.clone()
        self.failure_buffer = original_buffer.clone()

        # this is only for compatibility with CombinedBuffer
        self.old_buffer = self.success_buffer
        self.new_buffer = self.failure_buffer

        self.temp_input_output = []

        super().__init__(self.success_buffer, self.failure_buffer, ratio=ratio)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        self.temp_input_output.append((obs, next_obs, action, reward, done, infos))

        # if done, then we add it to the success buffer if success, otherwise we add it to failure buffer
        if any([d for d in done]):
            if any([info.get("success", False) for info in infos]):
                for (
                    obs,
                    next_obs,
                    action,
                    reward,
                    done,
                    infos,
                ) in self.temp_input_output:
                    self.success_buffer.add(obs, next_obs, action, reward, done, infos)
            else:
                for (
                    obs,
                    next_obs,
                    action,
                    reward,
                    done,
                    infos,
                ) in self.temp_input_output:
                    self.failure_buffer.add(obs, next_obs, action, reward, done, infos)
            #print(
            #    f"Success buffer {self.success_buffer.size()}, Failure buffer {self.failure_buffer.size()}"
            #)
            self.temp_input_output = []

    def clone(self):
        buf = SuccessFailSplitBuffer(
            original_buffer=self.success_buffer.clone(),
        )

        buf.success_buffer = self.success_buffer.clone()
        buf.failure_buffer = self.failure_buffer.clone()
        return buf

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        return self.success_buffer.size() + self.failure_buffer.size()

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        # print(
        #     "Sampling from SuccessFailSplitBuffer",
        #     self.success_buffer,
        #     self.success_buffer.size(),
        #     self.failure_buffer,
        #     self.failure_buffer.size(),
        # )
        old_batch_size = int(batch_size * self.ratio)
        new_batch_size = batch_size - old_batch_size

        old_size = self.success_buffer.size()
        new_size = self.failure_buffer.size()

        if old_size == 0 and new_size == 0:
            return CombinedBufferSamples(
                observations=th.empty(0),
                actions=th.empty(0),
                next_observations=th.empty(0),
                dones=th.empty(0),
                rewards=th.empty(0),
                mc_returns=th.empty(0),
                offline_data_mask=th.empty(0),
                valid_length=th.empty(0),
            )

        if old_size == 0:
            new_samples = self.failure_buffer.sample(batch_size, env=env)
            old_samples = None
        elif new_size == 0:
            old_samples = self.success_buffer.sample(batch_size, env=env)
            new_samples = None
        else:
            old_batch_size = int(batch_size * self.ratio)
            new_batch_size = batch_size - old_batch_size
            old_samples = self.success_buffer.sample(old_batch_size, env=env)
            new_samples = self.failure_buffer.sample(new_batch_size, env=env)
        # Concatenate the samples into old_samples
        cat_names = [
            "observations",
            "actions",
            "next_observations",
            "dones",
            "rewards",
            "mc_returns",
            "offline_data_mask",
            "valid_length",
        ]
        attributes = {}
        for name in cat_names:
            # NOTE: This is incorrect when using this as a success/fail buffer
            if name == "offline_data_mask":
                # 1 for the old data, 0 for the new data
                old_data = th.ones(old_batch_size, 1)
                new_data = th.zeros(new_batch_size, 1)

            # TODO: This is incorrect when using this as a success/fail buffer
            elif name == "mc_returns":
                if old_samples is not None:
                    old_data = getattr(old_samples, name)
                if new_samples is not None:
                    new_data = th.zeros_like(
                        old_data
                    )  # set all mc_returns to 0 for new data as it's currently not supported
            else:
                if old_samples is None:
                    old_data = th.empty(0)
                else:
                    try:
                        old_data = getattr(old_samples, name)
                    except AttributeError:
                        # print("old_samples failed on " + name)
                        old_data = th.ones(old_batch_size).to(old_data.device)
                if new_samples is None:
                    new_data = th.empty(0)
                else:
                    try:
                        new_data = getattr(new_samples, name)
                    except AttributeError:
                        # print("new_samples failed on " + name)
                        new_data = th.ones(new_batch_size).to(old_data.device)

            try:
                if old_samples is not None and new_samples is not None:
                    attributes[name] = th.cat((old_data, new_data), dim=0)
                elif old_samples is not None:
                    attributes[name] = old_data
                elif new_samples is not None:
                    attributes[name] = new_data
            except:
                breakpoint()

        old_samples = CombinedBufferSamples(**attributes)
        return old_samples


