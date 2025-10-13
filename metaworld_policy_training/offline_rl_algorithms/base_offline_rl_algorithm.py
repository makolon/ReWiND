from typing import Any, ClassVar, Dict, Optional, Tuple, Type, Union
import numpy as np

import torch as th
from gym import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    Schedule,
    RolloutReturn,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv

from offline_rl_algorithms.custom_policies import (
    CustomActor,
    CustomSACPolicy,
    CustomContinuousCritic,
    CustomCnnPolicy,
    CustomMlpPolicy,
    CustomRNNMlpPolicy,
    CustomMultiInputPolicy,
)

from offline_rl_algorithms.offline_replay_buffers import (
    CombinedBuffer,
    ActionChunkedReplayBuffer,
    SuccessFailSplitBuffer,
)

import gym
import threading
from concurrent import futures

import pathlib, io, functools, warnings

import copy

from stable_baselines3.common.utils import check_for_correct_spaces, get_system_info
from stable_baselines3.common.save_util import (
    load_from_zip_file,
)


def recursive_getattr(obj: Any, attr: str, *args) -> Any:
    """
    Recursive version of getattr
    taken from https://stackoverflow.com/questions/31174295

    Ex:
    > MyObject.sub_object = SubObject(name='test')
    > recursive_getattr(MyObject, 'sub_object.name')  # return test
    :param obj:
    :param attr: Attribute to retrieve
    :return: The attribute
    """

    def _getattr(obj: Any, attr: str) -> Any:
        try:
            return getattr(obj, attr, *args)
        except AttributeError:
            print("recursive_getattr failed on " + attr)
            return None

    return functools.reduce(_getattr, [obj, *attr.split(".")])


def recursive_setattr(obj: Any, attr: str, val: Any) -> None:
    """
    Recursive version of setattr
    taken from https://stackoverflow.com/questions/31174295

    Ex:
    > MyObject.sub_object = SubObject(name='test')
    > recursive_setattr(MyObject, 'sub_object.name', 'hello')
    :param obj:
    :param attr: Attribute to set
    :param val: New value of the attribute
    """
    pre, _, post = attr.rpartition(".")
    try:
        return setattr(recursive_getattr(obj, pre) if pre else obj, post, val)
    except AttributeError:
        print("recursive_setattr failed on " + attr)
        return None


from stable_baselines3.common.preprocessing import (
    get_action_dim,
    is_image_space,
    maybe_transpose,
    preprocess_obs,
)
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import (
    get_action_dim,
    is_image_space,
    maybe_transpose,
    preprocess_obs,
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.utils import (
    get_device,
    is_vectorized_observation,
    obs_as_tensor,
)


class ThreadSafePolicy:
    def __init__(self, actor, observation_space, device, _squash_output):
        self.actor = actor
        self.observation_space = observation_space
        self.action_space = actor.action_space
        self.device = device
        self._squash_output = _squash_output

    @property
    def squash_output(self) -> bool:
        """(bool) Getter for squash_output."""
        return self._squash_output

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale`
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def obs_to_tensor(
        self, observation: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Tuple[th.Tensor, bool]:
        """
        Convert an input observation to a PyTorch tensor that can be fed to a model.
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :return: The observation as PyTorch tensor
            and whether the observation is vectorized or not
        """
        vectorized_env = False
        if isinstance(observation, dict):
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                if is_image_space(obs_space):
                    obs_ = maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                vectorized_env = vectorized_env or is_vectorized_observation(
                    obs_, obs_space
                )
                # Add batch dimension if needed
                observation[key] = obs_.reshape(
                    (-1, *self.observation_space[key].shape)
                )

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            vectorized_env = is_vectorized_observation(
                observation, self.observation_space
            )
            # Add batch dimension if needed
            observation = observation.reshape((-1, *self.observation_space.shape))

        observation = obs_as_tensor(observation, self.device)
        return observation, vectorized_env

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """

        # Switch to eval mode (this affects batch norm / dropout)

        observation, vectorized_env = self.obs_to_tensor(observation)
        with th.no_grad():
            actions = self._predict(observation, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

        # now reshape it back to (batch_size, action_sequence_length, action_dim)
        actions = actions.reshape(
            -1, self.actor.action_sequence_length, *self.action_space.shape
        )

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)
        return actions, state

    def _predict(self, observation, deterministic):
        return self.actor(observation, deterministic)


# Thread-safe rollout collection function (not a class method)
def collect_rollouts_threadsafe(
    algo,
    env,
    callback,
    train_freq,
    replay_buffer,
    action_noise=None,
    learning_starts=0,
    log_interval=None,
    policy_lock=None,
    buffer_lock=None,
):
    """
    Collect experiences and store them into a ReplayBuffer.
    This function is designed to be run in a separate thread.
    Locks must be provided for policy and buffer.
    """
    # Switch to eval mode (affects batch norm / dropout)
    # with policy_lock:
    #     algo.policy.set_training_mode(False)
    num_collected_steps, num_collected_episodes = 0, 0
    assert train_freq.frequency > 0, "Should at least collect one step or episode."
    if env.num_envs > 1:
        assert train_freq.unit.name == "STEP", (
            "You must use only one env when doing episodic training."
        )
    if (
        action_noise is not None
        and env.num_envs > 1
        and not hasattr(action_noise, "vectorized")
    ):
        action_noise = VectorizedActionNoise(action_noise, env.num_envs)
    if hasattr(algo.policy, "use_sde") and algo.policy.use_sde:
        algo.policy.actor.reset_noise(env.num_envs)
    # callback.on_rollout_start()
    continue_training = True
    first_step = True
    with policy_lock:
        actor = copy.deepcopy(algo.policy.actor).cpu().to(algo.device)
        actor.eval()
        actor.set_training_mode(False)
        # make a new threadsafe policy that simply has a predict function that calls the actor
        threadsafe_policy = ThreadSafePolicy(
            actor, algo.observation_space, algo.device, algo.policy.squash_output
        )

    while should_collect_more_steps(
        train_freq, num_collected_steps, num_collected_episodes
    ):
        if (
            hasattr(algo.policy, "use_sde")
            and algo.policy.use_sde
            and hasattr(algo.policy, "sde_sample_freq")
            and algo.policy.sde_sample_freq > 0
            and num_collected_steps % algo.policy.sde_sample_freq == 0
        ):
            algo.policy.actor.reset_noise(env.num_envs)
        # Sample action
        with th.inference_mode():
            actions, buffer_actions = algo._sample_action(
                learning_starts,
                action_noise,
                env.num_envs,
                episode_start=np.array([first_step] * env.num_envs),
                policy_lock=None,
                policy=threadsafe_policy,
            )
        first_step = False
        new_obs, rewards, dones, infos = env.step(actions)
        # make fake ones
        # new_obs = np.zeros((env.num_envs, *env.observation_space.shape))
        # rewards = np.zeros(env.num_envs)
        # dones = np.zeros(env.num_envs)
        # infos = [{"action": action} for action in actions]

        if dones[0]:
            first_step = True
        # Action chunking logic if needed
        if hasattr(algo, "action_chunk_size") and algo.action_chunk_size > 1:
            assert "action" in infos[0], "Need action in infos"
            actual_action = np.array([info.get("action") for info in infos])
            buffer_actions = algo.policy.scale_action(actual_action)
        # Update counters
        algo.num_timesteps += env.num_envs

        # NOTE: num_timesteps and other global counters should be handled by main class, not here
        # Store transition in buffer
        # with buffer_lock:
        algo._store_transition(
            replay_buffer, buffer_actions, new_obs, rewards, dones, infos
        )
        callback.update_locals(locals())
        if callback.on_step() is False:
            return False
        algo._update_info_buffer(infos, dones)
        algo._update_current_progress_remaining(
            getattr(algo, "num_timesteps", 0), getattr(algo, "_total_timesteps", 0)
        )
        algo._on_step()
        for idx, done in enumerate(dones):
            if done and action_noise is not None:
                if hasattr(action_noise, "reset"):
                    action_noise.reset()
        num_collected_steps += 1
        for done in dones:
            if done:
                print("Reward:", rewards)
                num_collected_episodes += 1

    # clear memory and such with garbage collection
    import gc

    gc.collect()
    # clean up torch memory
    th.cuda.empty_cache()

    return replay_buffer, num_collected_steps


class OfflineRLAlgorithm(OffPolicyAlgorithm):
    """
    Base OfflineRL Algorithm Class

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param support_multi_env: Whether to support training with multiple environments
    :param warm_start_online_rl: If true, the online RL training will be warm started with the offline trained policy.
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": CustomMlpPolicy,
        "CnnPolicy": CustomCnnPolicy,
        "RnnMlpPolicy": CustomRNNMlpPolicy,
        "MultiInputPolicy": CustomMultiInputPolicy,
    }
    policy: CustomSACPolicy
    actor: CustomActor
    critic: CustomContinuousCritic
    critic_target: CustomContinuousCritic

    def __init__(
        self,
        policy: Union[str, Type[CustomSACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 0,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[spaces.Space]] = (spaces.Box,),
        support_multi_env: bool = True,
        warm_start_online_rl: bool = True,
        action_chunk_size: int = 3,
        success_bonus: float = 0.0,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=supported_action_spaces,
            support_multi_env=support_multi_env,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None

        self.warm_start_online_rl = warm_start_online_rl
        # self.learned_offline = False
        self.learned_offline = True

        self.action_chunk_size = action_chunk_size

        if _init_setup_model:
            self._setup_model()

        if action_chunk_size > 1:
            try:
                self.env.get_attr("chunk_size")
            except AttributeError:
                raise ValueError(
                    "Check if your env is wrapped with ActionChunkingWrapper"
                )
            self.replace_with_chunked_buffer(
                action_chunk_size,
                buffer_size,
                success_bonus=success_bonus,
                evenly_sample_success=True,
            )

    def replace_with_chunked_buffer(
        self,
        action_chunk_size: int,
        buffer_size: int,
        evenly_sample_success: bool = False,
        ratio: float = 0.5,
        success_bonus: float = 0.0,
        pad_action_chunk_with_last_action: bool = True,
    ):
        # Replace the replay buffer with ActionChunkedReplayBuffer
        self.replay_buffer = ActionChunkedReplayBuffer(
            action_chunk_size=action_chunk_size,
            pad_action_chunk_with_last_action=pad_action_chunk_with_last_action,
            buffer_size=buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            optimize_memory_usage=self.optimize_memory_usage,
            success_bonus=success_bonus,
        )

        if evenly_sample_success:
            self.replay_buffer = SuccessFailSplitBuffer(
                original_buffer=self.replay_buffer,
                ratio=ratio,
            )

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()

    def _create_aliases(self) -> None:
        raise NotImplementedError

    def learn_offline(
        self,
        train_steps: int,
        offline_replay_buffer: ReplayBuffer,
        batch_size: int = 64,
        callback: MaybeCallback = None,
    ) -> None:
        # Getting callbacks to work
        # Create eval callback if needed
        # total_timesteps = 0

        if "current_critic_update_ratio" in self.__dict__:
            # switch to offline
            self.critic_update_ratio = self.offline_critic_update_ratio

        total_timesteps, callback = self._setup_learn(
            total_timesteps=train_steps,
            callback=callback,
            reset_num_timesteps=False,
            tb_log_name="offline",
            progress_bar=False,
        )
        total_timesteps *= self.n_envs  # because of a progress bar issue

        callback = self._init_callback(callback, True)
        callback.on_training_start(locals(), globals())

        # Swap replay buffer for offline training
        old_replay_buffer = self.replay_buffer
        self.replay_buffer = offline_replay_buffer

        print("learning offline")
        self.learned_offline = True
        for _ in range(train_steps):
            metrics = self.train(1, batch_size=batch_size, logging_prefix="offline")
            # metrics is a local() which will be updated in callback.update_locals
            callback.update_locals(locals())  # a little hacky
            callback.on_step()  # because of locals, we have access to self.locals['metrics']

        callback.on_training_end()

        self.replay_buffer = old_replay_buffer

    def set_combined_buffer(
        self, offline_replay_buffer: ReplayBuffer, ratio=0.5
    ) -> None:
        self.replay_buffer = CombinedBuffer(
            old_buffer=offline_replay_buffer, new_buffer=self.replay_buffer, ratio=ratio
        )

    def train(
        self, gradient_steps: int, batch_size: int = 64, callback: MaybeCallback = None
    ) -> None:
        raise NotImplementedError

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        logger: Optional = None,
        parallelize: bool = False,
    ):
        if logger is not None:
            super().set_logger(logger)

        if "current_critic_update_ratio" in self.__dict__:
            # switch from offline to online
            self.critic_update_ratio = self.online_critic_update_ratio

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())
        # self.policy.actor.share_memory()

        if parallelize:
            # --- Threading primitives ---
            policy_lock = threading.Lock()
            buffer_lock = threading.Lock()
            steps_at_last_save = self.num_timesteps
            while self.num_timesteps < total_timesteps:
                # Start rollout collection in a thread
                print(self.num_timesteps, total_timesteps)

                empty_callback_list = CallbackList([])
                empty_callback_list = empty_callback_list.init_callback(self)

                with futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        collect_rollouts_threadsafe,
                        self,
                        self.env,
                        # empty_callback_list,
                        callback,
                        self.train_freq,
                        self.replay_buffer,
                        self.action_noise,
                        self.learning_starts,
                        log_interval,
                        policy_lock,
                        buffer_lock,
                    )

                    # Train in main thread while rollout is being collected
                    if self.num_timesteps >= self.learning_starts:
                        gradient_steps = (
                            self.gradient_steps
                            if self.gradient_steps >= 0
                            else self.train_freq.frequency
                        )
                        if gradient_steps > 0:
                            self.train(
                                batch_size=self.batch_size,
                                gradient_steps=gradient_steps,
                                policy_lock=policy_lock,
                            )

                    # Now let's save this!
                    if self.num_timesteps - steps_at_last_save >= 5000:
                        self.save(
                            "./checkpoints/checkpoint_{}".format(self.num_timesteps)
                        )
                        print("Saved checkpoint")
                        steps_at_last_save = self.num_timesteps

                    # Wait for rollout collection to finish and get the return value
                    # NOTE: this code only handles the train_freq=episodes case!
                    self.replay_buffer, num_collected_steps = future.result()

                    callback.update_locals(locals())
                    callback.on_step()

        else:
            # --- Non-Threaded Primitives ---
            while self.num_timesteps < total_timesteps:
                # Collect rollouts
                self.collect_rollouts(
                    self.env,
                    callback,
                    self.train_freq,
                    self.replay_buffer,
                    self.action_noise,
                    self.learning_starts,
                    log_interval,
                )
                # Train
                if self.num_timesteps >= self.learning_starts:
                    gradient_steps = (
                        self.gradient_steps
                        if self.gradient_steps >= 0
                        else self.train_freq.frequency
                    )
                    if gradient_steps > 0:
                        self.train(
                            batch_size=self.batch_size,
                            gradient_steps=gradient_steps,
                        )
        callback.on_training_end()
        return self

    # def _excluded_save_params(self) -> List[str]:
    #     raise NotImplementedError

    # def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
    #     raise NotImplementedError

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
        episode_start: bool = False,
        policy_lock: Optional[threading.Lock] = None,
        policy: Optional[BasePolicy] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This differs from the parent class in that if there are any offline training steps performed, we will warm start the online RL training with the pre-trained policy.

        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if (
            self.num_timesteps < learning_starts
            and not (self.use_sde and self.use_sde_at_warmup)
            and not (self.warm_start_online_rl and self.learned_offline)
        ):
            # Warmup phase
            unscaled_action = np.array(
                [self.action_space.sample() for _ in range(n_envs)]
            )
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            if policy_lock is not None:
                policy_lock.acquire()
            unscaled_action, _ = self.predict(
                self._last_obs,
                deterministic=True,
                episode_start=episode_start,
                policy=policy,
            )
            if policy_lock is not None:
                policy_lock.release()

        # Note: unscaled_action is only None when an existing chunk is being executed.
        # Exists more as a sanity check to ensure code breaks if this condition is true.
        if unscaled_action[0] is not None:
            # Rescale the action from [low, high] to [-1, 1]
            if isinstance(self.action_space, spaces.Box):
                # IF we have a chunked action, we turn it into a batch
                is_chunked = False
                original_shape = unscaled_action.shape
                if unscaled_action.ndim == 3:
                    # Should be of shape (n_envs*chunk_size, action_dim)
                    n_envs = unscaled_action.shape[0]
                    unscaled_action = unscaled_action.reshape(
                        n_envs * unscaled_action.shape[1],
                        unscaled_action.shape[2],
                    )
                    is_chunked = True

                scaled_action = self.policy.scale_action(unscaled_action)

                # print(scaled_action.shape, original_shape, action_noise().shape)
                # Add noise to the action (improve exploration)
                if action_noise is not None and action_noise._sigma > 0:
                    try:
                        if len(original_shape) == 3 and original_shape[0] != 1:
                            scaled_action = np.clip(
                                scaled_action.reshape(original_shape)
                                + action_noise()[:, None, :].repeat(
                                    self.action_chunk_size, axis=1
                                ),
                                -1,
                                1,
                            )

                        else:
                            scaled_action = np.clip(
                                scaled_action + action_noise(), -1, 1
                            )
                    except Exception as e:
                        print("Error adding action noise:", e)
                        breakpoint()

                # print(scaled_action.shape)

                # We store the scaled action in the buffer
                buffer_action = scaled_action

                # now to unscale it, we need to reshape it to be large again
                if len(original_shape) == 3:
                    scaled_action = scaled_action.reshape(
                        n_envs * original_shape[1], original_shape[2]
                    )
                action = self.policy.unscale_action(scaled_action)
                action = action.reshape(original_shape)

                # Now we unbatch the action if it was batched
                if is_chunked:
                    action = action.reshape(
                        n_envs,
                        self.action_chunk_size,
                        unscaled_action.shape[-1],
                    )
                    buffer_action = buffer_action.reshape(
                        n_envs,
                        self.action_chunk_size,
                        unscaled_action.shape[-1],
                    )

            else:
                # Discrete case, no need to normalize or clip
                # We do not support action chunking here yet, so this will crash.
                buffer_action = unscaled_action
                action = buffer_action
            return action, buffer_action

        # This is [None] when an existing chunk is being executed
        # info['action'] will be used to get the actual action later
        return unscaled_action, unscaled_action

    def get_log_prob(self, distribution, actions: th.Tensor) -> th.Tensor:
        # handles getting log prob even in action chunked case where we average among the chunk
        if actions.ndim == 3:
            # actions_for_logprob = actions.reshape(
            #     actions.shape[0] * actions.shape[1],
            #     actions.shape[2],
            # )
            log_prob = distribution.log_prob(actions)
            # log_prob = log_prob.reshape(
            #     actions.shape[0],
            #     actions.shape[1],
            # )
            log_prob = log_prob.mean(dim=1, keepdim=False)
        else:
            log_prob = distribution.log_prob(actions)
        return log_prob

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        env: Optional[GymEnv] = None,
        policy: Optional[BasePolicy] = None,
    ):
        if policy is None:
            policy = self.policy

        if env is None:
            env = self.env
        if self.action_chunk_size > 1:
            # assert self.n_envs == 1, "Action chunking only supported for single env"
            assert episode_start is not None, "Need episode_start for action chunking"
            envs_to_predict_for = []
            for i in range(env.num_envs):
                # episode_start[i] = True
                # print(f"dense_eval, {env.envs[0].dense_eval}")
                # print(f"episode_start: {episode_start[i]}, chunk: {env.envs[i].chunk}, is_chunk_empty: {env.envs[i].is_chunk_empty}")
                if env.envs[i].is_chunk_empty:
                    env.envs[i].chunk = []
                    envs_to_predict_for.append(i)
                    # print(f"envs_to_predict_for 1: {envs_to_predict_for}") # [0]
                # if env.envs[i].is_chunk_empty:
                #     # print("calling predict")
                #     envs_to_predict_for.append(i)
                #     print(f"envs_to_predict_for 2: {envs_to_predict_for}") # [0]
                    # try:
                    #     action, _ = super().predict(
                    #         observation, state, episode_start, deterministic
                    #     )
                    #     return action, _
                    # except Exception as e:
                    #     print("Exception in predict:", e)
                    #     return [None] * env.num_envs, None
                # else:
                #     # print("not calling predict")
                #     return [None] * env.num_envs, None
            # print("calling predict")
            #print(f"chunk: {env.envs[0].chunk}")
            # print(f"envs_to_predict_for 3: {envs_to_predict_for}") # [0]
            if len(envs_to_predict_for) == 0:
                #print("No envs to predict for")
                # breakpoint()
                return [None] * env.num_envs, None

        action, _ = policy.predict(observation, state, episode_start, deterministic)
        # action, _ = super().predict(observation, state, episode_start, deterministic)
        return action, _

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        # Only support 1 env
        # assert env.num_envs == 1, "Only support 1 env"

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, (
                "You must use only one env when doing episodic training."
            )

        # Vectorize action noise if needed
        if (
            action_noise is not None
            and env.num_envs > 1
            and not isinstance(action_noise, VectorizedActionNoise)
        ):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True
        first_step = True
        while should_collect_more_steps(
            train_freq, num_collected_steps, num_collected_episodes
        ):
            # print(
            #     train_freq.unit,
            #     train_freq.frequency,
            #     num_collected_steps,
            #     num_collected_episodes,
            # )

            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and num_collected_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(
                learning_starts,
                action_noise,
                env.num_envs,
                episode_start=np.array([first_step] * env.num_envs),
            )

            first_step = False
            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            # Reset
            if self.action_chunk_size > 1:
                # Check for infos['action']
                assert "action" in infos[0], "Need action in infos"

                # actual_action = infos[0].get("action")[None, :]
                # let's support multiple envs
                actual_action = np.array([info.get("action") for info in infos])
                buffer_actions = self.policy.scale_action(actual_action)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(
                    num_collected_steps * env.num_envs,
                    num_collected_episodes,
                    continue_training=False,
                )

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(
                replay_buffer, buffer_actions, new_obs, rewards, dones, infos
            )

            self._update_current_progress_remaining(
                self.num_timesteps, self._total_timesteps
            )

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if (
                        log_interval is not None
                        and self._episode_num % log_interval == 0
                    ):
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(
            num_collected_steps * env.num_envs,
            num_collected_episodes,
            continue_training,
        )

    def set_parameters(
        self,
        load_path_or_dict: Union[str, Dict[str, Dict]],
        exact_match: bool = True,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """
        Load parameters from a given zip-file or a nested dictionary containing parameters for
        different modules (see ``get_parameters``).

        :param load_path_or_iter: Location of the saved data (path or file-like, see ``save``), or a nested
            dictionary containing nn.Module parameters used by the policy. The dictionary maps
            object names to a state-dictionary returned by ``torch.nn.Module.state_dict()``.
        :param exact_match: If True, the given parameters should include parameters for each
            module and each of their parameters, otherwise raises an Exception. If set to False, this
            can be used to update only specific parameters.
        :param device: Device on which the code should run.
        """
        params = None
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        # Keep track which objects were updated.
        # `_get_torch_save_params` returns [params, other_pytorch_variables].
        # We are only interested in former here.
        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception as e:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name} is an invalid object name.") from e

            if isinstance(attr, th.optim.Optimizer):
                # Optimizers do not support "strict" keyword...
                # Seems like they will just replace the whole
                # optimizer state with the given one.
                # On top of this, optimizer state-dict
                # seems to change (e.g. first ``optim.step()``),
                # which makes comparing state dictionary keys
                # invalid (there is also a nesting of dictionaries
                # with lists with dictionaries with ...), adding to the
                # mess.
                #
                # TL;DR: We might not be able to reliably say
                # if given state-dict is missing keys.
                #
                # Solution: Just load the state-dict as is, and trust
                # the user has provided a sensible state dictionary.
                attr.load_state_dict(params[name])
            else:
                # Assume attr is th.nn.Module
                try:
                    attr.load_state_dict(params[name], strict=exact_match)
                except:
                    print("load_state_dict failed on " + name)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )

    def load(  # noqa: C901
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        load_torch_params_only: bool = False,
        **kwargs,
    ):
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        if not load_torch_params_only:
            # Remove stored device information and replace with ours
            if "policy_kwargs" in data:
                if "device" in data["policy_kwargs"]:
                    del data["policy_kwargs"]["device"]
                # backward compatibility, convert to new format
                if (
                    "net_arch" in data["policy_kwargs"]
                    and len(data["policy_kwargs"]["net_arch"]) > 0
                ):
                    saved_net_arch = data["policy_kwargs"]["net_arch"]
                    if isinstance(saved_net_arch, list) and isinstance(
                        saved_net_arch[0], dict
                    ):
                        data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

            if (
                "policy_kwargs" in kwargs
                and kwargs["policy_kwargs"] != data["policy_kwargs"]
            ):
                raise ValueError(
                    f"The specified policy kwargs do not equal the stored policy kwargs."
                    f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
                )

            if "observation_space" not in data or "action_space" not in data:
                raise KeyError(
                    "The observation_space and action_space were not given, can't verify new environments"
                )

            if env is not None:
                # Wrap first if needed
                env = self._wrap_env(env, data["verbose"])
                # Check if given env is valid
                check_for_correct_spaces(
                    env, data["observation_space"], data["action_space"]
                )
                # Discard `_last_obs`, this will force the env to reset before training
                # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
                if force_reset and data is not None:
                    data["_last_obs"] = None
                # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
                if data is not None:
                    data["n_envs"] = env.num_envs
            else:
                # Use stored env, if one exists. If not, continue as is (can be used for predict)
                if "env" in data:
                    env = data["env"]

            # noinspection PyArgumentList
            model = self.__class__(  # pytype: disable=not-instantiable,wrong-keyword-args
                policy=data["policy_class"],
                env=env,
                device=device,
                _init_setup_model=False,  # pytype: disable=not-instantiable,wrong-keyword-args
            )

            # load parameters
            model.__dict__.update(data)
            model.__dict__.update(kwargs)
            model._setup_model()
        else:
            model = self

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=False, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(
                e
            ) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e

        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, name + ".data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # pytype: disable=attribute-error
        return model
