from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F
from torch import nn

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from offline_rl_algorithms.base_offline_rl_algorithm import OfflineRLAlgorithm
from offline_rl_algorithms.custom_policies import create_mlp
from stable_baselines3.common.policies import BasePolicy, BaseModel
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from offline_rl_algorithms.custom_policies import (
    CustomActor,
    CustomSACPolicy,
    CustomCnnPolicy,
    CustomMlpPolicy,
    CustomRNNMlpPolicy,
    CustomMultiInputPolicy,
)

from copy import deepcopy

import threading


class ValueCritic(BaseModel):
    """
    Single Value network (state conditioned) for IQL.

    :param observation_space: Obervation space
    :param action_space: Action space (not used, only for compatibility)
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    features_extractor: BaseFeaturesExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        share_features_extractor: bool = False,
        lr_schedule: Schedule = None,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        use_layer_norm: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.share_features_extractor = share_features_extractor
        v_net_list = create_mlp(
            features_dim, 1, net_arch, activation_fn, use_layer_norm=use_layer_norm
        )
        v_net = nn.Sequential(*v_net_list)
        self.add_module(f"vf", v_net)
        self.optimizer = optimizer_class(
            self.vf.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **optimizer_kwargs,
        )

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        value_input = th.cat([features], dim=1)
        return self.vf(value_input)


class IQL(OfflineRLAlgorithm):
    """
    IQL https://arxiv.org/abs/2110.06169

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
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
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
    :param advantage_temp: IQL's advantage temperature
    :param expectile: IQL's expectile regression value
    :param clip_score: Clipping term on the advantage temp
    :param policy_extraction: ["awr", "ddpg"] policy extraction algorithm
    :param ddpg_bc_weight: DDPG's behavior cloning weight, only used when policy_extraction is "ddpg"
    :param critic_update_ratio: Number of critic updates per actor update
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
    # q_nets: ContinuousCritic
    # q_targets: ContinuousCritic
    v_net: ValueCritic

    def __init__(
        self,
        policy: Union[str, Type[CustomSACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 1,
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
        advantage_temp: float = 5.0,
        expectile: float = 0.7,
        clip_score: float = 100,
        policy_extraction: str = "awr",
        ddpg_bc_weight: float = 0.1,
        offline_critic_update_ratio: int = 1,  # number of critic updates per actor update
        online_critic_update_ratio: int = 1,  # number of critic updates per actor update
        n_critics_to_sample: int = 2,  # number of critics to sample from
        warm_start_online_rl: bool = True,
        action_chunk_size: int = 1,
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
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
            warm_start_online_rl=warm_start_online_rl,
            action_chunk_size=action_chunk_size,
            success_bonus=success_bonus,
        )

        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.target_update_interval = target_update_interval

        if _init_setup_model:
            self._setup_model()

        self.advantage_temp = advantage_temp
        self.expectile = expectile
        self.clip_score = clip_score
        assert policy_extraction in [
            "awr",
            "ddpg",
        ], "Policy extraction algorithm must be either 'awr' or 'ddpg'"
        self.policy_extraction = policy_extraction
        self.ddpg_bc_weight = ddpg_bc_weight
        self.online_critic_update_ratio = online_critic_update_ratio
        self.offline_critic_update_ratio = offline_critic_update_ratio
        self.current_critic_update_ratio = self.offline_critic_update_ratio
        self.n_critics_to_sample = n_critics_to_sample
        self.name = "iql"

    def _setup_model(self) -> None:
        super()._setup_model()

        # self.policy.actor = th.compile(self.policy.actor, mode="reduce-overhead")
        # self.policy.critic = th.compile(self.policy.critic, mode="reduce-overhead")
        # self.policy.critic_target = th.compile(
        # self.policy.critic_target, mode="reduce-overhead"
        # )

        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(
            self.critic_target, ["running_"]
        )

        self.v_net = ValueCritic(
            self.observation_space,
            self.action_space,
            self.policy.critic_kwargs["net_arch"],
            deepcopy(self.policy.critic.features_extractor),
            features_dim=self.policy.actor.latent_pi[0].in_features,
            activation_fn=self.policy.net_args["activation_fn"],
            normalize_images=self.policy.critic.normalize_images,
            share_features_extractor=self.policy.critic.share_features_extractor,
            lr_schedule=self.lr_schedule,
            optimizer_class=self.policy.optimizer_class,
            optimizer_kwargs=self.policy.optimizer_kwargs,
            use_layer_norm=self.policy.critic_kwargs["use_layer_norm"],
        ).to(self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic.to(th.float32)
        self.critic_target = self.policy.critic_target

    def train(
        self,
        gradient_steps: int,
        batch_size: int = 64,
        logging_prefix: str = "train",
        policy_lock=None,
    ) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        actor_losses, q_losses, v_losses = [], [], []
        actor_log_pis = []
        q_values = []
        v_next_values = []
        v_values = []
        q_target_values = []
        reward_values = []

        if gradient_steps != 1:
            # only so if we are doing per-step training, we don't overprint
            print(f"Going to take {gradient_steps} training steps")

        for gradient_step in range(gradient_steps):
            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            for _ in range(int(self.current_critic_update_ratio)):
                # Sample replay buffer
                replay_data = self.replay_buffer.sample(
                    batch_size, env=self._vec_normalize_env
                )  # type: ignore[union-attr]

                # Compute necessary values for the training update
                q_preds = th.cat(
                    self.critic(
                        replay_data.observations,
                        replay_data.actions,
                    ),
                    dim=1,
                )
                with th.no_grad():
                    # for generality with REDQ implmentation
                    critic_indices = th.randperm(self.policy_kwargs["n_critics"])[
                        : self.n_critics_to_sample
                    ].to(replay_data.observations.device)
                    target_q_preds = th.cat(
                        self.critic_target(
                            replay_data.observations,
                            replay_data.actions,
                            critic_indices=critic_indices,
                        ),
                        dim=1,
                    )

                    target_q_pred, _ = th.min(target_q_preds, dim=1)
                    target_q_pred = target_q_pred.reshape(-1, 1)
                    next_vf_pred = self.v_net(replay_data.next_observations)
                vf_pred = self.v_net(replay_data.observations)

                # Q value loss
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_vf_pred
                )
                q_loss = F.mse_loss(q_preds, target_q_values.expand_as(q_preds))

                # Value function expectile loss
                vf_err = vf_pred - target_q_pred
                vf_sign = (vf_err > 0).float()
                vf_weight = (1 - vf_sign) * self.expectile + vf_sign * (
                    1 - self.expectile
                )
                vf_loss = (vf_weight * (vf_err**2)).mean()

                # log q1 and q2 values
                q_values.append(q_preds.mean().item())

                # log v
                v_values.append(vf_pred.mean().item())

                # log target
                q_target_values.append(target_q_values.mean().item())

                # log next v
                v_next_values.append(next_vf_pred.mean().item())

                # log q and v losses
                q_losses.append(q_loss.item())
                v_losses.append(vf_loss.item())

                # Optimize the critic Q
                self.critic.optimizer.zero_grad()
                q_loss.backward()
                # Apply gradient clipping to improve stability
                th.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
                self.critic.optimizer.step()

                # Optimize the value function
                self.v_net.optimizer.zero_grad()
                vf_loss.backward()
                # Apply gradient clipping to improve stability
                th.nn.utils.clip_grad_norm_(self.v_net.parameters(), max_norm=10.0)
                self.v_net.optimizer.step()

                # Target network update
                if gradient_step % self.target_update_interval == 0:
                    polyak_update(
                        self.critic.parameters(),
                        self.critic_target.parameters(),
                        self.tau,
                    )
                    polyak_update(
                        self.batch_norm_stats, self.batch_norm_stats_target, 1.0
                    )

            # Policy loss
            if self.policy_extraction == "awr":
                advantage = target_q_pred - vf_pred.detach()
                weights = th.clamp(
                    th.exp(advantage * self.advantage_temp), 0, self.clip_score
                )
                if policy_lock is not None:
                    policy_lock.acquire()
                mean_actions, log_std, kwargs = self.actor.get_action_dist_params(
                    replay_data.observations
                )
                distribution = self.actor.action_dist.proba_distribution(
                    mean_actions, log_std
                )
                if policy_lock is not None:
                    policy_lock.release()
                log_prob = self.get_log_prob(distribution, replay_data.actions)
                log_prob = log_prob.reshape(-1, 1)
                policy_loss = -th.mean(weights * log_prob)
            elif self.policy_extraction == "ddpg":
                # autoscale the bc weight based on the average q value

                if policy_lock is not None:
                    policy_lock.acquire()

                with th.no_grad():
                    average_q_value = th.abs(th.min(q_preds, dim=1)).mean()
                    scaled_ddpg_bc_weight = self.ddpg_bc_weight / average_q_value
                mean_actions, log_std, _ = self.actor.get_action_dist_params(
                    replay_data.observations
                )
                distribution = self.actor.action_dist.proba_distribution(
                    mean_actions, log_std
                )
                if policy_lock is not None:
                    policy_lock.release()
                log_prob = self.get_log_prob(distribution, replay_data.actions)

                actions_pi = distribution.actions_from_params(mean_actions, log_std)

                critic_indices = th.randperm(self.policy_kwargs["n_critics"])[
                    : self.n_critics_to_sample
                ].to(replay_data.observations.device)
                q_values_pi = self.critic(
                    replay_data.observations, actions_pi, critic_indices=critic_indices
                )
                min_qf_pi = th.min(*q_values_pi).squeeze(-1)
                assert min_qf_pi.shape == log_prob.shape, (
                    f"{min_qf_pi.shape} != {log_prob.shape}"
                )
                policy_loss = -th.mean(min_qf_pi + scaled_ddpg_bc_weight * log_prob)
                # print proportion of policy loss contributed to by each term
                # policy_loss = -th.mean(min_qf_pi + self.ddpg_bc_weight * log_prob)

            # log average in batch reward
            reward_values.append(replay_data.rewards.mean().item())

            # Optimize the policy
            if policy_lock is not None:
                policy_lock.acquire()
            self.actor.optimizer.zero_grad()
            policy_loss.backward()
            self.actor.optimizer.step()
            if policy_lock is not None:
                policy_lock.release()

            # log actor stuff
            actor_losses.append(policy_loss.item())
            actor_log_pis.append(log_prob.mean().item())
        self._n_updates += gradient_steps

        metrics_dict = {
            f"{logging_prefix}/actor_loss": np.mean(actor_losses),
            f"{logging_prefix}/q_loss": np.mean(q_losses),
            f"{logging_prefix}/v_loss": np.mean(v_losses),
            f"{logging_prefix}/average_q_values": np.mean(q_values),
            f"{logging_prefix}/average_v_next_values": np.mean(v_next_values),
            f"{logging_prefix}/average_reward": np.mean(reward_values),
            f"{logging_prefix}/average_v_values": np.mean(v_values),
            f"{logging_prefix}/average_q1_target_values": np.mean(q_target_values),
            f"{logging_prefix}/average_actor_log_pis": np.mean(actor_log_pis),
        }

        for metric in metrics_dict:
            self.logger.record(metric, metrics_dict[metric])

        return metrics_dict

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "IQL",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        logger: Optional = None,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
            logger=logger,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + [
            "actor",
            "critic",
            "critic_target",
            "v_net",
        ]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = [
            "policy",
            "actor.optimizer",
            "critic.optimizer",
            "v_net.optimizer",
        ]
        saved_pytorch_variables = []
        return state_dicts, saved_pytorch_variables
