# TODO: if offline, just do BC. if online, do SAC
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    Iterable,
)
import io
import os
import pathlib

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from offline_rl_algorithms.base_offline_rl_algorithm import OfflineRLAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.common.distributions import kl_divergence
from offline_rl_algorithms.custom_policies import (
    CustomActor,
    CustomSACPolicy,
    CustomCnnPolicy,
    CustomMlpPolicy,
    CustomRNNMlpPolicy,
    CustomMultiInputPolicy,
)
import threading

import functools
from stable_baselines3.common.utils import check_for_correct_spaces, get_system_info
from stable_baselines3.common.save_util import (
    load_from_zip_file,
)

import warnings

from offline_rl_algorithms.iql import ValueCritic
from copy import deepcopy


class RLPD(OfflineRLAlgorithm):
    """
    RLPD https://arxiv.org/abs/2302.02948. This implementation uses BC for train_offline.

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
    :param critic_update_ratio: Number of critic updates per actor update
    :param n_critics_to_sample: Number of critics to sample from
    :param train_critic_with_entropy: Whether to train the critic with the entropy term
    :param warm_start_online_rl: Whether to warm start online RL with offline RL
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": CustomMlpPolicy,
        "CnnPolicy": CustomCnnPolicy,
        "RnnMlpPolicy": CustomRNNMlpPolicy,
        "MultiInputPolicy": CustomMultiInputPolicy,
    }
    policy: CustomSACPolicy
    actor: CustomActor
    v_net: ValueCritic

    def __init__(
        self,
        policy: Union[str, Type[CustomSACPolicy]],
        env: Union[GymEnv, str],
        offline_algo: OfflineRLAlgorithm = None,
        learning_rate: Union[float, Schedule] = 3e-4,
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
        offline_critic_update_ratio: int = 1,  # number of critic updates per actor update
        online_critic_update_ratio: int = 5,  # number of critic updates per actor update
        n_critics_to_sample: int = 2,  # number of critics to sample from
        train_critic_with_entropy: bool = False,  # whether to train the critic with the entropy term
        warm_start_online_rl: bool = True,
        action_chunk_size: int = 1,
        success_bonus: float = 0.0,
        use_kl_against_old: bool = False,
    ):
        # NOTE: Asserntions currently commonted out due to saving/loading logic. Must fix this later TODO
        # assert (
        #     policy_kwargs["n_critics"] > 2
        # ), "RLPD is made for more than 2 critics. Double check this."
        # assert (
        #     policy_kwargs["critic_layer_norm"] == True
        # ), "RLPD is made for layernorm critics. Double check this."
        # print(
        #     f"Mix offline and online buffers: {mix_offline_online_buffers}. RLPD assumes offline data is mixed with online data. Just printing for sanity."
        # )

        self.offline_algo = offline_algo

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

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None

        if _init_setup_model:
            self._setup_model()

        self.online_critic_update_ratio = online_critic_update_ratio
        self.offline_critic_update_ratio = offline_critic_update_ratio
        self.current_critic_update_ratio = self.offline_critic_update_ratio
        self.n_critics_to_sample = n_critics_to_sample
        self.train_critic_with_entropy = train_critic_with_entropy

        self.name = "rlpd"

        self.expectile = 0.7
        self.clip_score = 100
        self.policy_extraction = "awr"
        self.ddpg_bc_weight = 0.1
        self.advantage_temp = 1.0

        self.train = self.train_rlpd
        self.use_kl_against_old = use_kl_against_old

    def set_offline_algo(self, offline_algo):
        self.offline_algo = offline_algo

    def set_policies_with_offline(self, offline_algo=None):
        """
        Replace the RLPD actor and critic with the offline_algo's actor and critic.
        Replace their parameters so that the optimizer is still the same.
        """
        if offline_algo is None:
            offline_algo = self.offline_algo

        # if self.offline_algo is None:
        #     print("Offline algo is not set")
        #     return

        if self.offline_algo is not None:
            self.policy.actor = offline_algo.policy.actor
            self.policy.critic = offline_algo.policy.critic
            self.policy.critic_target = offline_algo.policy.critic_target

        # old actor will be used for KL divergence computation against the old policy
        print("setting old actor!")
        self.old_actor = th.compile(deepcopy(self.policy.actor).cpu().to(self.device))
        # self.old_actor = deepcopy(self.policy.actor).cpu().to(self.device)

        # self.policy.actor.optimizer = type(self.policy.actor.optimizer)(
        #     self.policy
        #     lr=self.policy.actor.optimizer.param_groups[0]["lr"],
        # )
        # self.policy.critic.optimizer = type(self.policy.critic.optimizer)(
        #     self.policy.critic.parameters(),
        #     lr=self.policy.critic.optimizer.param_groups[0]["lr"],
        # )

        # for layer in self.policy.actor.mu_processor.children():
        #     if hasattr(layer, "reset_parameters"):
        #         layer.reset_parameters()
        # for layer in self.policy.actor.log_std_processor.children():
        #     if hasattr(layer, "reset_parameters"):
        #         layer.reset_parameters()
        # for i in range(len(self.policy.critic.q_networks)):
        #     for layer in self.policy.critic.q_networks[i].q_network.children():
        #         if hasattr(layer, "reset_parameters"):
        #             layer.reset_parameters()

        return

        old_policy_optimizer = self.policy.actor.optimizer
        old_critic_optimizer = self.policy.critic.optimizer
        old_ent_coef_optimizer = self.ent_coef_optimizer

        # for name, param in self.policy.actor.named_parameters():
        #     if param.grad is None:
        #         print(f"{name} has no grad!")

        # self.policy.actor = offline_algo.policy.actor
        # self.policy.critic = offline_algo.policy.critic
        # self.policy.critic_target = offline_algo.policy.critic_target

        # self.policy.actor.optimizer = old_policy_optimizer
        # self.policy.critic.optimizer = old_critic_optimizer
        # self.ent_coef_optimizer = old_ent_coef_optimizer

        self.policy.actor.load_state_dict(offline_algo.policy.actor.state_dict())
        self.policy.critic.load_state_dict(offline_algo.policy.critic.state_dict())
        self.policy.critic_target.load_state_dict(
            offline_algo.policy.critic_target.state_dict()
        )

        # Recreate the optimizers with the right parameters
        old_actor_optim_state = old_policy_optimizer.state_dict()
        old_critic_optim_state = old_critic_optimizer.state_dict()

        # Re-init optimizers on new parameters
        self.policy.actor.optimizer = type(self.policy.actor.optimizer)(
            self.policy.actor.parameters(),
            lr=self.policy.actor.optimizer.param_groups[0]["lr"],
        )
        self.policy.critic.optimizer = type(self.policy.critic.optimizer)(
            self.policy.critic.parameters(),
            lr=self.policy.critic.optimizer.param_groups[0]["lr"],
        )

        # OPTIONAL: restore optimizer state if you want continuity
        self.policy.actor.optimizer.load_state_dict(old_actor_optim_state)
        self.policy.critic.optimizer.load_state_dict(old_critic_optim_state)

        # set to be trainable
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.critic_target.train()

        def compare_params(model, optimizer):
            model_params = set(p for p in model.parameters())
            opt_params = set(
                p for group in optimizer.param_groups for p in group["params"]
            )

            missing_in_optimizer = model_params - opt_params
            extra_in_optimizer = opt_params - model_params

            if missing_in_optimizer:
                print("❌ Parameters in model but not in optimizer:")
                for p in missing_in_optimizer:
                    print(f" - {p.shape}")
            if extra_in_optimizer:
                print("❌ Parameters in optimizer but not in model:")
                for p in extra_in_optimizer:
                    print(f" - {p.shape}")
            if not missing_in_optimizer and not extra_in_optimizer:
                print("✅ Optimizer is correctly tracking all model parameters.")

            return missing_in_optimizer, extra_in_optimizer

        print("actor")
        compare_params(self.policy.actor, self.policy.actor.optimizer)
        print("critic")
        compare_params(self.policy.critic, self.policy.critic.optimizer)

        # breakpoint()

        # This sets the optimizer to the offline_algo's optimizer
        # self.policy.actor.optimizer = offline_algo.policy.actor.optimizer
        # self.policy.critic.optimizer = offline_algo.policy.critic.optimizer

        # This replaces the optimizer with the old (new) optimizer
        # self.policy.actor.optimizer = old_policy_optimizer
        # self.policy.critic.optimizer = old_critic_optimizer

        if (
            hasattr(offline_algo, "ent_coef_optimizer")
            and offline_algo.ent_coef_optimizer is not None
        ):
            print(
                "Setting ent_coef_optimizer and ent coef to the old value of the offline algo"
            )
            # self.ent_coef_optimizer = offline_algo.ent_coef_optimizer
            # self.log_ent_coef = offline_algo.log_ent_coef
            # self.ent_coef_optimizer = old_ent_coef_optimizer
            # # self.log_ent_coef.load_state_dict(offline_algo.log_ent_coef.state_dict())
            # # self.ent_coef_optimizer.load_state_dict(
            # #     offline_algo.ent_coef_optimizer.state_dict()
            # # )

            # self.ent_coef_optimizer = type(self.ent_coef_optimizer)(
            #     [self.log_ent_coef], lr=self.lr_schedule(1)
            # )
            # compare_params(self.log_)

        elif hasattr(offline_algo, "ent_coef_tensor"):
            print(
                f"Setting ent_coef_tensor to the old value of the offline algo: {offline_algo.ent_coef_tensor.item()}"
            )
            self.ent_coef_tensor = offline_algo.ent_coef_tensor
        self.learned_offline = True

        self._create_aliases()

    def _setup_model(self) -> None:
        super()._setup_model()

        # self.policy.actor = th.compile(self.policy.actor, mode="reduce-overhead")
        # self.policy.critic = th.compile(self.policy.critic, mode="reduce-overhead")
        # self.policy.critic_target = th.compile(
        #     self.policy.critic_target, mode="reduce-overhead"
        # )

        # If there is a v_net, we can add one here
        # if hasattr(self.offline_algo, "v_net"): # not needed for online
        #    self.v_net = self.offline_algo.v_net # not needed for online

        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(
                -np.prod(self.env.action_space.shape).astype(np.float32)
            )  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, (
                    "The initial value of ent_coef must be greater than 0"
                )

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(
                th.ones(1, device=self.device) * init_value
            ).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam(
                [self.log_ent_coef], lr=self.lr_schedule(1)
            )
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

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
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def learn_offline(
        self,
        train_steps: int,
        offline_replay_buffer: ReplayBuffer,
        batch_size: int = 64,
        callback: MaybeCallback = None,
    ) -> None:
        if self.offline_algo is not None:
            # override train function to use BC offline training
            self.offline_algo.set_logger(self.logger)

            self.offline_algo.learn_offline(
                offline_replay_buffer=offline_replay_buffer,
                train_steps=train_steps,
                batch_size=batch_size,
                callback=callback,
            )

            # Set the policies with the offline_algo's policies
            self.set_policies_with_offline()

        else:
            # we do IQL in here!

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
                metrics = self.train_iql(
                    1, batch_size=batch_size, logging_prefix="offline"
                )
                # metrics is a local() which will be updated in callback.update_locals
                callback.update_locals(locals())  # a little hacky
                callback.on_step()  # because of locals, we have access to self.locals['metrics']

            callback.on_training_end()

            self.replay_buffer = old_replay_buffer

        # old_train_function = self.train
        # RLPD.train = self._train_offline
        # super().learn_offline(
        #     train_steps=train_steps,
        #     offline_replay_buffer=offline_replay_buffer,
        #     batch_size=batch_size,
        #     callback=callback,
        # )
        # # for online training we use RLPD's train function
        # RLPD.train = old_train_function

    def train_iql(
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

    def train_rlpd(
        self,
        gradient_steps: int,
        batch_size: int = 64,
        logging_prefix: str = "",
        policy_lock=None,
    ) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        # breakpoint()
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        # breakpoint()
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        # for name, param in self.actor.named_parameters():

        #     def make_hook(n):
        #         return lambda grad: print(
        #             f"Grad for {n}: {grad.norm() if grad is not None else 'None'}"
        #         )

        #     param.register_hook(make_hook(name))

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        actor_log_pis = []
        q_values_list = []
        q_next_values_list = []
        reward_values = []

        if gradient_steps != 1:
            # only so if we are doing per-step training, we don't overprint
            print(f"Going to take {gradient_steps} training steps")
            print(self.replay_buffer.size())

        # breakpoint()
        for gradient_step in range(gradient_steps):
            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
            else:
                ent_coef = self.ent_coef_tensor

            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )  # type: ignore[union-attr]
            if len(replay_data.observations) == 0:
                print("Buffer is still empty. Skipping this training step")
                return {}
            shuffled_indicies = np.random.permutation(batch_size)
            critic_batch_indicies = np.array_split(
                shuffled_indicies, self.current_critic_update_ratio
            )
            # Calculate KL divergence between old and current actor
            with th.no_grad():
                old_mean, old_log_std, _ = self.old_actor.get_action_dist_params(
                    replay_data.observations
                )
                old_actor_distribution = self.old_actor.action_dist.proba_distribution(
                    old_mean, old_log_std
                )

            # Action by the current actor for the sampled state
            if policy_lock is not None:
                policy_lock.acquire()
            # actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            mean_actions, log_std, kwargs = self.actor.get_action_dist_params(
                replay_data.observations
            )
            actions_pi, log_prob = self.actor.action_dist.log_prob_from_params(
                mean_actions, log_std, **kwargs
            )
            next_actions, next_log_prob = self.actor.action_log_prob(
                replay_data.next_observations
            )
            if policy_lock is not None:
                policy_lock.release()
            curr_actor_distribution = self.actor.action_dist.proba_distribution(
                mean_actions, log_std, **kwargs
            )
            kl_div = kl_divergence(
                curr_actor_distribution,
                old_actor_distribution,
            )
            kl_div = kl_div.sum(1).mean(-1, keepdim=True)
            for critic_update in range(self.current_critic_update_ratio):
                critic_next_obs = replay_data.next_observations[
                    critic_batch_indicies[critic_update]
                ]
                critic_observations = replay_data.observations[
                    critic_batch_indicies[critic_update]
                ]
                critic_dones = replay_data.dones[critic_batch_indicies[critic_update]]
                critic_next_actions = replay_data.actions[
                    critic_batch_indicies[critic_update]
                ]
                critic_actions = replay_data.actions[
                    critic_batch_indicies[critic_update]
                ]
                critic_rewards = replay_data.rewards[
                    critic_batch_indicies[critic_update]
                ]
                critic_valid_length = replay_data.valid_length[
                    critic_batch_indicies[critic_update]
                ]
                critic_kl_div = kl_div[critic_batch_indicies[critic_update]]

                with th.no_grad():
                    # Select action according to policy
                    # print("replay data shape", replay_data.next_observations.shape)
                    critic_next_actions = next_actions[
                        critic_batch_indicies[critic_update]
                    ]
                    critic_next_log_prob = next_log_prob[
                        critic_batch_indicies[critic_update]
                    ]
                    if critic_next_actions.ndim == 3:
                        # Take the mean of the logprob
                        critic_next_log_prob = critic_next_log_prob.mean(
                            dim=1, keepdim=True
                        )
                    # Compute the next Q values: min over all critics targets
                    # sample a random subset of self.n_critics_to_sample critics. no replacement
                    critic_indices = th.randperm(self.policy_kwargs["n_critics"])[
                        : self.n_critics_to_sample
                    ]
                    next_q_values = th.cat(
                        self.critic_target(
                            critic_next_obs,
                            critic_next_actions,
                            critic_indices=critic_indices,
                        ),
                        dim=1,
                    )
                    next_q_values = th.mean(next_q_values, dim=1, keepdim=True)

                    # add entropy term
                    if self.train_critic_with_entropy:
                        if self.use_kl_against_old:
                            next_q_values = (
                                next_q_values - ent_coef * critic_kl_div.detach()
                            )
                        else:
                            next_q_values = (
                                next_q_values
                                - ent_coef * critic_next_log_prob.reshape(-1, 1)
                            )

                    # td error + entropy term

                    if hasattr(replay_data, "valid_length"):
                        valid_lengths = critic_valid_length
                        discount = (self.gamma**valid_lengths).unsqueeze(1).float()
                    else:
                        discount = self.gamma
                    target_q_values = (
                        critic_rewards + (1 - critic_dones) * discount * next_q_values
                    )

                # Get current Q-values estimates for each critic network
                # using action from the replay buffer
                current_q_values = th.cat(
                    self.critic(critic_observations, critic_actions),
                    dim=1,
                )
                # Compute critic loss
                critic_loss = F.mse_loss(
                    current_q_values, target_q_values.expand_as(current_q_values)
                )
                assert isinstance(critic_loss, th.Tensor)  # for type checker
                critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

                # Optimize the critic
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                # Apply gradient clipping to improve stability
                # th.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
                self.critic.optimizer.step()

                # target network update
                if gradient_step % self.target_update_interval == 0:
                    polyak_update(
                        self.critic.parameters(),
                        self.critic_target.parameters(),
                        self.tau,
                    )
                    polyak_update(
                        self.batch_norm_stats, self.batch_norm_stats_target, 1.0
                    )

            if actions_pi.ndim == 3:
                # Take the mean of the logprob
                log_prob = log_prob.mean(dim=1, keepdim=True)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                assert not self.use_kl_against_old, "Not implemented yet"
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            # log average in batch reward
            reward_values.append(replay_data.rewards.mean().item())

            # log average q values
            q_values_list.append(np.mean([q.mean().item() for q in current_q_values]))

            # log average next q values
            q_next_values_list.append(next_q_values.mean().item())

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(
                self.critic(replay_data.observations, actions_pi), dim=1
            )
            mean_qf_pi = th.mean(q_values_pi, dim=1, keepdim=True)

            if policy_lock is not None:
                policy_lock.acquire()

            curr_actor_distribution = self.actor.action_dist.proba_distribution(
                mean_actions, log_std, **kwargs
            )

            kl_div = kl_divergence(
                curr_actor_distribution, old_actor_distribution
            )  # B x ACtion CHUnk x action dim
            kl_div = kl_div.sum(1).mean(-1, keepdim=True)

            if self.use_kl_against_old:
                actor_loss = (ent_coef * kl_div - mean_qf_pi).mean()
            else:
                actor_loss = (ent_coef * log_prob - mean_qf_pi).mean()

            # Log KL divergence
            self.logger.record(
                f"{logging_prefix}/kl_divergence", kl_div.detach().mean().item()
            )

            actor_losses.append(actor_loss.item())

            ### DEBUG
            # params_in_optimizer = set()
            # for group in self.actor.optimizer.param_groups:
            #     params_in_optimizer.update(group["params"])

            # target_param = self.actor.mu_processor[0]._parameters["weight"]
            # print("Tracking:", target_param in params_in_optimizer)  # Should be True
            # print("Grad:", self.actor.mu_processor[0]._parameters["weight"].grad)
            # print(
            #     "Requires grad:",
            #     self.actor.mu_processor[0]._parameters["weight"].requires_grad,
            # )

            # only do this if mu_processor exists
            if hasattr(self.actor, "mu_processor"):
                weights_before = (
                    self.actor.mu_processor[0]._parameters["weight"].detach().clone()
                )

            ### DEBUG END

            self.actor.optimizer.zero_grad()
            actor_loss.backward()

            # Apply gradient clipping to improve stability
            # th.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
            self.actor.optimizer.step()

            # only do this if mu_processor exists
            if hasattr(self.actor, "mu_processor"):
                weights_after = (
                    self.actor.mu_processor[0]._parameters["weight"].detach().clone()
                )

                # check if they are close
                weight_difference = weights_after - weights_before

            if policy_lock is not None:
                policy_lock.release()
            actor_log_pis.append(log_prob.mean().item())

        self._n_updates += gradient_steps

        metrics_dict = {
            f"{logging_prefix}/ent_coef": np.mean(ent_coefs),
            f"{logging_prefix}/actor_loss": np.mean(actor_losses),
            f"{logging_prefix}/critic_loss": np.mean(critic_losses),
            f"{logging_prefix}/average_q_values": np.mean(q_values_list),
            f"{logging_prefix}/average_q_next_values": np.mean(q_next_values_list),
            f"{logging_prefix}/average_reward": np.mean(reward_values),
            f"{logging_prefix}/average_actor_log_pis": np.mean(actor_log_pis),
        }

        if hasattr(self.actor, "mu_processor"):
            metrics_dict[f"{logging_prefix}/weight_difference_max"] = (
                weight_difference.max()
            )
            metrics_dict[f"{logging_prefix}/weight_difference_min"] = (
                weight_difference.min()
            )
            metrics_dict[f"{logging_prefix}/weight_difference_mean"] = (
                weight_difference.mean()
            )

        if len(ent_coef_losses) > 0:
            metrics_dict[f"{logging_prefix}/ent_coef_loss"] = np.mean(ent_coef_losses)

        for metric in metrics_dict:
            self.logger.record(metric, metrics_dict[metric])

        return metrics_dict

    def train_cql(
        self,
        gradient_steps: int,
        batch_size: int = 64,
        logging_prefix: str = "train",
        policy_lock=None,
    ) -> None:
        """
        CQL training loop, closely modeled after train_iql and CQL.train.
        """
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, cql_losses = [], [], []
        actor_log_pis = []
        q_values_list = []
        q_next_values_list = []
        reward_values = []

        if gradient_steps != 1:
            print(f"Going to take {gradient_steps} CQL training steps")

        for gradient_step in range(gradient_steps):
            if self.use_sde:
                self.actor.reset_noise()

            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
            else:
                ent_coef = self.ent_coef_tensor

            for critic_update in range(self.current_critic_update_ratio):
                replay_data = self.replay_buffer.sample(
                    batch_size, env=self._vec_normalize_env
                )
                if len(replay_data.observations) == 0:
                    print("Buffer is empty. Skipping this CQL training step")
                    return {}

                with th.no_grad():
                    if policy_lock is not None:
                        policy_lock.acquire()
                    next_actions, next_log_prob = self.actor.action_log_prob(
                        replay_data.next_observations
                    )
                    if policy_lock is not None:
                        policy_lock.release()
                    if next_actions.ndim == 3:
                        next_log_prob = next_log_prob.mean(dim=1, keepdim=True)
                    critic_indices = th.randperm(self.policy_kwargs["n_critics"])[
                        : self.n_critics_to_sample
                    ]
                    next_q_values = th.cat(
                        self.critic_target(
                            replay_data.next_observations,
                            next_actions,
                            critic_indices=critic_indices,
                        ),
                        dim=1,
                    )
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(
                        -1, 1
                    )
                    target_q_values = (
                        replay_data.rewards
                        + (1 - replay_data.dones) * self.gamma * next_q_values
                    )

                # --- CQL Loss ---
                random_actions = (
                    th.FloatTensor(replay_data.actions.shape)
                    .uniform_(-1, 1)
                    .to(self.device)
                )
                current_actions, current_log_pis = self.actor.action_log_prob(
                    replay_data.observations
                )
                next_actions_cql, next_log_pis = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                # current_log_pis = current_log_pis.reshape(-1, 1)
                # next_log_pis = next_log_pis.reshape(-1, 1)
                if next_actions_cql.ndim == 3:
                    next_log_pis = next_log_pis.mean(dim=1, keepdim=True)
                    current_log_pis = current_log_pis.mean(dim=1, keepdim=True)

                q_rand = th.cat(
                    self.critic(replay_data.observations, random_actions), 1
                )
                q_current_actions = th.cat(
                    self.critic(replay_data.observations, current_actions), 1
                )
                q_next_actions = th.cat(
                    self.critic(replay_data.observations, next_actions_cql), 1
                )
                random_density = np.log(0.5 ** current_actions.shape[-1])

                # Expand log_pis to match q_* shape if needed
                n_critics = q_rand.shape[1]
                if current_log_pis.shape[1] == 1:
                    current_log_pis = current_log_pis.repeat(1, n_critics)
                if next_log_pis.shape[1] == 1:
                    next_log_pis = next_log_pis.repeat(1, n_critics)

                # CQL regularizer (logsumexp)
                # breakpoint()
                cql_cat = th.cat(
                    [
                        q_rand - random_density,
                        q_next_actions - next_log_pis.detach(),
                        q_current_actions - current_log_pis.detach(),
                    ],
                    1,
                )
                cql_loss = (
                    th.logsumexp(cql_cat, dim=1).mean() - q_current_actions.mean()
                )
                cql_losses.append(cql_loss.item())

                # Critic loss (TD + CQL)
                current_q_values = th.cat(
                    self.critic(replay_data.observations, replay_data.actions),
                    dim=1,
                )
                critic_loss = (
                    F.mse_loss(
                        current_q_values, target_q_values.expand_as(current_q_values)
                    )
                    + cql_loss
                )
                critic_losses.append(critic_loss.item())

                # Optimize the critic
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()

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

                # log average q values
                q_values_list.append(current_q_values.mean().item())
                q_next_values_list.append(next_q_values.mean().item())
                reward_values.append(replay_data.rewards.mean().item())

            # Actor loss (same as SAC)
            if policy_lock is not None:
                policy_lock.acquire()
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            if policy_lock is not None:
                policy_lock.release()
            if actions_pi.ndim == 3:
                log_prob = log_prob.mean(dim=1, keepdim=True)
            log_prob = log_prob.reshape(-1, 1)
            q_values_pi = th.cat(
                self.critic(replay_data.observations, actions_pi), dim=1
            )
            mean_qf_pi = th.mean(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - mean_qf_pi).mean()
            actor_losses.append(actor_loss.item())
            actor_log_pis.append(log_prob.mean().item())
            # Optimize actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Entropy coefficient loss (if applicable)
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

        self._n_updates += gradient_steps
        metrics_dict = {
            f"{logging_prefix}/ent_coef": np.mean(ent_coefs),
            f"{logging_prefix}/actor_loss": np.mean(actor_losses),
            f"{logging_prefix}/critic_loss": np.mean(critic_losses),
            f"{logging_prefix}/cql_loss": np.mean(cql_losses),
            f"{logging_prefix}/average_q_values": np.mean(q_values_list),
            f"{logging_prefix}/average_q_next_values": np.mean(q_next_values_list),
            f"{logging_prefix}/average_reward": np.mean(reward_values),
            f"{logging_prefix}/average_actor_log_pis": np.mean(actor_log_pis),
        }
        if len(ent_coef_losses) > 0:
            metrics_dict[f"{logging_prefix}/ent_coef_loss"] = np.mean(ent_coef_losses)
        for metric in metrics_dict:
            self.logger.record(metric, metrics_dict[metric])
        return metrics_dict

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "RLPD",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        logger: Optional = None,
        parallelize: bool = False,
    ):
        self.current_critic_update_ratio = self.online_critic_update_ratio
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
            logger=logger,
            parallelize=parallelize,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + [
            "offline_algo",  # Exclude the offline algorithm
            "reward_model",  # Exclude the reward model
            "combined_buffer",  # Exclude the combined buffer
            "online_buffer",  # Exclude the online buffer
            "offline_buffer",  # Exclude the offline buffer
            "logger",  # Exclude the logger
            "train",
        ]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
