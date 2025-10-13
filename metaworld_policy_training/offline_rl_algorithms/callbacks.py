import wandb

from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import imageio
import wandb
from wandb.integration.sb3 import WandbCallback
import io
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.vec_env import sync_envs_normalization

import gym

def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(observations, state=states, episode_start=episode_starts, deterministic=deterministic, env=env)
        # print(f"in eval: {env.envs[0].chunk}, actions: {actions}")
        # print(f"is_chunk_empty: {env.envs[0].is_chunk_empty}")
        # print(f"dense_eval: {env.envs[0].dense_eval}")
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


class OfflineEvalCallback(EvalCallback):
    def __init__(self, *args, video_freq, **kwargs):
        super(OfflineEvalCallback, self).__init__(*args, **kwargs)
        self.video_freq = video_freq
        # we need to overide num_timesteps as EvalCallback uses it to align the built in logger's x-axis
        # we are using wandb so now we're using self.n_calls as the step for everything
        self.num_timesteps = lambda x: self.n_calls  # convert num_timst

    def _on_step(self) -> bool:
        # print(self.n_calls, self.n_calls % self.video_freq)
        # Log policy gradients
        if self.n_calls % 500 == 0:
            try:
                policy_gradients = [
                    param.grad.view(-1)
                    .detach()
                    .cpu()
                    .numpy()  # Flatten each gradient tensor
                    for param in self.model.policy.actor.parameters()
                    if param.grad is not None
                ]
                if len(policy_gradients) != 0:
                    all_gradients = np.concatenate(policy_gradients)
                    self.logger.record(
                        "grad/policy_histogram", wandb.Histogram(all_gradients)
                    )
                    # Log critic gradients
                    critic_gradients = [
                        param.grad.view(-1)
                        .detach()
                        .cpu()
                        .numpy()  # Flatten each gradient tensor
                        for param in self.model.policy.critic.parameters()
                        if param.grad is not None
                    ]
                    if len(critic_gradients) != 0:
                        all_gradients = np.concatenate(critic_gradients)
                        self.logger.record(
                            "grad/critic_histogram", wandb.Histogram(all_gradients)
                        )

                    # Log critic_target gradients
                    critic_target_gradients = [
                        param.grad.view(-1)
                        .detach()
                        .cpu()
                        .numpy()  # Flatten each gradient tensor
                        for param in self.model.policy.critic_target.parameters()
                        if param.grad is not None
                    ]
                    if len(critic_target_gradients) != 0:
                        all_gradients = np.concatenate(critic_target_gradients)
                        self.logger.record(
                            "grad/critic_target_histogram",
                            wandb.Histogram(all_gradients),
                        )

                    # Log critic weights
                    critic_weights = [
                        param.data.view(-1)
                        .detach()
                        .cpu()
                        .numpy()  # Flatten each weight tensor
                        for param in self.model.policy.critic.parameters()
                    ]

                    try:
                        if len(critic_weights) != 0:
                            all_weights = np.concatenate(critic_weights)
                            self.logger.record(
                                "weights/critic_histogram", wandb.Histogram(all_weights)
                            )
                    except:
                        print("NaN detected in critic weights. Skipping logging")

                # Log critic_target weights
                critic_target_weights = [
                    param.data.view(-1)
                    .detach()
                    .cpu()
                    .numpy()  # Flatten each weight tensor
                    for param in self.model.policy.critic_target.parameters()
                ]

                try:
                    if len(critic_target_weights) != 0:
                        all_weights = np.concatenate(critic_target_weights)
                        self.logger.record(
                            "weights/critic_target_histogram",
                            wandb.Histogram(all_weights),
                        )
                        # Log critic_target weights
                        critic_target_weights = [
                            param.data.view(-1)
                            .detach()
                            .cpu()
                            .numpy()  # Flatten each weight tensor
                            for param in self.model.policy.critic_target.parameters()
                        ]
                        if len(critic_target_weights) != 0:
                            all_weights = np.concatenate(critic_target_weights)
                            self.logger.record(
                                "weights/critic_target_histogram",
                                wandb.Histogram(all_weights),
                            )
                except:
                    print("NaN detected in critic_target weights. Skipping logging")

                # Log v_net weights and gradients
                if hasattr(self.model, "v_net"):
                    # Log v_net gradients
                    v_net_gradients = [
                        param.grad.view(-1)
                        .detach()
                        .cpu()
                        .numpy()  # Flatten each gradient tensor
                        for param in self.model.v_net.parameters()
                        if param.grad is not None
                    ]
                    if len(v_net_gradients) != 0:
                        all_gradients = np.concatenate(v_net_gradients)
                        self.logger.record(
                            "grad/v_net_histogram", wandb.Histogram(all_gradients)
                        )

                    # Log v_net weights
                    v_net_weights = [
                        param.data.view(-1)
                        .detach()
                        .cpu()
                        .numpy()  # Flatten each weight tensor
                        for param in self.model.v_net.parameters()
                    ]
                    if len(v_net_weights) != 0:
                        all_weights = np.concatenate(v_net_weights)
                        self.logger.record(
                            "weights/v_net_histogram", wandb.Histogram(all_weights)
                        )

                # Log policy weights
                actor_weights = [
                    param.data.view(-1)
                    .detach()
                    .cpu()
                    .numpy()  # Flatten each weight tensor
                    for param in self.model.policy.actor.parameters()
                ]
                if len(actor_weights) != 0:
                    all_weights = np.concatenate(actor_weights)
                    self.logger.record(
                        "weights/policy_histogram", wandb.Histogram(all_weights)
                    )
            except:
                print(
                    "Something went wrong when logging gradients/weights. Skipping logging"
                )

        if self.video_freq > 0 and (
            self.n_calls % self.video_freq == 0 or self.n_calls == 1
        ):
            print("Logging video")
            video_buffer = self.record_video()
            # self.logger.record({f"evaluation_video": wandb.Video(video_buffer, fps=20, format="mp4")}, commit=False)
            self.logger.record(
                "eval/evaluation_video", wandb.Video(video_buffer, fps=20, format="mp4")
            )
            # self.logger.record({f"eval/evaluate_succ": success}, step = self.n_calls)
            print("video logged")

        self.logger.record("num_timesteps", self.num_timesteps)

        # result = super(OfflineEvalCallback, self)._on_step()

        continue_training = True
        # print(self.eval_env.envs[0].chunk)
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            print("Evaluating")
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []
            
            # breakpoint()
            try:
                episode_rewards, episode_lengths = evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    render=self.render,
                    deterministic=self.deterministic,
                    return_episode_rewards=True,
                    warn=self.warn,
                    callback=self._log_success_callback,
                )
            except:
                breakpoint()

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training



        result = True

        return result

    def record_video(self):
        frames = []
        obs = self.eval_env.reset()
        first_step = True
        for _ in range(
            self.eval_env.get_attr("max_episode_steps")[0]
        ):  # You can adjust the number of steps for recording
            frame = self.eval_env.render(mode="rgb_array")
            # downsample frame
            frame = frame[::3, ::3, :3]
            frames.append(frame)
            try:
                action, _ = self.model.predict(
                    obs,
                    deterministic=True,
                    episode_start=np.array([first_step]),
                    env=self.eval_env,
                )
            except Exception as e:
                print("Exception in predict:", e)
                breakpoint()
            obs, reward, done, info = self.eval_env.step(action)
            first_step = False
            if done:
                break

        video_buffer = io.BytesIO()

        with imageio.get_writer(video_buffer, format="mp4", fps=20) as writer:
            for frame in frames:
                writer.append_data(frame)

        video_buffer.seek(0)
        return video_buffer


class CustomWandbCallback(WandbCallback):
    def _on_step(self):
        if "metrics" in self.locals:
            self.logger.record_dict(self.locals["metrics"])
        self.logger.dump(
            self.n_calls
        )  # this ensures that dump gets called, otherwise it's only called in EvalCallback whenever an eval happens
