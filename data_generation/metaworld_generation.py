import os
import cv2
import sys
import h5py
import imageio
import argparse
import functools
import numpy as np
from tqdm import tqdm
from generation_config import env_config
import metaworld 
sys.path.insert(0, "metaworld_policy_training/Metaworld")

from tests.metaworld.envs.mujoco.sawyer_xyz.test_scripted_policies import ALL_ENVS, test_cases_latest_nonoise

# from metaworld.envs.metaworld_envs.metaworld import create_wrapped_env, environment_to_instruction

resolution = (640, 480)
camera = "corner2"  # one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
flip = False  # if True, flips output image 180 degrees



def trajectory_generator(env, policy, act_noise_pct, res=(640, 480), camera="corner2"):
    action_space_ptp = env.action_space.high - env.action_space.low

    env.reset()
    env.reset_model()
    o = env.reset()

    for _ in range(env.max_path_length):
        a = policy.get_action(o)
        a = np.random.normal(a, act_noise_pct * action_space_ptp)

        o, r, done, info = env.step(a)
        # Camera is one of ['corner', 'topview', 'behindGripper', 'gripperPOV']
        yield (
            r,
            done,
            info,
            env.sim.render(*res, mode="offscreen", camera_name=camera)[:, :, ::-1],
        )



def main(args):
    collect_num = args.collect_num
    config_range = (0, len(env_config))
    base_path = args.save_path

    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    print("Path is", os.path.join(base_path, f"metaworld_generation.h5"))
    h5_traj = h5py.File(
        os.path.join(base_path, f"metaworld_generation.h5"), "w"
    )

    for config_idx in tqdm(range(config_range[0], config_range[1])):
        env_name, noise, cycles, quit_on_success = env_config[config_idx]

        policy = functools.reduce(
            lambda a, b: a if a[0] == env_name else b, test_cases_latest_nonoise
        )[1]
        env = ALL_ENVS[env_name]()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True

        success_num = 0
        # env_name += '-goal-hidden'

        for i in range(collect_num + 10):

            env._partially_observable = False
            env._freeze_rand_vec = False
            env._set_task_called = True
            env.seed(i)
            env.reset()

            env.reset_model()
            o = env.reset()
            rollout_success = False
            imgs = [
                env.sim.render(
                    *resolution, mode="offscreen", camera_name=camera
                ).astype(np.uint8)
            ]
            action_space_ptp = env.action_space.high - env.action_space.low

            temp_state_list = []
            temp_next_state_list = []
            temp_action_list = []
            temp_reward_list = []
            temp_done_list = []
            temp_string_list = []

            for step in range(env.max_path_length):
                temp_state_list.append(o)
                a = policy.get_action(o)
                a = np.random.normal(a, noise * action_space_ptp)
                # clip action to be within the action space
                a = np.clip(a, env.action_space.low, env.action_space.high)

                o, r, done, info = env.step(a)

                imgs.append(
                    env.sim.render(*resolution, mode="offscreen", camera_name=camera)
                )
                temp_next_state_list.append(o)
                temp_action_list.append(a)

                if info["success"]:
                    rollout_success = True
                    success_num += 1
                    done = 1
                    temp_reward_list.append(r)
                    temp_done_list.append(done)
                    break
                else:
                    done = 0
                    temp_reward_list.append(r)
                    temp_done_list.append(done)

            if rollout_success:

                # corrupts quickly, so let's save here instead
                if env_name not in h5_traj:
                    h5_traj.create_group(env_name)
                h5_traj[env_name].create_group(str(i))
                h5_traj[env_name][str(i)]["state"] = np.array(temp_state_list)
                h5_traj[env_name][str(i)]["next_state"] = np.array(temp_next_state_list)
                h5_traj[env_name][str(i)]["action"] = np.array(temp_action_list)
                h5_traj[env_name][str(i)]["reward"] = np.array(temp_reward_list)
                h5_traj[env_name][str(i)]["done"] = np.array(temp_done_list)
                h5_traj[env_name][str(i)]["string"] = np.array(temp_string_list).astype("S")
                h5_traj[env_name][str(i)]["env_id"] = np.array(
                    [env_name] * len(temp_string_list)
                ).astype("S")
                h5_traj[env_name][str(i)]["img"] = np.array(imgs)
                
                if args.save_video:
                    folder_name = os.path.join(args.video_path, env_name)
                    if not os.path.exists(folder_name):
                        os.makedirs(folder_name)
                    video_path = os.path.join(folder_name, f"{env_name}_{i}.mp4")
                    imageio.mimwrite(
                        video_path,
                        np.array(imgs),
                        fps=30,
                    )

            if success_num >= collect_num:
                break
    h5_traj.close()


if __name__ == "__main__":

    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--collect_num", type=int, default=5, help="number of demos to collect"
    )
    argparse.add_argument(
        "--save_path", type=str, default="datasets", help="path to save the data"
    )
    argparse.add_argument(
        "--save_video", action="store_true", help="whether to save the video"
    )
    argparse.add_argument(
        "--video_path", type=str, default="datasets/videos", help="path to save the videos"
    )
    args = argparse.parse_args()
    main(args)