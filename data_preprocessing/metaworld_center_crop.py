import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm


TRAIN_TASK_LIST = ['button-press-topdown-wall-v2', 'button-press-v2', 'coffee-pull-v2', 'dial-turn-v2', 'door-open-v2', 
             'door-unlock-v2', 'drawer-close-v2', 'faucet-open-v2', 'hand-insert-v2', 'handle-press-v2', 
             'handle-pull-side-v2', 'peg-insert-side-v2', 'pick-place-v2', 'plate-slide-back-side-v2', 'plate-slide-v2', 
             'push-v2', 'reach-v2', 'stick-pull-v2', 'stick-push-v2', 'window-open-v2'] # list of tasks to process
EVAL_TASK_LIST = ['button-press-topdown-v2', 'button-press-wall-v2', 'coffee-button-v2', 'coffee-push-v2', 'door-close-v2', 
                  'door-lock-v2', 'faucet-close-v2', 'handle-press-side-v2', 'handle-pull-v2', 'pick-place-wall-v2', 
                  'plate-slide-back-v2', 'plate-slide-side-v2', 'push-back-v2', 'reach-wall-v2', 'soccer-v2', 
                  'sweep-into-v2', 'window-close-v2']
VIDEO_ID = ['0', '1', '2', '3', '4'] # type the video id you want to process

def center_crop(image, size):

    h, w = image.shape[:2]
    x = (w - size) // 2
    y = (h - size) // 2
    return image[y:y+size, x:x+size]

def process_video(args, center_crop_path, task_list, id_list=VIDEO_ID):
    """
    Process the video data from the given path and save it to a new h5 file.
    
    Args:
        path (str): Path to the input h5 file containing video data.
        new_h5 (str): Path to the output h5 file where processed data will be saved.
        task_list (list): List of tasks to process.
        id_list (list): List of video IDs to process.
    """
    video_path = os.path.join(args.video_path, "metaworld_generation.h5")
    h5_file = h5py.File(video_path, 'r')
    new_h5_file = h5py.File(center_crop_path, 'w')

    for key in tqdm(h5_file.keys()):
        if key not in task_list:
            continue
        group = h5_file[key]

        if key not in new_h5_file:
            new_h5_file.create_group(key)

        for idx in list(group.keys()):
            if idx not in id_list:
                continue
            videos = np.asarray(group[idx]["img"])

            indices = np.linspace(
                0,
                len(videos) - 1,
                args.max_length,
                dtype=int,
            )
            sampled_images = [videos[i] for i in indices]
            sampled_images = [center_crop(img, 224) for img in sampled_images]

            new_h5_file[key].create_dataset(idx, data=sampled_images)
    new_h5_file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="datasets/")
    parser.add_argument("--target_path", type=str, default="datasets/")
    parser.add_argument("--max_length", type=int, default=32)
    args = parser.parse_args()

    train_h5_path = os.path.join(args.target_path, f"metaworld_centercrop_{args.max_length}_train.h5")
    eval_h5_path = os.path.join(args.target_path, f"metaworld_centercrop_{args.max_length}_eval.h5")

    process_video(args, train_h5_path, task_list=TRAIN_TASK_LIST, id_list=VIDEO_ID)
    print("Center cropping completed and saved to:", train_h5_path)
    process_video(args, eval_h5_path, task_list=EVAL_TASK_LIST, id_list=VIDEO_ID)
    print("Center cropping completed and saved to:", eval_h5_path)