'''
This file can directly run on non-centercroped videos. 
We have a centercrop video just for visulization. 
'''
import os
import h5py
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from utils.processing_utils import dino_load_image, mean_pooling
from task_annotation import TRAIN_GT_ANN, EVAL_GT_ANN, GENERATE_TRAIN_ANN, EVAL_ANN_1, EVAL_ANN_2, EVAL_ANN_3
TARGET_PATH = "./"
DINO_BATCH_SIZE = 32
MAX_NUM_FRAMES_PER_EPISODE = 32


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", force_reload=False)
dinov2_vits14 = dinov2_model.to(device)
minilm_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L12-v2"
)
minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(
    device
)


def embedding_videos(h5_path, new_h5_path, split = "train"):
    '''
    this function is used to process images to dino embeddings
    '''

    h5_file = h5py.File(h5_path, 'r')
    new_h5_file = h5py.File(new_h5_path, 'w')

    for key in tqdm(h5_file.keys()):
        group = h5_file[key]

        if key not in new_h5_file:
            new_h5_file.create_group(key)

        for idx in list(group.keys()):
            videos = np.asarray(group[idx])
            sampled_images = [videos[i] for i in range(len(videos))]

            with torch.inference_mode():
                # batch it
                episode_images_dino = [
                    dino_load_image(img) for img in sampled_images
                ]
                
                episode_images_dino = [
                    torch.concatenate(
                        episode_images_dino[i : i + DINO_BATCH_SIZE]
                    )
                    for i in range(
                        0, len(episode_images_dino), DINO_BATCH_SIZE
                    )
                ]
                embedding_list = []
                for batch in episode_images_dino:
                    episode_image_embeddings = (
                        dinov2_vits14(batch.to(device))
                        .squeeze()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    embedding_list.append(episode_image_embeddings)
                episode_image_embeddings = np.concatenate(embedding_list)
            new_h5_file[key].create_dataset(
                idx,
                data=episode_image_embeddings,
            )

        # Add language embedding
        lang_embeddings = list()
        if key in TRAIN_GT_ANN:
            anns = [TRAIN_GT_ANN[key]] + GENERATE_TRAIN_ANN[key]
        else:
            anns = [EVAL_GT_ANN[key], EVAL_ANN_1[key], EVAL_ANN_2[key], EVAL_ANN_3[key]] # ad human gen ann for eval

        for task in anns:
            encoded_input = minilm_tokenizer(
                [task], padding=False, truncation=True, return_tensors="pt"
            ).to(device)

            model_output = minilm_model(**encoded_input)
            minlm_task_embedding = (
                mean_pooling(model_output, encoded_input["attention_mask"])
                .cpu()
                .detach()
                .numpy()
            )
            lang_embeddings.append(minlm_task_embedding)

        if split == "train":
            lang_embeddings = np.concatenate(lang_embeddings, axis=0)
            new_h5_file[key].create_dataset("minilm_lang_embedding", data=lang_embeddings)
        else:
            new_h5_file[key].create_dataset("minilm_lang_embedding", data=lang_embeddings[0:1])
            new_h5_file[key].create_dataset("minilm_lang_embedding_1", data=lang_embeddings[1:2])
            new_h5_file[key].create_dataset("minilm_lang_embedding_2", data=lang_embeddings[2:3])
            new_h5_file[key].create_dataset("minilm_lang_embedding_3", data=lang_embeddings[3:4])


    new_h5_file.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--video_path_folder", type=str, default="datasets")
    argparser.add_argument("--target_path", type=str, default="datasets")
    argparser.add_argument("--max_length", type=int, default=32)
    args = argparser.parse_args()

    train_video_path = os.path.join(args.video_path_folder, f"metaworld_centercrop_{args.max_length}_train.h5")
    eval_video_path = os.path.join(args.video_path_folder, f"metaworld_centercrop_{args.max_length}_eval.h5")
    embedding_videos(train_video_path, os.path.join(args.target_path, "metaworld_embeddings_train.h5"), split = "train")
    embedding_videos(eval_video_path, os.path.join(args.target_path, "metaworld_embeddings_eval.h5"), split = "eval")
    print("Finished processing dino embeddings!")
    

