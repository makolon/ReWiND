# ReWiND: Language-Guided Rewards Teach Robot Policies without New Demonstrations. (Oral Presentation @ CoRL 2025) 

![ReWiND Teaser](rewind_teaser.png)

<p align="center">
  <a href="https://arxiv.org/abs/2505.10911">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv2505.10911-b31b1b.svg">
  </a>
   <a href="https://rewind-reward.github.io/">
   <img alt="Website" src="https://img.shields.io/badge/Website-rewind--reward.github.io-blue">
   </a>
</p>

We provide code to train ReWiND reward models and policies on MetaWorld.
The overall pipeline is as follows:
- Train the ReWiND Reward Model on MetaWorld + OXE data.
- Label the offline training dataset with the ReWiND Reward Model.
- Train the ReWiND Policy with offline to online RL for new tasks.

## Installation Instructions:
```bash
git clone git@github.com:Jiahui-3205/ReWiND_Release.git
cd ReWiND_Release/
```


### Create Environment
```bash
# Run the setup script to create environment and install all dependencies
bash -i setup_ReWiND_env.sh
conda activate rewind
```



### WandB Configuration

This project uses Weights & Biases (WandB) for experiment tracking. Before running experiments:

1. **For Policy Training**: Edit `metaworld_policy_training/configs/base_config.yaml` lines 15-16:
   ```yaml
   wandb_entity_name: your-wandb-entity
   wandb_project_name: rewind-policy-training
   ```

2. **To Disable WandB**: Set `logging.wandb=false` when running policy training commands.

## Data Preparation (We recommend to run it with the Default path)

**Data Processing (Recommend run with Default path)**
```bash
# Download preprocessed OpenX DinoV2 Embeddings
python download_data.py --download_path DOWNLOADPATH(Default:datasets)
```

**Generate MetaWorld Trajectories for ReWiND Reward Training (Recommend run with Default path)**
```bash
# Generate Metaworld trajectories
python data_generation/metaworld_generation.py --save_path SAVE_DATA_PATH(Default:datasets)
# Centercrop the videos and convert to DinoV2 features
python data_preprocessing/metaworld_center_crop.py --video_path SAVE_DATA_PATH(Default:datasets) --target_path TARGET_DATASET_PATH(Default:datasets)  
python data_preprocessing/generate_dino_embeddings.py --video_path_folder TARGET_DATASET_PATH(Default:datasets) --target_path EMBEDDING_TARGET_PATH(Default:datasets)
```

## ReWiND Reward Model Training 
```bash
# require wandb entity
python train_reward.py --wandb_entity YOUR_WANDB_ENTITY(Required) \
--wandb_project WANDB_Project_NAME(Default:rewind-reward-training) \
--rewind \
--subsample_video \
--clip_grad \
--cosine_scheduler \
--batch_size 1024 \
--worker 1
```



## ReWiND Metaworld Policy Training

### Label Offline Dataset (Recommend run with default path)
```bash
# Relabel the dataset we collect with ReWiND reward model
python data_preprocessing/metaworld_label_reward.py --reward_model_path CHECKPOINT_PATH --h5_video_path GENERATION_PATH --h5_embedding_path EMBEDDING_TARGET_PATH --output_path OUTPUT_PATH
```

Note:
- `OUTPUT_PATH`: The labeled dataset file path (default: `datasets/metaworld_labeled.h5`). This will be used as `<OUTPUT_PATH>` in [Offline Training](#offline-training) and [Online Training](#online-training) below.

```bash
cd metaworld_policy_training
```


### Policy Offline to Online RL Training
```bash
python train_policy.py metaworld=off_on_15 \
algorithm=wsrl_iql \
reward=rewind_metaworld \
offline_training.offline_training_steps=15000 \
general_training.seed=42 \
environment.env_id=<ENV_ID> \
offline_training.offline_h5_path=<OUTPUT_PATH> \
reward_model.model_path=<CHECKPOINT_PATH>
```

- `<ENV_ID>`: the Metaworld task you want to train online, e.g., `button-press-wall-v2`, `window-close-v2`. Full list of our (not in training data) evaluation tasks in the paper is: [`window-close-v2`, `reach-wall-v2`, `faucet-close-v2`, `coffee-button-v2`, `button-press-wall-v2`, `door-lock-v2`, `handle-press-side-v2`, `sweep-into-v2`]
- `<OFFLINE_CKPT_PATH>`: path to your offline-trained checkpoint directory (often contains `last_offline`) to warm-start online training. If set to `null`, the run will first execute the offline phase for `offline_training.offline_training_steps` steps on the dataset, and then proceed to the online phase. 
- To skip offline learning entirely, set `offline_training.offline_training_steps=0`.


### Optional: Policy Offline Training
We also provide code to just train the policy offline, so that you can load the same offline policy checkpoint for online RL to multiple new tasks downstream.
You only need to set `online_training.total_time_steps=0`. 

After offline training completes, check the `model_dir` in your wandb log to find the `<OFFLINE_CKPT_PATH>` for online training (see [Online Training](#online-training) below).

Then, run the above offline to online RL training command with `offline_training.ckpt_path=<OFFLINE_CKPT_PATH>` as an extra argument to perform online RL directly with the same offline policy.

Note:
- In offline training, `environment.env_id` is not important; the agent is trained over all training tasks found in your offline dataset.
- `<OUTPUT_PATH>` should point to your labeled offline dataset (see [Label Offline Dataset](#label-offline-dataset-recommend-run-with-default-path) above).


## FAQ & Debugging

### Mujoco Installation

1. Download mujoco210 from [mujoco-py installation guide](https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco)
2. Extract the downloaded mujoco210 directory into `~/.mujoco/mujoco210`
3. Add the following lines to `~/.bashrc`:
   ```bash
   export LD_LIBRARY_PATH=~/.mujoco/mujoco210/bin
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
   ```
4. Reload your shell configuration:
   ```bash
   source ~/.bashrc

### Debug
```
fatal error: GL/glew.h: No such file or directory 4 | #include <GL/glew.h>
```
Solution:
check https://github.com/openai/mujoco-py/issues/745


## 📄 Citation
```bibtex
  @inproceedings{
      zhang2025rewind,
      title={ReWi{ND}: Language-Guided Rewards Teach Robot Policies without New Demonstrations},
      author={Jiahui Zhang and Yusen Luo and Abrar Anwar and Sumedh Anand Sontakke and Joseph J Lim and Jesse Thomason and Erdem Biyik and Jesse Zhang},
      booktitle={9th Annual Conference on Robot Learning},
      year={2025},
      url={https://openreview.net/forum?id=XjjXLxfPou}
    }
```
