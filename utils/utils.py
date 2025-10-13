import torch
import torch.nn.functional as F
import numpy as np
import wandb
import h5py
import os
import json
import pickle
import matplotlib.pyplot as plt
import textwrap
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

# from models import ClassProgressTransformer
# from model_pe import ClassProgressTransformer

from PIL import Image
from torchvision import transforms as T
from transformers import AutoTokenizer, AutoModel
import io
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

minilm_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L12-v2"
)
minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(
    device
)

DINO_BATCH_SIZE = 128
MAX_NUM_FRAMES_PER_EPISODE = 128


def animate_incremental(frames_tensor, incremental_rewards, fps=15, fig_path="ax2_plot_rewind.png", epoch=0, suboptimal_type="all_fail"):
    """
    Create an animation that shows frames on the left and the incremental reward curve on the right.

    Args:
        frames_tensor: torch.Tensor of shape [N, C, H, W], pixel range [0,1].
        incremental_rewards: list or array of length N, each element is a reward.
        fps: GIF framerate.

    Returns:
        gif_buffer: in-memory BytesIO containing the GIF.
    """
    # 1) Convert frames to numpy for display
    # if isinstance(frames_tensor, torch.Tensor):
    #     frames_np = frames_tensor.cpu().numpy()
    # else:
    #     frames_np = frames_tensor
    # frames_np = (frames_np * 255).astype(np.uint8)
    # frames_np = np.transpose(frames_np, (0, 2, 3, 1))

    # 2) Prepare figure
    # n = len(frames_np)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # image_plot = ax1.imshow(frames_np[0])
    ax1.set_title('Video Frames')
    ax1.axis('off')

    # ax2.set_title('Incremental Rewards')
    ax2.set_xlim(0, len(incremental_rewards) - 1)
    y_min, y_max = min(incremental_rewards), max(incremental_rewards)
    y_range = y_max - y_min if y_max != y_min else 1
    # ax2.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    ax2.set_ylim(-1,1)

    line_plot, = ax2.plot([], [], lw=2, color='blue')
    scat = ax2.scatter([], [], color='red', zorder=5)

    gif_buffer = io.BytesIO()
    images = []

    # 3) Update frames one by one
    for frame_idx in range(len(incremental_rewards)):
        # image_plot.set_array(frames_np[frame_idx])
        line_plot.set_data(np.arange(frame_idx + 1), incremental_rewards[:frame_idx + 1])
        scat.set_offsets(np.array([[frame_idx, incremental_rewards[frame_idx]]]))
        fig.canvas.draw()

        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img_array = img_array.reshape((h, w, 3))
        images.append(Image.fromarray(img_array))

    # 4) Save to GIF
    images[0].save(
        gif_buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=1000 // fps,
        loop=0,
    )
    gif_buffer.seek(0)

    # change above to english comments
    # --------new: save ax2 (Incremental Rewards) subplot as separate PNG --------
    # First ensure the last frame (i.e., the final curve) has been fully drawn
    # (Actually, the above for loop has already drawn it, so we can directly get the ax2 bbox here)
    # Note: Sometimes text or ticks may be clipped too tightly, you can use expanded() to leave some margin



    ax2.tick_params(labelleft=False, left=False)  # Hide left axis

    # 1) Get the bounding box of ax2 relative to the entire figure (in pixels)
    extent_px = ax2.get_window_extent()

    # 2) (fig.dpi_scale_trans) Convert from pixels to inches
    extent_in = extent_px.transformed(fig.dpi_scale_trans.inverted())
    
    # extract width, height and compute center
    width = extent_in.width
    height = extent_in.height
    center_x = extent_in.x0 + width / 2.0
    center_y = extent_in.y0 + height / 2.0

    # 3) Determine the side length of the square bbox
    side = max(width, height)

    # 4) Construct a new square bbox centered at (center_x, center_y)
    x0 = center_x - side / 2.0
    x1 = center_x + side / 2.0
    y0 = center_y - side / 2.0
    y1 = center_y + side / 2.0

    # first construct a square bbox without expansion
    square_extent = Bbox.from_extents(x0, y0, x1, y1)
    
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches=square_extent.expanded(1.2, 1.2))
    buf.seek(0)  # put the cursor to the beginning of the buffer
    pil_image = Image.open(buf)
    pil_image.save("output.png", format="PNG")
    # 4. add to wandb
    fig_path = fig_path.split(".pn")[0] + f"_{suboptimal_type}"
    wandb.log({fig_path: wandb.Image(pil_image, caption=f"EPOCH {epoch}")})

    # ---------------------------------------------------------------------


    plt.close(fig)
    return gif_buffer



def rank_comparison(cm1, cm2, cm3, tasks, epoch=0):

    '''
    Only compare the diagonal values of the confusion matrices (the number of correct predictions for each task),
    give the ranking of each matrix on each task (1=best, 3=worst),
    and count the probability of each matrix getting ranks 1, 2, and 3, and finally upload to W&B.
    Additionally, add a "GT Ranking" metric, which is the success rate of tasks where the diagonal satisfies cm1 < cm2 < cm3.
    New:
    - Calculate the Spearman correlation coefficient with GT=[3,2,1] and take the average over all tasks.
    '''

    # 1. Extract the diagonals of the three matrices
    diag1 = np.diag(cm1)
    diag2 = np.diag(cm2)
    diag3 = np.diag(cm3)

    total_max = np.maximum(diag1, np.maximum(diag2, diag3))
    total_min = np.minimum(diag1, np.minimum(diag2, diag3))

    normed_diag1 = (diag1 - total_min) / (total_max - total_min)
    normed_diag2 = (diag2 - total_min) / (total_max - total_min)
    normed_diag3 = (diag3 - total_min) / (total_max - total_min)

    diff_close_fail = diag2 - diag1 # difference close_succ - all_fail
    diff_success_close = diag3 - diag2 # difference success - close_succ

    avg_close_fail = np.mean(diff_close_fail)
    avg_success_close = np.mean(diff_success_close)
    avg_total = (avg_close_fail + avg_success_close) / 2 # this is also diag3 - diag1 ???

    normed_diff_close_fail = normed_diag2 - normed_diag1
    normed_diff_success_close = normed_diag3 - normed_diag2

    normed_avg_close_fail = np.mean(normed_diff_close_fail)
    normed_avg_success_close = np.mean(normed_diff_success_close)
    normed_avg_total = (normed_avg_close_fail + normed_avg_success_close) / 2

    # for i, env_name in enumerate(tasks):
    #     wandb.log({f"suboptimal_reward/{env_name}_all_fail": diag1[i], f"suboptimal_reward/{env_name}_close_success": diag2[i], f"suboptimal_reward/{env_name}_success": diag3[i], "epoch": epoch})
    #     wandb.log({f"no_norm_self_collected/{env_name}_close_fail_difference": diff_close_fail[i], f"no_norm_self_collected/{env_name}_success_close_difference": diff_success_close[i], f"no_norm_self_collected{env_name}_avg_diff": (diff_success_close[i] +diff_close_fail[i]) / 2,  "epoch": epoch})
    #     wandb.log({f"norm_self_collected/{env_name}_close_fail_difference": normed_diff_close_fail[i], f"norm_self_collected/{env_name}_success_close_difference": normed_diff_success_close[i], f"norm_self_collected/{env_name}_avg_diff": (normed_diff_close_fail[i] + normed_diff_success_close[i]) / 2 , "epoch": epoch})

    num_tasks = len(diag1)

    # ranks[i, j] represents the rank (1/2/3) of the j-th matrix for the i-th task
    ranks = np.zeros((num_tasks, 3), dtype=np.int32)

    # 2. Rank each task by comparing the three diagonals
    gt_count = 0 # count how many times cm1 < cm2 < cm3

    for i in range(num_tasks):
        d1, d2, d3 = diag1[i], diag2[i], diag3[i]
        values = [d1, d2, d3]
        
        # sorted_indices[0] -> index of smallest value
        sorted_indices = np.argsort(values)  # ascending order
        
        # sorted_indices = [index_of_min, index_of_mid, index_of_max]
        # we want ranks[i, index_of_max] = 1, ranks[i, index_of_mid] = 2, ranks[i, index_of_min] = 3

        ranks[i, sorted_indices[0]] = 3  # smallest value => rank 3
        ranks[i, sorted_indices[1]] = 2
        ranks[i, sorted_indices[2]] = 1  # maximum value => rank 1

        # check if d1 < d2 < d3
        if d1 < d2 < d3:
            gt_count += 1

    # 3. compute average ranks (axis=0 => average over rows for each column)
    avg_ranks = np.mean(ranks, axis=0)  # shape (3,)

    # 4. count how many times each matrix gets rank 1, 2, and 3 across all tasks
    rank1_counts = np.sum(ranks == 1, axis=0)  # (3,)
    rank2_counts = np.sum(ranks == 2, axis=0)
    rank3_counts = np.sum(ranks == 3, axis=0)

    # change to calculate probabilities (count / num_tasks)
    rank1_probs = rank1_counts / num_tasks
    rank2_probs = rank2_counts / num_tasks
    rank3_probs = rank3_counts / num_tasks

    # 5. compute "GT Ranking" success rate
    gt_success_rate = gt_count / num_tasks if num_tasks > 0 else 0.0

    # ========== new: Calculate Spearman correlation coefficient (with GT=[3,2,1]) and average over all tasks ==========
    # ground truth ranking
    gt_ranks = np.array([3, 2, 1])

    spearman_sum = 0.0
    # for every task [cm1_rank, cm2_rank, cm3_rank], compute Spearman correlation with [3,2,1]
    for i in range(num_tasks):
        predicted = ranks[i, :]  # e.g. [3,2,1] / [2,1,3] etc.
        rho, p_value = spearmanr(gt_ranks, predicted)
        spearman_sum += rho

    # get average
    spearman_avg = spearman_sum / num_tasks if num_tasks > 0 else 0.0

    # 7. upload to wandb
    wandb.log({
        f"Average_Reward_Rank/all_fail": float(avg_ranks[0]),
        f"Average_Reward_Rank/close_success": float(avg_ranks[1]),
        f"Average_Reward_Rank/Ground_Truth": float(avg_ranks[2]),

        # rank=1 
        f"Rank1_Probability/all_fail": float(rank1_probs[0]),
        f"Rank1_Probability/close_success": float(rank1_probs[1]),
        f"Rank1_Probability/Ground_Truth": float(rank1_probs[2]),

        # rank=2 
        f"Rank2_Probability/all_fail": float(rank2_probs[0]),
        f"Rank2_Probability/close_success": float(rank2_probs[1]),
        f"Rank2_Probability/Ground_Truth": float(rank2_probs[2]),

        # rank=3 
        f"Rank3_Probability/all_fail": float(rank3_probs[0]),
        f"Rank3_Probability/close_success": float(rank3_probs[1]),
        f"Rank3_Probability/Ground_Truth": float(rank3_probs[2]),

        # New "GT Ranking" success rate
        f"Policy_Rollout_Ranking/Ground_Truth_Ranking_Success_Rate": float(gt_success_rate),

        # New: Spearman correlation coefficient (average)
        f"Policy_Rollout_Ranking/Average_Reward_Ranking (GT=[3,2,1])": float(spearman_avg),

        # f"no_norm_self_collected/avg_close_fail_diff": avg_close_fail,
        # f"no_norm_self_collected/avg_success_close_diff": avg_success_close,
        # f"no_norm_self_collected/avg_total_diff": avg_total,

        # f"Policy_Rollout_Ranking/avg_close_fail_diff": normed_avg_close_fail,
        # f"Policy_Rollout_Ranking/avg_success_close_diff": normed_avg_success_close,
        f"Policy_Rollout_Ranking/Average_Reward_Difference": normed_avg_total,

        "epoch": epoch
    })
    
    return avg_ranks, rank1_probs, rank2_probs, rank3_probs, gt_success_rate, spearman_avg



def sample_embedding_frames(embeddings, num_frames = 32):
    total_frames = embeddings.shape[0]
    if total_frames > num_frames:
        index = np.linspace(0, total_frames-1, num_frames).astype(int)
        embeddings = embeddings[index]

    else:
        # padding 1st frame
        padding_num = num_frames - total_frames
        first_frame = embeddings[0].unsqueeze(0)
        padding_frames = first_frame.repeat(padding_num, 1)
        embeddings = torch.cat([padding_frames, embeddings], dim=0)
    return embeddings




def compute_rewind_reward(rewind_model, args, episode_image_embeddings, lang_embeddings, pdf_path="reward_rewind.png", epoch=0, suboptimal_type = "all_fail"):
    

    reward_seq = []

    for i in range(1, episode_image_embeddings.shape[0]+1):
        partial_image_embeddings = episode_image_embeddings[:i]
        if args.subsample_video:
            processed_video_embedding = sample_embedding_frames(
                partial_image_embeddings, args.max_length
            ).unsqueeze(0)

        pred_class = rewind_model(processed_video_embedding, lang_embeddings)
        pred_class = pred_class 
        predicted_classes = np.array(pred_class.squeeze().detach().cpu().numpy())
        predicted_classes = predicted_classes[1:]  # Remove first frame prediction


        video_reward = predicted_classes[-1] if len(predicted_classes) > 0 else 0.0
        reward_seq.append(video_reward)
    
    gif_buffer = animate_incremental(None, reward_seq, fps=15, fig_path=pdf_path, epoch=epoch, suboptimal_type=suboptimal_type)

    return reward_seq


def generate_rewind_data(
    h5_file,
    task_subset,
    set_type: str,
    rewind_model,
    device="cuda",
    args=None,
    annotation = None,
):

    '''
    # translated comments to english
    Similar to generate_gemini_data, iterate over (environment, text) combinations to generate:
      1) (N x N) confusion matrix (single value at the last frame)
      2) (N complete sequences, each containing incremental results for 5 demos), only perform "frame-by-frame" or "incremental" inference for i==j combinations
    Returns:
        (confusion_matrix, predicted_sequences, tasks, text_list)
            confusion_matrix: shape=(N,N), float, representing [0..1] averaged over 5 demos
            predicted_sequences: List[Optional[List[List[float]]]]
                # predicted_sequences[i] = [ seq_demo_0, seq_demo_1, ..., seq_demo_4 ]
                # each seq_demo_k is a segment of frame-by-frame/incremental results
            tasks: list of task names
            text_list: list of text descriptions (here it is minilm_lang_embedding, shape=[N, 1, 384] etc.)
    '''

    # 1) from read tasks
    if set_type == "train":
        tasks = task_subset["training_tasks"]
    elif set_type == "eval":
        tasks = task_subset["eval_tasks"]
    else:
        tasks = task_subset["test_tasks"]
    num_tasks = len(tasks)
    # 2) generate metrics
    # confusion matrix shape: (num_tasks, num_tasks)
    confusion_matrix = np.zeros((num_tasks, num_tasks), dtype=np.float32)

    # predicted_sequences[i] => None or [[seq_for_demo0], [seq_for_demo1], ... , [seq_for_demo4]]
    predicted_sequences = [None] * num_tasks

    # 3) Read all texts (corresponding to tasks) => text_list
    text_list = []
    for env in tasks:
        # Read minilm_lang_embedding for each env
        group = h5_file[env]
        if annotation != None:
            text_arr = np.asarray(group[f"minilm_lang_embedding_{annotation}"])  # shape (1, 384)
        else:
            text_arr = np.asarray(group["minilm_lang_embedding"])
        text_list.append(text_arr)
    # e.g. shape: (N, 1, 384) after stacking
    text_list = np.stack(text_list)
    text_list = torch.tensor(text_list).to(device).float()


    # 4) Iterate over all (i, j) combinations
    for i, env_video_name in enumerate(tasks):
        
        group = h5_file[env_video_name]
        # Gather all 5 embeddings
        # Some HDF5 might not have exactly 5, so we do `range(5)` but check if dataset exists.
        # 1) find all digit keys, sort and take first 5
        digit_keys = sorted([k for k in group.keys() if k.isdigit()], key=lambda x: int(x))
        selected_keys = digit_keys[:5]

        # 2) Collect their corresponding embeddings
        all_video_embeddings = []
        for key_name in selected_keys:
            embed_arr = np.asarray(group[key_name])  # shape (32, 768)
            video_emb = torch.tensor(embed_arr).to(device).float()
            all_video_embeddings.append(video_emb)

        if len(all_video_embeddings) == 0:
            # If no embeddings found, skip
            print(f"Warning: no embeddings for {env_video_name}")
            continue

        # For partial-sequence saving:
        # predicted_seq_for_5 = list of 5 partial sequences
        predicted_seq_for_5 = [[] for _ in range(len(all_video_embeddings))]

        # 4.2) Iterate over all text embeddings j
        for j, text_embedding in enumerate(text_list):
            text_embedding = text_embedding.squeeze(0)  # shape (1, 384)
            # ---- A) Compute final reward for 5 demos and average ----
            final_rewards = []
            reward_seq = []
            for demo_id, video_embedding in enumerate(all_video_embeddings):
                # here the example just does sample_embedding_frames => forward
                if args.subsample_video:
                    processed_video_embedding = sample_embedding_frames(
                        video_embedding, args.max_length
                    ).unsqueeze(0)

                # forward
                if len(text_embedding.shape) == 1:
                    text_embedding = text_embedding.unsqueeze(0)
                pred_class = rewind_model(processed_video_embedding, text_embedding)
                predicted_classes = np.array(pred_class.squeeze().detach().cpu().numpy())
                predicted_classes = predicted_classes[1:]  # Remove first frame prediction

                reward_seq.append(predicted_classes)
                # final = last index
                video_reward = predicted_classes[-1] if len(predicted_classes) > 0 else 0.0

                final_rewards.append(video_reward)

            # average final reward for confusion_matrix
            if len(final_rewards) > 0:
                avg_reward = float(sum(final_rewards) / len(final_rewards))
            else:
                avg_reward = 0.0

            confusion_matrix[i, j] = avg_reward
            
            # ---- B) If i == j => frame-by-frame/incremental prediction sequences (for each of the 5 demos) ----
            if i == j:
                for demo_id, video_embedding in enumerate(all_video_embeddings):
                    # Build partial seq: for partial_count in [1..args.max_length) 
                    partial_seq = []
                    # We can reuse the same "predicted_classes" from above or re-run.
                    # For simplicity, we show re-running in partial. But you could store them once above if you prefer.
                    
                    # Here we assume your model always expects full sequence,
                    # so we do sample_embedding_frames(video_embedding, partial_count).
                    # If your partial is truly partial frames, you do something else.
                    for partial_count in range(1, args.max_length):       
                        sub_reward = reward_seq[demo_id][partial_count-1] 
                        partial_seq.append(sub_reward)

                    # store partial_seq
                    predicted_seq_for_5[demo_id] = partial_seq

                # Now assign the 5 partial sequences to predicted_sequences[i]
                predicted_sequences[i] = predicted_seq_for_5
    return confusion_matrix, predicted_sequences, tasks, text_list




def compute_correlation_from_sequences(
    all_seqs,
    env_names,
    set_type: str,
    epoch = 0
):

    '''
    now assume:
        all_seqs[i] => a list of length 5, 
                         where each element seq_demo_k => shape like [val0, val1, ..., valM].
        env_names[i] => the name of the i-th environment (str).
    for each environment i:
        1) iterate over its 5 sequences, calculate the Pearson correlation coefficient for each sequence
           - if the sequence is all zeros or length < 2, then correlation coefficient = 0
        2) calculate the mean and standard deviation of the correlation coefficients of the 5 sequences -> wandb.log()
        3) save the mean value to env_avg_correlations
    finally, average env_avg_correlations (overall_avg) and upload to wandb.
    '''

    if len(all_seqs) != len(env_names):
        print("[!] all_seqs and env_names length mismatch, cannot correspond one-to-one.")
        return 0.0, []
    pearson_env_avg_correlations = [] 
    spearmans_env_avg_correlations = []

    for i, five_seqs in enumerate(all_seqs):
        env_name = env_names[i]

        # five_seqs = [seq_demo0, seq_demo1, seq_demo2, seq_demo3, seq_demo4]
        pearson_cor_vals = []
        spearman_cor_vals = []

        for seq_demo in five_seqs:
            if len(seq_demo) < 2:
                # if length < 2, cannot compute correlation coefficient => 0
                pearson_cor_vals.append(0.0)
                spearman_cor_vals.append(0.0)
                continue

            pred_array = np.array(seq_demo, dtype=np.float32)

            # if all elements are 0, set correlation coefficient to 0
            if np.allclose(pred_array, 0):
                pearson_cor_vals.append(0.0)
                spearman_cor_vals.append(0.0)
                continue

            n = len(pred_array)
            # construct GT from 0..1
            gt_array = np.linspace(0, 1, n, dtype=np.float32)

            pearson, _ = pearsonr(pred_array, gt_array)
            spearman, _ = spearmanr(pred_array, gt_array)
            pearson_cor_vals.append(float(pearson))
            spearman_cor_vals.append(float(spearman))

        # compute mean & std of these 5 demos
        if len(pearson_cor_vals) > 0:
            pearson_avg = float(np.mean(pearson_cor_vals))
            spearman_avg = float(np.mean(spearman_cor_vals))
            pearson_std = float(np.std(pearson_cor_vals))
            spearman_std = float(np.std(spearman_cor_vals))
        else:
            pearson_avg = 0.0
            pearson_std = 0.0
            spearman_avg = 0.0
            spearman_std = 0.0

        # wandb.log({
        #     f"{set_type}_pearson_correlation_thrd/{env_name}_avg": pearson_avg,
        #     f"{set_type}_pearson_correlation_thrd/{env_name}_std": pearson_std,
        #     f"{set_type}_spearman_correlation_thrd/{env_name}_avg": spearman_avg,
        #     f"{set_type}_spearman_correlation_thrd/{env_name}_std": spearman_std,
        #     "epoch": epoch
        # })

        pearson_env_avg_correlations.append(pearson_avg)
        spearmans_env_avg_correlations.append(spearman_avg)

    if len(pearson_env_avg_correlations) > 0:
        pearson_overall_avg = float(np.mean(pearson_env_avg_correlations))
        spearman_overall_avg = float(np.mean(spearmans_env_avg_correlations))
    else:
        pearson_overall_avg = 0.0
        spearman_overall_avg = 0.0

    # log overall
    wandb.log({f"{set_type}_Demo_Reward_Alignment/Average_Pearson": pearson_overall_avg, "epoch": epoch})
    wandb.log({f"{set_type}_Demo_Reward_Alignment/Average_Spearman": spearman_overall_avg, "epoch": epoch})
    return pearson_overall_avg, pearson_env_avg_correlations, spearman_overall_avg, spearmans_env_avg_correlations


def compute_mse_from_sequences(
    all_seqs,
    env_names,
    set_type: str,
    epoch = 0
):
    '''
    Now all_seqs[i] => a "list of demos", e.g. 5 sequences:
        all_seqs[i] = [ seq_demo0, seq_demo1, ..., seq_demo4 ]
      Each sequence is [val0, val1, ..., valN], values in range (0..100).
    We compute Overall MSE & Final-frame MSE for multiple demos under the same environment, then average them.
    Then we plot only one figure on wandb, but it contains curves for multiple demos.
    '''

    # 1) check length
    if len(all_seqs) != len(env_names):
        print("[!] all_seqs and env_names length mismatch, cannot correspond one-to-one.")
        return 0.0, 0.0, [], []

    # store each environment's (after aggregating multiple demos) MSE and final-MSE
    mse_list = []
    final_mse_list = []

    # 2) Iterate over each environment i
    for i, demo_seqs in enumerate(all_seqs):
        env_name = env_names[i]
        local_mses = []
        local_final_mses = []
        fig, ax = plt.subplots(figsize=(6, 4))

        # 3) Iterate over multiple demos under the same environment
        for demo_idx, seq in enumerate(demo_seqs):
            if len(seq) < 2:
                # if length smaller than 2, cannot compute MSE
                mse_val = 0.0
                final_mse_val = 0.0
                ax.text(0.5, 0.5, f"Demo {demo_idx} <2 frames", ha="center", va="center")
            else:
                pred_array = np.array(seq, dtype=np.float32)
                n = len(pred_array)
                gt_array = np.linspace(0, 1, n + 1, dtype=np.float32)[1:]
                mse_val = mean_squared_error(gt_array, pred_array)
                final_mse_val = mean_squared_error([gt_array[-1]], [pred_array[-1]])

                frame_numbers = np.arange(n)
                ax.plot(frame_numbers, gt_array, label=f"GT_demo{demo_idx}", linestyle="dashed")
                ax.plot(frame_numbers, pred_array, label=f"Pred_demo{demo_idx}", linestyle="-")

            local_mses.append(mse_val)
            local_final_mses.append(final_mse_val)

        # 4) Average MSE across multiple demos under the same environment
        if len(local_mses) > 0:
            env_mse_val = float(np.mean(local_mses))
            env_final_mse_val = float(np.mean(local_final_mses))
        else:
            env_mse_val = 0.0
            env_final_mse_val = 0.0

        mse_list.append(env_mse_val)
        final_mse_list.append(env_final_mse_val)

        wandb.log({f"{set_type}_overall_mse_thrd/{env_name}": env_mse_val, "epoch": epoch})
        wandb.log({f"{set_type}_final_mse_thrd/{env_name}": env_final_mse_val, "epoch": epoch})

        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Reward (0..1)")
        ax.set_title(f"{env_name} Prediction vs. GT (Multiple Demos)")
        ax.legend()
        plt.tight_layout()
        wandb.log({f"{set_type}_curve_thrd/{env_name}": wandb.Image(fig, caption=f"Epoch {epoch}")})
        plt.close(fig)

    avg_mse = float(np.mean(mse_list)) if mse_list else 0.0
    avg_final_mse = float(np.mean(final_mse_list)) if final_mse_list else 0.0

    wandb.log({f"{set_type}_mse_thrd/average_overall_mse": avg_mse, "epoch": epoch})
    wandb.log({f"{set_type}_mse_thrd/average_final_mse": avg_final_mse, "epoch": epoch})

    return avg_mse, avg_final_mse, mse_list, final_mse_list




def compute_correlation_multi_annotations(
    all_seqs_a,  
    all_seqs_b,  
    all_seqs_c,  
    all_seqs_d,  
    env_names,    
    set_type: str,
    epoch = 0
):
    '''
    for each task i:
        1) from all_seqs_a[i], take out 5 sequences => compute spearman correlation for each => average => corr_a
        2) from all_seqs_b[i], take out 5 sequences => compute spearman correlation for each => average => corr_b
        3) from all_seqs_c[i], take out 5 sequences => compute spearman correlation for each => average => corr_c
        4) from all_seqs_d[i], take out 5 sequences => compute spearman correlation for each => average => corr_d
        Then compute the mean of [corr_a, corr_b, corr_c, corr_d] => task_avg_corr
                                and compute the variance => task_avg_var (sample variance, ddof=1).
        Finally, compute the overall average of each task's (avg_corr, avg_var) => overall_avg_corr, overall_avg_var
    '''

    num_tasks = len(env_names)

    if not (len(all_seqs_a) == len(all_seqs_b) == len(all_seqs_c) == len(all_seqs_d) == num_tasks):
        print("[Error] four all_seqs length and env_names do not match")
        return 0.0, 0.0

    task_corrs = []
    task_vars = []

    for i in range(num_tasks):
        env_name = env_names[i]

        corr_a = compute_avg_spearman(all_seqs_a[i])
        corr_b = compute_avg_spearman(all_seqs_b[i])
        corr_c = compute_avg_spearman(all_seqs_c[i])
        corr_d = compute_avg_spearman(all_seqs_d[i])

        approach_corrs = np.array([corr_a, corr_b, corr_c, corr_d], dtype=np.float32)

        task_avg_corr = float(approach_corrs.mean())

        task_avg_var = float(np.var(approach_corrs, ddof=1))

        # wandb.log({
        #     f"{set_type}_multi_spearman_correlation_thrd/{env_name}_avg_corr": task_avg_corr,
        #     f"{set_type}_multi_spearman_correlation_thrd/{env_name}_avg_var": task_avg_var,
        #     "epoch": epoch})
        

        task_corrs.append(task_avg_corr)
        task_vars.append(task_avg_var)

    if len(task_corrs) > 0:
        overall_avg_corr = float(np.mean(task_corrs))
        overall_avg_var  = float(np.mean(task_vars))
    else:
        overall_avg_corr = 0.0
        overall_avg_var  = 0.0

    wandb.log({
        f"{set_type}_Input_Robustness/Average_Spearman_Correlation": overall_avg_corr,
        f"{set_type}_Input_Robustness/Average_Variance": overall_avg_var,
        "epoch": epoch
    })


    return overall_avg_corr, overall_avg_var


def compute_avg_spearman(five_seqs):
    """
    given five sequence, compute spearman correlation for each, then average.
    if len(seq)<2 or all zeros, set correlation to 0.
    """
    cor_vals = []
    for seq in five_seqs:
        if len(seq) < 2:
            cor_vals.append(0.0)
            continue

        arr = np.array(seq, dtype=np.float32)
        if np.allclose(arr, 0):
            cor_vals.append(0.0)
            continue

        n = len(arr)
        gt = np.linspace(1, n, n, dtype=np.float32)
        r, _ = spearmanr(arr, gt)
        if np.isnan(r):
            r = 0.0
        cor_vals.append(float(r))

    if len(cor_vals) == 0:
        return 0.0
    return float(np.mean(cor_vals))


