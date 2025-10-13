import os
from utils.utils import generate_rewind_data
from utils.utils import compute_mse_from_sequences
from utils.utils import compute_correlation_multi_annotations, rank_comparison, compute_correlation_from_sequences



os.environ["TOKENIZERS_PARALLELISM"] = "False"


def compute_metrics_multi(args, rewind_model, gt_data, close_success_data, all_fail_data, task_list, epoch = None):
    print("Computing corrlation metrics...")
    confusion_matrix, all_seqs, tasks, _ = generate_rewind_data(
        h5_file=gt_data,
        task_subset = task_list,
        set_type="eval",
        rewind_model=rewind_model,
        args = args,
    )

    _, all_seqs1, _, _ = generate_rewind_data(
            h5_file=gt_data,
            task_subset=task_list,
            set_type="eval",
            rewind_model=rewind_model,
            args = args,
            annotation = 1,
        )

    _, all_seqs2, _, _ = generate_rewind_data(
        h5_file=gt_data,
        task_subset=task_list,
        set_type="eval",
        rewind_model=rewind_model,
        args = args,
        annotation = 2,
    )

    _, all_seqs3, _, _ = generate_rewind_data(
        h5_file=gt_data,
        task_subset=task_list,
        set_type="eval",
        rewind_model=rewind_model,
        args = args,
        annotation = 3,
    )

    confusion_matrix_all_fail, _, _, _ = generate_rewind_data(
        h5_file=all_fail_data,
        task_subset=task_list,
        set_type="eval",
        rewind_model=rewind_model,
        args = args,
    )

    confusion_matrix_close_success, _, _, _ = generate_rewind_data(
        h5_file=close_success_data,
        task_subset=task_list,
        set_type="eval",
        rewind_model=rewind_model,
        args = args,
    )

    compute_correlation_from_sequences(
        all_seqs=all_seqs,
        env_names=tasks,
        set_type="eval",
        epoch=epoch
    )

    # compute_mse_from_sequences(
    #     all_seqs=all_seqs,
    #     env_names=tasks,
    #     set_type="eval",
    #     epoch=epoch
    # )

    compute_correlation_multi_annotations(
        all_seqs_a=all_seqs1,
        all_seqs_b=all_seqs2,
        all_seqs_c=all_seqs3,
        all_seqs_d=all_seqs,
        env_names=tasks,
        set_type="eval",
        epoch=epoch
    )

    rank_comparison(confusion_matrix_all_fail, confusion_matrix_close_success, confusion_matrix, tasks, epoch=epoch)
    print("Finished computing metrics.")




