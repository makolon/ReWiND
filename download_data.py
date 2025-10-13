import argparse
from huggingface_hub import hf_hub_download, snapshot_download

arg = argparse.ArgumentParser()
arg.add_argument("--download_path", type=str, default="datasets", help="Path to download the datasets.")
args = arg.parse_args()

openx_embedding_path = "HenryZhang/rewind_raw_data5"
load_dir = snapshot_download(repo_id=openx_embedding_path, repo_type="dataset", local_dir=args.download_path)
print(f"Downloaded dataset to {load_dir}")