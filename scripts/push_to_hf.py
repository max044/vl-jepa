"""Upload dataset to Hugging Face for fast cloud downloads."""

import os
import argparse
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Upload Charades data to HF")
    parser.add_argument("--repo-id", type=str, required=True, help="HF repo ID (e.g. username/dataset-name)")
    parser.add_argument("--data-dir", type=str, default="./data", help="Local data directory")
    args = parser.parse_args()

    api = HfApi()

    # Create repo if it doesn't exist
    print(f"▸ Ensuring dataset repo '{args.repo_id}' exists...")
    try:
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"  Note: Could not create/verify repo (might already exist or permission issue): {e}")

    print(f"▸ Uploading folder '{args.data_dir}' to HF...")
    
    # We upload the content of the data folder
    api.upload_folder(
        folder_path=args.data_dir,
        repo_id=args.repo_id,
        repo_type="dataset",
        # Ignore checkpoints if they are in data
        ignore_patterns=["*.pth", ".ipynb_checkpoints", "__pycache__"]
    )

    print("\n" + "="*50)
    print(f"✓ Upload complete! Repo: https://huggingface.co/datasets/{args.repo_id}")
    print("="*50)
    print("\nNext step (on cloud instance):")
    print(f"uv run huggingface-cli download {args.repo_id} --local-dir data --repo-type dataset")

if __name__ == "__main__":
    main()
