"""Download Charades-STA annotations from Hugging Face Hub."""

import os
import argparse

DATA_DIR = "./data"
REPO_ID = "max044/Charades_v1_480"
FILES = [
    "charades_sta_train.txt",
    "charades_sta_test.txt"
]

def main():
    parser = argparse.ArgumentParser(description="Download Charades-STA annotations")
    parser.add_argument("--repo-id", type=str, default=REPO_ID, help="Hugging Face dataset repo ID")
    parser.add_argument("--local-dir", type=str, default=DATA_DIR, help="Local directory to save files")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub is not installed.")
        print("Install it with: uv pip install huggingface_hub")
        return

    print(f"Downloading annotations from {args.repo_id}...")
    
    for filename in FILES:
        dest = os.path.join(args.local_dir, filename)
        if os.path.exists(dest):
            print(f"  Already exists: {dest}")
            continue
            
        try:
            downloaded_path = hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=args.local_dir
            )
            print(f"  ✓ Downloaded {filename} -> {downloaded_path}")
        except Exception as e:
            print(f"  ❌ Failed to download {filename}: {e}")
            print("  Make sure you are authenticated if the repository is private:")
            print("  hf auth login --token YOUR_TOKEN")

if __name__ == "__main__":
    main()
