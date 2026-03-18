"""Download Charades-STA annotations from Hugging Face Storage bucket.

This script downloads annotations from the private HF Storage bucket:
hf://buckets/max044/charades-sta-storage
"""

import os
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    HAS_HF = True
except ImportError:
    HAS_HF = False

DATA_DIR = "./data"
BUCKET_URL = "hf://buckets/max044/charades-sta-storage/Charades_v1_480"


def main():
    if not HAS_HF:
        print("Error: huggingface_hub not installed. Run: pip install huggingface-hub")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    # Download train annotations
    train_dest = os.path.join(DATA_DIR, "charades_sta_train.txt")
    _download_from_bucket("charades_sta_train.txt", train_dest)

    # Download test annotations
    test_dest = os.path.join(DATA_DIR, "charades_sta_test.txt")
    _download_from_bucket("charades_sta_test.txt", test_dest)

    print("\n✓ All annotations downloaded successfully!")


def _download_from_bucket(filename: str, dest: str):
    """Download a file from the HF Storage bucket."""
    try:
        downloaded_path = hf_hub_download(
            repo_id="max044/charades-sta-storage",
            filename=f"Charades_v1_480/{filename}",
            repo_type="dataset",
            local_dir=DATA_DIR,
            local_dir_use_symlinks=False,
        )

        # Rename to expected location if needed
        if downloaded_path != dest:
            Path(downloaded_path).rename(dest)

        # Count lines
        with open(dest, "r") as f:
            lines = sum(1 for _ in f)

        print(f"✓ Downloaded {filename} ({lines} entries)")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        print("Trying fallback to GitHub...")
        _download_from_github(filename, dest)


def _download_from_github(filename: str, dest: str):
    """Fallback: download from GitHub MESM repo."""
    import urllib.request

    url = f"https://raw.githubusercontent.com/lntzm/MESM/main/data/charades/annotations/{filename}"

    try:
        urllib.request.urlretrieve(url, dest)

        with open(dest, "r") as f:
            lines = sum(1 for _ in f)

        print(f"✓ Downloaded {filename} from GitHub ({lines} entries)")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")


if __name__ == "__main__":
    main()
