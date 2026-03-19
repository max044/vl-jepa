"""
Download full Charades-STA dataset for training.
Downloads all annotations and videos from Hugging Face Storage.

Usage:
    uv run training/download_data.py

Data stored in data/ (15GB total)
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
HF_BUCKET = "max044/charades-sta-storage"
BASE_URL = "https://raw.githubusercontent.com/max044/vl-jepa/main/data"

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_annotations():
    """Download train/test annotations."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    files = ["charades_sta_train.txt", "charades_sta_test.txt"]
    for fname in files:
        fpath = DATA_DIR / fname
        if fpath.exists():
            print(f"  ✓ {fname} already exists")
            continue
        
        url = f"{BASE_URL}/{fname}"
        print(f"  📥 Downloading {fname}...")
        subprocess.run(["curl", "-L", "-o", str(fpath), url], check=True)
    
    print(f"✅ Annotations ready")


def download_all_videos():
    """Download all videos from HF Storage (9,848 videos, ~15GB)."""
    video_dir = DATA_DIR / "Charades_v1_480"
    
    if video_dir.exists() and len(list(video_dir.glob("*.mp4"))) >= 9000:
        print(f"✓ Videos already downloaded ({len(list(video_dir.glob('*.mp4')))} files)")
        return
    
    print("📥 Downloading all videos from Hugging Face Storage...")
    print("   This will download ~15GB of data (9,848 videos)")
    print("   Source: max044/charades-sta-storage")
    print()
    
    try:
        # Use hf sync to download all videos
        cmd = [
            "hf", "sync",
            f"hf://buckets/{HF_BUCKET}/Charades_v1_480",
            str(video_dir),
            "--include", "*.mp4"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        num_videos = len(list(video_dir.glob("*.mp4")))
        print(f"✅ Downloaded {num_videos} videos")
        
    except Exception as e:
        print(f"⚠️  Error downloading videos: {e}")
        print("   You can also download manually from:")
        print(f"   https://huggingface.co/datasets/{HF_BUCKET}")
        sys.exit(1)


def verify_setup():
    """Verify that all data is ready."""
    train_file = DATA_DIR / "charades_sta_train.txt"
    test_file = DATA_DIR / "charades_sta_test.txt"
    video_dir = DATA_DIR / "Charades_v1_480"
    
    if not train_file.exists() or not test_file.exists():
        print("❌ Annotations missing")
        return False
    
    videos = list(video_dir.glob("*.mp4"))
    if len(videos) < 9000:
        print(f"⚠️  Only {len(videos)} videos found (expected 9,848)")
        return False
    
    print(f"✅ Setup verified: {len(videos)} videos, annotations ready")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download full Charades-STA dataset")
    parser.add_argument("--verify-only", action="store_true", 
                       help="Only verify setup, don't download")
    args = parser.parse_args()

    print("=" * 60)
    print("VL-JEPA: Full Dataset Download")
    print("=" * 60)
    print()

    if args.verify_only:
        if verify_setup():
            print("\n✅ All data is ready!")
            sys.exit(0)
        else:
            print("\n❌ Data incomplete")
            sys.exit(1)

    # Step 1: Download annotations
    print("Step 1: Downloading annotations...")
    download_annotations()
    print()

    # Step 2: Download all videos
    print("Step 2: Downloading videos...")
    download_all_videos()
    print()

    # Step 3: Verify
    print("Step 3: Verifying setup...")
    if verify_setup():
        print("\n" + "=" * 60)
        print("✅ Done! Dataset ready for training.")
        print("=" * 60)
        print(f"\nLocation: {DATA_DIR}")
        print(f"Training samples: ~12,000")
        print(f"Videos: 9,848")
        print(f"\nNext step: bash scripts/cloud_train.sh")
    else:
        print("\n❌ Setup incomplete. Please check errors above.")
        sys.exit(1)
