"""
One-time data preparation for VL-JEPA autoresearch.
Downloads annotations and a subset of video data for quick experiments.

Usage:
    uv run prepare.py              # Full prep
    uv run prepare.py --subset 100 # Only 100 videos for testing

Data stored in data/autoresearch/
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
MAX_FRAMES = 16          # frames per video
FRAME_SIZE = 224         # frame resolution

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/autoresearch")
CACHE_DIR = Path.home() / ".cache" / "vl-jepa"
BASE_URL = "https://raw.githubusercontent.com/max044/vl-jepa/main/data"
HF_BUCKET = "max044/charades-sta-storage"

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
            print(f"  {fname} already exists")
            continue
        
        url = f"{BASE_URL}/{fname}"
        print(f"  Downloading {fname}...")
        subprocess.run(["curl", "-L", "-o", str(fpath), url], check=True)
    
    print(f"✅ Annotations ready at {DATA_DIR}")


def download_video_subset(num_videos=500):
    """Download a subset of videos from HF Storage."""
    video_dir = DATA_DIR / "Charades_v1_480"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Check existing videos
    existing = list(video_dir.glob("*.mp4"))
    if len(existing) >= num_videos:
        print(f"✅ {len(existing)} videos already exist")
        return
    
    print(f"📥 Downloading video subset ({num_videos} videos)...")
    print("   Using Hugging Face Storage (XET)...")
    
    # Download first N videos using hf sync
    # For autoresearch, we download a fixed subset
    try:
        # First, check if hf CLI is available
        result = subprocess.run(
            ["which", "hf"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("⚠️  hf CLI not found. Installing...")
            subprocess.run(["pip", "install", "huggingface-hub[cli]"], check=True)
        
        # Download subset using hf sync
        # Note: For autoresearch we download a small fixed set
        cmd = [
            "hf", "sync",
            f"hf://buckets/{HF_BUCKET}/Charades_v1_480",
            str(video_dir),
            "--include", "*.mp4"
        ]
        
        print(f"   Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # Limit to num_videos if we got more
        all_videos = list(video_dir.glob("*.mp4"))
        if len(all_videos) > num_videos:
            print(f"   Trimming to {num_videos} videos...")
            for vid in sorted(all_videos)[num_videos:]:
                vid.unlink()
        
        print(f"✅ {min(len(all_videos), num_videos)} videos ready")
        
    except Exception as e:
        print(f"⚠️  Error downloading videos: {e}")
        print("   Will use lazy loading (slower but works)")


def verify_setup():
    """Verify that data is ready."""
    train_file = DATA_DIR / "charades_sta_train.txt"
    test_file = DATA_DIR / "charades_sta_test.txt"
    video_dir = DATA_DIR / "Charades_v1_480"
    
    if not train_file.exists() or not test_file.exists():
        print("❌ Annotations missing. Run prepare.py first.")
        return False
    
    videos = list(video_dir.glob("*.mp4"))
    if len(videos) == 0:
        print("⚠️  No videos found. Will use lazy loading (slower).")
    else:
        print(f"✅ Setup verified: {len(videos)} videos, annotations ready")
    
    return True


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class CharadesSubset:
    """Wrapper for autoresearch dataset with fixed subset."""
    
    def __init__(self, split="train", data_dir=DATA_DIR):
        self.split = split
        self.data_dir = Path(data_dir)
        self.annotations = self._load_annotations()
        
    def _load_annotations(self):
        """Load annotations for videos that exist locally."""
        ann_file = self.data_dir / f"charades_sta_{self.split}.txt"
        video_dir = self.data_dir / "Charades_v1_480"
        
        if not ann_file.exists():
            return []
        
        # Get list of available videos
        available_videos = {v.stem for v in video_dir.glob("*.mp4")}
        
        annotations = []
        with open(ann_file) as f:
            for line in f:
                parts = line.strip().split("##")
                if len(parts) != 2:
                    continue
                
                video_query = parts[0]
                times_label = parts[1]
                
                video_id = video_query.split(" ")[0]
                
                # Only include if video exists locally (for autoresearch)
                if available_videos and video_id not in available_videos:
                    continue
                
                annotations.append(line.strip())
        
        return annotations
    
    def __len__(self):
        return len(self.annotations)


def get_dataset_info():
    """Get info about the prepared dataset."""
    train_ann = DATA_DIR / "charades_sta_train.txt"
    video_dir = DATA_DIR / "Charades_v1_480"
    
    info = {
        "data_dir": str(DATA_DIR),
        "train_annotations": train_ann.exists(),
        "num_videos": len(list(video_dir.glob("*.mp4"))) if video_dir.exists() else 0,
    }
    
    if info["train_annotations"]:
        with open(train_ann) as f:
            info["total_annotations"] = len(f.readlines())
    
    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for VL-JEPA autoresearch")
    parser.add_argument("--subset", type=int, default=500, 
                       help="Number of videos to download (-1 = all)")
    args = parser.parse_args()

    print(f"Cache directory: {DATA_DIR}")
    print()

    # Step 1: Download annotations
    download_annotations()
    print()

    # Step 2: Download video subset
    if args.subset > 0:
        download_video_subset(args.subset)
    else:
        print("Skipping video download (will use lazy loading)")
    print()

    # Step 3: Verify
    if verify_setup():
        print("\n✅ Done! Ready to train.")
        print(f"   Data location: {DATA_DIR}")
        print(f"   Time budget: {TIME_BUDGET}s per experiment")
    else:
        print("\n❌ Setup incomplete. Please fix errors above.")
        sys.exit(1)
