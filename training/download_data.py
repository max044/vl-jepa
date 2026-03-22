"""
Download Charades-STA dataset for VL-JEPA training.

Annotations and videos are stored together in the HF bucket.
A single `hf sync` downloads everything (resumable).

Usage:
    uv run training/download_data.py
    uv run training/download_data.py --verify-only
"""

import sys
import subprocess
from pathlib import Path
import argparse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR  = Path("data")
VIDEO_DIR = DATA_DIR / "Charades_v1_480"
ANNO_DIR  = DATA_DIR / "Charades_v1_480"

HF_BUCKET_ID    = "max044/charades-sta-storage"
ANNO_FILES      = ["charades_sta_train.txt", "charades_sta_test.txt"]
EXPECTED_VIDEOS = 9848


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def download_all():
    """Download everything (annotations + videos) via a single hf sync."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  📥 Syncing from hf://buckets/{HF_BUCKET_ID} → {DATA_DIR}")
    print(f"     (~15 GB, resumable)")
    print()

    cmd = [
        "hf", "sync",
        f"hf://buckets/{HF_BUCKET_ID}",
        str(DATA_DIR),
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ hf sync failed (exit code {result.returncode})")
        print(f"\nManual:")
        print(f"  hf sync hf://buckets/{HF_BUCKET_ID} {DATA_DIR}")
        sys.exit(1)

    print(f"\n✅ Sync complete")


def verify():
    """Verify annotations and video count."""
    ok = True

    for fname in ANNO_FILES:
        fpath = ANNO_DIR / fname
        if fpath.exists():
            n_lines = sum(1 for line in fpath.open() if line.strip())
            print(f"  ✓ {fname} ({n_lines} annotations)")
        else:
            print(f"  ❌ {fname} missing")
            ok = False

    n_videos = len(list(VIDEO_DIR.glob("*.mp4"))) if VIDEO_DIR.exists() else 0
    if n_videos >= EXPECTED_VIDEOS:
        print(f"  ✓ {n_videos} videos in {VIDEO_DIR}")
    else:
        print(f"  ⚠️  {n_videos}/{EXPECTED_VIDEOS} videos in {VIDEO_DIR}")
        ok = False

    return ok




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Charades-STA for VL-JEPA")
    parser.add_argument("--verify-only", action="store_true", help="Only verify, don't download")
    args = parser.parse_args()

    print("=" * 55)
    print("VL-JEPA — Charades-STA Download")
    print("=" * 55)
    print()

    if args.verify_only:
        sys.exit(0 if verify() else 1)

    print("Step 1: Download")
    download_all()
    print()

    print("Step 2: Verification")
    if verify():
        print()
        print("=" * 55)
        print("✅ Dataset ready for training.")
        print("=" * 55)
    else:
        print("\n❌ Setup incomplete.")
        sys.exit(1)