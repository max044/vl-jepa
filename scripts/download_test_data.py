"""Download only the videos required for evaluation (Test Set).

This avoids downloading the full 15GB dataset when you only want to run tests.
Uses hf_transfer for high-speed parallel downloads if installed.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

# Enable fast HF downloads if hf_transfer is installed
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("❌ Error: huggingface_hub not installed. Run: uv add huggingface_hub")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

# Add current dir to path to import vljepa
sys.path.append(os.getcwd())
from vljepa.config import Config
from vljepa.dataset import CharadesSTADataset


def download_video(video_id, repo_id, dest_dir):
    """Download a single video file."""
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=f"{video_id}.mp4",
            repo_type="dataset",
            local_dir=dest_dir,
            local_dir_use_symlinks=False
        )
        return True
    except Exception as e:
        # Some videos might be missing or have different extensions, though unlikely in Charades
        return False


def main():
    config = Config()
    
    # 1. Load test annotations to find required videos
    if not os.path.exists(config.anno_test):
        print(f"❌ Error: Annotations not found at {config.anno_test}")
        print("Run 'uv run download_annotations.py' first.")
        return

    print(f"🔍 Reading test annotations from {config.anno_test}...")
    dataset = CharadesSTADataset(config.anno_test, config.videos_dir, config, split="test")
    
    # 2. Identify unique video IDs
    video_ids = list(set([s["video_id"] for s in dataset.samples]))
    print(f"✅ Found {len(video_ids)} unique test videos.")

    # 3. Check what's already downloaded
    os.makedirs(config.videos_dir, exist_ok=True)
    to_download = []
    for vid in video_ids:
        if not os.path.exists(os.path.join(config.videos_dir, f"{vid}.mp4")):
            to_download.append(vid)

    if not to_download:
        print("🎉 All test videos are already present in data/Charades_v1_480/")
        return

    print(f"📥 Need to download {len(to_download)} videos...")

    # 4. Parallel download
    # We use a ThreadPoolExecutor because hf_hub_download is I/O bound
    # hf_transfer (if installed) will handle the per-file speed.
    max_workers = 8  # Adjust based on your bandwidth
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_video, vid, config.hf_dataset_id, config.videos_dir): vid for vid in to_download}
        
        results = []
        for future in tqdm(futures, total=len(to_download), desc="Downloading"):
            results.append(future.result())

    success_count = sum(results)
    print(f"\n✅ Finished! {success_count}/{len(to_download)} videos downloaded successfully.")
    print(f"Total videos now available: {len(os.listdir(config.videos_dir))}")


if __name__ == "__main__":
    main()
