"""
Precompute and cache V-JEPA 2 features for all videos in the dataset.

Since V-JEPA is frozen during training, recomputing features every epoch
is pure waste. This script runs once and saves (video_id -> sv tensor) to disk.
The dataset then loads cached tensors instead of decoding raw videos.

Usage:
    uv run training/precompute_features.py
    uv run training/precompute_features.py --batch-size 8 --workers 16

Output:
    data/features/
        {video_id}.pt   # torch tensor (1, x_dim) = mean-pooled V-JEPA embedding
        manifest.json   # list of all cached video_ids

After running this script, set in config:
    use_precomputed_features = True
    features_dir = "./data/features"
"""

import os
import json
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from vljepa.config import Config
from vljepa.models import XEncoder


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------

def load_video_frames(video_path: Path, num_frames: int = 64) -> np.ndarray | None:
    """Decode a video and uniformly sample num_frames frames using decord.

    decord is 5-10x faster than cv2 for batch frame sampling — it loads all
    requested frames in a single operation instead of seeking frame by frame.

    Returns (num_frames, H, W, 3) uint8 array or None on failure.
    """
    try:
        from decord import VideoReader, cpu
        vr      = VideoReader(str(video_path), ctx=cpu(0))
        total   = len(vr)
        if total <= 0:
            return None
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames  = vr.get_batch(indices).asnumpy()  # (T, H, W, 3) uint8
        return frames
    except Exception as e:
        print(f"  ⚠️  Failed to load {video_path.name}: {e}")
        return None


def preprocess_frames(frames_np: np.ndarray) -> torch.Tensor:
    """CPU-only preprocessing — resize avec PIL, normalise en numpy."""
    from PIL import Image
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    resized = []
    for frame in frames_np:
        img = Image.fromarray(frame).resize((224, 224), Image.BILINEAR)
        resized.append(np.array(img, dtype=np.float32) / 255.0)

    t = np.stack(resized)               # (T, 224, 224, 3)
    t = (t - mean) / std
    t = t.transpose(3, 0, 1, 2)        # (3, T, 224, 224)
    return torch.from_numpy(t).unsqueeze(0)  # (1, 3, T, 224, 224)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Videos per GPU batch (default: 4)")
    parser.add_argument("--workers",    type=int, default=8,
                        help="CPU workers for video loading (default: 8)")
    parser.add_argument("--videos-dir", type=str, default=None,
                        help="Override videos directory from config")
    parser.add_argument("--output-dir", type=str, default="data/features",
                        help="Where to save .pt feature files (default: data/features)")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Recompute even if .pt already exists")
    return parser.parse_args()


def main():
    args   = parse_args()
    config = Config()
    device = config.device

    videos_dir  = Path(args.videos_dir or config.videos_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all video files
    video_files = sorted(videos_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} videos in {videos_dir}")

    if not video_files:
        print("❌ No .mp4 files found. Check --videos-dir.")
        return

    # Load X-Encoder (frozen V-JEPA 2)
    print(f"\nLoading X-Encoder ({config.clip_model})...")
    encoder = XEncoder(config)
    encoder.eval()
    print(f"  ✓ X-Encoder loaded on {device}")

    # Check which videos are already cached
    if not args.overwrite:
        existing = {p.stem for p in output_dir.glob("*.pt")}
        todo     = [v for v in video_files if v.stem not in existing]
        print(f"\n{len(existing)} already cached, {len(todo)} remaining.")
    else:
        todo = video_files
        print(f"\nOverwrite mode: processing all {len(todo)} videos.")

    if not todo:
        print("✅ All videos already cached.")
        _write_manifest(output_dir, video_files)
        return

    # Process videos in batches, loading frames in parallel with threads.
    # decord releases the GIL so ThreadPoolExecutor gives real parallelism here.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    t0     = time.time()
    done   = 0
    failed = 0

    print(f"\nExtracting features (device={device}, gpu_batch={args.batch_size}, cpu_workers={args.workers})...")

    def load_one(video_path):
        frames = load_video_frames(video_path, num_frames=config.num_frames)
        return video_path, frames

    batch_paths   = []
    batch_tensors = []

    def flush_batch():
        nonlocal done
        if not batch_tensors:
            return
        batch = torch.cat(batch_tensors, dim=0)  # (B, C, T, H, W)
        with torch.no_grad():
            sv = encoder(batch)                  # (B, x_dim)
        for path, feat in zip(batch_paths, sv):
            torch.save(feat.cpu(), output_dir / f"{path.stem}.pt")
            done += 1
        batch_paths.clear()
        batch_tensors.clear()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(load_one, v): v for v in todo}

        for i, future in enumerate(as_completed(futures)):
            video_path, frames = future.result()

            if frames is None:
                failed += 1
                continue

            try:
                tensor = preprocess_frames(frames, device)
                batch_paths.append(video_path)
                batch_tensors.append(tensor)
            except Exception as e:
                failed += 1
                print(f"  ⚠️  Preprocess failed {video_path.name}: {e}")
                continue

            if len(batch_tensors) >= args.batch_size:
                flush_batch()

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate    = (i + 1) / elapsed if elapsed > 0 else 0
                eta     = (len(todo) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1:5d}/{len(todo)}]  done={done}  failed={failed}  "
                      f"rate={rate:.1f} vid/s  ETA={eta/60:.1f}min")

    # Flush remaining
    flush_batch()

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"✅ Done: {done} features saved, {failed} failed  ({elapsed/60:.1f} min)")
    print(f"   Output: {output_dir.resolve()}")

    _write_manifest(output_dir, video_files)


def _write_manifest(output_dir: Path, video_files: list[Path]):
    """Write manifest.json listing all successfully cached video_ids."""
    cached = [p.stem for p in output_dir.glob("*.pt")]
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(sorted(cached), indent=2))
    print(f"   Manifest: {len(cached)} entries → {manifest_path}")


if __name__ == "__main__":
    main()