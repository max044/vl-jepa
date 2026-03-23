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
    """Decode a video and uniformly sample num_frames frames.

    Returns (num_frames, H, W, 3) uint8 array or None on failure.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None

        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames  = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                # Repeat last frame on read failure
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((256, 256, 3), dtype=np.uint8))
            else:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return np.stack(frames)  # (num_frames, H, W, 3)

    except Exception as e:
        print(f"  ⚠️  Failed to load {video_path.name}: {e}")
        return None


def preprocess_frames(frames_np: np.ndarray, device: str) -> torch.Tensor:
    """(T, H, W, 3) uint8 → (1, T, C, H, W) float normalised."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    t = torch.tensor(frames_np, dtype=torch.float32, device=device)  # (T, H, W, 3)
    t = t.permute(0, 3, 1, 2) / 255.0                                # (T, 3, H, W)
    t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    t = (t - mean) / std                                              # (T, 3, 224, 224)

    # V-JEPA expects (B, C, T, H, W)
    t = t.permute(1, 0, 2, 3).unsqueeze(0)                           # (1, C, T, H, W)
    return t


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

    # Process videos
    t0      = time.time()
    done    = 0
    failed  = 0

    print(f"\nExtracting features ({device}, batch_size={args.batch_size})...")
    print(f"{'Progress':>10}  {'Video':>40}  {'Status'}")
    print("-" * 65)

    batch_paths   = []
    batch_tensors = []

    def flush_batch():
        nonlocal done
        if not batch_tensors:
            return

        # Stack into (B, C, T, H, W)
        batch = torch.cat(batch_tensors, dim=0)  # (B, C, T, H, W)

        with torch.no_grad():
            sv = encoder(batch)  # (B, x_dim)

        # Save each video's feature individually
        for path, feat in zip(batch_paths, sv):
            out_path = output_dir / f"{path.stem}.pt"
            torch.save(feat.cpu(), out_path)
            done += 1

        batch_paths.clear()
        batch_tensors.clear()

    for i, video_path in enumerate(todo):
        frames = load_video_frames(video_path, num_frames=config.num_frames)

        if frames is None:
            failed += 1
            print(f"  [{i+1:5d}/{len(todo)}]  {video_path.name:>40}  ❌ failed")
            continue

        try:
            tensor = preprocess_frames(frames, device)  # (1, C, T, H, W)
            batch_paths.append(video_path)
            batch_tensors.append(tensor)
        except Exception as e:
            failed += 1
            print(f"  [{i+1:5d}/{len(todo)}]  {video_path.name:>40}  ❌ {e}")
            continue

        # Flush when batch is full
        if len(batch_tensors) >= args.batch_size:
            flush_batch()

        # Progress every 50 videos
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate    = done / elapsed if elapsed > 0 else 0
            eta     = (len(todo) - done) / rate if rate > 0 else 0
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