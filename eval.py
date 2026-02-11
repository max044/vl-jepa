"""VL-JEPA Evaluation on Charades-STA.

Evaluates moment retrieval performance using standard metrics:
  - R@1 IoU=0.3
  - R@1 IoU=0.5
  - R@1 IoU=0.7
  - mIoU (mean IoU of top-1 predictions)

Usage:
    python eval.py --checkpoint checkpoints/best.pth
    python eval.py --checkpoint checkpoints/best.pth --device cuda
"""

import argparse
import torch
from tqdm import tqdm

from vljepa.config import Config
from vljepa.dataset import CharadesSTADataset
from vljepa.models import VLJepa
from vljepa.utils import temporal_iou, get_video_duration

from infer import retrieve_moments


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VL-JEPA")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit eval samples")
    return parser.parse_args()


def main():
    args = parse_args()

    config = Config()
    if args.device:
        config.device = args.device

    print(f"Device: {config.device}")
    print()

    # Load model
    print("Loading model...")
    model = VLJepa(config)

    ckpt = torch.load(args.checkpoint, map_location=config.device, weights_only=True)
    model.predictor.load_state_dict(ckpt["predictor_state_dict"])
    model.y_encoder.projection.load_state_dict(ckpt["y_projection_state_dict"])

    model.predictor.to(config.device)
    model.y_encoder.projection.to(config.device)
    model.predictor.eval()

    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})\n")

    # Load test dataset
    test_dataset = CharadesSTADataset(
        config.anno_test, config.videos_dir, config, split="test"
    )

    samples = test_dataset.samples
    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Evaluating on {len(samples)} samples...\n")

    # Metrics accumulators
    ious = []
    recalls = {0.3: 0, 0.5: 0, 0.7: 0}
    total = 0
    skipped = 0

    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        video_path = sample["video_path"]
        caption = sample["caption"]
        gt_start = sample["start"]
        gt_end = sample["end"]

        # Use the ground-truth caption as the query for moment retrieval
        try:
            results = retrieve_moments(model, video_path, caption, config)
        except Exception as e:
            skipped += 1
            continue

        if len(results) == 0:
            skipped += 1
            continue

        # Top-1 prediction
        pred_start = results[0]["start"]
        pred_end = results[0]["end"]

        iou = temporal_iou(pred_start, pred_end, gt_start, gt_end)
        ious.append(iou)

        for thresh in recalls:
            if iou >= thresh:
                recalls[thresh] += 1

        total += 1

    # Print results
    print(f"\n{'═' * 50}")
    print(f"VL-JEPA Evaluation Results")
    print(f"{'═' * 50}")
    print(f"Evaluated: {total} / {len(samples)} samples (skipped {skipped})")
    print()

    if total > 0:
        for thresh, count in sorted(recalls.items()):
            r = count / total * 100
            print(f"  R@1 IoU={thresh:.1f}:  {r:6.2f}%  ({count}/{total})")

        mean_iou = sum(ious) / len(ious) * 100
        print(f"\n  mIoU:          {mean_iou:6.2f}%")
    else:
        print("  No samples evaluated successfully.")


if __name__ == "__main__":
    main()
