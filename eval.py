"""VL-JEPA Evaluation on Charades-STA.

Evaluates moment retrieval performance using standard metrics:
  - R@1 IoU=0.3
  - R@1 IoU=0.5
  - R@1 IoU=0.7
  - mIoU (mean IoU of top-1 predictions)

Usage:
    python eval.py --checkpoint checkpoints/best.pth
    python eval.py --checkpoint checkpoints/best.pth --device cuda
    python eval.py --checkpoint checkpoints/best.pth --wandb-project vl-jepa
"""

import argparse
import torch
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

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
    # W&B arguments
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="vl-jepa", help="W&B project name")
    parser.add_argument("--wandb-run-path", type=str, default=None, help="Attach to existing W&B run (entity/project/run_id)")
    return parser.parse_args()


def main():
    args = parse_args()

    config = Config()
    if args.device:
        config.device = args.device

    # ── W&B Init ──────────────────────────────────────────
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        if args.wandb_run_path:
            # Resume/attach to existing training run
            wandb.init(
                project=args.wandb_project,
                id=args.wandb_run_path.split("/")[-1],
                resume="allow",
                tags=["eval"],
            )
        else:
            wandb.init(
                project=args.wandb_project,
                job_type="eval",
                config={"checkpoint": args.checkpoint, "device": config.device},
                tags=["eval"],
            )
        print(f"W&B run: {wandb.run.url}")

    print(f"Device: {config.device}")
    print()

    # Load model
    print("Loading model...")
    model = VLJepa(config)

    checkpoint_path = args.checkpoint
    # Check if it's a W&B artifact path
    if (":" in checkpoint_path or "/" in checkpoint_path) and not __import__("os").path.exists(checkpoint_path):
        if use_wandb:
            print(f"📥 Downloading checkpoint from W&B Artifact: {checkpoint_path}")
            try:
                artifact = wandb.run.use_artifact(checkpoint_path, type='model')
                artifact_dir = artifact.download()
                checkpoint_path = __import__("os").path.join(artifact_dir, "best.pth")
                if not __import__("os").path.exists(checkpoint_path):
                    pths = [f for f in __import__("os").listdir(artifact_dir) if f.endswith(".pth")]
                    if pths:
                        checkpoint_path = __import__("os").path.join(artifact_dir, pths[0])
            except Exception as e:
                print(f"❌ Failed to download artifact: {e}")
                return

    ckpt = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
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
        eval_metrics = {}
        for thresh, count in sorted(recalls.items()):
            r = count / total * 100
            print(f"  R@1 IoU={thresh:.1f}:  {r:6.2f}%  ({count}/{total})")
            eval_metrics[f"eval/R@1_IoU={thresh:.1f}"] = r

        mean_iou = sum(ious) / len(ious) * 100
        print(f"\n  mIoU:          {mean_iou:6.2f}%")
        eval_metrics["eval/mIoU"] = mean_iou

        # ── W&B: log eval metrics ────────────────────────
        if use_wandb:
            wandb.log(eval_metrics)
            for k, v in eval_metrics.items():
                wandb.summary[k] = v
            wandb.finish()
    else:
        print("  No samples evaluated successfully.")
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
