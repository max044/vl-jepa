"""VL-JEPA Evaluation on Charades-STA test set.

Metrics: R@1 IoU={0.3, 0.5, 0.7}, mIoU  (standard Charades-STA protocol)

Usage:
    uv run training/eval.py --checkpoint checkpoints/best.pt
    uv run training/eval.py --checkpoint checkpoints/best.pt --max-samples 500
"""

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from vljepa.config import Config
from vljepa.dataset import CharadesSTADataset
from vljepa.models import VLJepa
from vljepa.utils import temporal_iou, sliding_window_proposals, nms, load_video_to_ram

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

NEUTRAL_QUERY = "What is happening in this video?"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VL-JEPA on Charades-STA")
    parser.add_argument("--checkpoint",     type=str, required=True)
    parser.add_argument("--device",         type=str, default=None)
    parser.add_argument("--max-samples",    type=int, default=None)
    parser.add_argument("--window-stride",  type=float, default=None)
    parser.add_argument("--window-sizes",   type=str, default=None, help="e.g. 4.0,8.0,16.0")
    parser.add_argument("--no-wandb",       action="store_true")
    parser.add_argument("--save-report",    type=str, default="eval_report.txt")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(results: dict, duration_sec: float, total: int, skipped: int, checkpoint: str):
    w = 58
    print("\n" + "═" * w)
    print(f"║ {'VL-JEPA EVALUATION REPORT':^{w-2}} ║")
    print("═" * w)
    print(f"║  Checkpoint : {os.path.basename(checkpoint):<{w-18}} ║")
    print(f"║  Samples    : {total:<6}  Skipped: {skipped:<6}  Time: {duration_sec/60:>5.1f}m  ║")
    print("╟" + "─" * (w-2) + "╢")
    metrics = [
        ("R@1  IoU≥0.3", results.get("eval/R@1_IoU=0.3", 0)),
        ("R@1  IoU≥0.5", results.get("eval/R@1_IoU=0.5", 0)),
        ("R@1  IoU≥0.7", results.get("eval/R@1_IoU=0.7", 0)),
        ("mIoU",         results.get("eval/mIoU",        0)),
    ]
    for name, val in metrics:
        bar = "█" * int(val / 2)
        print(f"║  {name:<14} {val:>6.2f}%  {bar:<{w-28}} ║")
    print("═" * w + "\n")


def save_report(results: dict, duration_sec: float, total: int, skipped: int,
                checkpoint: str, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"VL-JEPA Evaluation Report — {time.ctime()}\n")
        f.write(f"Checkpoint : {checkpoint}\n")
        f.write(f"Samples    : {total}  Skipped: {skipped}  Duration: {duration_sec/60:.2f}m\n\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"📝 Report saved to: {path}")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def main():
    args   = parse_args()
    config = Config()

    # ── Load checkpoint ────────────────────────────────────────────────
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return

    print(f"📂 Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Restore config from checkpoint, then apply CLI overrides
    if "config" in ckpt:
        saved = ckpt["config"]
        for k, v in saved.items():
            if hasattr(config, k):
                setattr(config, k, v)

    if args.device:
        config.device = args.device
    if args.window_stride:
        config.window_stride = args.window_stride
    if args.window_sizes:
        config.window_sizes = [float(x) for x in args.window_sizes.split(",")]

    # ── Model ──────────────────────────────────────────────────────────
    print("\nLoading model...")
    model = VLJepa(config).to(config.device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  ✓ Weights loaded (val_loss: {ckpt.get('best_val_loss', 'N/A')})")
    else:
        print("❌ Unknown checkpoint format (expected 'model_state_dict')")
        return

    model.eval()

    # ── W&B ───────────────────────────────────────────────────────────
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "vl-jepa"),
            entity=os.getenv("WANDB_ENTITY", ""),
            job_type="eval",
            tags=["eval"],
            config={"checkpoint": ckpt_path, **{k: getattr(config, k) for k in ["window_sizes", "window_stride", "nms_threshold"]}},
        )

    # ── Dataset ───────────────────────────────────────────────────────
    print(f"\nLoading test set from: {config.anno_test}")
    test_dataset = CharadesSTADataset(config.anno_test, config.videos_dir, config, split="test")
    samples = test_dataset.samples
    if args.max_samples:
        samples = samples[: args.max_samples]

    print(f"\nEval config:")
    print(f"  device         : {config.device}")
    print(f"  window_sizes   : {config.window_sizes}s")
    print(f"  window_stride  : {config.window_stride}s")
    print(f"  inference_batch: {config.inference_batch_size}")
    print(f"  samples        : {len(samples)}")

    # ── Group samples by video to load each video once ─────────────────
    video_to_queries: dict[str, list] = defaultdict(list)
    for s in samples:
        video_to_queries[s["video_path"]].append(s)

    # ── Evaluation loop ────────────────────────────────────────────────
    ious: list[float] = []
    recalls = {0.3: 0, 0.5: 0, 0.7: 0}
    total = skipped = 0
    bs = config.inference_batch_size
    start_time = time.time()

    with tqdm(total=len(samples), desc="Evaluating", dynamic_ncols=True) as pbar:
        for video_path, group in video_to_queries.items():

            if not os.path.exists(video_path):
                skipped += len(group)
                pbar.update(len(group))
                continue

            # 1. Load full video into RAM once
            v_data = load_video_to_ram(video_path)
            if not v_data:
                skipped += len(group)
                pbar.update(len(group))
                continue

            fps       = v_data["fps"]
            frames_np = v_data["frames"]
            duration  = len(frames_np) / fps

            # 2. Generate sliding window proposals
            proposals = sliding_window_proposals(duration, config.window_sizes, config.window_stride)
            if not proposals:
                skipped += len(group)
                pbar.update(len(group))
                continue

            # 3. Extract visual features for all proposals
            # Build numpy clips from frames_np, then call preprocess_frames
            # exactly as in training — no extra function needed.
            all_sv: list[torch.Tensor] = []
            valid_proposals: list[tuple] = []

            for i in range(0, len(proposals), bs):
                batch_props = proposals[i : i + bs]
                clips = []
                for p_start, p_end in batch_props:
                    start_f = max(0, int(p_start * fps))
                    end_f   = min(len(frames_np) - 1, int(p_end * fps))
                    if end_f <= start_f:
                        continue
                    indices = np.linspace(start_f, end_f, config.num_frames, dtype=int)
                    clips.append([frames_np[idx] for idx in indices])  # list of (H,W,3)
                    valid_proposals.append((p_start, p_end))

                if clips:
                    pixel_values = model.x_encoder.preprocess_frames(clips, device=config.device)
                    all_sv.append(model.x_encoder(pixel_values))

            if not all_sv:
                skipped += len(group)
                pbar.update(len(group))
                continue

            sv_full = torch.cat(all_sv, dim=0)  # (N_proposals, x_dim)

            # 4. Predict embeddings for all proposals using neutral query
            nq_tokens  = model.query_encoder.tokenize([NEUTRAL_QUERY], device=config.device)
            nq_ids_one = nq_tokens["input_ids"]    # (1, T_q)
            nq_mask_one = nq_tokens["attention_mask"]

            all_sy_hat: list[torch.Tensor] = []
            for j in range(0, sv_full.size(0), bs):
                b_sv        = sv_full[j : j + bs]
                current_bs  = b_sv.size(0)
                nq_ids      = nq_ids_one.expand(current_bs, -1)
                nq_mask     = nq_mask_one.expand(current_bs, -1)
                outputs     = model.predictor(b_sv, nq_ids, nq_mask)
                all_sy_hat.append(F.normalize(outputs["sy_hat"], dim=-1))

            sy_hat_full = torch.cat(all_sy_hat, dim=0)  # (N_proposals, embed_dim)

            # 5. Encode ground-truth captions and compute similarities
            captions  = [s["caption"] for s in group]
            sy_refs   = F.normalize(model.encode_text(captions, device=config.device), dim=-1)  # (N_queries, embed_dim)
            sims      = (sy_hat_full @ sy_refs.T).cpu().float().numpy()  # (N_proposals, N_queries)

            # 6. Score each query, apply NMS, compute IoU
            for q_idx, sample in enumerate(group):
                scores = sims[:, q_idx].tolist()
                top_k  = nms(valid_proposals, scores, config.nms_threshold)

                if not top_k:
                    skipped += 1
                    pbar.update(1)
                    continue

                pred_start, pred_end = valid_proposals[top_k[0]]
                iou = temporal_iou(pred_start, pred_end, sample["start"], sample["end"])

                ious.append(iou)
                for thresh in recalls:
                    if iou >= thresh:
                        recalls[thresh] += 1
                total += 1
                pbar.update(1)

    # ── Results ────────────────────────────────────────────────────────
    eval_duration = time.time() - start_time

    if total == 0:
        print("❌ No samples evaluated — check video paths and annotations.")
        if use_wandb:
            wandb.finish()
        return

    results = {
        f"eval/R@1_IoU={t}": (recalls[t] / total) * 100
        for t in recalls
    }
    results["eval/mIoU"] = (sum(ious) / len(ious)) * 100

    print_results(results, eval_duration, total, skipped, ckpt_path)

    if args.save_report:
        report_path = os.path.join(config.checkpoint_dir, args.save_report)
        save_report(results, eval_duration, total, skipped, ckpt_path, report_path)

    if use_wandb:
        wandb.log(results)
        wandb.finish()


if __name__ == "__main__":
    main()