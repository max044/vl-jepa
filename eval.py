"""VL-JEPA Evaluation on Charades-STA.

Evaluates moment retrieval performance using standard metrics:
  - R@1 IoU=0.3
  - R@1 IoU=0.5
  - R@1 IoU=0.7
  - mIoU (mean IoU of top-1 predictions)

Usage:
    python eval.py --checkpoint checkpoints/best.pth
"""

import argparse
import os

# Enable fast HF downloads if hf_transfer is installed
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from collections import defaultdict
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from dotenv import load_dotenv
load_dotenv()

from vljepa.config import Config
from vljepa.dataset import CharadesSTADataset
from vljepa.models import VLJepa
from vljepa.utils import temporal_iou, get_video_duration, sliding_window_proposals, load_video_frames, nms

try:
    from huggingface_hub import hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


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


@torch.no_grad()
def main():
    args = parse_args()

    config = Config()
    if args.device:
        config.device = args.device

    # ── W&B Init ──────────────────────────────────────────
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        if args.wandb_run_path:
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

    print(f"Device: {config.device}")

    # Load model
    print("Loading model...")
    model = VLJepa(config)

    checkpoint_path = args.checkpoint
    
    # 1. Basic validation
    if not checkpoint_path:
        print("❌ Error: Checkpoint path is empty. Provide --checkpoint or set CHECKPOINT env var.")
        return

    # 2. Handle W&B Artifacts
    if (":" in checkpoint_path or "/" in checkpoint_path) and not os.path.exists(checkpoint_path):
        if use_wandb:
            print(f"📥 Downloading checkpoint from W&B Artifact: {checkpoint_path}")
            try:
                artifact = wandb.run.use_artifact(checkpoint_path, type='model')
                artifact_dir = artifact.download()
                
                # Look for .pth files in the downloaded dir
                potential_pths = [os.path.join(artifact_dir, f) for f in os.listdir(artifact_dir) if f.endswith(".pth")]
                if not potential_pths:
                    print(f"❌ Error: No .pth files found in artifact directory: {artifact_dir}")
                    return
                
                # Preference for best.pth, else first .pth found
                best_pth = os.path.join(artifact_dir, "best.pth")
                checkpoint_path = best_pth if os.path.exists(best_pth) else potential_pths[0]
                print(f"✅ Using artifact checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"❌ Failed to download W&B artifact: {e}")
                return
        else:
            print("❌ W&B is disabled, cannot download artifact.")
            return

    # 3. Final existance check
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint file not found: {checkpoint_path}")
        return

    print(f"📂 Loading weights from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
    model.predictor.load_state_dict(ckpt["predictor_state_dict"])
    model.y_encoder.projection.load_state_dict(ckpt["y_projection_state_dict"])
    model.predictor.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})\n")

    # Load dataset
    test_dataset = CharadesSTADataset(config.anno_test, config.videos_dir, config, split="test")
    samples = test_dataset.samples
    if args.max_samples:
        samples = samples[:args.max_samples]

    # 🚀 Optimization: Group queries by video
    video_to_queries = defaultdict(list)
    for s in samples:
        video_to_queries[s["video_path"]].append(s)

    print(f"Evaluating {len(samples)} queries across {len(video_to_queries)} unique videos...\n")

    ious = []
    recalls = {0.3: 0, 0.5: 0, 0.7: 0}
    total = 0
    skipped = 0

    pbar = tqdm(total=len(samples), desc="Evaluating")
    
    for video_path, group in video_to_queries.items():
        # 1. Lazy Loading / Verification
        if not os.path.exists(video_path) and config.hf_dataset_id:
            if HAS_HF_HUB:
                try:
                    video_id = group[0].get("video_id") or os.path.basename(video_path).replace(".mp4", "")
                    video_path = hf_hub_download(
                        repo_id=config.hf_dataset_id,
                        filename=f"{video_id}.mp4",
                        repo_type="dataset",
                        local_dir=config.videos_dir,
                    )
                except Exception as e:
                    print(f"❌ Error downloading {video_id}: {e}")
                    skipped += len(group)
                    pbar.update(len(group))
                    continue
            else:
                skipped += len(group)
                pbar.update(len(group))
                continue

        # 2. Extract visual features for all windows of this video ONCE
        duration = get_video_duration(video_path)
        if duration <= 0:
            skipped += len(group)
            pbar.update(len(group))
            continue

        proposals = sliding_window_proposals(duration, config.window_sizes, config.window_stride)
        
        all_sv = []
        valid_proposals = []
        batch_size = config.inference_batch_size
        
        for i in range(0, len(proposals), batch_size):
            batch_props = proposals[i : i + batch_size]
            frames_batch = []
            prop_indices = []
            for start, end in batch_props:
                frames = load_video_frames(video_path, start, end, config.num_frames)
                if frames:
                    frames_batch.append(frames)
                    prop_indices.append((start, end))
            
            if not frames_batch:
                continue
                
            pixel_values = model.x_encoder.preprocess_frames(frames_batch, device=config.device)
            sv = model.x_encoder(pixel_values) # (B, hidden)
            all_sv.append(sv)
            valid_proposals.extend(prop_indices)

        if not all_sv:
            skipped += len(group)
            pbar.update(len(group))
            continue
            
        sv_combined = torch.cat(all_sv, dim=0) # (TotalProposals, hidden)

        # 3. For each query, just run the Predictor (Fast)
        for sample in group:
            caption = sample["caption"]
            gt_start = sample["start"]
            gt_end = sample["end"]

            try:
                # Text ref
                sy_ref = model.encode_text([caption], device=config.device)
                sy_ref = F.normalize(sy_ref, dim=-1)
                
                # Predictor pass for all proposals
                query_tokens = model.query_encoder.tokenize([caption], device=config.device)
                all_sims = []
                for i in range(0, sv_combined.size(0), batch_size):
                    batch_sv = sv_combined[i : i + batch_size]
                    B = batch_sv.size(0)
                    q_ids = query_tokens["input_ids"].expand(B, -1)
                    q_mask = query_tokens["attention_mask"].expand(B, -1)
                    
                    sy_hat = model.predictor(batch_sv, q_ids, q_mask)
                    sy_hat = F.normalize(sy_hat, dim=-1)
                    sims = (sy_hat @ sy_ref.T).squeeze(-1)
                    all_sims.append(sims)
                
                scores = torch.cat(all_sims, dim=0).cpu().numpy()
                kept_indices = nms(valid_proposals, scores.tolist(), config.nms_threshold)
                
                if kept_indices:
                    pred_start, pred_end = valid_proposals[kept_indices[0]]
                    iou = temporal_iou(pred_start, pred_end, gt_start, gt_end)
                    ious.append(iou)
                    for thresh in recalls:
                        if iou >= thresh:
                            recalls[thresh] += 1
                    total += 1
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
            
            pbar.update(1)

    pbar.close()

    # 📊 Results
    print(f"\n{'═' * 50}\nVL-JEPA Evaluation Results\n{'═' * 50}")
    print(f"Evaluated: {total} / {len(samples)} (skipped {skipped})")
    if total > 0:
        metrics = {}
        for t, c in sorted(recalls.items()):
            r = c / total * 100
            print(f"  R@1 IoU={t:.1f}:  {r:6.2f}%")
            metrics[f"eval/R@1_IoU={t}"] = r
        m_iou = sum(ious) / len(ious) * 100
        print(f"  mIoU:          {m_iou:6.2f}%")
        metrics["eval/mIoU"] = m_iou
        if use_wandb:
            wandb.log(metrics)
            wandb.finish()
    else:
        if use_wandb: wandb.finish()


if __name__ == "__main__":
    main()
