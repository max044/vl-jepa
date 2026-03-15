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
import cv2
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

# Enable fast HF downloads if hf_transfer is installed
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from vljepa.config import Config
from vljepa.dataset import CharadesSTADataset
from vljepa.models import VLJepa
from vljepa.utils import temporal_iou, sliding_window_proposals, nms, load_video_seg_from_cap

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
    parser.add_argument("--wandb-run-path", type=str, default=None, help="Attach to existing W&B run")
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
            wandb.init(project=args.wandb_project, id=args.wandb_run_path.split("/")[-1], resume="allow", tags=["eval"])
        else:
            wandb.init(project=args.wandb_project, job_type="eval", tags=["eval"])

    print(f"Device: {config.device}")

    # Load model
    print("Loading model...")
    model = VLJepa(config)

    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        print("❌ Error: Checkpoint path is empty.")
        return

    # Handle W&B Artifacts
    if (":" in checkpoint_path or "/" in checkpoint_path) and not os.path.exists(checkpoint_path):
        if use_wandb:
            print(f"📥 Downloading checkpoint from W&B Artifact: {checkpoint_path}")
            artifact = wandb.run.use_artifact(checkpoint_path, type='model')
            artifact_dir = artifact.download()
            pths = [os.path.join(artifact_dir, f) for f in os.listdir(artifact_dir) if f.endswith(".pth")]
            checkpoint_path = pths[0] if pths else ""
        else:
            print("❌ W&B is disabled, cannot download artifact.")
            return

    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint file not found: {checkpoint_path}")
        return

    print(f"📂 Loading weights from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
    model.predictor.load_state_dict(ckpt["predictor_state_dict"])
    model.y_encoder.projection.load_state_dict(ckpt["y_projection_state_dict"])
    model.predictor.eval()

    # Load dataset
    test_dataset = CharadesSTADataset(config.anno_test, config.videos_dir, config, split="test")
    samples = test_dataset.samples
    if args.max_samples:
        samples = samples[:args.max_samples]

    # Group by video
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
        # 1. Verification / Lazy Loading
        if not os.path.exists(video_path) and config.hf_dataset_id:
            if HAS_HF_HUB:
                try:
                    video_id = group[0].get("video_id") or os.path.basename(video_path).replace(".mp4", "")
                    video_path = hf_hub_download(repo_id=config.hf_dataset_id, filename=f"{video_id}.mp4", repo_type="dataset", local_dir=config.videos_dir)
                except Exception:
                    skipped += len(group)
                    pbar.update(len(group))
                    continue
            else:
                skipped += len(group)
                pbar.update(len(group))
                continue

        # 2. Optimized Reading: Open cap ONCE
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            skipped += len(group)
            pbar.update(len(group))
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = count / fps if fps > 0 else 0
        
        if duration <= 0:
            cap.release()
            skipped += len(group)
            pbar.update(len(group))
            continue

        # Extract features for all windows
        proposals = sliding_window_proposals(duration, config.window_sizes, config.window_stride)
        all_sv = []
        valid_proposals = []
        bs = config.inference_batch_size
        
        for i in range(0, len(proposals), bs):
            batch_props = proposals[i : i + bs]
            fb = []
            valid_p = []
            for start, end in batch_props:
                frames = load_video_seg_from_cap(cap, start, end, config.num_frames)
                if frames:
                    fb.append(frames)
                    valid_p.append((start, end))
            
            if not fb: continue
            
            pv = model.x_encoder.preprocess_frames(fb, device=config.device)
            sv = model.x_encoder(pv)
            all_sv.append(sv)
            valid_proposals.extend(valid_p)

        cap.release()

        if not all_sv:
            skipped += len(group)
            pbar.update(len(group))
            continue
            
        sv_full = torch.cat(all_sv, dim=0)

        # 3. Process queries
        for sample in group:
            try:
                sy_ref = F.normalize(model.encode_text([sample["caption"]], device=config.device), dim=-1)
                qt = model.query_encoder.tokenize([sample["caption"]], device=config.device)
                
                sims_list = []
                for j in range(0, sv_full.size(0), bs):
                    b_sv = sv_full[j : j + bs]
                    B = b_sv.size(0)
                    sy_hat = F.normalize(model.predictor(b_sv, qt["input_ids"].expand(B, -1), qt["attention_mask"].expand(B, -1)), dim=-1)
                    sims_list.append((sy_hat @ sy_ref.T).squeeze(-1))
                
                scores = torch.cat(sims_list, dim=0).cpu().numpy().tolist()
                k = nms(valid_proposals, scores, config.nms_threshold)
                
                if k:
                    iou = temporal_iou(valid_proposals[k[0]][0], valid_proposals[k[0]][1], sample["start"], sample["end"])
                    ious.append(iou)
                    for t in recalls:
                        if iou >= t: recalls[t] += 1
                    total += 1
                else: skipped += 1
            except Exception: skipped += 1
            pbar.update(1)

    pbar.close()
    
    print(f"\nFinal: {total}/{len(samples)} (skipped {skipped})")
    if total > 0:
        for t, c in sorted(recalls.items()): print(f"  R@1 IoU={t}: {c/total*100:.2f}%")
        print(f"  mIoU: {sum(ious)/len(ious)*100:.2f}%")
        if use_wandb:
            wandb.log({f"eval/R@1_IoU={t}": c/total*100 for t, c in recalls.items()} | {"eval/mIoU": sum(ious)/len(ious)*100})
            wandb.finish()


if __name__ == "__main__":
    main()
