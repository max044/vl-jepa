"""VL-JEPA Evaluation on Charades-STA (Optimized)."""

import argparse
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from vljepa.config import Config
from vljepa.dataset import CharadesSTADataset
from vljepa.models import VLJepa
from vljepa.utils import temporal_iou, sliding_window_proposals, nms, load_video_to_ram

try:
    from huggingface_hub import hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VL-JEPA")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="vl-jepa")
    parser.add_argument("--wandb-run-path", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    config = Config()
    if args.device:
        config.device = args.device

    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        if args.wandb_run_path:
            wandb.init(project=args.wandb_project, id=args.wandb_run_path.split("/")[-1], resume="allow", tags=["eval"])
        else:
            wandb.init(project=args.wandb_project, job_type="eval", tags=["eval"])

    # Load model & move to half precision for A100 speedup
    model = VLJepa(config)
    if config.device == "cuda":
        model = model.half()
    
    checkpoint_path = args.checkpoint
    if (":" in checkpoint_path or "/" in checkpoint_path) and not os.path.exists(checkpoint_path):
        if use_wandb:
            print(f"📥 Downloading checkpoint from W&B Artifact: {checkpoint_path}")
            try:
                artifact = wandb.run.use_artifact(checkpoint_path, type='model')
                artifact_dir = artifact.download()
                pths = [os.path.join(artifact_dir, f) for f in os.listdir(artifact_dir) if f.endswith(".pth")]
                if not pths:
                    print(f"❌ Error: No .pth files found in artifact {checkpoint_path}")
                    return
                checkpoint_path = pths[0]
            except Exception as e:
                print(f"❌ Failed to download artifact: {e}")
                return
        else:
            print("❌ W&B is disabled, cannot download artifact.")
            return

    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"📂 Loading weights from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=config.device, weights_only=True)
    model.predictor.load_state_dict(ckpt["predictor_state_dict"])
    model.y_encoder.projection.load_state_dict(ckpt["y_projection_state_dict"])
    model.predictor.eval()

    # Dataset
    test_dataset = CharadesSTADataset(config.anno_test, config.videos_dir, config, split="test")
    samples = test_dataset.samples[:args.max_samples] if args.max_samples else test_dataset.samples

    video_to_queries = defaultdict(list)
    for s in samples:
        video_to_queries[s["video_path"]].append(s)

    ious, recalls = [], {0.3: 0, 0.5: 0, 0.7: 0}
    total, skipped = 0, 0

    pbar = tqdm(total=len(samples), desc="Evaluating")
    
    for video_path, group in video_to_queries.items():
        # 1. Ensure video exists
        if not os.path.exists(video_path):
            if HAS_HF_HUB and config.hf_dataset_id:
                try:
                    vid = group[0].get("video_id") or os.path.basename(video_path).replace(".mp4", "")
                    video_path = hf_hub_download(config.hf_dataset_id, f"{vid}.mp4", repo_type="dataset", local_dir=config.videos_dir)
                except Exception:
                    skipped += len(group); pbar.update(len(group)); continue
            else:
                skipped += len(group); pbar.update(len(group)); continue

        # 2. LOAD BRUT VIDEO TO RAM
        v_data = load_video_to_ram(video_path)
        if not v_data:
            skipped += len(group); pbar.update(len(group)); continue

        fps = v_data["fps"]
        frames_np = v_data["frames"]
        duration = len(frames_np) / fps
        
        # 🚀 PREPROCESS FULL VIDEO ON GPU ONCE (IN FP16 + BGR->RGB)
        frames_gpu = model.x_encoder.preprocess_video(frames_np, device=config.device)
        
        proposals = sliding_window_proposals(duration, config.window_sizes, config.window_stride)
        
        # Batch extract visual features
        all_sv, valid_p = [], []
        bs = config.inference_batch_size
        
        for i in range(0, len(proposals), bs):
            batch_props = proposals[i:i+bs]
            fb_list = []
            for start, end in batch_props:
                start_f = max(0, int(start * fps))
                end_f = min(len(frames_gpu) - 1, int(end * fps))
                if end_f <= start_f: continue
                indices = torch.linspace(start_f, end_f, config.num_frames, device=config.device).long()
                fb_list.append(frames_gpu[indices])
                valid_p.append((start, end))
            
            if fb_list:
                pixel_values = torch.stack(fb_list, dim=0)
                all_sv.append(model.x_encoder(pixel_values))
        
        if not all_sv:
            skipped += len(group); pbar.update(len(group)); continue
            
        sv_full = torch.cat(all_sv, dim=0) # (NumProposals, Hidden)

        # 3. Predict for each query vectorially
        captions = [s["caption"] for s in group]
        # Pre-compute all query references for the video
        sy_refs = F.normalize(model.encode_text(captions, device=config.device), dim=-1) # (NumQueries, Embed)
        
        # Tokenize queries once
        qt = model.query_encoder.tokenize(captions, device=config.device)
        
        # We need to compute sy_hat for each (proposal, query) pair.
        # But wait, Sy_hat depends on both the video segment (sv) AND the query text!
        # So it's (NumProposals * NumQueries) forward passes of the Predictor.
        # However, Predictor is super fast.
        
        for q_idx, sample in enumerate(group):
            sy_ref = sy_refs[q_idx:q_idx+1]
            q_ids = qt["input_ids"][q_idx:q_idx+1].expand(bs, -1)
            q_mask = qt["attention_mask"][q_idx:q_idx+1].expand(bs, -1)
            
            sims_list = []
            for j in range(0, sv_full.size(0), bs):
                b_sv = sv_full[j : j + bs]
                current_bs = b_sv.size(0)
                # Expand query to match batch size
                b_q_ids = q_ids[:current_bs]
                b_q_mask = q_mask[:current_bs]
                
                sy_hat = F.normalize(model.predictor(b_sv, b_q_ids, b_q_mask), dim=-1)
                sims_list.append((sy_hat @ sy_ref.T).squeeze(-1))
            
            scores = torch.cat(sims_list, dim=0).cpu().numpy().tolist()
            k = nms(valid_p, scores, config.nms_threshold)
            if k:
                iou = temporal_iou(valid_p[k[0]][0], valid_p[k[0]][1], sample["start"], sample["end"])
                ious.append(iou)
                for t in recalls:
                    if iou >= t: recalls[t] += 1
                total += 1
            else: skipped += 1
            pbar.update(1)

    # Summary
    if total > 0:
        res = {f"eval/R@1_IoU={t}": (c/total)*100 for t, c in recalls.items()} | {"eval/mIoU": (sum(ious)/len(ious))*100}
        print("\nResults:", res)
        if use_wandb: wandb.log(res); wandb.finish()


if __name__ == "__main__":
    main()
