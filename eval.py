"""VL-JEPA Evaluation on Charades-STA (Optimized)."""

import argparse
import os
import cv2
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from vljepa.config import Config
from vljepa.dataset import CharadesSTADataset
from vljepa.models import VLJepa
from vljepa.utils import temporal_iou, sliding_window_proposals, nms, load_video_to_ram, sample_frames_from_array

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

    # Load model
    model = VLJepa(config)
    ckpt = torch.load(args.checkpoint, map_location=config.device, weights_only=True)
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

        # 2. LOAD FULL VIDEO TO RAM (The Speed Up)
        v_data = load_video_to_ram(video_path)
        if not v_data:
            skipped += len(group); pbar.update(len(group)); continue

        duration = len(v_data["frames"]) / v_data["fps"]
        proposals = sliding_window_proposals(duration, config.window_sizes, config.window_stride)
        
        # Batch extract visual features
        all_sv, valid_p = [], []
        bs = config.inference_batch_size
        for i in range(0, len(proposals), bs):
            fb = []
            for start, end in proposals[i:i+bs]:
                f = sample_frames_from_array(v_data, start, end, config.num_frames)
                if f: fb.append(f); valid_p.append((start, end))
            if fb:
                pv = model.x_encoder.preprocess_frames(fb, device=config.device)
                all_sv.append(model.x_encoder(pv))
        
        if not all_sv:
            skipped += len(group); pbar.update(len(group)); continue
            
        sv_full = torch.cat(all_sv, dim=0)

        # 3. Predict for each query
        for sample in group:
            try:
                sy_ref = F.normalize(model.encode_text([sample["caption"]], device=config.device), dim=-1)
                qt = model.query_encoder.tokenize([sample["caption"]], device=config.device)
                sims = []
                for j in range(0, sv_full.size(0), bs):
                    b_sv = sv_full[j : j + bs]
                    B = b_sv.size(0)
                    sy_hat = F.normalize(model.predictor(b_sv, qt["input_ids"].expand(B, -1), qt["attention_mask"].expand(B, -1)), dim=-1)
                    sims.append((sy_hat @ sy_ref.T).squeeze(-1))
                
                scores = torch.cat(sims, dim=0).cpu().numpy().tolist()
                k = nms(valid_p, scores, config.nms_threshold)
                if k:
                    iou = temporal_iou(valid_p[k[0]][0], valid_p[k[0]][1], sample["start"], sample["end"])
                    ious.append(iou)
                    for t in recalls:
                        if iou >= t: recalls[t] += 1
                    total += 1
                else: skipped += 1
            except Exception: skipped += 1
            pbar.update(1)

    # Summary
    if total > 0:
        res = {f"eval/R@1_IoU={t}": (c/total)*100 for t, c in recalls.items()} | {"eval/mIoU": (sum(ious)/len(ious))*100}
        print("\nResults:", res)
        if use_wandb: wandb.log(res); wandb.finish()


if __name__ == "__main__":
    main()
