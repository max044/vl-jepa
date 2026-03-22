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
    parser.add_argument("--save-report", type=str, default="eval_report.txt")
    parser.add_argument("--window-stride", type=float, default=None)
    parser.add_argument("--window-sizes", type=str, default=None, help="Comma-separated float list (e.g. 4.0,8.0,16.0)")
    return parser.parse_args()


def print_results(results, duration_sec, total_count, skipped_count, checkpoint):
    """Print a beautiful ASCII table and summary."""
    print("\n" + "═"*60)
    print(f"║ {'VL-JEPA EVALUATION REPORT':^56} ║")
    print("═"*60)
    print(f"║ Checkpoint: {os.path.basename(checkpoint):<44} ║")
    print("╟" + "─"*58 + "╢")
    
    metrics = [
        ("R@1, IoU=0.3", results.get("eval/R@1_IoU=0.3", 0)),
        ("R@1, IoU=0.5", results.get("eval/R@1_IoU=0.5", 0)),
        ("R@1, IoU=0.7", results.get("eval/R@1_IoU=0.7", 0)),
        ("Mean IoU (mIoU)", results.get("eval/mIoU", 0)),
    ]
    
    print(f"║ {'Metric':<30} │ {'Value':>25} ║")
    print("╟" + "─"*31 + "┼" + "─"*26 + "╢")
    for name, val in metrics:
        print(f"║ {name:<30} │ {val:>24.2f}% ║")
    
    print("╟" + "─"*31 + "┴" + "─"*26 + "╢")
    print(f"║ Samples: {total_count:<10} │ Skipped: {skipped_count:<10} │ Time: {duration_sec/60:>6.1f}m ║")
    print("═"*60 + "\n")


@torch.no_grad()
def main():
    args = parse_args()
    # Load base config
    config = Config()
    
    # Load from YAML if specified (or try default)
    import yaml
    base_config_path = os.path.join("configs", "base.yaml")
    if os.path.exists(base_config_path):
        print(f"Loading config from {base_config_path}")
        with open(base_config_path, "r") as f:
            yaml_config = yaml.safe_load(f)
            for k, v in yaml_config.items():
                if hasattr(config, k):
                    setattr(config, k, v)

    # CLI Overrides
    if args.device:
        config.device = args.device
    if args.window_stride:
        config.window_stride = args.window_stride
    if args.window_sizes:
        config.window_sizes = [float(x.strip()) for x in args.window_sizes.split(",")]

    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        entity = os.getenv("WANDB_ENTITY", "maxence-cabiddu-maxence-cabiddu")
        if args.wandb_run_path:
            wandb.init(project=args.wandb_project, entity=entity, id=args.wandb_run_path.split("/")[-1], resume="allow", tags=["eval"])
        else:
            wandb.init(project=args.wandb_project, entity=entity, config=config.__dict__, job_type="eval", tags=["eval"])

    checkpoint_path = args.checkpoint
    if ":" in checkpoint_path and not os.path.exists(checkpoint_path):
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
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False) # Load to CPU first to parse config
    
    # Handle config in checkpoint
    if "config" in ckpt:
        print("  ✓ Found config in checkpoint, updating current config...")
        ckpt_config = ckpt["config"]
        for k, v in ckpt_config.items():
            if hasattr(config, k):
                setattr(config, k, v)
    
    # Initialize model with correct architecture from checkpoint
    model = VLJepa(config).to(config.device)

    # 2nd Pass: Allow base.yaml/CLI to override inference-specific params (for speed/stride)
    # Priority: CLI > base.yaml > Checkpoint
    inference_params = ["window_sizes", "window_stride", "nms_threshold", "inference_batch_size"]
    
    # 1. From base.yaml
    if os.path.exists(base_config_path):
        with open(base_config_path, "r") as f:
            yaml_config = yaml.safe_load(f)
            for k in inference_params:
                if k in yaml_config and hasattr(config, k):
                    setattr(config, k, yaml_config[k])
    
    # 2. From CLI (Highest Priority)
    if args.window_stride:
        config.window_stride = args.window_stride
    if args.window_sizes:
        config.window_sizes = [float(x.strip()) for x in args.window_sizes.split(",")]
    if args.device:
        config.device = args.device

    # Handle different checkpoint formats
    if "model_state_dict" in ckpt:
        # Full model state dict format (from training/train.py)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  ✓ Loaded full model state (best val_loss: {ckpt.get('best_val_loss', 'N/A')})")
    elif "predictor_state_dict" in ckpt:
        # Component-specific format
        model.predictor.load_state_dict(ckpt["predictor_state_dict"])
        model.y_encoder.projection.load_state_dict(ckpt["y_projection_state_dict"])
        print("  ✓ Loaded predictor and y_encoder projection")
    else:
        print("❌ Unknown checkpoint format")
        return
    
    model.eval()

    print("\n" + "─"*40)
    print(f"🚀 Evaluation Config:")
    print(f"   Device: {config.device}")
    print(f"   Windows: {config.window_sizes}s (stride {config.window_stride}s)")
    print(f"   Regression Head: {'ON' if getattr(config, 'use_regression', False) else 'OFF'}")
    print(f"   Inference Batch Size: {config.inference_batch_size}")
    print("─"*40 + "\n")

    # Dataset
    test_dataset = CharadesSTADataset(config.anno_test, config.videos_dir, config, split="test")
    samples = test_dataset.samples[:args.max_samples] if args.max_samples else test_dataset.samples

    video_to_queries = defaultdict(list)
    for s in samples:
        video_to_queries[s["video_path"]].append(s)

    ious, recalls = [], {0.3: 0, 0.5: 0, 0.7: 0}
    total, skipped = 0, 0

    import time
    start_time = time.time()
    
    torch.set_grad_enabled(False)
    
    with tqdm(total=len(samples), desc="Evaluating", dynamic_ncols=True) as pbar:
        for video_path, group in video_to_queries.items():
            # 1. Ensure video exists (download from HF Storage if needed)
            if not os.path.exists(video_path):
                if HAS_HF_HUB and config.use_hf_storage:
                    try:
                        vid = group[0].get("video_id") or os.path.basename(video_path).replace(".mp4", "")
                        video_path = hf_hub_download(
                            config.hf_bucket_id,
                            f"Charades_v1_480/{vid}.mp4",
                            repo_type="dataset",
                            local_dir=config.videos_dir,
                            local_dir_use_symlinks=False
                        )
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

            # 3. Predict once for all proposals in this video using Neutral Query
            neutral_query = "What is happening in this video?"
            nq_t = model.query_encoder.tokenize([neutral_query], device=config.device)
            nq_ids = nq_t["input_ids"].expand(bs, -1)
            nq_mask = nq_t["attention_mask"].expand(bs, -1)
            
            all_sy_hat = []
            all_offsets = []
            
            for j in range(0, sv_full.size(0), bs):
                b_sv = sv_full[j : j + bs]
                current_bs = b_sv.size(0)
                outputs = model.predictor(b_sv, nq_ids[:current_bs], nq_mask[:current_bs])
                
                all_sy_hat.append(F.normalize(outputs["sy_hat"], dim=-1))
                if "offsets" in outputs and getattr(config, "use_regression", False):
                    all_offsets.append(outputs["offsets"])
            
            sy_hat_full = torch.cat(all_sy_hat, dim=0) # (NumProposals, Embed)
            
            # 4. Compute similarities for each query in the group
            captions = [s["caption"] for s in group]
            sy_refs = F.normalize(model.encode_text(captions, device=config.device), dim=-1) # (NumQueries, Embed)
            
            # (NumProposals, Embed) @ (Embed, NumQueries) -> (NumProposals, NumQueries)
            all_sims = (sy_hat_full @ sy_refs.T).cpu().numpy()
            
            for q_idx, sample in enumerate(group):
                scores = all_sims[:, q_idx].tolist()
                
                # Refine proposals if regression was used
                if all_offsets:
                    offsets_full = torch.cat(all_offsets, dim=0).cpu().numpy()
                    refined_proposals = []
                    for idx, (p_start, p_end) in enumerate(valid_p):
                        dur = p_end - p_start
                        o_start, o_end = offsets_full[idx]
                        refined_proposals.append((
                            max(0, p_start + o_start * dur),
                            min(duration, p_end + o_end * dur)
                        ))
                else:
                    refined_proposals = valid_p
                
                k = nms(refined_proposals, scores, config.nms_threshold)
                if k:
                    iou = temporal_iou(refined_proposals[k[0]][0], refined_proposals[k[0]][1], sample["start"], sample["end"])
                    ious.append(iou)
                    for t in recalls:
                        if iou >= t: recalls[t] += 1
                    total += 1
                else: skipped += 1
                pbar.update(1)

    eval_duration = time.time() - start_time
    # Summary
    if total > 0:
        res = {f"eval/R@1_IoU={t}": (c/total)*100 for t, c in recalls.items()} | {"eval/mIoU": (sum(ious)/len(ious))*100}
        
        print_results(res, eval_duration, total, skipped, checkpoint_path)
        
        if args.save_report:
            report_path = os.path.join(config.checkpoint_dir, args.save_report)
            with open(report_path, "w") as f:
                f.write(f"Evaluation Report - {time.ctime()}\n")
                f.write(f"Checkpoint: {checkpoint_path}\n")
                f.write(f"Results: {res}\n")
                f.write(f"Samples: {total}, Skipped: {skipped}, Duration: {eval_duration/60:.2f}m\n")
            print(f"📝 Report saved to: {report_path}")

        if use_wandb:
            wandb.log(res)
            wandb.finish()


if __name__ == "__main__":
    main()
