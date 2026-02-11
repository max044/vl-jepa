"""VL-JEPA Moment Retrieval Inference.

Given a video and a text query, retrieve the most relevant temporal moments.

Usage:
    python infer.py --video data/Charades_v1_480/3MSZA.mp4 \\
                    --query "person turns on the light" \\
                    --checkpoint checkpoints/best.pth
"""

import argparse
import torch
import torch.nn.functional as F

from vljepa.config import Config
from vljepa.models import VLJepa
from vljepa.utils import (
    load_video_frames,
    get_video_duration,
    sliding_window_proposals,
    nms,
)


def parse_args():
    parser = argparse.ArgumentParser(description="VL-JEPA Moment Retrieval")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--query", type=str, required=True, help="Text query")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--window-sizes", type=float, nargs="+", default=None)
    parser.add_argument("--stride", type=float, default=None)
    return parser.parse_args()


@torch.no_grad()
def retrieve_moments(
    model: VLJepa,
    video_path: str,
    query: str,
    config: Config,
) -> list[dict]:
    """Retrieve top-K temporal moments from a video matching the query.

    Process:
    1. Generate sliding-window proposals over the full video
    2. For each proposal, extract frames → X-Encoder → Predictor → Ŝy
    3. Encode query via Y-Encoder to get target reference Sy
    4. Compute cosine similarity between each Ŝy and Sy
    5. Apply NMS, return top-K ranked moments

    Returns:
        List of dicts with keys: start, end, score, rank
    """
    device = config.device

    # Get video duration
    duration = get_video_duration(video_path)
    if duration <= 0:
        print(f"Error: Cannot read video {video_path}")
        return []

    print(f"Video duration: {duration:.1f}s")

    # Generate proposals
    proposals = sliding_window_proposals(
        duration, config.window_sizes, config.window_stride
    )
    print(f"Generated {len(proposals)} proposals")

    # Encode the query text to get reference embedding
    sy_ref = model.encode_text([query], device=device)  # (1, embed_dim)
    sy_ref = F.normalize(sy_ref, dim=-1)

    # Tokenize query for predictor input
    query_tokens = model.query_encoder.tokenize([query], device=device)

    # Score each proposal
    scores = []
    valid_proposals = []

    # Process proposals in batches for efficiency
    # Reduced batch_size significantly to avoid MPS OOM with V-JEPA 2 ViT-L
    batch_size = 2
    for i in range(0, len(proposals), batch_size):
        batch_props = proposals[i : i + batch_size]

        frames_batch = []
        batch_valid = []

        for start, end in batch_props:
            frames = load_video_frames(
                video_path, start, end, config.num_frames
            )
            if frames is not None and len(frames) > 0:
                frames_batch.append(frames)
                batch_valid.append((start, end))

        if len(frames_batch) == 0:
            continue

        # Preprocess and encode
        pixel_values = model.x_encoder.preprocess_frames(frames_batch, device=device)

        # Repeat query tokens for batch
        B = pixel_values.size(0)
        q_ids = query_tokens["input_ids"].expand(B, -1)
        q_mask = query_tokens["attention_mask"].expand(B, -1)

        # Get predictions
        sy_hat = model.encode_video_query(pixel_values, q_ids, q_mask)
        sy_hat = F.normalize(sy_hat, dim=-1)

        # Cosine similarity with reference
        sims = (sy_hat @ sy_ref.T).squeeze(-1)  # (B,)

        for j, (start, end) in enumerate(batch_valid):
            scores.append(sims[j].item())
            valid_proposals.append((start, end))

    if len(valid_proposals) == 0:
        print("No valid proposals found!")
        return []

    # Apply NMS
    kept_indices = nms(valid_proposals, scores, config.nms_threshold)

    # Return top-K
    results = []
    for rank, idx in enumerate(kept_indices[: config.top_k]):
        results.append({
            "rank": rank + 1,
            "start": valid_proposals[idx][0],
            "end": valid_proposals[idx][1],
            "score": scores[idx],
        })

    return results


def main():
    args = parse_args()

    config = Config()
    if args.device:
        config.device = args.device
    if args.top_k:
        config.top_k = args.top_k
    if args.window_sizes:
        config.window_sizes = args.window_sizes
    if args.stride:
        config.window_stride = args.stride

    print(f"Device: {config.device}")
    print(f"Query: \"{args.query}\"")
    print(f"Video: {args.video}")
    print()

    # Load model
    print("Loading model...")
    model = VLJepa(config)

    # Load trained weights
    ckpt = torch.load(args.checkpoint, map_location=config.device, weights_only=True)
    model.predictor.load_state_dict(ckpt["predictor_state_dict"])
    model.y_encoder.projection.load_state_dict(ckpt["y_projection_state_dict"])

    model.predictor.to(config.device)
    model.y_encoder.projection.to(config.device)
    model.predictor.eval()

    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})\n")

    # Retrieve moments
    results = retrieve_moments(model, args.video, args.query, config)

    # Display results
    print(f"\n{'═' * 50}")
    print(f"Top-{config.top_k} moments for: \"{args.query}\"")
    print(f"{'═' * 50}")

    for r in results:
        start_fmt = f"{int(r['start']//60):02d}:{r['start']%60:05.2f}"
        end_fmt = f"{int(r['end']//60):02d}:{r['end']%60:05.2f}"
        print(
            f"  #{r['rank']}  [{start_fmt} → {end_fmt}]  "
            f"score={r['score']:.4f}  "
            f"(dur={r['end']-r['start']:.1f}s)"
        )

    if not results:
        print("  No moments found.")


if __name__ == "__main__":
    main()
