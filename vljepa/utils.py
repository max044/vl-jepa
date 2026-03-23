"""Utility functions: video I/O, temporal IoU, NMS, sliding windows."""

import cv2
import numpy as np


def load_video_frames(
    video_path: str,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    num_frames: int = 16,
) -> list[np.ndarray] | None:
    try:
        from decord import VideoReader, cpu
        vr    = VideoReader(video_path, ctx=cpu(0))
        fps   = vr.get_avg_fps()
        total = len(vr)

        start_frame = max(0, int(start_sec * fps))
        end_frame   = min(total - 1, int(end_sec * fps) if end_sec is not None else total - 1)
        if end_frame <= start_frame:
            return None

        indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
        frames  = vr.get_batch(indices).asnumpy()  # (T, H, W, 3) — une seule op
        return list(frames)
    except Exception:
        return None


def load_video_to_ram(video_path: str) -> dict | None:
    """Load an entire video into RAM as a single RGB numpy array.

    Returns dict with 'frames' (N, H, W, 3) uint8 RGB and 'fps', or None.
    Used by eval.py to load each video once before sliding window scoring.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # BGR → RGB

    cap.release()

    if not frames:
        return None

    return {"frames": np.array(frames), "fps": fps}


def temporal_iou(
    pred_start: float,
    pred_end: float,
    gt_start: float,
    gt_end: float,
) -> float:
    """Temporal Intersection over Union between two segments."""
    inter = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = (pred_end - pred_start) + (gt_end - gt_start) - inter
    return inter / union if union > 0 else 0.0


def nms(
    proposals: list[tuple[float, float]],
    scores: list[float],
    iou_threshold: float = 0.5,
) -> list[int]:
    """Non-maximum suppression for temporal proposals.

    Returns kept indices sorted by score descending.
    """
    if not proposals:
        return []

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept  = []

    for i in order:
        if all(
            temporal_iou(proposals[i][0], proposals[i][1],
                         proposals[j][0], proposals[j][1]) <= iou_threshold
            for j in kept
        ):
            kept.append(i)

    return kept


def sliding_window_proposals(
    duration: float,
    window_sizes: list[float],
    stride: float = 1.0,
) -> list[tuple[float, float]]:
    """Generate temporal proposals via sliding windows.

    For each window size, slides across the video with the given stride.
    If a window is larger than the video, a single proposal covers the whole video.

    Returns list of (start, end) tuples in seconds.
    """
    proposals = []
    for ws in window_sizes:
        if ws >= duration:
            proposals.append((0.0, duration))
            continue
        start = 0.0
        while start + ws <= duration + 1e-6:
            proposals.append((start, min(start + ws, duration)))
            start += stride
    return proposals