"""Utility functions: video I/O, temporal IoU, NMS, sliding windows."""

import cv2
import numpy as np
import torch


def load_video_frames(
    video_path: str,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    num_frames: int = 16,
) -> list[np.ndarray] | None:
    """Load uniformly sampled RGB frames from a video segment.

    Args:
        video_path: path to .mp4 file
        start_sec: start of segment in seconds
        end_sec: end of segment in seconds (None = end of video)
        num_frames: number of frames to sample

    Returns:
        List of RGB numpy arrays (H, W, 3), or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        return None

    duration = total_frames / fps
    if end_sec is None:
        end_sec = duration

    start_frame = max(0, int(start_sec * fps))
    end_frame = min(total_frames - 1, int(end_sec * fps))

    if end_frame <= start_frame:
        cap.release()
        return None

    n_available = end_frame - start_frame + 1
    n_sample = min(num_frames, n_available)
    indices = np.linspace(start_frame, end_frame, n_sample, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if len(frames) == 0:
        return None

    return frames


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        return 0.0
    return total_frames / fps


def temporal_iou(
    pred_start: float,
    pred_end: float,
    gt_start: float,
    gt_end: float,
) -> float:
    """Compute temporal Intersection over Union between two segments."""
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    inter = max(0.0, inter_end - inter_start)
    union = (pred_end - pred_start) + (gt_end - gt_start) - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms(
    proposals: list[tuple[float, float]],
    scores: list[float],
    iou_threshold: float = 0.5,
) -> list[int]:
    """Non-maximum suppression for temporal proposals.

    Args:
        proposals: list of (start, end) tuples
        scores: corresponding scores
        iou_threshold: suppress proposals with IoU above this

    Returns:
        List of kept indices (sorted by score descending).
    """
    if len(proposals) == 0:
        return []

    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept = []

    for i in sorted_idx:
        should_keep = True
        for j in kept:
            iou = temporal_iou(
                proposals[i][0], proposals[i][1],
                proposals[j][0], proposals[j][1],
            )
            if iou > iou_threshold:
                should_keep = False
                break
        if should_keep:
            kept.append(i)

    return kept


def sliding_window_proposals(
    duration: float,
    window_sizes: list[float],
    stride: float = 1.0,
) -> list[tuple[float, float]]:
    """Generate candidate temporal proposals using sliding windows.

    Args:
        duration: total video duration in seconds
        window_sizes: list of window durations to use
        stride: step size in seconds

    Returns:
        List of (start, end) proposals.
    """
    proposals = []
    for ws in window_sizes:
        if ws > duration:
            # Single proposal covering the whole video
            proposals.append((0.0, duration))
            continue
        start = 0.0
        while start + ws <= duration + 0.01:  # small epsilon for float
            end = min(start + ws, duration)
            proposals.append((start, end))
            start += stride
    return proposals
