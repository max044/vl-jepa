"""Utility functions: video I/O, temporal IoU, NMS, sliding windows."""

import cv2
import numpy as np
import torch


def load_video_to_ram(video_path: str) -> dict | None:
    """Load an entire video into a numpy array in RAM.
    
    Returns:
        dict with 'frames' (N, H, W, 3) and 'fps', or None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    
    if not frames:
        return None
        
    return {
        "frames": np.array(frames),
        "fps": fps
    }


def sample_frames_from_array(video_data: dict, start_sec: float, end_sec: float, num_frames: int = 16) -> list[np.ndarray] | None:
    """Sample frames from a pre-loaded numpy array."""
    frames = video_data["frames"]
    fps = video_data["fps"]
    total_frames = len(frames)
    
    start_frame = max(0, int(start_sec * fps))
    end_frame = min(total_frames - 1, int(end_sec * fps))
    
    if end_frame <= start_frame:
        return None
        
    indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    return [frames[idx] for idx in indices]



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
