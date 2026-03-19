"""Evaluation metrics for Temporal Moment Retrieval."""

import torch
import numpy as np
from typing import Tuple, List, Dict


def compute_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """Compute Intersection over Union for temporal segments.
    
    Args:
        pred_start: Predicted start time (seconds)
        pred_end: Predicted end time (seconds)
        gt_start: Ground truth start time (seconds)
        gt_end: Ground truth end time (seconds)
        
    Returns:
        IoU value between 0 and 1
    """
    # Intersection
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0.0, inter_end - inter_start)
    
    # Union
    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union = union_end - union_start
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def compute_temporal_metrics(
    batch_predictions: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute mIoU, R@1, and R@5 from batch predictions.
    
    Args:
        batch_predictions: List of dicts with keys:
            - 'gt_start': Ground truth start
            - 'gt_end': Ground truth end  
            - 'pred_start': Predicted start (or list for top-k)
            - 'pred_end': Predicted end (or list for top-k)
            - 'top_k_predictions': Optional list of (start, end) tuples for R@5
        iou_threshold: Threshold for Recall calculation (default 0.5)
        
    Returns:
        Dict with 'mIoU', 'R@1', 'R@5'
    """
    if not batch_predictions:
        return {"mIoU": 0.0, "R@1": 0.0, "R@5": 0.0}
    
    ious = []
    recall_1 = []
    recall_5 = []
    
    for pred in batch_predictions:
        gt_start = pred["gt_start"]
        gt_end = pred["gt_end"]
        
        # Single prediction (regression) or best from top-k
        pred_start = pred["pred_start"]
        pred_end = pred["pred_end"]
        
        # Compute IoU for best prediction
        iou = compute_iou(pred_start, pred_end, gt_start, gt_end)
        ious.append(iou)
        
        # R@1: best prediction IoU > threshold
        recall_1.append(1.0 if iou >= iou_threshold else 0.0)
        
        # R@5: any of top 5 predictions has IoU > threshold
        if "top_k_predictions" in pred and len(pred["top_k_predictions"]) > 0:
            top_5_ious = [
                compute_iou(p[0], p[1], gt_start, gt_end)
                for p in pred["top_k_predictions"][:5]
            ]
            recall_5.append(1.0 if any(i >= iou_threshold for i in top_5_ious) else 0.0)
        else:
            # Only one prediction, R@5 = R@1
            recall_5.append(1.0 if iou >= iou_threshold else 0.0)
    
    return {
        "mIoU": float(np.mean(ious)),
        "R@1": float(np.mean(recall_1)),
        "R@5": float(np.mean(recall_5)),
    }


def predict_from_offsets(
    offset_predictions: torch.Tensor,
    window_start: float,
    window_end: float,
) -> Tuple[float, float]:
    """Convert offset predictions to absolute timestamps.
    
    Args:
        offset_predictions: Tensor of shape [2] with [start_offset, end_offset]
                           Offsets are typically in range [-1, 1] or similar
        window_start: Start time of the reference window
        window_end: End time of the reference window
        
    Returns:
        (pred_start, pred_end) in absolute time
    """
    window_duration = window_end - window_start
    
    # Convert normalized offsets to absolute timestamps
    # Assuming offsets are in range [-1, 1] where -1 = window_start, 1 = window_end
    start_offset = offset_predictions[0].item()
    end_offset = offset_predictions[1].item()
    
    # Denormalize: offset in [-1, 1] -> timestamp in [window_start, window_end]
    pred_start = window_start + (start_offset + 1) / 2 * window_duration
    pred_end = window_start + (end_offset + 1) / 2 * window_duration
    
    # Ensure valid order
    if pred_start >= pred_end:
        pred_end = pred_start + 1.0
    
    return pred_start, pred_end
