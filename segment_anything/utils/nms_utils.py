"""
nms_utils.py

Non-Maximum Suppression (NMS) utilities for mask post-processing.
Handles overlapping masks by removing duplicates based on IoU overlap and confidence scores.
"""

import numpy as np
from typing import List, Tuple, Optional


def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate IoU (Intersection over Union) between two binary masks.
    
    Args:
        mask1: Binary mask as numpy array (H, W)
        mask2: Binary mask as numpy array (H, W)
    
    Returns:
        IoU value between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def get_mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the centroid (center of mass) of a binary mask.
    
    Args:
        mask: Binary mask as numpy array (H, W)
    
    Returns:
        Tuple of (x, y) centroid coordinates
    """
    if mask.sum() == 0:
        return (0.0, 0.0)
    
    y_coords, x_coords = np.where(mask > 0)
    
    centroid_x = float(np.mean(x_coords))
    centroid_y = float(np.mean(y_coords))
    
    return (centroid_x, centroid_y)


def calculate_centroid_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: (x, y) coordinates
        point2: (x, y) coordinates
    
    Returns:
        Euclidean distance
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def nms_masks(
    masks: List[np.ndarray],
    scores: List[float],
    points: List[Tuple[float, float]],
    iou_threshold: float = 0.3,
    min_centroid_distance: Optional[float] = None,
    use_soft_nms: bool = False,
    soft_nms_sigma: float = 0.5
) -> Tuple[List[np.ndarray], List[float], List[Tuple[float, float]], List[int]]:
    """
    Apply Non-Maximum Suppression to remove overlapping masks.
    
    Args:
        masks: List of binary masks (each is H x W numpy array)
        scores: List of confidence scores (predicted IoU values)
        points: List of prompt points (x, y) tuples
        iou_threshold: IoU threshold for suppression (default: 0.3)
        min_centroid_distance: Minimum distance between mask centroids (optional)
        use_soft_nms: If True, use soft NMS instead of hard suppression
        soft_nms_sigma: Sigma parameter for Gaussian soft NMS
    
    Returns:
        Tuple of (filtered_masks, filtered_scores, filtered_points, kept_indices)
    """
    if len(masks) == 0:
        return [], [], [], []
    
    # Convert to numpy arrays for easier manipulation
    scores_np = np.array(scores)
    
    # Sort indices by score (descending)
    sorted_indices = np.argsort(scores_np)[::-1]
    
    kept_indices = []
    suppressed = np.zeros(len(masks), dtype=bool)
    
    # Calculate centroids if needed
    centroids = None
    if min_centroid_distance is not None:
        centroids = [get_mask_centroid(mask) for mask in masks]
    
    for i in sorted_indices:
        if suppressed[i]:
            continue
        
        kept_indices.append(i)
        
        # Check overlap with this mask
        for j in sorted_indices:
            if i == j or suppressed[j]:
                continue
            
            # Calculate IoU between masks
            iou = calculate_mask_iou(masks[i], masks[j])
            
            # Check centroid distance if enabled
            centroid_too_close = False
            if min_centroid_distance is not None and centroids is not None:
                dist = calculate_centroid_distance(centroids[i], centroids[j])
                centroid_too_close = dist < min_centroid_distance
            
            # Suppress if IoU exceeds threshold or centroids too close
            if iou > iou_threshold or centroid_too_close:
                if use_soft_nms:
                    # Soft NMS: reduce score instead of removing
                    scores_np[j] *= np.exp(-(iou ** 2) / soft_nms_sigma)
                else:
                    # Hard NMS: mark for removal
                    suppressed[j] = True
    
    # If using soft NMS, re-sort and filter by threshold
    if use_soft_nms:
        # Keep masks with score above a threshold after soft suppression
        final_threshold = 0.01  # Very low threshold to keep most masks
        kept_indices = [i for i in range(len(masks)) if scores_np[i] > final_threshold]
        kept_indices = sorted(kept_indices, key=lambda i: scores_np[i], reverse=True)
    
    # Filter results
    filtered_masks = [masks[i] for i in kept_indices]
    filtered_scores = [float(scores_np[i]) for i in kept_indices]
    filtered_points = [points[i] for i in kept_indices]
    
    return filtered_masks, filtered_scores, filtered_points, kept_indices


def nms_masks_advanced(
    masks: List[np.ndarray],
    scores: List[float],
    points: List[Tuple[float, float]],
    iou_threshold: float = 0.3,
    min_centroid_distance: Optional[float] = 5.0,
    score_threshold: float = 0.5
) -> Tuple[List[np.ndarray], List[float], List[Tuple[float, float]], List[int], dict]:
    """
    Advanced NMS with additional statistics and multi-stage filtering.
    
    Stages:
    1. Pre-filter by score threshold
    2. Apply IoU-based NMS
    3. Apply centroid distance filtering
    4. Return filtered results with statistics
    
    Args:
        masks: List of binary masks
        scores: List of confidence scores
        points: List of prompt points
        iou_threshold: IoU threshold for overlap suppression
        min_centroid_distance: Minimum centroid distance
        score_threshold: Minimum score to consider
    
    Returns:
        Tuple of (filtered_masks, filtered_scores, filtered_points, kept_indices, statistics)
    """
    initial_count = len(masks)
    
    if initial_count == 0:
        return [], [], [], [], {
            "initial_count": 0,
            "after_score_filter": 0,
            "after_nms": 0,
            "after_centroid_filter": 0,
            "removed_by_score": 0,
            "removed_by_iou": 0,
            "removed_by_centroid": 0
        }
    
    # Stage 1: Pre-filter by score
    valid_indices = [i for i, score in enumerate(scores) if score >= score_threshold]
    filtered_masks = [masks[i] for i in valid_indices]
    filtered_scores = [scores[i] for i in valid_indices]
    filtered_points = [points[i] for i in valid_indices]
    
    after_score_count = len(filtered_masks)
    removed_by_score = initial_count - after_score_count
    
    if len(filtered_masks) == 0:
        return [], [], [], [], {
            "initial_count": initial_count,
            "after_score_filter": 0,
            "after_nms": 0,
            "after_centroid_filter": 0,
            "removed_by_score": removed_by_score,
            "removed_by_iou": 0,
            "removed_by_centroid": 0
        }
    
    # Stage 2: Apply NMS
    final_masks, final_scores, final_points, final_indices = nms_masks(
        filtered_masks,
        filtered_scores,
        filtered_points,
        iou_threshold=iou_threshold,
        min_centroid_distance=min_centroid_distance,
        use_soft_nms=False
    )
    
    after_nms_count = len(final_masks)
    removed_by_iou = after_score_count - after_nms_count
    
    # Map back to original indices
    final_kept_indices = [valid_indices[i] for i in final_indices]
    
    statistics = {
        "initial_count": initial_count,
        "after_score_filter": after_score_count,
        "after_nms": after_nms_count,
        "after_centroid_filter": after_nms_count,  # Already handled in NMS
        "removed_by_score": removed_by_score,
        "removed_by_iou": removed_by_iou,
        "removed_by_centroid": 0,  # Handled in NMS
        "reduction_percentage": float((initial_count - after_nms_count) / initial_count * 100) if initial_count > 0 else 0.0
    }
    
    return final_masks, final_scores, final_points, final_kept_indices, statistics


def visualize_nms_comparison(
    original_masks: List[np.ndarray],
    filtered_masks: List[np.ndarray],
    original_scores: List[float],
    filtered_scores: List[float]
) -> dict:
    """
    Generate statistics comparing before/after NMS.
    
    Args:
        original_masks: Masks before NMS
        filtered_masks: Masks after NMS
        original_scores: Scores before NMS
        filtered_scores: Scores after NMS
    
    Returns:
        Dictionary with comparison statistics
    """
    return {
        "original_count": len(original_masks),
        "filtered_count": len(filtered_masks),
        "removed_count": len(original_masks) - len(filtered_masks),
        "removal_percentage": float((len(original_masks) - len(filtered_masks)) / len(original_masks) * 100) if len(original_masks) > 0 else 0.0,
        "original_avg_score": float(np.mean(original_scores)) if original_scores else 0.0,
        "filtered_avg_score": float(np.mean(filtered_scores)) if filtered_scores else 0.0,
        "score_improvement": float(np.mean(filtered_scores) - np.mean(original_scores)) if (filtered_scores and original_scores) else 0.0
    }
