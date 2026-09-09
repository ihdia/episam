from __future__ import annotations

import os
import json
import logging
import argparse
import itertools
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as torchF
from torch.utils.data import Dataset as TorchDataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

log = logging.getLogger("two_hop_graph")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────

class TwoHopGraph:
    """
    Lightweight container for a two-hop graph over N character nodes.

    Attributes
    ----------
    num_nodes      : int
    edges          : np.ndarray  shape (E, 2)  – each row is (i, j), i < j
    edge_sim       : np.ndarray  shape (E,)    – symmetric similarity score
    edge_hop       : np.ndarray  shape (E,)    – 1 or 2 (which hop added it)
    edge_gt        : np.ndarray  shape (E,) or None – 1 if same line, else 0
    adj            : dict[int, set[int]]       – adjacency (both directions)
    centroids      : np.ndarray  shape (N, 2)
    target_masks   : list of (256,256) uint8
    line_numbers   : np.ndarray  shape (N,) or None
    gt_line_masks  : list of (H, W) bool or None – pixel-level GT line masks
    image_id       : str
    """
    __slots__ = (
        'num_nodes', 'edges', 'edge_sim', 'edge_hop',
        'edge_gt', 'adj', 'centroids', 'target_masks',
        'line_numbers', 'gt_line_masks', 'image_id',
    )

    def __init__(self):
        for s in self.__slots__:
            object.__setattr__(self, s, None)


# ─────────────────────────────────────────────────────────────────────────────
# Similarity helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sim(mask_a: np.ndarray, mask_b: np.ndarray, method: str) -> float:
    """Compute a single directed similarity score between two binary masks."""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    if method == 'iou':
        union = np.logical_or(a, b).sum()
        return float(inter / (union + 1e-6))
    elif method == 'ios':
        smaller = min(a.sum(), b.sum())
        return float(inter / (smaller + 1e-6))
    elif method == 'dice':
        return float(2.0 * inter / (a.sum() + b.sum() + 1e-6))
    raise ValueError(f"Unknown similarity method: {method!r}")


def symmetric_similarity(
    i: int,
    j: int,
    target_masks: List[np.ndarray],
    left_masks:   List[np.ndarray],
    right_masks:  List[np.ndarray],
    method: str,
) -> float:
    """
    Symmetric similarity between nodes i and j.

    We compute seven directional scores and take the maximum:

      • target_i vs target_j  — direct mask overlap
      • left_i   vs target_j  — i predicts j as its left  neighbour
      • right_i  vs target_j  — i predicts j as its right neighbour
      • target_i vs left_j    — j predicts i as its left  neighbour
      • target_i vs right_j   — j predicts i as its right neighbour
      • left_i   vs right_j   — i's left prediction overlaps j's right
                                 prediction.  Handles the case where a
                                 character between i and j is missing from
                                 the graph: both i and j predict a neighbour
                                 in the gap, so their neighbour masks point
                                 at the same empty region and overlap.
      • right_i  vs left_j    — same logic mirrored (j is to the left of i).

    Taking the max means an edge is strong if *any* of these directional
    signals agrees, which is maximally robust to missing characters and
    weak individual predictions.
    """
    tm_i, tm_j = target_masks[i], target_masks[j]
    lm_i, lm_j = left_masks[i],   left_masks[j]
    rm_i, rm_j = right_masks[i],  right_masks[j]

    scores = [
        _sim(tm_i, tm_j, method),   # target  vs target
        _sim(lm_i, tm_j, method),   # i-left  vs j-target
        _sim(rm_i, tm_j, method),   # i-right vs j-target
        _sim(tm_i, lm_j, method),   # i-target vs j-left
        _sim(tm_i, rm_j, method),   # i-target vs j-right
        _sim(lm_i, rm_j, method),   # i-left  vs j-right  ← gap bridging
        _sim(rm_i, lm_j, method),   # i-right vs j-left   ← gap bridging
    ]
    return float(max(scores))


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_two_hop_graph(
    centroids:     np.ndarray,
    target_masks:  List[np.ndarray],
    left_masks:    List[np.ndarray],
    right_masks:   List[np.ndarray],
    line_numbers:  Optional[np.ndarray],
    gt_line_masks: Optional[List[np.ndarray]] = None,  
    k:             int                  = 10,
    similarity_method: str              = 'iou',
    image_id:      str                  = '',
) -> TwoHopGraph:
    """
    Build a two-hop graph from character centroids and SAM masks.

    Step 1 – 1-hop kNN
        For each node i, find its k nearest neighbours by L2 centroid distance.
        This defines the base 1-hop adjacency.

    Step 2 – 2-hop extension
        For each 1-hop edge (i,j), add edges from i to all of j's 1-hop
        neighbours (and vice versa) that are not already 1-hop neighbours
        of i.  These new edges are tagged hop=2.

    Step 3 – similarity scores
        For every edge (i,j) (1-hop and 2-hop), compute the symmetric
        similarity score: max over all five directional mask overlaps.

    Step 4 – GT labels (if line_numbers provided)
        edge_gt[e] = 1 iff line_numbers[i] == line_numbers[j] and both ≥ 0.

    Parameters
    ----------
    centroids         : (N, 2) float  – character centroids in SAM 1024-space
    target_masks      : N × (256,256) uint8 {0,1}
    left_masks        : N × (256,256) uint8 {0,1}
    right_masks       : N × (256,256) uint8 {0,1}
    line_numbers      : (N,) int  or None
    k                 : number of 1-hop nearest neighbours
    similarity_method : 'iou' | 'ios' | 'dice'
    image_id          : label stored in graph for logging

    Returns
    -------
    TwoHopGraph
    """
    N = len(centroids)
    
    g = TwoHopGraph()
    g.num_nodes    = N
    g.centroids    = centroids.copy()
    g.target_masks = target_masks
    g.line_numbers = line_numbers
    g.image_id     = image_id
    g.gt_line_masks = gt_line_masks

    if N < 2:
        g.edges    = np.zeros((0, 2), dtype=np.int32)
        g.edge_sim = np.zeros(0,      dtype=np.float32)
        g.edge_hop = np.zeros(0,      dtype=np.int8)
        g.edge_gt  = np.zeros(0,      dtype=np.int8)
        g.adj      = {i: set() for i in range(N)}
        return g

    # ── Step 1: 1-hop kNN ────────────────────────────────────────────────────
    diff = centroids[:, None, :] - centroids[None, :, :]   # (N, N, 2)
    dist = np.linalg.norm(diff, axis=-1)                    # (N, N)
    np.fill_diagonal(dist, np.inf)

    adj_1hop: Dict[int, set] = {i: set() for i in range(N)}
    for i in range(N):
        nn = np.argsort(dist[i])[:k]
        for j in nn:
            if i != j:  # ADD THIS CHECK
                adj_1hop[i].add(int(j))
                adj_1hop[int(j)].add(i)

    # ── Step 2: 2-hop extension ───────────────────────────────────────────────
    # edge_set maps frozenset({i,j}) → hop_label (1 takes priority over 2)
    edge_hop_map: Dict[frozenset, int] = {}
    for i in range(N):
        for j in adj_1hop[i]:
            key = frozenset({i, j})
            edge_hop_map[key] = 1                           # 1-hop

    for i in range(N):
        for j in adj_1hop[i]:
            for m in adj_1hop[j]:                           # 2-hop from i
                if m == i:
                    continue
                key = frozenset({i, m})
                if key not in edge_hop_map:
                    edge_hop_map[key] = 2

    # ── Step 3: build edge arrays ─────────────────────────────────────────────
    adj_full: Dict[int, set] = {i: set() for i in range(N)}
    edge_list:  List[Tuple[int, int]] = []
    edge_sims:  List[float]           = []
    edge_hops:  List[int]             = []

    for key, hop in edge_hop_map.items():
        if len(key) != 2:  # ADD THIS CHECK
            log.warning("Skipping malformed edge: %s", key)
            continue
        i, j = sorted(key)
        if i == j:  # ADD THIS CHECK (redundant but safe)
            continue
        adj_full[i].add(j)
        adj_full[j].add(i)
        sim = symmetric_similarity(
            i, j, target_masks, left_masks, right_masks, similarity_method
        )
        edge_list.append((i, j))
        edge_sims.append(sim)
        edge_hops.append(hop)

    g.adj      = adj_full
    g.edges    = np.array(edge_list,  dtype=np.int32)   if edge_list else np.zeros((0,2), dtype=np.int32)
    g.edge_sim = np.array(edge_sims,  dtype=np.float32) if edge_sims else np.zeros(0, dtype=np.float32)
    g.edge_hop = np.array(edge_hops,  dtype=np.int8)    if edge_hops else np.zeros(0, dtype=np.int8)

    # ── Step 4: GT labels ─────────────────────────────────────────────────────
    if line_numbers is not None and len(edge_list) > 0:
        lbl = []
        for (i, j) in edge_list:
            li, lj = int(line_numbers[i]), int(line_numbers[j])
            lbl.append(1 if (li >= 0 and li == lj) else 0)
        g.edge_gt = np.array(lbl, dtype=np.int8)
    else:
        g.edge_gt = None

    log.debug(
        "Built 2-hop graph: image=%s  N=%d  1-hop=%d  2-hop=%d  total_edges=%d",
        image_id, N,
        int((g.edge_hop == 1).sum()),
        int((g.edge_hop == 2).sum()),
        len(edge_list),
    )



    return g


# ─────────────────────────────────────────────────────────────────────────────
# Method 1 – Manual threshold with common-neighbour veto
# ─────────────────────────────────────────────────────────────────────────────

def classify_edges_manual(
    graph:     TwoHopGraph,
    threshold: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Label every edge using manual threshold + common-neighbour veto.

    Rule
    ----
    For edge (A, B):
      1.  If sim(A,B) ≤ t  →  label = 0  (not same line)
      2.  For every common neighbour C of A and B:
              if sim(A,C) < t  AND  sim(B,C) > t:
                  label = 0  (veto: C is on B's line, not A's)
                  break
      3.  Otherwise  →  label = 1  (same line)

    Returns
    -------
    labels     : (E,) int8  – 0 or 1
    confidence : (E,) float – raw similarity score (used as confidence)
    """
    if len(graph.edges) == 0:
        return np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.float32)

    E = len(graph.edges)
    labels     = np.zeros(E, dtype=np.int8)
    confidence = graph.edge_sim.copy()

    # Build edge_sim lookup: (i,j) with i<j → sim
    sim_lookup: Dict[Tuple[int,int], float] = {}
    for idx, (i, j) in enumerate(graph.edges):
        sim_lookup[(int(i), int(j))] = float(graph.edge_sim[idx])

    def get_sim(a: int, b: int) -> float:
        key = (min(a, b), max(a, b))
        return sim_lookup.get(key, 0.0)

    for idx, (i, j) in enumerate(graph.edges):
        i, j = int(i), int(j)
        sim_ij = float(graph.edge_sim[idx])

        # Rule 1: direct threshold
        if sim_ij <= threshold:
            continue                        # label stays 0

        # Rule 2: common-neighbour veto
        common = graph.adj[i] & graph.adj[j]
        vetoed = False
        for c in common:
            sim_ic = get_sim(i, c)
            sim_jc = get_sim(j, c)
            # C is firmly connected to j but not to i → veto
            if sim_ic < threshold and sim_jc > threshold:
                log.debug(
                    "Veto: (%d,%d) via common nbr %d  "
                    "sim(%d,%d)=%.3f sim(%d,%d)=%.3f",
                    i, j, c, i, c, sim_ic, j, c, sim_jc,
                )
                vetoed = True
                break

        if not vetoed:
            labels[idx] = 1

    n_pos = labels.sum()
    log.debug("Manual classify: t=%.3f  pos=%d/%d", threshold, n_pos, E)
    return labels, confidence


# ─────────────────────────────────────────────────────────────────────────────
# Line extraction from labelled edges
# ─────────────────────────────────────────────────────────────────────────────

def edges_to_lines(
    graph:  TwoHopGraph,
    labels: np.ndarray,
    num_nodes: int,
) -> List[List[int]]:
    """
    BFS connected-components on the positive-labelled edges.
    Returns list of lists of node indices (one per line).
    """
    adj: Dict[int, set] = {i: set() for i in range(num_nodes)}
    for idx, (i, j) in enumerate(graph.edges):
        if labels[idx] == 1:
            adj[int(i)].add(int(j))
            adj[int(j)].add(int(i))

    visited = [False] * num_nodes
    lines   = []
    for start in range(num_nodes):
        if visited[start]:
            continue
        queue   = [start]
        visited[start] = True
        comp    = []
        while queue:
            node = queue.pop()
            comp.append(node)
            for nb in adj[node]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        lines.append(comp)
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Line mask creation and line-level metrics
# ─────────────────────────────────────────────────────────────────────────────

def load_gt_line_masks(record: dict):
    """
    Load GT line masks from the record's 'line_masks_npz' key.

    Returns list of (H, W) bool arrays, or None if unavailable.
    """
    key = 'line_masks_npz'
    if key not in record or record[key] is None or not os.path.exists(record[key]):
        return None
    try:
        data = np.load(record[key])
        arr = data['masks']          # (L, H, W)
        masks = [
            (arr[i] > 0).astype(bool)
            for i in range(arr.shape[0])
            if arr[i].sum() > 0
        ]
        return masks if masks else None
    except Exception as e:
        print(f"  [WARN] Could not load GT line masks: {e}")
        return None

def _unpad_and_resize_mask(
    mask: np.ndarray,
    input_size: Tuple[int, int],
    original_size: Tuple[int, int],
) -> np.ndarray:
    """
    Reverse SAM padding + ResizeLongestSide to map a 256x256 decoder mask
    back to original image pixel space.

    Steps:
      1. Scale 256x256 -> 1024x1024 (decoder -> padded input space).
      2. Crop [:ih, :iw] to remove zero-padding.
      3. Resize to original_size.
    """
    sam_size = 1024
    mask_255 = (mask.astype(np.uint8) * 255)
    mask_1024 = cv2.resize(mask_255, (sam_size, sam_size),
                           interpolation=cv2.INTER_NEAREST)
    ih = min(int(input_size[0]), sam_size)
    iw = min(int(input_size[1]), sam_size)
    mask_cropped = mask_1024[:ih, :iw]
    orig_h, orig_w = original_size
    mask_orig = cv2.resize(mask_cropped, (orig_w, orig_h),
                           interpolation=cv2.INTER_NEAREST)
    return (mask_orig > 127).astype(np.uint8)


def _get_mask_center_orig(mask: np.ndarray) -> Optional[np.ndarray]:
    """Get the center of mass of a binary mask (original image space)."""
    moments = cv2.moments(mask)
    if moments['m00'] > 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return np.array([cx, cy])
    else:
        # Fallback to geometric center if moments fail
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) > 0:
            return np.array([np.mean(x_coords), np.mean(y_coords)])
    return None


def _get_mask_dimensions_orig(mask: np.ndarray) -> Tuple[float, float]:
    """Get the width and height of a binary mask's bounding box (original image space)."""
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) > 0 and len(y_coords) > 0:
        width = float(x_coords.max() - x_coords.min())
        height = float(y_coords.max() - y_coords.min())
        return width, height
    return 0.0, 0.0


def _calculate_average_mask_height_orig(char_masks: List[np.ndarray]) -> float:
    """Calculate average height of all character masks in original image space."""
    heights = []
    for mask in char_masks:
        _, h = _get_mask_dimensions_orig(mask)
        if h > 0:
            heights.append(h)
    avg_height = np.mean(heights) if heights else 50.0
    return avg_height


def create_line_mask_from_characters(
    line_char_indices: List[int],
    target_masks: List[np.ndarray],
    original_size: Tuple[int, int],
    input_size: Optional[Tuple[int, int]] = None,
    size_multiplier: float = 1.0,
) -> np.ndarray:
    """
    Create a line mask by connecting character centroids with lines.
    
    Similar to create_line_mask_from_centers from create_line_masks_improved.py.
    This method:
    1. Starts with union of character masks as the base
    2. Gets the center of each character mask
    3. Connects centers with lines (thickness = average character height)
    4. Adds circles at connection points for smooth transitions

    Parameters
    ----------
    line_char_indices : List[int] - indices of characters in this line
    target_masks      : List of (256, 256) uint8 masks in SAM decoder space
    original_size     : (H, W) size of original image
    input_size        : (h, w) after ResizeLongestSide, before SAM padding.
                        If None, estimated from original_size.
    size_multiplier   : Multiplier for average character height (for line thickness)

    Returns
    -------
    Line mask (H, W) uint8 in original image space with connected character centers
    """
    if not line_char_indices:
        return np.zeros(original_size, dtype=np.uint8)

    if input_size is None:
        H, W = original_size
        scale = 1024 / max(H, W)
        input_size = (int(H * scale), int(W * scale))

    oh, ow = original_size
    
    # Step 1: Convert all character masks to original image space
    char_masks_orig = []
    for char_idx in line_char_indices:
        if char_idx >= len(target_masks):
            continue
        m = target_masks[char_idx]
        if isinstance(m, np.ndarray) and m.dtype == object:
            m = m.item()
        m = np.asarray(m).squeeze()
        if m.ndim != 2 or m.size == 0:
            continue
        # Normalise to {0,1}
        if m.dtype != np.uint8:
            m = (m > 0.5).astype(np.uint8) if m.max() <= 1.0 else (m > 127).astype(np.uint8)
        elif m.max() > 1:
            m = (m > 127).astype(np.uint8)
        m_orig = _unpad_and_resize_mask(m, input_size, original_size)
        char_masks_orig.append(m_orig)
    
    if not char_masks_orig:
        return np.zeros(original_size, dtype=np.uint8)
    
    # Step 2: Start with union of all character masks
    result_mask = np.zeros((oh, ow), dtype=np.uint8)
    for mask in char_masks_orig:
        result_mask = cv2.bitwise_or(result_mask, mask)
    
    # Step 3: Calculate average character height
    avg_height = _calculate_average_mask_height_orig(char_masks_orig)
    avg_height *= size_multiplier
    
    # Step 4: Get centers of all character masks
    centers = []
    for mask in char_masks_orig:
        center = _get_mask_center_orig(mask)
        if center is not None:
            centers.append(center)
    
    if len(centers) < 2:
        # Only one character, return union of character masks
        return result_mask
    
    centers = np.array(centers)
    
    # Step 5: Sort centers by x-coordinate (left to right)
    sorted_indices = np.argsort(centers[:, 0])
    sorted_centers = centers[sorted_indices]
    
    # Step 6: Use average height as line thickness
    thickness = max(1, int(avg_height))
    
    # Step 7: Draw lines connecting consecutive centers on result mask
    for i in range(len(sorted_centers) - 1):
        pt1 = tuple(sorted_centers[i].astype(int))
        pt2 = tuple(sorted_centers[i + 1].astype(int))
        cv2.line(result_mask, pt1, pt2, 255, thickness)
    
    # Step 8: Add small circles at connection points for smoother transitions
    radius = max(1, thickness // 2)
    for center in sorted_centers:
        cv2.circle(result_mask, tuple(center.astype(int)), radius, 255, -1)
    
    return result_mask

def _mask_to_contour_points(mask: np.ndarray, max_points: int = 1000) -> np.ndarray:
    """Convert mask to contour point array (N,2)."""
    mask_uint = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    
    pts = contours[0].reshape(-1, 2).astype(np.float32)
    
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    
    return pts


def _hausdorff95(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    """Compute 95-percentile Hausdorff distance between two masks."""
    from scipy.spatial.distance import cdist
    
    pts_gt = _mask_to_contour_points(mask_gt)
    pts_pr = _mask_to_contour_points(mask_pred)
    
    if len(pts_gt) == 0 and len(pts_pr) == 0:
        return 0.0
    if len(pts_gt) == 0 or len(pts_pr) == 0:
        return float("inf")
    
    distances = cdist(pts_gt, pts_pr)
    min_gt = np.min(distances, axis=1)
    min_pr = np.min(distances, axis=0)
    
    hd95 = max(np.percentile(min_gt, 95), np.percentile(min_pr, 95))
    return float(hd95)


def _iou_score(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask_gt, mask_pred).sum()
    union = np.logical_or(mask_gt, mask_pred).sum()
    if union == 0:
        return 1.0
    return float(intersection / (union + 1e-6))


def _dice_score(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    """Compute Dice coefficient between two binary masks."""
    intersection = np.logical_and(mask_gt, mask_pred).sum()
    denom = mask_gt.sum() + mask_pred.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * intersection / (denom + 1e-6))


def compute_line_metrics(
    gt_line_masks: np.ndarray,
    pred_lines: List[List[int]],
    target_masks: List[np.ndarray],
    original_size: Tuple[int, int],
    input_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, float]:
    """
    Compute line-level metrics by matching GT and predicted lines.
    
    Parameters
    ----------
    gt_line_masks  : (num_gt_lines, H, W) binary masks
    pred_lines     : List of character index lists (predicted lines)
    target_masks   : Character masks in SAM 256x256 decoder space
    original_size  : (H, W) original image size
    input_size     : (h, w) after ResizeLongestSide, before SAM padding.
                     If None, estimated from original_size.
    
    Returns
    -------
    metrics : Dictionary with keys 'mean_iou', 'mean_dice', 'mean_hausdorff95', etc.
    """
    if gt_line_masks is None or len(gt_line_masks) == 0:
        return {}
    
    # Create predicted line masks
    pred_masks = []
    for line_chars in pred_lines:
        mask = create_line_mask_from_characters(line_chars, target_masks, original_size, input_size)
        pred_masks.append(mask)
    
    if not pred_masks:
        return {}
    
    # Match GT and predicted lines using Hungarian algorithm
    from scipy.optimize import linear_sum_assignment
    
    num_gt = len(gt_line_masks)
    num_pred = len(pred_masks)
    
    # Compute IoU matrix
    iou_matrix = np.zeros((num_gt, num_pred), dtype=np.float32)
    for i, gt_mask in enumerate(gt_line_masks):
        for j, pred_mask in enumerate(pred_masks):
            iou_matrix[i, j] = _iou_score(gt_mask, pred_mask)
    
    # Hungarian matching
    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)
    
    # Compute metrics for matched pairs
    ious = []
    dices = []
    hds = []
    
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        gt_mask = gt_line_masks[gt_idx]
        pred_mask = pred_masks[pred_idx]
        
        ious.append(_iou_score(gt_mask, pred_mask))
        dices.append(_dice_score(gt_mask, pred_mask))
        hds.append(_hausdorff95(gt_mask, pred_mask))
    
    # Handle unmatched
    unmatched_gt = num_gt - len(gt_indices)
    unmatched_pred = num_pred - len(pred_indices)
    
    metrics = {
        'mean_iou': float(np.mean(ious)) if ious else 0.0,
        'mean_dice': float(np.mean(dices)) if dices else 0.0,
        'mean_hausdorff95': float(np.mean(hds)) if hds else float('inf'),
        'num_gt_lines': num_gt,
        'num_pred_lines': num_pred,
        'num_matched': len(ious),
        'num_unmatched_gt': unmatched_gt,
        'num_unmatched_pred': unmatched_pred,
    }
    
    return metrics


def visualize_line_comparison(
    image_np: np.ndarray,
    gt_line_masks: Optional[np.ndarray],
    pred_lines: List[List[int]],
    target_masks: List[np.ndarray],
    original_size: Tuple[int, int],
    save_path: str,
    title: str = '',
    input_size: Optional[Tuple[int, int]] = None,
):
    """
    Visualize GT line masks (left) vs predicted line masks (right) side by side.
    
    Parameters
    ----------
    image_np       : Original image
    gt_line_masks  : (num_gt_lines, H, W) binary masks or None
    pred_lines     : List of character index lists
    target_masks   : Character masks in SAM 256x256 decoder space
    original_size  : (H, W) original image size
    save_path      : Path to save visualization
    title          : Title for the figure
    input_size     : (h, w) after ResizeLongestSide, before SAM padding.
                     If None, estimated from original_size.
    """
    print("Inside the visualization function...")
    H, W = image_np.shape[:2]
    
    # Create predicted line masks
    pred_masks = []
    for line_chars in pred_lines:
        print(f"  Creating mask for predicted line with characters: {line_chars}")
        mask = create_line_mask_from_characters(line_chars, target_masks, original_size, input_size)
        pred_masks.append(mask)
    
    # Generate colors for lines
    num_lines_gt = len(gt_line_masks) if gt_line_masks is not None else 0
    num_lines_pred = len(pred_masks)
    max_lines = max(num_lines_gt, num_lines_pred)
    
    colors = [
        (255, 0, 0), (0, 200, 0), (0, 0, 255),
        (255, 200, 0), (200, 0, 255), (0, 220, 220),
        (255, 128, 0), (128, 0, 255), (0, 128, 255),
        (255, 0, 128), (128, 255, 0), (0, 255, 128),
    ] * 3  # Repeat colors if needed
    
    # Create figure with two subplots
    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: GT line masks
    if gt_line_masks is not None and len(gt_line_masks) > 0:
        print(f"  Visualizing {num_lines_gt} GT lines...")
        overlay_gt = image_np.copy().astype(np.float32)
        for line_idx, gt_mask in enumerate(gt_line_masks):
            color = np.array(colors[line_idx % len(colors)], dtype=np.float32)
            overlay_region = np.zeros_like(image_np, dtype=np.float32)
            overlay_region[gt_mask > 0] = color
            overlay_gt = cv2.addWeighted(overlay_gt, 1.0, overlay_region, 0.3, 0)
        
        overlay_gt = np.clip(overlay_gt, 0, 255).astype(np.uint8)
        ax_gt.imshow(overlay_gt)
        ax_gt.set_title(f'GT Lines ({num_lines_gt})', fontsize=11, fontweight='bold')
    else:
        ax_gt.imshow(image_np)
        ax_gt.set_title('GT Lines (N/A)', fontsize=11, fontweight='bold')
    
    ax_gt.axis('off')
    
    # Right panel: Predicted line masks
    if pred_masks:
        overlay_pred = image_np.copy().astype(np.float32)
        for line_idx, pred_mask in enumerate(pred_masks):
            color = np.array(colors[line_idx % len(colors)], dtype=np.float32)
            overlay_region = np.zeros_like(image_np, dtype=np.float32)
            overlay_region[pred_mask > 0] = color
            overlay_pred = cv2.addWeighted(overlay_pred, 1.0, overlay_region, 0.3, 0)
        
        overlay_pred = np.clip(overlay_pred, 0, 255).astype(np.uint8)
        ax_pred.imshow(overlay_pred)
        ax_pred.set_title(f'Predicted Lines ({num_lines_pred})', fontsize=11, fontweight='bold')
    else:
        ax_pred.imshow(image_np)
        ax_pred.set_title('Predicted Lines (0)', fontsize=11, fontweight='bold')
    
    ax_pred.axis('off')
    
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"  Visualization saved to: {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SAM inverse-transform helpers (self-contained copy so this module can run
# standalone without importing from inference_gnn)
# ─────────────────────────────────────────────────────────────────────────────

def _unpad_and_resize_mask(mask, input_size, original_size):
    sam_size = 1024
    m255 = (mask * 255).astype(np.uint8)
    m1024 = cv2.resize(m255, (sam_size, sam_size), interpolation=cv2.INTER_NEAREST)
    ih = min(int(input_size[0]), sam_size)
    iw = min(int(input_size[1]), sam_size)
    cropped = m1024[:ih, :iw]
    oh, ow = original_size
    orig = cv2.resize(cropped, (ow, oh), interpolation=cv2.INTER_NEAREST)
    return (orig > 127).astype(np.uint8)


def _untransform_centroid(cx, cy, input_size, original_size):
    oh, ow = original_size
    ih, iw = input_size
    scale  = max(ih / oh, iw / ow)
    return cx / scale, cy / scale


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    (220, 20, 60), (30, 144, 255), (50, 205, 50), (255, 165, 0),
    (138, 43, 226), (0, 206, 209), (255, 20, 147), (255, 215, 0),
    (0, 128, 128), (255, 127, 80), (106, 90, 205), (60, 179, 113),
]


def visualize_two_hop_graph(
    graph:       TwoHopGraph,
    image_np:    np.ndarray,
    input_size:  Tuple[int, int],
    save_path:   str,
    labels:      Optional[np.ndarray] = None,
    confidence:  Optional[np.ndarray] = None,
    title:       str                  = '',
    show_hop2:   bool                 = True,
):
    """
    Render the two-hop graph over the image.

    Layout (4 panels in a 2×2 grid)
    --------------------------------
    Top-left  : 1-hop edges only, coloured by similarity, nodes by GT line
    Top-right : 2-hop edges only (if show_hop2=True), else same as left
    Bottom-left : classified edges (green=same line, red=different)
                  only drawn if labels is not None
    Bottom-right: degree distribution histogram + stats table

    Edge colour scheme
    ------------------
    • Unclassified: YlOrRd cmap, opacity ∝ similarity
    • Classified  : green (label=1), red (label=0), width ∝ confidence
    """
    H, W = image_np.shape[:2]
    original_size = (H, W)

    # Map centroids from SAM 1024-space to original image space
    cxs, cys = [], []
    for (cx, cy) in graph.centroids:
        ox, oy = _untransform_centroid(cx, cy, input_size, original_size)
        cxs.append(ox); cys.append(oy)
    cxs = np.array(cxs); cys = np.array(cys)

    # Node colours by GT line
    ln = graph.line_numbers
    if ln is not None:
        uniq = sorted(set(int(l) for l in ln if l >= 0))
        c_map = {l: _PALETTE[i % len(_PALETTE)] for i, l in enumerate(uniq)}
        node_col = [
            tuple(c / 255 for c in c_map.get(int(ln[i]), (128, 128, 128)))
            for i in range(graph.num_nodes)
        ]
    else:
        node_col = [(0.2, 0.6, 1.0)] * graph.num_nodes

    n_panels = 4 if labels is not None else 3
    fig = plt.figure(figsize=(22, 14))
    gs  = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.15)
    ax_1hop  = fig.add_subplot(gs[0, 0])
    ax_2hop  = fig.add_subplot(gs[0, 1])
    ax_class = fig.add_subplot(gs[1, 0])
    ax_stats = fig.add_subplot(gs[1, 1])

    edge_cmap = cm.RdYlGn

    def _draw_nodes(ax):
        for i in range(graph.num_nodes):
            ax.plot(cxs[i], cys[i], 'o',
                    color=node_col[i], markersize=7,
                    markeredgecolor='white', markeredgewidth=1.0, zorder=4)
            ax.text(cxs[i]+3, cys[i]-3, str(i),
                    fontsize=5, color='white', zorder=5,
                    bbox=dict(fc=node_col[i], ec='none',
                              boxstyle='round,pad=0.1', alpha=0.7))

    def _draw_edges_by_sim(ax, mask):
        for idx in np.where(mask)[0]:
            i, j = int(graph.edges[idx, 0]), int(graph.edges[idx, 1])
            sim  = float(graph.edge_sim[idx])
            col  = edge_cmap(0.3 + 0.7 * sim)
            ax.plot([cxs[i], cxs[j]], [cys[i], cys[j]],
                    color=col, alpha=max(0.15, sim), linewidth=1.0, zorder=2)

    # Panel 1 – 1-hop
    ax_1hop.imshow(image_np, alpha=0.55, zorder=0)
    _draw_edges_by_sim(ax_1hop, graph.edge_hop == 1)
    _draw_nodes(ax_1hop)
    ax_1hop.set_title(f'1-hop edges  (k={graph.edges.shape[0]})', fontsize=11, fontweight='bold')
    ax_1hop.axis('off')

    # Panel 2 – 2-hop
    ax_2hop.imshow(image_np, alpha=0.55, zorder=0)
    if show_hop2:
        _draw_edges_by_sim(ax_2hop, graph.edge_hop == 2)
    _draw_edges_by_sim(ax_2hop, graph.edge_hop == 1)   # 1-hop on top, thinner
    _draw_nodes(ax_2hop)
    n1 = int((graph.edge_hop == 1).sum())
    n2 = int((graph.edge_hop == 2).sum())
    ax_2hop.set_title(f'1-hop ({n1}) + 2-hop ({n2})  total={n1+n2}', fontsize=11, fontweight='bold')
    ax_2hop.axis('off')

    # Panel 3 – classified edges
    ax_class.imshow(image_np, alpha=0.55, zorder=0)
    # if labels is not None:
    #     conf = confidence if confidence is not None else np.ones(len(labels))
    #     for idx, (i, j) in enumerate(graph.edges):
    #         i, j = int(i), int(j)
    #         lw   = 1.0 + 2.0 * float(conf[idx])
    #         col  = (0.1, 0.8, 0.1) if labels[idx] == 1 else (0.9, 0.1, 0.1)
    #         ax_class.plot([cxs[i], cxs[j]], [cys[i], cys[j]],
    #                       color=col, alpha=0.7, linewidth=lw, zorder=2)
    if labels is not None:
        conf = confidence if confidence is not None else np.ones(len(labels))
        for idx, (i, j) in enumerate(graph.edges):
            if labels[idx] != 1:  # Skip non-same-line edges
                continue
            i, j = int(i), int(j)
            lw   = 1.0 + 2.0 * float(conf[idx])
            col  = (0.1, 0.8, 0.1)  # green only
            ax_class.plot([cxs[i], cxs[j]], [cys[i], cys[j]],
                          color=col, alpha=0.7, linewidth=lw, zorder=2)
    _draw_nodes(ax_class)
    ax_class.set_title('Classified edges  (green=same line  red=diff)', fontsize=11, fontweight='bold')
    ax_class.axis('off')

    # Panel 4 – stats
    ax_stats.axis('off')
    degrees = np.array([len(graph.adj[i]) for i in range(graph.num_nodes)])

    # Mini histogram inline
    ax_hist = ax_stats.inset_axes([0.0, 0.45, 1.0, 0.5])
    bins = np.arange(0, degrees.max() + 2) - 0.5
    ax_hist.hist(degrees, bins=bins, color='steelblue', edgecolor='white', rwidth=0.8)
    ax_hist.axvline(degrees.mean(), color='tomato', ls='--', lw=1.5,
                    label=f'mean={degrees.mean():.1f}')
    ax_hist.set_xlabel('Degree'); ax_hist.set_ylabel('Count')
    ax_hist.set_title('Degree distribution', fontsize=9)
    ax_hist.legend(fontsize=7)

    # Text stats
    lines_info = []
    lines_info.append(f'Image : {title or graph.image_id}')
    lines_info.append(f'Nodes : {graph.num_nodes}')
    lines_info.append(f'Edges : {len(graph.edges)}  (1-hop={n1}  2-hop={n2})')
    lines_info.append(f'Degree — mean:{degrees.mean():.1f}  max:{degrees.max()}  min:{degrees.min()}')
    if graph.edge_gt is not None:
        pos = int(graph.edge_gt.sum())
        lines_info.append(f'GT pos edges : {pos}/{len(graph.edges)}  ({100*pos/max(len(graph.edges),1):.1f}%)')
    if labels is not None:
        pred_pos = int(labels.sum())
        lines_info.append(f'Pred pos edges : {pred_pos}/{len(labels)}')
        if graph.edge_gt is not None:
            tp = int(((labels == 1) & (graph.edge_gt == 1)).sum())
            fp = int(((labels == 1) & (graph.edge_gt == 0)).sum())
            fn = int(((labels == 0) & (graph.edge_gt == 1)).sum())
            pr = tp / (tp + fp + 1e-6); rc = tp / (tp + fn + 1e-6)
            f1 = 2*pr*rc/(pr+rc+1e-6)
            lines_info.append(f'TP={tp}  FP={fp}  FN={fn}')
            lines_info.append(f'P={pr:.3f}  R={rc:.3f}  F1={f1:.3f}')

    ax_stats.text(0.02, 0.42, '\n'.join(lines_info),
                  transform=ax_stats.transAxes,
                  fontsize=8, va='top', fontfamily='monospace',
                  bbox=dict(fc='#f5f5f5', ec='#cccccc', boxstyle='round'))

    # Colourbar
    sm = cm.ScalarMappable(cmap=edge_cmap,
                           norm=matplotlib.colors.Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.01, 0.55, 0.012, 0.3])
    fig.colorbar(sm, cax=cbar_ax, label='Similarity')
    cbar_ax.yaxis.set_label_position('left')
    cbar_ax.yaxis.label.set_size(7)

    fig.suptitle(f'Two-hop graph — {title or graph.image_id}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("Saved graph vis → %s", save_path)


def visualize_k_sweep(
    npz_path:         str,
    image_np:         np.ndarray,
    input_size:       Tuple[int, int],
    k_values:         List[int],
    similarity_method: str,
    output_dir:       str,
    image_id:         str = '',
):
    """
    Build and visualise graphs for multiple values of k side-by-side.
    Produces one PNG per k value, and one summary comparing edge counts.
    """
    from PIL import Image as _PIL
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)
    centroids   = data['centroids']
    t_masks     = list(data['target_masks'])
    l_masks     = list(data['left_neighbor_masks'])
    r_masks     = list(data['right_neighbor_masks'])
    line_nums   = data['line_numbers'] if 'line_numbers' in data else None

    summary = []
    for k in k_values:
        g = build_two_hop_graph(
            centroids, t_masks, l_masks, r_masks,
            line_nums, k=k, similarity_method=similarity_method,
            image_id=image_id,
        )
        save_path = os.path.join(output_dir, f'{image_id}_k{k:03d}.png')
        visualize_two_hop_graph(
            graph=g, image_np=image_np,
            input_size=input_size,
            save_path=save_path,
            title=f'k={k}',
        )
        n1 = int((g.edge_hop == 1).sum())
        n2 = int((g.edge_hop == 2).sum())
        summary.append({'k': k, '1hop': n1, '2hop': n2, 'total': n1+n2,
                        'nodes': g.num_nodes})
        log.info("  k=%3d  nodes=%d  1-hop=%d  2-hop=%d  total=%d",
                 k, g.num_nodes, n1, n2, n1+n2)

    # Summary chart
    ks      = [s['k'] for s in summary]
    totals  = [s['total'] for s in summary]
    hop1s   = [s['1hop']  for s in summary]
    hop2s   = [s['2hop']  for s in summary]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, totals, 'k-o', label='total')
    ax.plot(ks, hop1s,  'b--s', label='1-hop')
    ax.plot(ks, hop2s,  'r--^', label='2-hop')
    ax.set_xlabel('k'); ax.set_ylabel('Edge count')
    ax.set_title(f'Edge count vs k  ({image_id})')
    ax.legend(); ax.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{image_id}_k_sweep_summary.png'), dpi=120)
    plt.close(fig)

    # Save summary JSON
    with open(os.path.join(output_dir, f'{image_id}_k_sweep.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    log.info("k-sweep summary → %s", output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Batch evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_graphs(
    graphs:     List[TwoHopGraph],
    method:     str,                 # 'manual' | 'learned'
    output_dir: str,
    threshold:  float                         = 0.4,
    model:      Optional[TwoHopEdgeClassifier] = None,
    device:     str                           = 'cpu',
    image_root: str                           = '',
    input_sizes: Optional[Dict[str, Tuple]] = None,
) -> Dict:
    """
    Run edge classification on all graphs, compute P/R/F1 per image and
    aggregate, save per-image JSON and vis.

    Returns aggregate metrics dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    per_image = []

    for g in graphs:
        if method == 'manual':
            labels, conf = classify_edges_manual(g, threshold=threshold)
        elif method == 'learned':
            assert model is not None, "Need a trained model for method='learned'"
            labels, conf = classify_edges_learned(g, model, device=device,
                                                  threshold=0.5)
        else:
            raise ValueError(f"Unknown method: {method!r}")

        m = _edge_metrics(g, labels)
        m['image_id'] = g.image_id
        per_image.append(m)

        # Visualise
        img_path = os.path.join(image_root, g.image_id)
        if os.path.exists(img_path):
            image_np = np.array(Image.open(img_path).convert('RGB'))
            H, W = image_np.shape[:2]
            if input_sizes and g.image_id in input_sizes:
                isize = input_sizes[g.image_id]
            else:
                sc    = 1024 / max(H, W)
                isize = (int(H * sc), int(W * sc))
            stem = g.image_id.replace('.png','').replace('.jpg','')
            visualize_two_hop_graph(
                graph=g, image_np=image_np,
                input_size=isize,
                save_path=os.path.join(vis_dir, f'{stem}_{method}.png'),
                labels=labels, confidence=conf,
                title=g.image_id,
            )
            # Also create line comparison visualization
            print(f"Evaluating line metrics for {g.image_id}...")
            print(f"len(g.gt_line_masks)={len(g.gt_line_masks) if g.gt_line_masks is not None else 'None'}  len(pred_lines)={len(edges_to_lines(g, labels, num_nodes=g.num_nodes))}")
            if g.gt_line_masks is not None and len(g.gt_line_masks) > 0:
                print(f"Creating line comparison vis for {g.image_id}...")
                # Extract predicted lines from classified edges
                pred_lines = edges_to_lines(g, labels, num_nodes=g.num_nodes)
                print("Now going to visualize line comparison...")
                
                visualize_line_comparison(
                    image_np=image_np,
                    gt_line_masks=np.array(g.gt_line_masks),
                    pred_lines=pred_lines,
                    target_masks=g.target_masks,
                    original_size=(H, W),
                    save_path=os.path.join(vis_dir, f'{stem}_{method}_lines.png'),
                    title=f'{g.image_id} - {method}',
                    input_size=isize,
                )
            else:
                print(f"No GT line masks for {g.image_id}, skipping line comparison vis.")
                log.debug("No GT line masks available for %s", g.image_id)

        # Need to compute line-level metrics here and log them, since they won't be in the edge-level JSON
        if g.gt_line_masks is not None and len(g.gt_line_masks) > 0:
            pred_lines = edges_to_lines(g, labels, num_nodes=g.num_nodes)
            line_metrics = compute_line_metrics(
                gt_line_masks=np.array(g.gt_line_masks),
                pred_lines=pred_lines,
                target_masks=g.target_masks,
                original_size=(H, W),
                input_size=isize,
            )
            m.update(line_metrics)
            log.info("  Line metrics: IoU=%.3f  Dice=%.3f  HD95=%.1f",
                     line_metrics.get('mean_iou', 0.0),
                     line_metrics.get('mean_dice', 0.0),
                     line_metrics.get('mean_hausdorff95', float('inf')))

    # Aggregate
    agg = _aggregate_edge_metrics(per_image)
    agg['method'] = method
    agg['threshold'] = threshold
    agg['n_images'] = len(per_image)

    log.info("=== %s method  threshold=%.3f ===", method, threshold)
    log.info("  P=%.4f  R=%.4f  F1=%.4f  Acc=%.4f",
             agg['precision'], agg['recall'], agg['f1'], agg['accuracy'])

    with open(os.path.join(output_dir, f'results_{method}.json'), 'w') as f:
        json.dump({'summary': agg, 'per_image': per_image}, f, indent=2)

    return agg


def _edge_metrics(g: TwoHopGraph, labels: np.ndarray) -> Dict:
    if g.edge_gt is None or len(labels) == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0}
    gt = g.edge_gt.astype(int)
    pr = labels.astype(int)
    tp = int(((pr == 1) & (gt == 1)).sum())
    fp = int(((pr == 1) & (gt == 0)).sum())
    fn = int(((pr == 0) & (gt == 1)).sum())
    tn = int(((pr == 0) & (gt == 0)).sum())
    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    f1   = 2*prec*rec/(prec+rec+1e-6)
    acc  = (tp + tn) / (tp + fp + fn + tn + 1e-6)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=float(prec), recall=float(rec),
                f1=float(f1), accuracy=float(acc))


def _aggregate_edge_metrics(per_image: List[Dict]) -> Dict:
    keys = ['precision', 'recall', 'f1', 'accuracy']
    agg  = {k: float(np.mean([m[k] for m in per_image])) for k in keys}
    agg['total_tp'] = sum(m['tp'] for m in per_image)
    agg['total_fp'] = sum(m['fp'] for m in per_image)
    agg['total_fn'] = sum(m['fn'] for m in per_image)
    agg['total_tn'] = sum(m['tn'] for m in per_image)
    tp = agg['total_tp']; fp = agg['total_fp']
    fn = agg['total_fn']
    agg['global_precision'] = tp / (tp + fp + 1e-6)
    agg['global_recall']    = tp / (tp + fn + 1e-6)
    p, r = agg['global_precision'], agg['global_recall']
    agg['global_f1'] = 2*p*r/(p+r+1e-6)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Batch graph loading
# ─────────────────────────────────────────────────────────────────────────────

def load_index_records(index_file: str) -> Dict[str, dict]:
    """
    Load full records from index JSON file, including line numbers and masks.
    Skips records with missing or invalid line_numbers.
    """
    records_map = {}
    try:
        with open(index_file, 'r') as f:
            records = json.load(f)
        for record in records:
            if 'image_path' not in record or 'line_numbers' not in record:
                continue
            line_numbers = record['line_numbers']
            if (line_numbers is None or
                not isinstance(line_numbers, list) or
                any(x is None for x in line_numbers)):
                log.warning(f"Skipping {record.get('image_path', '?')}: invalid line_numbers")
                continue
            image_path = record['image_path']
            image_id = os.path.basename(image_path)
            print(f"[DEBUG] index image_path='{image_path}'  → image_id='{image_id}'  line_masks_npz='{record.get('line_masks_npz')}'")
            line_nums = np.array(line_numbers, dtype=np.int32)
            records_map[image_id] = {
                'image_path': image_path,
                'line_numbers': line_nums,
                'line_masks_npz': record.get('line_masks_npz', None),
            }
        log.info("Loaded records for %d images from %s", len(records_map), index_file)
    except Exception as e:
        log.warning("Could not load index file %s: %s", index_file, e)
    return records_map

# def load_gt_line_masks_from_npz(npz_path: str) -> Optional[List[np.ndarray]]:
#     """
#     Load GT line masks from precomputed NPZ file.
    
#     Parameters
#     ----------
#     npz_path : str
#         Path to NPZ file containing line masks
        
#     Returns
#     -------
#     masks : List[np.ndarray] or None
#         List of (H, W) boolean arrays, one per GT line.
#         Empty masks are filtered out.
#         Returns None if file not found or cannot be loaded.
#     """
#     if not npz_path or not os.path.exists(npz_path):
#         return None
    
#     try:
#         data = np.load(npz_path, allow_pickle=True)
#         if 'masks' in data:
#             arr = data['masks']  # (L, H, W)
#             masks = [
#                 (arr[i] > 0).astype(bool)
#                 for i in range(arr.shape[0])
#                 if arr[i].sum() > 0
#             ]
#             return masks if masks else None
#         elif 'line_masks' in data:
#             # Fallback for older format
#             arr = data['line_masks']
#             masks = [m.astype(bool) for m in arr if m.sum() > 0]
#             return masks if masks else None
#     except Exception as e:
#         log.debug("Could not load line masks from %s: %s", npz_path, e)
    
#     return None

def load_gt_line_masks(record: dict):
    """
    Load GT line masks from the record's 'line_masks_npz' key.
    Returns list of (H, W) bool arrays, or None if unavailable.
    """
    key = 'line_masks_npz'
    if key not in record or record[key] is None or not os.path.exists(record[key]):
        print(f"GT line masks NPZ not found for {record.get('image_path', '?')}: {record.get(key, 'N/A')}")
        log.warning("GT line masks NPZ not found for %s: %s",
                    record.get('image_path', '?'), record.get(key, 'N/A'))
        return None
    try:
        data = np.load(record[key])
        arr = data['masks']          # (L, H, W)
        masks = [
            (arr[i] > 0).astype(bool)
            for i in range(arr.shape[0])
            if arr[i].sum() > 0
        ]
        return masks if masks else None
    except Exception as e:
        print(f"Could not load GT line masks for {record.get('image_path', '?')}: {e}")
        log.warning(f"Could not load GT line masks: {e}")
        return None

def load_graphs_from_dir(
    precomputed_dir: str,
    k:               int   = 10,
    similarity_method: str = 'iou',
    max_images:      Optional[int] = None,
    index_file:      Optional[str] = None,
) -> List[TwoHopGraph]:
    """
    Load all .npz files in precomputed_dir and build TwoHopGraphs.
    
    Parameters
    ----------
    precomputed_dir : str
        Directory containing precomputed .npz files
    k : int
        Number of nearest neighbors for 1-hop graph
    similarity_method : str
        Similarity metric: 'iou', 'ios', or 'dice'
    max_images : Optional[int]
        Maximum number of images to load
    index_file : Optional[str]
        Path to JSON index file with line number annotations and GT masks.
        If provided, line numbers and masks from this file will be loaded.
        
    Returns
    -------
    graphs : List[TwoHopGraph]
        List of loaded and constructed graphs with GT masks attached
    """
    # Load index records if provided
    index_records = {}
    if index_file:
        index_records = load_index_records(index_file)
        if index_records:
            sample_keys = list(index_records.keys())[:3]
            print(f"[DEBUG] index_records has {len(index_records)} entries. Sample keys: {sample_keys}")
        else:
            print("[DEBUG] index_records is EMPTY after load_index_records!")
    
    npz_files = sorted(f for f in os.listdir(precomputed_dir) if f.endswith('.npz'))
    print("="*80)
    print(f"Found {len(npz_files)} NPZ files in {precomputed_dir}")
    print("="*80)

    if max_images:
        npz_files = npz_files[:max_images]
    graphs = []
    for fname in npz_files:
        print(f"[DEBUG] Processing NPZ file: {fname}")
        path = os.path.join(precomputed_dir, fname)
        try:
            data = np.load(path, allow_pickle=True)
            N = len(data['centroids'])
            if N < 2:
                continue
            
            image_id = fname[:-4]
            print(f"[DEBUG] NPZ image_id='{image_id}'  in index_records={image_id in index_records}")
            
            # Use line numbers from index if available, otherwise from NPZ
            line_numbers = None
            if image_id in index_records:
                line_numbers = index_records[image_id]['line_numbers']
                log.debug("Using line numbers from index for %s", image_id)
            elif 'line_numbers' in data:
                line_numbers = data['line_numbers']
                log.debug("Using line numbers from NPZ for %s", image_id)
            
            # Load GT line masks from index if available
            gt_line_masks = None
            if image_id in index_records:
                line_masks_npz = index_records[image_id]['line_masks_npz']
                print("="*40)
                print(line_masks_npz)
                gt_line_masks = load_gt_line_masks(index_records[image_id])
                if gt_line_masks:
                    print(f"Loaded {len(gt_line_masks)} GT line masks for {image_id} from index")
                    log.debug("Loaded %d GT line masks from %s", len(gt_line_masks), image_id)
                else:
                    print(f"No valid GT line masks found for {image_id} in index")
                    log.debug("No GT line masks found for %s in index", image_id)
            
            g = build_two_hop_graph(
                centroids      = data['centroids'],
                target_masks   = list(data['target_masks']),
                left_masks     = list(data['left_neighbor_masks']),
                right_masks    = list(data['right_neighbor_masks']),
                gt_line_masks = gt_line_masks,
                line_numbers   = line_numbers,
                k              = k,
                similarity_method = similarity_method,
                image_id       = image_id,
            )
            graphs.append(g)
        except Exception as e:
            log.warning("Skipping %s: %s", fname, e)
    log.info("Loaded %d graphs from %s", len(graphs), precomputed_dir)
    return graphs


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Two-hop graph edge classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--precomputed_dir',  required=True)
    p.add_argument('--image_root',       required=True)
    p.add_argument('--index_file',       default=None,
                   help='Optional: JSON index file with line number annotations')
    p.add_argument('--output_dir',       default='./graph_lines_two_hop_results')
    p.add_argument('--k',                type=int,   default=5)
    p.add_argument('--similarity_method', default='dice',
                   choices=['iou', 'ios', 'dice'])
    p.add_argument('--method',           default='manual',
                   choices=['manual', 'learned', 'both'])
    p.add_argument('--threshold',        type=float, default=0.4,
                   help='Manual threshold t')
    p.add_argument('--vis_k_values',     type=int,   nargs='+',
                   default=None,
                   help='If set, also produce k-sweep visualisations for '
                        'these k values on the first --max_vis_sweep images')
    p.add_argument('--max_vis_sweep',    type=int,   default=5,
                   help='Number of images to run the k-sweep on')
    p.add_argument('--max_images',       type=int,   default=None)
    p.add_argument('--log_level',        default='INFO',
                   choices=['DEBUG','INFO','WARNING'])
    return p.parse_args()


def main():
    args = _parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load val graphs ───────────────────────────────────────────────────────
    log.info("Loading graphs from %s", args.precomputed_dir)
    if args.index_file:
        log.info("Using index file for line annotations: %s", args.index_file)
    val_graphs = load_graphs_from_dir(
        args.precomputed_dir, k=args.k,
        similarity_method=args.similarity_method,
        max_images=args.max_images,
        index_file=args.index_file,
    )

    # ── k-sweep visualisation (optional) ─────────────────────────────────────
    if args.vis_k_values:
        log.info("Running k-sweep for k in %s", args.vis_k_values)
        sweep_dir = os.path.join(args.output_dir, 'k_sweep')
        npz_files = sorted(
            f for f in os.listdir(args.precomputed_dir) if f.endswith('.npz')
        )[:args.max_vis_sweep]

        for fname in npz_files:
            npz_path  = os.path.join(args.precomputed_dir, fname)
            image_id  = fname[:-4]
            img_path  = os.path.join(args.image_root, image_id)
            if not os.path.exists(img_path):
                log.warning("Image not found for k-sweep: %s", img_path)
                continue
            image_np = np.array(Image.open(img_path).convert('RGB'))
            data     = np.load(npz_path, allow_pickle=True)
            if 'input_size' in data:
                isize = tuple(int(x) for x in data['input_size'])
            else:
                H, W = image_np.shape[:2]
                sc   = 1024 / max(H, W)
                isize = (int(H * sc), int(W * sc))

            visualize_k_sweep(
                npz_path=npz_path,
                image_np=image_np,
                input_size=isize,
                k_values=args.vis_k_values,
                similarity_method=args.similarity_method,
                output_dir=os.path.join(sweep_dir, image_id),
                image_id=image_id,
            )

    # ── Manual method ─────────────────────────────────────────────────────────
    log.info("=== Manual threshold  t=%.3f ===", args.threshold)
    # Build input_size lookup
    input_sizes = {}
    for g in val_graphs:
        fn = g.image_id + '.npz'
        path = os.path.join(args.precomputed_dir, fn)
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            if 'input_size' in d:
                input_sizes[g.image_id] = tuple(int(x) for x in d['input_size'])

    evaluate_graphs(
        graphs=val_graphs,
        method='manual',
        output_dir=os.path.join(args.output_dir, 'manual'),
        threshold=args.threshold,
        image_root=args.image_root,
        input_sizes=input_sizes,
    )

    log.info("Done.  Results in %s", args.output_dir)


if __name__ == '__main__':
    main()