"""
Evaluation script for SAM model - generates predictions and visualizations.
"""
import torch
import argparse
from tqdm import tqdm
from segment_anything import sam_model_registry
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
from torch.nn import functional as F
import random
import skimage.measure
from scipy.optimize import linear_sum_assignment

from data_loader import SimplePreprocessedSamDataset
from utils import dice_loss

def mask_nms_ios(masks, scores, ios_threshold=0.8):
    """
    NMS using Intersection over Smaller mask.
    If a smaller mask is mostly contained in a larger one, suppress it.
    """
    masks = np.array(masks)
    scores = np.array(scores)
    order = scores.argsort()[::-1]
    keep = []
    suppressed = np.zeros(len(masks), dtype=bool)

    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        mask_i = masks[i]
        area_i = mask_i.sum()
        
        for j in order:
            if i == j or suppressed[j]:
                continue
            mask_j = masks[j]
            area_j = mask_j.sum()
            
            intersection = np.logical_and(mask_i, mask_j).sum()
            smaller_area = min(area_i, area_j)
            
            # If most of the smaller mask is inside the other, suppress the smaller one
            ios = intersection / (smaller_area + 1e-6)
            if ios > ios_threshold:
                # Suppress the one with lower confidence (which is j, since we iterate by score)
                suppressed[j] = True
    return keep

# ============================================================
# Metric helpers (adapted from metrics.py)
# ============================================================

def _dice_coef(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    intersection = np.logical_and(mask_gt, mask_pred).sum()
    denom = mask_gt.sum() + mask_pred.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * intersection / (denom + 1e-6))


def _iou_score(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    intersection = np.logical_and(mask_gt, mask_pred).sum()
    union = np.logical_or(mask_gt, mask_pred).sum()
    if union == 0:
        return 1.0
    return float(intersection / (union + 1e-6))


def _compute_iou_matrix(gt_masks, pred_masks):
    if len(gt_masks) == 0 or len(pred_masks) == 0:
        return np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    iou_mat = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    for i, gt in enumerate(gt_masks):
        for j, pr in enumerate(pred_masks):
            iou_mat[i, j] = _iou_score(gt, pr)
    return iou_mat


def _hungarian_match(iou_matrix, iou_threshold=0.0):
    matches = []
    if iou_matrix.size == 0:
        return matches
    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        iou = iou_matrix[gt_idx, pred_idx]
        if iou >= iou_threshold:
            matches.append((gt_idx, pred_idx, float(iou)))
    return matches


def calculate_metrics(pred_masks, gt_masks):
    """
    Full metrics: fg_iou, fg_dice, mean_iou, mean_dice.
    pred_masks / gt_masks: list of boolean (H, W) numpy arrays.
    """
    pred_list = [m.astype(bool) for m in pred_masks]
    gt_list   = [m.astype(bool) for m in gt_masks]

    if len(pred_list) == 0 and len(gt_list) == 0:
        return {"fg_iou": 1.0, "fg_dice": 1.0, "mean_iou": 1.0, "mean_dice": 1.0}

    ref_shape = pred_list[0].shape if pred_list else gt_list[0].shape
    img_h, img_w = ref_shape

    # Foreground (combined) metrics
    fg_pred = np.zeros((img_h, img_w), dtype=bool)
    for m in pred_list:
        fg_pred |= m
    fg_gt = np.zeros((img_h, img_w), dtype=bool)
    for m in gt_list:
        fg_gt |= m
    fg_iou  = _iou_score(fg_gt, fg_pred)
    fg_dice = _dice_coef(fg_gt, fg_pred)

    # Instance matching
    iou_mat = _compute_iou_matrix(gt_list, pred_list)
    matches = _hungarian_match(iou_mat, iou_threshold=0.0)

    matched_ious  = [iou for _, _, iou in matches]
    matched_dices = [_dice_coef(gt_list[gi], pred_list[pi]) for gi, pi, _ in matches]
    mean_iou  = float(np.mean(matched_ious))  if matched_ious  else 0.0
    mean_dice = float(np.mean(matched_dices)) if matched_dices else 0.0

    return {
        "fg_iou":   round(fg_iou,  4),
        "fg_dice":  round(fg_dice, 4),
        "mean_iou":  round(mean_iou,  4),
        "mean_dice": round(mean_dice, 4),
    }


# ============================================================

def create_overlay_image(image_np, masks_list, points_list, bbox_list=None):
    """Creates overlay image with masks, points, and bboxes."""
    overlay_img = Image.fromarray(image_np).convert("RGBA")
    overlay = Image.new('RGBA', overlay_img.size, (0, 0, 0, 0))
    
    # Add masks
    for mask_np in masks_list:
        if mask_np.sum() == 0:
            continue
        color = (random.randint(50, 255), random.randint(50, 255), 
                random.randint(50, 255), 128)
        mask_rgba = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
        mask_rgba[mask_np > 0] = color
        overlay = Image.alpha_composite(overlay, Image.fromarray(mask_rgba, 'RGBA'))
    
    result = Image.alpha_composite(overlay_img, overlay)
    draw = ImageDraw.Draw(result)
    
    # Add points
    for point in points_list:
        x, y = int(point[0]), int(point[1])
        draw.ellipse((x-3, y-3, x+3, y+3), fill='white', outline='black')
    
    # Add bboxes
    if bbox_list is not None:
        colors = ["lime", "cyan", "yellow", "magenta", "orange"]
        for i, bbox in enumerate(bbox_list):
            if bbox.sum() > 0:
                x_min, y_min, x_max, y_max = bbox
                color = colors[i % len(colors)]
                draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
    
    return result

def save_feature_map(map_tensor, save_path, title):
    arr = map_tensor.cpu().detach().numpy()
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    plt.figure(figsize=(4, 4))
    plt.imshow(arr_norm, cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def get_random_point_in_bbox(bbox, image_size=1024):
    """
    Generate a random point inside a bounding box.
    Args:
        bbox: [x_min, y_min, x_max, y_max] format
        image_size: Size to ensure point is within bounds
    Returns:
        torch.Tensor of shape (2,) with random point [x, y]
    """
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(image_size - 1, int(x_max))
    y_max = min(image_size - 1, int(y_max))
    
    # Add small padding to avoid edges
    x_min = min(x_min + 1, x_max - 1)
    y_min = min(y_min + 1, y_max - 1)
    
    random_x = np.random.uniform(x_min, x_max)
    random_y = np.random.uniform(y_min, y_max)
    return torch.tensor([random_x, random_y], dtype=torch.float32)


def evaluate(sam, eval_loader, device, output_dir,
             per_char_visualization=False, pred_iou_threshold=0.0,
             use_random_bbox_point=False):
    """Evaluate model and save visualizations and predicted masks."""
    sam.eval()
    os.makedirs(output_dir, exist_ok=True)
    feature_dir = os.path.join(output_dir, "feature_maps")
    os.makedirs(feature_dir, exist_ok=True)

    # Accumulators for aggregation across images
    # Each entry: dict with keys fg_iou, fg_dice, mean_iou, mean_dice
    per_image_results = {          # strategy -> list of per-image metric dicts
        "dice_pre_nms":    [],
        "prediou_pre_nms": [],
        "dice_post_nms":   [],
        "prediou_post_nms": [],
    }
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            images = batch['image'].to(device)
            gt_masks_256 = batch['gt_masks']
            gt_points = batch['gt_points']
            bbox_prompts = batch.get('bbox_prompts', None)
            
            image_embeddings = sam.image_encoder(images)
            print(f"image_embeddings shape: {image_embeddings.shape}")
            summary_map_mean = image_embeddings[0].mean(dim=0)  # shape: [64, 64]
            summary_map_max = image_embeddings[0].max(dim=0).values  # shape: [64, 64]
            summary_map_sum = image_embeddings[0].sum(dim=0)  # shape: [64, 64]

            # Save the maps for this image
            save_feature_map(summary_map_mean, os.path.join(feature_dir, f"img_{idx:04d}_mean.png"), "Mean")
            save_feature_map(summary_map_max, os.path.join(feature_dir, f"img_{idx:04d}_max.png"), "Max")
            save_feature_map(summary_map_sum, os.path.join(feature_dir, f"img_{idx:04d}_sum.png"), "Sum")
            
            for i in range(len(images)):
                image_tensor = images[i]
                points_i = gt_points[i]
                masks_gt_256_i = gt_masks_256[i].to(device)
                bbox_prompts_i = bbox_prompts[i] if bbox_prompts else None

                pred_masks_list = []        # best-by-dice
                pred_masks_low_list = []    # low-res tensors stored as numpy
                target_masks_1024_list = []

                pred_masks_prediou_list = []   # best-by-pred-iou (after threshold filter)
                confidences_prediou_list = []
                # GT masks that survive the pred-iou threshold (parallel to pred_masks_prediou_list)
                gt_masks_prediou_list = []

                confidences_list = []
                
                # Denormalize image
                mean = sam.pixel_mean.to(device).view(-1, 1, 1)
                std = sam.pixel_std.to(device).view(-1, 1, 1)
                image_np = (image_tensor * std + mean).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                
                # Upsample GT masks to 1024x1024
                gt_masks_1024 = F.interpolate(
                    masks_gt_256_i.unsqueeze(1), (1024, 1024),
                    mode='bilinear', align_corners=False
                ).squeeze(1)
                gt_masks_list = [(m > 0.5).cpu().numpy() for m in gt_masks_1024]
                random_points = []
                
                # Generate predictions
                for j in range(len(points_i)):
                    # prompt_point = points_i[j].unsqueeze(0).unsqueeze(0).to(device)
                    if use_random_bbox_point and bbox_prompts_i is not None:
                        bbox = bbox_prompts_i[j].cpu().numpy()
                        random_point = get_random_point_in_bbox(bbox, image_size=1024)
                        random_points.append(random_point)
                        prompt_point = random_point.unsqueeze(0).unsqueeze(0).to(device)
                    else:
                        prompt_point = points_i[j].unsqueeze(0).unsqueeze(0).to(device)
                    prompt_label = torch.tensor([[1]], device=device)
                    target_mask = masks_gt_256_i[j].unsqueeze(0).unsqueeze(0)
                    
                    bbox_input = None
                    if bbox_prompts_i is not None:
                        bbox_input = bbox_prompts_i[j].unsqueeze(0).to(device)
                    
                    sparse_emb, dense_emb = sam.prompt_encoder(
                        points=(prompt_point, prompt_label),
                        boxes=bbox_input,
                        masks=None
                    )
                    
                    pred_masks_low, pred_iou = sam.mask_decoder(
                        image_embeddings=image_embeddings[i].unsqueeze(0),
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=True,
                    )
                    
                    # For best-by-dice
                    dice_scores = torch.stack([1 - dice_loss(m, target_mask) 
                                             for m in pred_masks_low.squeeze(0)])
                    best_idx = torch.argmax(dice_scores)

                    # For best-by-predicted-iou
                    best_prediou_idx = torch.argmax(pred_iou[0]).item()
                    
                    pred_mask_1024 = sam.postprocess_masks(
                        pred_masks_low[:, best_idx:best_idx+1, :, :],
                        input_size=images[i].shape[-2:],
                        original_size=images[i].shape[-2:]
                    )

                    # For predicted IoU selection
                    pred_mask_1024_prediou = sam.postprocess_masks(
                        pred_masks_low[:, best_prediou_idx:best_prediou_idx+1, :, :],
                        input_size=images[i].shape[-2:],
                        original_size=images[i].shape[-2:]
                    )
                    
                    # Convert to numpy immediately (move off GPU)
                    pred_mask_np = (pred_mask_1024 > sam.mask_threshold).squeeze().cpu().numpy()
                    pred_mask_np_prediou = (pred_mask_1024_prediou > sam.mask_threshold).squeeze().cpu().numpy()

                    # best-by-dice: always kept
                    pred_masks_list.append(pred_mask_np)

                    # best-by-pred-iou: filter by threshold
                    raw_prediou_score = pred_iou[0, best_prediou_idx].item()
                    if raw_prediou_score >= pred_iou_threshold:
                        pred_masks_prediou_list.append(pred_mask_np_prediou)
                        confidences_prediou_list.append(raw_prediou_score)
                        # Store GT mask paired with this prediction
                        target_1024_np = F.interpolate(
                            target_mask, (1024, 1024), mode='bilinear', align_corners=False
                        ).squeeze().cpu().numpy()
                        gt_masks_prediou_list.append((target_1024_np > 0.5))

                    confidence = pred_iou[0, best_idx].item()
                    # print confidence
                    # print("*"*20)
                    # print(f"Confidence: {confidence}")  
                    confidences_list.append(confidence)

                    # Upsampled target for NMS-metrics path (best-by-dice)
                    target_1024 = F.interpolate(
                        target_mask, (1024, 1024), mode='bilinear', align_corners=False
                    ).to(device)
                    target_masks_1024_list.append((target_1024 > 0.5).squeeze().cpu().numpy())

                    # Store pred low-res as numpy for post-NMS metric reuse
                    pred_masks_low_list.append(pred_mask_np)

                    # Delete GPU tensors immediately
                    del pred_masks_low, pred_iou, sparse_emb, dense_emb, pred_mask_1024, target_1024
                    del pred_mask_1024_prediou
                    torch.cuda.empty_cache()

                # points_np = points_i.cpu().numpy()
                if use_random_bbox_point is not None and bbox_prompts_i is not None:
                    print("why is it here")
                    points_np = np.stack([p.cpu().numpy() for p in random_points])
                else:
                    points_np = points_i.cpu().numpy()
                bbox_np = None
                if bbox_prompts_i is not None:
                    bbox_np = [b.cpu().numpy() for b in bbox_prompts_i]

                # --- Confidence-based border visualization (best-by-dice) ---
                confs = np.array(confidences_list)
                if len(confs) > 1:
                    conf_min, conf_max = confs.min(), confs.max()
                    conf_range = conf_max - conf_min if conf_max > conf_min else 1.0
                    norm_confs = (confs - conf_min) / conf_range
                else:
                    norm_confs = np.ones_like(confs)

                # ----------------------------------------------------------------
                # Per-image metrics  (best-by-dice, BEFORE NMS)
                # ----------------------------------------------------------------
                m_dice_pre = calculate_metrics(
                    pred_masks_list, gt_masks_list
                )
                per_image_results["dice_pre_nms"].append(m_dice_pre)

                # ----------------------------------------------------------------
                # Per-image metrics  (best-by-pred-iou, BEFORE NMS)
                # pred_masks_prediou_list already filtered by pred_iou_threshold;
                # gt_masks_prediou_list contains only matched GT masks;
                # unmatched GTs (filtered out) are counted as FN via the full
                # gt_masks_list passed below.
                # ----------------------------------------------------------------
                m_prediou_pre = calculate_metrics(
                    pred_masks_prediou_list, gt_masks_list
                )
                per_image_results["prediou_pre_nms"].append(m_prediou_pre)

                # Compose GT and prompt images
                gt_img = create_overlay_image(image_np, gt_masks_list, [], None)
                prompt_img = create_overlay_image(image_np, [], points_np, bbox_np)

                if per_char_visualization:
                    # Make subdir for this image
                    char_dir = os.path.join(output_dir, f"eval_{idx:04d}_chars")
                    os.makedirs(char_dir, exist_ok=True)
                    for j, (mask_np, conf) in enumerate(zip(pred_masks_list, norm_confs)):
                        char_img = Image.fromarray(image_np).convert("RGB")
                        draw = ImageDraw.Draw(char_img)
                        mask_rgba = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
                        mask_rgba[mask_np > 0] = (0, 255, 0, 128)
                        mask_overlay = Image.fromarray(mask_rgba, 'RGBA')
                        char_img = Image.alpha_composite(char_img.convert("RGBA"), mask_overlay).convert("RGB")
                        draw = ImageDraw.Draw(char_img)
                        contours = skimage.measure.find_contours(mask_np, 0.5)
                        r = int(255 * (1 - conf))
                        g = int(255 * conf)
                        color = (r, g, 0)
                        for contour in contours:
                            contour_xy = [(float(x[1]), float(x[0])) for x in contour]
                            if len(contour_xy) > 1:
                                draw.line(contour_xy, fill=color, width=4)
                        x, y = int(points_np[j][0]), int(points_np[j][1])
                        draw.ellipse((x-3, y-3, x+3, y+3), fill='white', outline='black')
                        char_img.save(os.path.join(char_dir, f"char_{j:02d}.png"))
                else:
                    # --- NMS on predicted masks (best-by-dice) ---
                    keep_indices = mask_nms_ios(pred_masks_list, confidences_list, ios_threshold=0.7)
                    pred_masks_list_nms = [pred_masks_list[k] for k in keep_indices]
                    norm_confs_nms = norm_confs[keep_indices]
                    points_np_nms = points_np[keep_indices]
                    bbox_np_nms = [bbox_np[k] for k in keep_indices] if bbox_np is not None else None

                    # Per-image metrics (best-by-dice, AFTER NMS)
                    m_dice_post = calculate_metrics(
                        pred_masks_list_nms, gt_masks_list
                    )
                    per_image_results["dice_post_nms"].append(m_dice_post)

                    # --- NMS on predicted masks (best-by-pred-iou) ---
                    if pred_masks_prediou_list:
                        keep_indices_prediou = mask_nms_ios(
                            pred_masks_prediou_list, confidences_prediou_list, ios_threshold=0.5
                        )
                        pred_masks_prediou_list_nms = [pred_masks_prediou_list[k] for k in keep_indices_prediou]
                    else:
                        pred_masks_prediou_list_nms = []

                    # Per-image metrics (best-by-pred-iou, AFTER NMS)
                    m_prediou_post = calculate_metrics(
                        pred_masks_prediou_list_nms, gt_masks_list
                    )
                    per_image_results["prediou_post_nms"].append(m_prediou_post)

                    # --- Per-image console printout ---
                    print(f"  [Image {idx:04d}] "
                          f"dice_pre: iou={m_dice_pre['mean_iou']:.4f} dice={m_dice_pre['mean_dice']:.4f} | "
                          f"prediou_pre(thr={pred_iou_threshold}): iou={m_prediou_pre['mean_iou']:.4f} dice={m_prediou_pre['mean_dice']:.4f}")

                    # Build visualization (best-by-dice post-NMS)
                    overlay_img = Image.fromarray(image_np).convert("RGBA")
                    overlay = Image.new('RGBA', overlay_img.size, (0, 0, 0, 0))
                    for mask_np in pred_masks_list_nms:
                        if mask_np.sum() == 0:
                            continue
                        color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255), 128)
                        mask_rgba = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
                        mask_rgba[mask_np > 0] = color
                        overlay = Image.alpha_composite(overlay, Image.fromarray(mask_rgba, 'RGBA'))
                    pred_img = Image.alpha_composite(overlay_img, overlay).convert("RGB")
                    draw = ImageDraw.Draw(pred_img)
                    for j, mask_np in enumerate(pred_masks_list_nms):
                        if mask_np.sum() == 0:
                            continue
                        contours = skimage.measure.find_contours(mask_np, 0.5)
                        conf = norm_confs_nms[j]
                        r = int(255 * (1 - conf))
                        g = int(255 * conf)
                        color = (r, g, 0)
                        for contour in contours:
                            contour_xy = [(float(x[1]), float(x[0])) for x in contour]
                            if len(contour_xy) > 1:
                                draw.line(contour_xy, fill=color, width=4)
                    for point in points_np_nms:
                        x, y = int(point[0]), int(point[1])
                        draw.ellipse((x-3, y-3, x+3, y+3), fill='white', outline='black')
                    combined = Image.new('RGB', (gt_img.width * 3, gt_img.height))
                    combined.paste(gt_img, (0, 0))
                    combined.paste(prompt_img, (gt_img.width, 0))
                    combined.paste(pred_img, (gt_img.width * 2, 0))
                    draw_comb = ImageDraw.Draw(combined)
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                    except:
                        font = ImageFont.load_default()
                    draw_comb.text((10, 10), "Ground Truth", fill='white', font=font)
                    draw_comb.text((gt_img.width + 10, 10), "Prompts", fill='white', font=font)
                    draw_comb.text((gt_img.width * 2 + 10, 10), "Predictions", fill='white', font=font)
                    save_path = os.path.join(output_dir, f"eval_{idx:04d}.png")
                    combined.save(save_path)

                    # Save masks + metadata
                    npz_path = os.path.join(output_dir, f"eval_{idx:04d}.npz")
                    np.savez_compressed(
                        npz_path,
                        pred_masks=np.stack(pred_masks_list_nms) if pred_masks_list_nms else np.array([]),
                        gt_masks=np.stack(gt_masks_list) if gt_masks_list else np.array([]),
                        points=points_np_nms,
                        bboxes=np.stack(bbox_np_nms) if bbox_np_nms is not None else None,
                        confidences=norm_confs_nms
                    )

                # Clean up after each image
                del masks_gt_256_i
                torch.cuda.empty_cache()

            # Clean up image embedding after all images in the batch are processed
            del image_embeddings
            torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Aggregate metrics across all images
    # ----------------------------------------------------------------
    def _aggregate(results_list, label):
        """Print per-image-averaged metrics for one strategy."""
        n = len(results_list)
        if n == 0:
            print(f"  [{label}] No images evaluated.")
            return

        keys = ["fg_iou", "fg_dice", "mean_iou", "mean_dice"]
        avgs = {k: float(np.mean([r[k] for r in results_list])) for k in keys}

        print(f"\n  [{label}]  ({n} images)")
        print(f"    fg_iou={avgs['fg_iou']:.4f}  fg_dice={avgs['fg_dice']:.4f}")
        print(f"    mean_iou={avgs['mean_iou']:.4f}  mean_dice={avgs['mean_dice']:.4f}")

    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    _aggregate(per_image_results["dice_pre_nms"],    "best-by-dice,    PRE-NMS ")
    _aggregate(per_image_results["prediou_pre_nms"],  f"best-by-predIoU (thr={pred_iou_threshold}), PRE-NMS ")
    if per_image_results["dice_post_nms"]:
        _aggregate(per_image_results["dice_post_nms"],  "best-by-dice,    POST-NMS")
        _aggregate(per_image_results["prediou_post_nms"], f"best-by-predIoU (thr={pred_iou_threshold}), POST-NMS")
    print(f"\n  Results saved to: {output_dir}")
    print(f"{'='*70}\n")

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device)
    sam.eval()
    
    # Load dataset
    dataset = SimplePreprocessedSamDataset(
        index_file=args.index_file,
        image_size=sam.image_encoder.img_size,
        pixel_mean=sam.pixel_mean,
        pixel_std=sam.pixel_std,
        use_line_masks=args.use_line_masks,
        use_bbox_prompt=args.use_bbox_prompt
    )
    
    def collate_fn(batch):
        return {
            'image': torch.stack([s['image'] for s in batch]),
            'gt_masks': [s['gt_masks'] for s in batch],
            'gt_points': [s['gt_points'] for s in batch],
            'bbox_prompts': [s['bbox_prompts'] for s in batch] if 'bbox_prompts' in batch[0] else None
        }
    
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=4
    )
    
    # Run evaluation
    evaluate(sam, loader, device, args.output_dir,
         per_char_visualization=args.per_char_visualization,
         pred_iou_threshold=args.pred_iou_threshold,
         use_random_bbox_point=args.use_random_bbox_point)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate SAM model')
    parser.add_argument('--model_type', type=str, default='vit_b')
    parser.add_argument('--checkpoint', type=str, default="best_model.pth", help='Model checkpoint path')
    parser.add_argument('--index_file', type=str, default="master_index_from_associations_end_val.json", help='Dataset index JSON')
    parser.add_argument('--output_dir', type=str, default='./analysis_final/', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--use_line_masks', action='store_true')
    parser.add_argument('--use_bbox_prompt', action='store_true', default=True)
    parser.add_argument('--per_char_visualization', action='store_true', help='Store per-character prediction visualizations')
    parser.add_argument('--pred_iou_threshold', type=float, default=0.3,
                        help='Minimum predicted IoU score to keep a mask (pred-iou selection path)')
    parser.add_argument('--use_random_bbox_point', default=True, action='store_true', 
                    help='Use random point inside bbox instead of centroid as prompt')
    args = parser.parse_args()
    main(args)