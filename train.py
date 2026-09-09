# train.py
import torch
import argparse
from tqdm import tqdm
from segment_anything import sam_model_registry
import numpy as np
from PIL import Image, ImageDraw
import cv2
import wandb
import random
import os
import matplotlib.pyplot as plt
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont


from data_loader import SimplePreprocessedSamDataset
from utils import SequentialCharacterLoss, dice_loss, compute_inter_line_overlap, compute_char_vs_other_lines_penalty, CombinedLoss

# --- Debug Functions for Line Assignment ---
def debug_log_training_batch_lines(batch, batch_idx, num_chars_to_show=20):
    """
    Log which characters are being trained on which lines.
    """
    print(f"\n{'='*80}")
    print(f"TRAINING BATCH {batch_idx} - Line Assignment Verification")
    print(f"{'='*80}")
    
    line_numbers_batch = batch.get('line_numbers', None)
    
    if line_numbers_batch is None:
        print("WARNING: No line_numbers found in batch!")
        return
    
    for img_idx in range(len(batch['gt_masks'])):
        masks = batch['gt_masks'][img_idx]
        points = batch['gt_points'][img_idx]
        line_nums = line_numbers_batch[img_idx] if img_idx < len(line_numbers_batch) else None
        
        print(f"\nImage {img_idx} in batch:")
        print(f"  Total characters: {len(masks)}")
        
        if line_nums is not None:
            # Convert to list if tensor
            if isinstance(line_nums, torch.Tensor):
                line_nums = line_nums.cpu().numpy()
            
            # Count per line
            line_counts = {}
            for ln in line_nums:
                line_counts[ln] = line_counts.get(ln, 0) + 1
            
            print(f"  Characters per line:")
            for line_num in sorted(line_counts.keys()):
                status = "BACKGROUND" if line_num == -1 else f"Line {line_num}"
                print(f"    {status}: {line_counts[line_num]} characters")
            
            # Show detailed assignments
            print(f"\n  First {min(num_chars_to_show, len(masks))} character assignments:")
            for i in range(min(num_chars_to_show, len(masks))):
                ln = line_nums[i]
                mask = masks[i]
                point = points[i]
                mask_pixels = mask.sum().item() if isinstance(mask, torch.Tensor) else mask.sum()
                bg_marker = " [BACKGROUND - Should have zero mask]" if ln == -1 else ""
                zero_mask = " **ZERO MASK**" if mask_pixels == 0 else ""
                print(f"    Char {i:3d}: Line {ln:2d}, Mask Pixels: {mask_pixels:8.0f}, "
                      f"Point: ({point[0].item():.1f}, {point[1].item():.1f}){bg_marker}{zero_mask}")
        else:
            print("  No line number information available!")
    
    print(f"{'='*80}\n")

def save_checkpoint(sam, optimizer, scheduler, epoch, best_val_iou, checkpoint_path):
    """
    Save model, optimizer, scheduler, and training metadata.
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': sam.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_iou': best_val_iou,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(sam, optimizer, scheduler, checkpoint_path, device):
    """
    Load model, optimizer, scheduler, and training metadata.
    Returns: epoch, best_val_iou
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sam.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    best_val_iou = checkpoint.get('best_val_iou', 0.0)
    print(f"Checkpoint loaded from epoch {epoch} with best_val_iou={best_val_iou}")
    return epoch, best_val_iou

# --- Visualization Utility Functions ---
def create_overlay_image(image_np, masks_list, points_list, bbox_list=None):
    """
    Creates an overlay image with all character masks colored randomly.
    Optionally overlays bounding boxes and points.
    """
    overlay_img = Image.fromarray(image_np).convert("RGBA")
    overlay = Image.new('RGBA', overlay_img.size, (0, 0, 0, 0))
    
    # Add masks if provided
    for mask_np in masks_list:
        if mask_np.sum() == 0:
            continue
        color = (
            random.randint(50, 255),
            random.randint(50, 255), 
            random.randint(50, 255),
            128
        )
        mask_rgba = np.zeros((*mask_np.shape, 4), dtype=np.uint8)
        mask_rgba[mask_np > 0] = color
        mask_pil = Image.fromarray(mask_rgba, 'RGBA')
        overlay = Image.alpha_composite(overlay, mask_pil)
    
    result = Image.alpha_composite(overlay_img, overlay)
    draw = ImageDraw.Draw(result)
    
    # Add points if provided
    for point in points_list:
        point_x, point_y = point[0], point[1]
        radius = 3
        draw.ellipse(
            (point_x - radius, point_y - radius, point_x + radius, point_y + radius),
            fill='white',
            outline='black'
        )
    
    # Add bounding boxes if provided (show ALL boxes, including empty ones for debugging)
    if bbox_list is not None:
        for i, bbox in enumerate(bbox_list):
            if bbox.sum() > 0:  # Non-empty bbox
                x_min, y_min, x_max, y_max = bbox
                # Use different colors for different bboxes
                colors = ["lime", "cyan", "yellow", "magenta", "orange"]
                color = colors[i % len(colors)]
                draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
                # Add bbox index label
                draw.text((x_min, y_min - 15), f"BBox {i}", fill=color, font=None)
            else:
                # For debugging: show empty bboxes as small red dots at points
                if i < len(points_list):
                    point = points_list[i]
                    draw.ellipse(
                        (point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2),
                        fill='red', outline='darkred'
                    )
                    draw.text((point[0] + 5, point[1] - 10), f"Empty {i}", fill='red', font=None)
    
    return result

def eval_and_save_images(sam, eval_loader, device, args):
    """
    Evaluate on the eval_loader and save visualizations for every image.
    Now uses SequentialCharacterLoss and handles neighbor masks.
    """
    sam.eval()
    os.makedirs(args.eval_results_dir, exist_ok=True)
    
    # Create loss function
    loss_fn = SequentialCharacterLoss(focal_weight=20.0, overlap_weight=1.0)
    
    img_idx = 0
    total_loss = 0
    total_iou = 0
    num_chars = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Eval-Only: Saving Results"):
            images = batch['image'].to(device)
            gt_masks_256 = batch['gt_masks']
            gt_points = batch['gt_points']
            batch_bbox_prompts = batch.get('bbox_prompts', None)
            
            # NEW: Get neighbor masks
            left_neighbor_masks = batch.get('left_neighbor_masks', None)
            right_neighbor_masks = batch.get('right_neighbor_masks', None)

            image_embeddings = sam.image_encoder(images)

            for i in range(len(images)):
                image_tensor = images[i]
                points_i = gt_points[i]
                masks_gt_256_i = gt_masks_256[i].to(device)
                
                # Get neighbor masks for this image
                left_neighbors_i = None
                right_neighbors_i = None
                if left_neighbor_masks is not None and i < len(left_neighbor_masks):
                    left_neighbors_i = left_neighbor_masks[i].to(device)
                if right_neighbor_masks is not None and i < len(right_neighbor_masks):
                    right_neighbors_i = right_neighbor_masks[i].to(device)
                
                bbox_prompts_i = None
                if batch_bbox_prompts is not None:
                    bbox_prompts_i = batch_bbox_prompts[i]

                # Denormalize image
                mean = sam.pixel_mean.to(device)
                std = sam.pixel_std.to(device)
                if mean.dim() == 1:
                    mean = mean[:, None, None]
                if std.dim() == 1:
                    std = std[:, None, None]
                image_np = (image_tensor * std + mean).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

                gt_masks_1024_tensor = F.interpolate(
                    masks_gt_256_i.unsqueeze(1),
                    size=(1024, 1024),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                gt_masks_1024_list = [(m > 0.5).cpu().numpy() for m in gt_masks_1024_tensor]

                pred_masks_1024_list = []
                
                for j in range(len(points_i)):
                    prompt_point = points_i[j].unsqueeze(0).unsqueeze(0).to(device)
                    prompt_label = torch.tensor([[1]], device=device)
                    target_mask_256 = masks_gt_256_i[j].unsqueeze(0).unsqueeze(0)
                    
                    # Get neighbor masks for this character
                    left_mask_256 = torch.zeros_like(target_mask_256)
                    right_mask_256 = torch.zeros_like(target_mask_256)
                    if left_neighbors_i is not None and j < len(left_neighbors_i):
                        left_mask_256 = left_neighbors_i[j].unsqueeze(0).unsqueeze(0)
                    if right_neighbors_i is not None and j < len(right_neighbors_i):
                        right_mask_256 = right_neighbors_i[j].unsqueeze(0).unsqueeze(0)
                    
                    bbox_input = None
                    if bbox_prompts_i is not None and j < len(bbox_prompts_i):
                        bbox_input = bbox_prompts_i[j].unsqueeze(0).to(device)
                    
                    sparse_emb, dense_emb = sam.prompt_encoder(
                        points=(prompt_point, prompt_label),
                        boxes=bbox_input,
                        masks=None
                    )
                    
                    pred_masks_low_res, _ = sam.mask_decoder(
                        image_embeddings=image_embeddings[i].unsqueeze(0),
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=True,
                    )
                    
                    # Calculate loss using new loss function
                    loss, loss_dict = loss_fn(pred_masks_low_res, target_mask_256, left_mask_256, right_mask_256)
                    total_loss += loss.item()
                    num_chars += 1
                    
                    # Use best target mask for visualization
                    best_mask_idx = loss_dict['best_target_idx']
                    predicted_mask_1024 = sam.postprocess_masks(
                        pred_masks_low_res[:, best_mask_idx, :, :].unsqueeze(1),
                        input_size=images[i].shape[-2:],
                        original_size=images[i].shape[-2:]
                    )
                    predicted_mask_np = (predicted_mask_1024 > sam.mask_threshold).squeeze().cpu().numpy()
                    pred_masks_1024_list.append(predicted_mask_np)
                    
                    # Calculate IoU
                    pred_binary = (predicted_mask_1024 > sam.mask_threshold).float()
                    gt_1024 = F.interpolate(target_mask_256, size=(1024, 1024), mode='bilinear', align_corners=False)
                    intersection = (pred_binary * gt_1024).sum()
                    union = ((pred_binary + gt_1024) > 0).float().sum()
                    iou = (intersection / (union + 1e-6)).item()
                    total_iou += iou

                    pred_masks = pred_masks.to('cpu')
                    # torch.cuda.empty_cache()

                points_list_np = points_i.cpu().numpy()
                bbox_list_np = None
                if bbox_prompts_i is not None:
                    bbox_list_np = [b.cpu().numpy() for b in bbox_prompts_i]

                # Create visualizations
                gt_overlay = create_overlay_image(image_np, gt_masks_1024_list, [], bbox_list=None)
                prompts_overlay = create_overlay_image(image_np, [], points_list_np, bbox_list=bbox_list_np)
                pred_overlay = create_overlay_image(image_np, pred_masks_1024_list, [], bbox_list=None)

                from PIL import ImageDraw, ImageFont, Image as PILImage
                combined_img = PILImage.new('RGB', (gt_overlay.width * 3, gt_overlay.height))
                combined_img.paste(gt_overlay, (0, 0))
                combined_img.paste(prompts_overlay, (gt_overlay.width, 0))
                combined_img.paste(pred_overlay, (gt_overlay.width * 2, 0))
                draw = ImageDraw.Draw(combined_img)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                except:
                    font = ImageFont.load_default()
                draw.text((10, 10), "Ground Truth", fill='white', font=font)
                draw.text((gt_overlay.width + 10, 10), "Prompts (Points + BBox)", fill='white', font=font)
                draw.text((gt_overlay.width * 2 + 10, 10), "Predictions", fill='white', font=font)

                save_path = os.path.join(args.eval_results_dir, f"eval_image_{img_idx+1}_comparison.png")
                combined_img.save(save_path)
                img_idx += 1


                # --- Save 5-panel neighbor visualization for a few random characters ---
                num_chars_for_neighbors = 5
                char_indices = list(range(len(points_i)))
                random.shuffle(char_indices)
                for char_idx in char_indices[:num_chars_for_neighbors]:
                    # Prepare masks for this character and its neighbors
                    prompt_point = points_i[char_idx].unsqueeze(0).unsqueeze(0).to(device)
                    prompt_label = torch.tensor([[1]], device=device)
                    target_mask_256 = masks_gt_256_i[char_idx].unsqueeze(0).unsqueeze(0)
                    left_mask_256 = left_neighbors_i[char_idx].unsqueeze(0).unsqueeze(0) if left_neighbors_i is not None else torch.zeros_like(target_mask_256)
                    right_mask_256 = right_neighbors_i[char_idx].unsqueeze(0).unsqueeze(0) if right_neighbors_i is not None else torch.zeros_like(target_mask_256)
                    bbox_input = None
                    if bbox_prompts_i is not None and char_idx < len(bbox_prompts_i):
                        bbox_input = bbox_prompts_i[char_idx].unsqueeze(0).to(device)
                    sparse_emb, dense_emb = sam.prompt_encoder(
                        points=(prompt_point, prompt_label),
                        boxes=bbox_input,
                        masks=None
                    )
                    pred_masks_low_res, _ = sam.mask_decoder(
                        image_embeddings=image_embeddings[i].unsqueeze(0),
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=True,
                    )
                    # 0-2: target, 3: left, 4: right
                    target_preds = pred_masks_low_res[:, 0:3, :, :]
                    left_pred = pred_masks_low_res[:, 3, :, :]
                    right_pred = pred_masks_low_res[:, 4, :, :]
                    dice_scores = torch.stack([1 - dice_loss(m, target_mask_256) for m in target_preds.squeeze(0)])
                    best_mask_idx = torch.argmax(dice_scores)
                    # Upsample all masks
                    def upsample(mask):
                        return (sam.postprocess_masks(
                            mask.unsqueeze(1),
                            input_size=images[i].shape[-2:],
                            original_size=images[i].shape[-2:]
                        ) > sam.mask_threshold).squeeze().cpu().numpy()
                    gt_mask_1024 = (F.interpolate(target_mask_256, size=(1024, 1024), mode='bilinear', align_corners=False) > 0.5).squeeze().cpu().numpy()
                    left_mask_1024 = (F.interpolate(left_mask_256, size=(1024, 1024), mode='bilinear', align_corners=False) > 0.5).squeeze().cpu().numpy()
                    right_mask_1024 = (F.interpolate(right_mask_256, size=(1024, 1024), mode='bilinear', align_corners=False) > 0.5).squeeze().cpu().numpy()
                    pred_target_1024 = upsample(target_preds[:, best_mask_idx, :, :])
                    pred_left_1024 = upsample(left_pred)
                    pred_right_1024 = upsample(right_pred)
                    # Single point and bbox for this char
                    point_np = points_i[char_idx].cpu().numpy().reshape(1, 2)
                    bbox_np = [bbox_prompts_i[char_idx].cpu().numpy()] if bbox_prompts_i is not None else None
                    # Compose 5-panel image
                    gt_overlay = create_overlay_image(image_np, [gt_mask_1024], [], bbox_list=None)
                    prompts_overlay = create_overlay_image(image_np, [], point_np, bbox_list=bbox_np)
                    pred_overlay = create_overlay_image(image_np, [pred_target_1024], [], bbox_list=None)
                    left_overlay = create_overlay_image(image_np, [pred_left_1024], [], bbox_list=None)
                    right_overlay = create_overlay_image(image_np, [pred_right_1024], [], bbox_list=None)
                    width, height = gt_overlay.width, gt_overlay.height
                    combined_neighbor_img = Image.new('RGB', (width * 5, height))
                    combined_neighbor_img.paste(gt_overlay, (0, 0))
                    combined_neighbor_img.paste(prompts_overlay, (width, 0))
                    combined_neighbor_img.paste(pred_overlay, (width * 2, 0))
                    combined_neighbor_img.paste(left_overlay, (width * 3, 0))
                    combined_neighbor_img.paste(right_overlay, (width * 4, 0))
                    draw = ImageDraw.Draw(combined_neighbor_img)
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                    except:
                        font = ImageFont.load_default()
                    draw.text((10, 10), "GT", fill='white', font=font)
                    draw.text((width + 10, 10), "Prompt", fill='white', font=font)
                    draw.text((width * 2 + 10, 10), "Target Pred", fill='white', font=font)
                    draw.text((width * 3 + 10, 10), "Left Pred", fill='white', font=font)
                    draw.text((width * 4 + 10, 10), "Right Pred", fill='white', font=font)
                    save_path = os.path.join(args.eval_results_dir, f"eval_image_{img_idx}_neighbor_char_{char_idx+1}.png")
                    combined_neighbor_img.save(save_path)
                    print(f"Char {char_idx}: left_mask_256 sum={left_mask_256.sum().item()}, right_mask_256 sum={right_mask_256.sum().item()}")
                    print(f"Pred left_pred sum={left_pred.sum().item()}, right_pred sum={right_pred.sum().item()}")
    
    # Print evaluation summary
    avg_loss = total_loss / num_chars if num_chars > 0 else 0
    avg_iou = total_iou / num_chars if num_chars > 0 else 0
    print(f"\nEvaluation Summary:")
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Avg IoU: {avg_iou:.4f}")
    print(f"  Total Characters: {num_chars}")
    print(f"  Images Processed: {img_idx}")

def log_comparison_images(epoch, sam, train_loader, device, args):
    """
    Generate and log TWO TYPES of comparison images:
    1. Main visualization: GT | Prompts | Target Predictions (3-panel, multiple images)
    2. Neighbor visualization: Character + Left/Right neighbor predictions (5-panel, 5 characters)
    """
    sam.eval()
    
    main_logged_images = []
    neighbor_logged_images = []
    num_images_to_visualize = 3  # Number of full images for main visualization
    num_chars_for_neighbors = 5  # Number of individual characters for neighbor visualization
    
    loader_iter = iter(train_loader)
    random.seed(42 + epoch)
    
    # ========== 1. MAIN VISUALIZATION ==========
    with torch.no_grad():
        for img_idx in range(num_images_to_visualize):
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            
            images = batch['image'].to(device)
            gt_masks_256 = batch['gt_masks']
            gt_points = batch['gt_points']
            batch_bbox_prompts = batch.get('bbox_prompts', None)
            left_neighbor_masks = batch.get('left_neighbor_masks', None)
            right_neighbor_masks = batch.get('right_neighbor_masks', None)
            
            image_embeddings = sam.image_encoder(images)
            i = 0  # batch_size=1
            image_tensor = images[i]
            points_i = gt_points[i]
            masks_gt_256_i = gt_masks_256[i].to(device)
            bbox_prompts_i = batch_bbox_prompts[i] if batch_bbox_prompts is not None else None
            left_neighbors_i = left_neighbor_masks[i].to(device) if left_neighbor_masks is not None else None
            right_neighbors_i = right_neighbor_masks[i].to(device) if right_neighbor_masks is not None else None

            # Denormalize image
            mean = sam.pixel_mean.to(device)
            std = sam.pixel_std.to(device)
            if mean.dim() == 1:
                mean = mean[:, None, None]
            if std.dim() == 1:
                std = std[:, None, None]
            image_np = (image_tensor * std + mean).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            
            # Upsample GT masks to 1024
            gt_masks_1024_tensor = F.interpolate(
                masks_gt_256_i.unsqueeze(1),
                size=(1024, 1024),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            gt_masks_1024_list = [(m > 0.5).cpu().numpy() for m in gt_masks_1024_tensor]
            
            # Get predictions for TARGET character only (token 0-2, best one)
            pred_masks_1024_list = []
            for j in range(len(points_i)):
                prompt_point = points_i[j].unsqueeze(0).unsqueeze(0).to(device)
                prompt_label = torch.tensor([[1]], device=device)
                target_mask_256 = masks_gt_256_i[j].unsqueeze(0).unsqueeze(0)
                bbox_input = None
                if bbox_prompts_i is not None and j < len(bbox_prompts_i):
                    bbox_input = bbox_prompts_i[j].unsqueeze(0).to(device)
                sparse_emb, dense_emb = sam.prompt_encoder(
                    points=(prompt_point, prompt_label),
                    boxes=bbox_input,
                    masks=None
                )
                pred_masks_low_res, _ = sam.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=True,
                )
                # Find best TARGET mask (tokens 0-2)
                target_preds = pred_masks_low_res[:, 0:3, :, :]
                dice_scores = torch.stack([1 - dice_loss(m, target_mask_256) for m in target_preds.squeeze(0)])
                best_mask_idx = torch.argmax(dice_scores)
                # Upsample best target prediction
                predicted_mask_1024 = sam.postprocess_masks(
                    target_preds[:, best_mask_idx, :, :].unsqueeze(1),
                    input_size=images[i].shape[-2:],
                    original_size=images[i].shape[-2:]
                )
                predicted_mask_np = (predicted_mask_1024 > sam.mask_threshold).squeeze().cpu().numpy()
                pred_masks_1024_list.append(predicted_mask_np)
            
            points_list_np = points_i.cpu().numpy()
            bbox_list_np = None
            if bbox_prompts_i is not None:
                bbox_list_np = [b.cpu().numpy() for b in bbox_prompts_i]
            
            # Create 3-panel visualization (GT | Prompts | Target Predictions)
            gt_overlay = create_overlay_image(image_np, gt_masks_1024_list, [], bbox_list=None)
            prompts_overlay = create_overlay_image(image_np, [], points_list_np, bbox_list=bbox_list_np)
            pred_overlay = create_overlay_image(image_np, pred_masks_1024_list, [], bbox_list=None)
            
            combined_img = Image.new('RGB', (gt_overlay.width * 3, gt_overlay.height))
            combined_img.paste(gt_overlay, (0, 0))
            combined_img.paste(prompts_overlay, (gt_overlay.width, 0))
            combined_img.paste(pred_overlay, (gt_overlay.width * 2, 0))
            draw = ImageDraw.Draw(combined_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                font = ImageFont.load_default()
            draw.text((10, 10), "Ground Truth", fill='white', font=font)
            draw.text((gt_overlay.width + 10, 10), "Prompts", fill='white', font=font)
            draw.text((gt_overlay.width * 2 + 10, 10), "Target Predictions", fill='white', font=font)
            os.makedirs(args.comparison_images_dir, exist_ok=True)
            save_path = f"{args.comparison_images_dir}/epoch_{epoch}_main_{img_idx+1}.png"
            combined_img.save(save_path)
            main_logged_images.append(wandb.Image(combined_img, caption=f"Epoch {epoch} - Main {img_idx+1}"))

            # ========== 2. NEIGHBOR VISUALIZATION (5-panel) ==========
            # Pick up to num_chars_for_neighbors random characters
            char_indices = list(range(len(points_i)))
            random.shuffle(char_indices)
            for char_idx in char_indices[:num_chars_for_neighbors]:
                # Prepare masks for this character and its neighbors
                prompt_point = points_i[char_idx].unsqueeze(0).unsqueeze(0).to(device)
                prompt_label = torch.tensor([[1]], device=device)
                target_mask_256 = masks_gt_256_i[char_idx].unsqueeze(0).unsqueeze(0)
                left_mask_256 = left_neighbors_i[char_idx].unsqueeze(0).unsqueeze(0) if left_neighbors_i is not None else torch.zeros_like(target_mask_256)
                right_mask_256 = right_neighbors_i[char_idx].unsqueeze(0).unsqueeze(0) if right_neighbors_i is not None else torch.zeros_like(target_mask_256)
                bbox_input = None
                if bbox_prompts_i is not None and char_idx < len(bbox_prompts_i):
                    bbox_input = bbox_prompts_i[char_idx].unsqueeze(0).to(device)
                sparse_emb, dense_emb = sam.prompt_encoder(
                    points=(prompt_point, prompt_label),
                    boxes=bbox_input,
                    masks=None
                )
                pred_masks_low_res, _ = sam.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=True,
                )
                # 0-2: target, 3: left, 4: right
                target_preds = pred_masks_low_res[:, 0:3, :, :]
                left_pred = pred_masks_low_res[:, 3, :, :]
                right_pred = pred_masks_low_res[:, 4, :, :]
                dice_scores = torch.stack([1 - dice_loss(m, target_mask_256) for m in target_preds.squeeze(0)])
                best_mask_idx = torch.argmax(dice_scores)
                # Upsample all masks
                def upsample(mask):
                    return (sam.postprocess_masks(
                        mask.unsqueeze(1),
                        input_size=images[i].shape[-2:],
                        original_size=images[i].shape[-2:]
                    ) > sam.mask_threshold).squeeze().cpu().numpy()
                gt_mask_1024 = (F.interpolate(target_mask_256, size=(1024, 1024), mode='bilinear', align_corners=False) > 0.5).squeeze().cpu().numpy()
                left_mask_1024 = (F.interpolate(left_mask_256, size=(1024, 1024), mode='bilinear', align_corners=False) > 0.5).squeeze().cpu().numpy()
                right_mask_1024 = (F.interpolate(right_mask_256, size=(1024, 1024), mode='bilinear', align_corners=False) > 0.5).squeeze().cpu().numpy()
                pred_target_1024 = upsample(target_preds[:, best_mask_idx, :, :])
                pred_left_1024 = upsample(left_pred)
                pred_right_1024 = upsample(right_pred)
                # Single point and bbox for this char
                point_np = points_i[char_idx].cpu().numpy().reshape(1, 2)
                bbox_np = [bbox_prompts_i[char_idx].cpu().numpy()] if bbox_prompts_i is not None else None
                # Compose 5-panel image
                gt_overlay = create_overlay_image(image_np, [gt_mask_1024], [], bbox_list=None)
                prompts_overlay = create_overlay_image(image_np, [], point_np, bbox_list=bbox_np)
                pred_overlay = create_overlay_image(image_np, [pred_target_1024], [], bbox_list=None)
                left_overlay = create_overlay_image(image_np, [pred_left_1024], [], bbox_list=None)
                right_overlay = create_overlay_image(image_np, [pred_right_1024], [], bbox_list=None)
                width, height = gt_overlay.width, gt_overlay.height
                combined_neighbor_img = Image.new('RGB', (width * 5, height))
                combined_neighbor_img.paste(gt_overlay, (0, 0))
                combined_neighbor_img.paste(prompts_overlay, (width, 0))
                combined_neighbor_img.paste(pred_overlay, (width * 2, 0))
                combined_neighbor_img.paste(left_overlay, (width * 3, 0))
                combined_neighbor_img.paste(right_overlay, (width * 4, 0))
                draw = ImageDraw.Draw(combined_neighbor_img)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                except:
                    font = ImageFont.load_default()
                draw.text((10, 10), "GT", fill='white', font=font)
                draw.text((width + 10, 10), "Prompt", fill='white', font=font)
                draw.text((width * 2 + 10, 10), "Target Pred", fill='white', font=font)
                draw.text((width * 3 + 10, 10), "Left Pred", fill='white', font=font)
                draw.text((width * 4 + 10, 10), "Right Pred", fill='white', font=font)
                os.makedirs(args.comparison_images_dir, exist_ok=True)
                save_path = f"{args.comparison_images_dir}/epoch_{epoch}_neighbor_{img_idx+1}_char_{char_idx+1}.png"
                combined_neighbor_img.save(save_path)
                neighbor_logged_images.append(wandb.Image(combined_neighbor_img, caption=f"Epoch {epoch} - Neighbor {img_idx+1} Char {char_idx+1}"))

    # Upload to wandb if enabled
    if getattr(args, "use_wandb", False):
        wandb.log({
            f"comparison/main_epoch_{epoch}": main_logged_images,
            f"comparison/neighbor_epoch_{epoch}": neighbor_logged_images
        })


def visualize_prompts_debug(train_loader, sam, device, save_dir="./debug_prompts", num_images=3):
    """
    Visualize all point prompts (centroids) and bounding box prompts for each image in one panel.
    """
    from matplotlib.patches import Rectangle, Circle
    from matplotlib.lines import Line2D

    os.makedirs(save_dir, exist_ok=True)

    mean = sam.pixel_mean.cpu()
    std = sam.pixel_std.cpu()

    if mean.dim() == 1:
        mean = mean[:, None, None]
    if std.dim() == 1:
        std = std[:, None, None]

    loader_iter = iter(train_loader)

    for img_idx in range(num_images):
        try:
            batch = next(loader_iter)
        except StopIteration:
            break

        image_tensor = batch['image'][0]
        points_i = batch['gt_points'][0].cpu().numpy()
        masks_i = batch['gt_masks'][0].cpu().numpy()
        bbox_prompts_i = None
        if 'bbox_prompts' in batch:
            bbox_prompts_i = batch['bbox_prompts'][0].cpu().numpy()

        image_np = (image_tensor * std + mean).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image_np)

        # Show all GT mask contours
        for char_idx in range(len(points_i)):
            gt_mask_256 = masks_i[char_idx]
            gt_mask_1024 = cv2.resize(gt_mask_256, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            ax.contour(gt_mask_1024, colors=['blue'], linewidths=1, levels=[0.5])

        # Show all bbox prompts as rectangles
        if bbox_prompts_i is not None:
            for char_idx, bbox in enumerate(bbox_prompts_i):
                if bbox.sum() > 0:
                    x_min, y_min, x_max, y_max = bbox
                    width = x_max - x_min
                    height = y_max - y_min
                    rect = Rectangle((x_min, y_min), width, height,
                                     linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
                    ax.add_patch(rect)
                    # Optionally, label the bbox
                    ax.text(x_min, y_min - 5, f"{char_idx}", color='red', fontsize=8)

        # Show all point prompts as small filled circles
        for char_idx, point in enumerate(points_i):
            circle = Circle((point[0], point[1]), radius=5, color='lime', fill=True, linewidth=2)
            ax.add_patch(circle)
            # Optionally, label the point
            ax.text(point[0] + 5, point[1] + 5, f"{char_idx}", color='lime', fontsize=8)

        ax.set_title(f'Image {img_idx + 1}: All Points and BBox Prompts')
        ax.axis('off')

        # Add legend
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=2, label='GT Mask'),
            Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='BBox Prompt'),
            Line2D([0], [0], marker='o', color='lime', markersize=10, label='Point Prompt', linestyle='None')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()
        save_path = f"{save_dir}/prompts_image_{img_idx + 1}_all.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved bbox prompt debug visualization: {save_path}")

def custom_collate_fn(batch):
    """Custom collate function to handle variable number of characters per image."""
    images = []
    gt_masks = []
    gt_points = []
    bbox_prompts = []
    left_neighbor_masks = [] 
    right_neighbor_masks = [] 
    line_numbers = []
    line_masks = []
    dataset_indices = []
    
    for sample in batch:
        images.append(sample['image'])
        gt_masks.append(sample['gt_masks'])
        gt_points.append(sample['gt_points'])
        dataset_indices.append(sample['dataset_idx'])
        if 'bbox_prompts' in sample:
            bbox_prompts.append(sample['bbox_prompts'])
        # NEW: Add neighbor masks
        if 'left_neighbor_masks' in sample:
            left_neighbor_masks.append(sample['left_neighbor_masks'])
        if 'right_neighbor_masks' in sample:
            right_neighbor_masks.append(sample['right_neighbor_masks'])
        if 'line_numbers' in sample and sample['line_numbers'] is not None:
            line_numbers.append(sample['line_numbers'])
        if 'line_masks' in sample and sample['line_masks'] is not None:  # <--- ADD THIS
            line_masks.append(sample['line_masks'])
    
    images = torch.stack(images, dim=0)
    
    result = {
        'image': images,
        'gt_masks': gt_masks,
        'gt_points': gt_points,
        'dataset_indices': dataset_indices
    }
    
    if bbox_prompts:
        result['bbox_prompts'] = bbox_prompts
    if left_neighbor_masks:
        result['left_neighbor_masks'] = left_neighbor_masks
    if right_neighbor_masks:
        result['right_neighbor_masks'] = right_neighbor_masks
    if line_numbers:
        result['line_numbers'] = line_numbers  # NEW
    if line_masks:  # <--- ADD THIS
        result['line_masks'] = line_masks
    
    return result

def evaluate(sam, eval_loader, loss_fn, device):
    """
    Evaluate the model on validation/test set using SequentialCharacterLoss.
    Returns average loss, IoU, and Dice score.
    """
    sam.eval()
    total_loss = 0
    total_target_loss = 0
    total_left_loss = 0
    total_right_loss = 0
    total_overlap_loss = 0
    total_iou = 0
    total_iou_loss = 0
    total_dice = 0
    num_chars_total = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            gt_masks = batch['gt_masks']
            gt_points = batch['gt_points']
            
            # NEW: Get neighbor masks
            left_neighbor_masks = batch.get('left_neighbor_masks', None)
            right_neighbor_masks = batch.get('right_neighbor_masks', None)
            batch_bbox_prompts = batch.get('bbox_prompts', None)
            
            image_embeddings = sam.image_encoder(images)
            
            for i in range(len(images)):
                
                points_i = gt_points[i]
                masks_i = gt_masks[i]
                
                # Get neighbor masks for this image
                left_neighbors_i = None
                right_neighbors_i = None
                if left_neighbor_masks is not None and i < len(left_neighbor_masks):
                    left_neighbors_i = left_neighbor_masks[i].to(device)
                if right_neighbor_masks is not None and i < len(right_neighbor_masks):
                    right_neighbors_i = right_neighbor_masks[i].to(device)
                
                bbox_prompts_i = None
                if batch_bbox_prompts is not None:
                    bbox_prompts_i = batch_bbox_prompts[i]
                
                # Normalize masks
                if torch.max(masks_i) > 1.0:
                    masks_i = masks_i / 255.0
                if left_neighbors_i is not None and torch.max(left_neighbors_i) > 1.0:
                    left_neighbors_i = left_neighbors_i / 255.0
                if right_neighbors_i is not None and torch.max(right_neighbors_i) > 1.0:
                    right_neighbors_i = right_neighbors_i / 255.0
                
                num_prompts = min(len(points_i), len(masks_i))
                
                for j in range(num_prompts):
                    prompt_point = points_i[j].unsqueeze(0).unsqueeze(0).to(device)
                    prompt_label = torch.tensor([[1]], device=device)
                    target_mask = masks_i[j].unsqueeze(0).unsqueeze(0).to(device)
                    
                    # Get neighbor masks for this character (default to zeros if missing)
                    left_mask = torch.zeros_like(target_mask)
                    right_mask = torch.zeros_like(target_mask)
                    if left_neighbors_i is not None and j < len(left_neighbors_i):
                        left_mask = left_neighbors_i[j].unsqueeze(0).unsqueeze(0).to(device)
                    if right_neighbors_i is not None and j < len(right_neighbors_i):
                        right_mask = right_neighbors_i[j].unsqueeze(0).unsqueeze(0).to(device)
                    
                    bbox_input = None
                    if bbox_prompts_i is not None and j < len(bbox_prompts_i):
                        bbox_input = bbox_prompts_i[j].unsqueeze(0).to(device)
                    
                    num_chars_total += 1
                    
                    sparse_emb, dense_emb = sam.prompt_encoder(
                        points=(prompt_point, prompt_label), 
                        boxes=bbox_input, 
                        masks=None
                    )
                    
                    pred_masks, pred_iou = sam.mask_decoder(
                        image_embeddings=image_embeddings[i].unsqueeze(0),
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=True,
                    )
                    
                    # NEW: Use SequentialCharacterLoss
                    loss, loss_dict = loss_fn(pred_masks, target_mask, left_mask, right_mask, pred_iou)
                    total_loss += loss.item()
                    total_target_loss += loss_dict['target_loss']
                    total_left_loss += loss_dict['left_loss']
                    total_right_loss += loss_dict['right_loss']
                    total_overlap_loss += loss_dict['overlap_loss']
                    total_iou_loss += loss_dict['iou_loss']
                    
                    # Calculate IoU using best target mask
                    best_target_idx = loss_dict['best_target_idx']
                    best_pred_mask = pred_masks[:, best_target_idx, :, :].unsqueeze(1)
                    
                    pred_binary = (best_pred_mask > 0.5).float()
                    intersection = (pred_binary * target_mask).sum()
                    union = ((pred_binary + target_mask) > 0).float().sum()
                    iou = (intersection / (union + 1e-6)).item()
                    total_iou += iou
                    
                    # Calculate Dice score
                    dice = (2 * intersection / (pred_binary.sum() + target_mask.sum() + 1e-6)).item()
                    total_dice += dice
    
    avg_loss = total_loss / num_chars_total if num_chars_total > 0 else 0
    avg_iou = total_iou / num_chars_total if num_chars_total > 0 else 0
    avg_dice = total_dice / num_chars_total if num_chars_total > 0 else 0

    
    print(f"  Detailed Eval Metrics:")
    print(f"    Target Loss: {total_target_loss / num_chars_total:.4f}")
    print(f"    Left Loss: {total_left_loss / num_chars_total:.4f}")
    print(f"    Right Loss: {total_right_loss / num_chars_total:.4f}")
    print(f"    Overlap Loss: {total_overlap_loss / num_chars_total:.4f}")
    print(f"    IoU Loss: {total_iou_loss / num_chars_total:.4f}") 
    
    return avg_loss, avg_iou, avg_dice

def main(args):
    # Initialize wandb with a run name only if use_wandb is True
    if args.use_wandb:
        if args.run_name:
            run_name = args.run_name
        else:
            mask_suffix = "_char"
            run_name = f"sam_{args.model_type}_lr{args.lr}_epochs{args.epochs}_bs{args.batch_size}{mask_suffix}"
        
        if args.wandb_run_id:
            wandb.init(
                project="sam-finetuning-sristi",
                config=args,
                id=args.wandb_run_id,
                resume="must",
            )
        else:
            wandb.init(
                project="sam-finetuning-sristi",
                config=args,
                name=run_name,
            )
    
    # Use the device argument with fallback to CPU
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Setup Model
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device)

    # Enable gradient checkpointing on mask decoder to save VRAM
    if hasattr(sam.mask_decoder, 'transformer'):
        from torch.utils.checkpoint import checkpoint
        sam.mask_decoder.transformer.gradient_checkpointing = True
    
    # Freeze image encoder
    for name, param in sam.named_parameters():
        if 'image_encoder' in name:
            param.requires_grad = False
        elif 'mask_decoder' in name:
            param.requires_grad = True
        elif 'prompt_encoder' in name:
            param.requires_grad = False
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

    # 2. Setup Optimizer and Loss
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, sam.parameters()),
        lr=args.lr
    )

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ============================================================
    # LOSS FUNCTION SETUP - Two modes
    # ============================================================
    if args.use_legacy_loss:
        print("\n" + "="*80)
        print("LEGACY TRAINING MODE (Old CombinedLoss, No Neighbors)")
        print("="*80)
        print(f"  Using focal_weight=20.0")
        print(f"  LR: {args.lr}")
        print("="*80 + "\n")
        
        # Use CombinedLoss for legacy mode
        loss_fn = CombinedLoss(focal_weight=20.0)

    else:
        print("CHARACTER TRAINING MODE")
        loss_fn = SequentialCharacterLoss(
            focal_weight=20.0,
            overlap_weight=1.0,
            iou_loss_weight=5.0,
            iou_loss_type=args.iou_loss_type,
        )

    # ============================================================
    # DATASET SETUP
    # ============================================================

    train_dataset = SimplePreprocessedSamDataset(
        index_file=args.train_index_file,
        image_size=sam.image_encoder.img_size,
        pixel_mean=sam.pixel_mean,
        pixel_std=sam.pixel_std,
        use_bbox_prompt=args.use_bbox_prompt,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True
    )

    eval_loader = None
    if args.val_index_file:
        val_dataset = SimplePreprocessedSamDataset(
            index_file=args.val_index_file,
            image_size=sam.image_encoder.img_size,
            pixel_mean=sam.pixel_mean,
            pixel_std=sam.pixel_std,
            use_bbox_prompt=args.use_bbox_prompt
        )
        eval_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=4
        )

    if getattr(args, "eval_only", False):
        print("Running in eval-only mode!")
        eval_and_save_images(sam, eval_loader, device, args)
        return

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    best_val_iou = 0.0
    scaler = torch.cuda.amp.GradScaler()
    start_epoch = 0

    # Load checkpoint if resuming
    resume_checkpoint = getattr(args, 'resume_from', None)
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        start_epoch, best_val_iou = load_checkpoint(sam, optimizer, scheduler, resume_checkpoint, device)
        start_epoch += 1  # Start from next epoch


    for epoch in range(args.epochs):
        sam.train()
        epoch_loss = 0
        epoch_target_loss = 0
        epoch_left_loss = 0
        epoch_right_loss = 0
        epoch_overlap_loss = 0
        epoch_iou_loss = 0
        epoch_iou = 0
        epoch_line_penalty = 0
        num_train_chars = 0
        num_train_batches = 0
        metrics = {}

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            if batch_idx % 50 == 0 and epoch == 0:
                debug_log_training_batch_lines(batch, batch_idx, num_chars_to_show=10)

            images = batch['image'].to(device)
            gt_points = batch['gt_points']
            gt_masks = batch['gt_masks']
            batch_bbox_prompts = batch.get('bbox_prompts', None)
            line_numbers_batch = batch.get('line_numbers', None)


            # ============================================================
            # LEGACY MODE - Simple per-character training
            # ============================================================
            if args.use_legacy_loss:
                with torch.no_grad():
                    image_embeddings = sam.image_encoder(images)
                
                for i in range(len(images)):
                    image_embedding = image_embeddings[i:i+1]
                    points_i = gt_points[i]
                    masks_i = gt_masks[i]
                    bbox_prompts_i = batch_bbox_prompts[i] if batch_bbox_prompts is not None else None
                    
                    if torch.max(masks_i) > 1.0:
                        masks_i = masks_i / 255.0
                    
                    optimizer.zero_grad()
                    image_total_loss = 0
                    image_num_chars = 0
                    
                    num_prompts = min(len(points_i), len(masks_i))
                    for j in range(min(num_prompts, 200)):
                        prompt_point = points_i[j].unsqueeze(0).unsqueeze(0).to(device)
                        prompt_label = torch.tensor([[1]], device=device)
                        target_mask = masks_i[j].unsqueeze(0).unsqueeze(0).to(device)
                        bbox_input = None
                        if bbox_prompts_i is not None and torch.sum(target_mask) > 0:
                            bbox_input = bbox_prompts_i[j].unsqueeze(0).to(device)
                        
                        with torch.cuda.amp.autocast():
                            sparse_emb, dense_emb = sam.prompt_encoder(
                                points=(prompt_point, prompt_label), boxes=bbox_input, masks=None)
                            pred_masks, _ = sam.mask_decoder(
                                image_embeddings=image_embedding,
                                image_pe=sam.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_emb,
                                dense_prompt_embeddings=dense_emb,
                                multimask_output=True,
                            )
                            # Use only first mask (original SAM behavior)
                            loss = loss_fn(pred_masks[:, 0, :, :], target_mask)
                        
                        image_total_loss = image_total_loss + loss
                        image_num_chars += 1
                        
                        with torch.no_grad():
                            pred_binary = (torch.sigmoid(pred_masks[:, 0, :, :]) > 0.5).float()
                            target_binary = (target_mask > 0.5).float()
                            inter = (pred_binary * target_binary).sum().item()
                            uni = ((pred_binary + target_binary) > 0).float().sum().item()
                            epoch_iou += inter / uni if uni > 0 else 0.0
                        
                        del pred_masks, sparse_emb, dense_emb
                    
                    if image_num_chars > 0:
                        avg_image_loss = image_total_loss / image_num_chars
                        scaler.scale(avg_image_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        
                        epoch_loss += avg_image_loss.item()
                        num_train_chars += image_num_chars
                        num_train_batches += 1

         
            # ============================================================
            # DEFAULT CHARACTER MODE - GRADIENT ACCUMULATION
            # ============================================================
            else:
                left_neighbor_masks_b = batch.get('left_neighbor_masks', None)
                right_neighbor_masks_b = batch.get('right_neighbor_masks', None)

                with torch.no_grad():
                    image_embeddings = sam.image_encoder(images)

                for i in range(len(images)):
                    image_embedding = image_embeddings[i:i+1]
                    points_i = gt_points[i]
                    masks_i = gt_masks[i]
                    left_neighbors_i = left_neighbor_masks_b[i].to(device) if left_neighbor_masks_b is not None and i < len(left_neighbor_masks_b) else torch.zeros_like(masks_i).to(device)
                    right_neighbors_i = right_neighbor_masks_b[i].to(device) if right_neighbor_masks_b is not None and i < len(right_neighbor_masks_b) else torch.zeros_like(masks_i).to(device)
                    bbox_prompts_i = batch_bbox_prompts[i] if batch_bbox_prompts is not None else None

                    if torch.max(masks_i) > 1.0: masks_i = masks_i / 255.0
                    if torch.max(left_neighbors_i) > 1.0: left_neighbors_i = left_neighbors_i / 255.0
                    if torch.max(right_neighbors_i) > 1.0: right_neighbors_i = right_neighbors_i / 255.0

                    # ===== PER-IMAGE GRADIENT ACCUMULATION =====
                    optimizer.zero_grad()
                    image_total_loss = 0
                    image_num_chars = 0
                    
                    num_prompts = min(len(points_i), len(masks_i))
                    for j in range(min(num_prompts, 400)):
                        prompt_point = points_i[j].unsqueeze(0).unsqueeze(0).to(device)
                        prompt_label = torch.tensor([[1]], device=device)
                        target_mask = masks_i[j].unsqueeze(0).unsqueeze(0).to(device)
                        left_mask = left_neighbors_i[j].unsqueeze(0).unsqueeze(0).to(device)
                        right_mask = right_neighbors_i[j].unsqueeze(0).unsqueeze(0).to(device)
                        bbox_input = None
                        if bbox_prompts_i is not None:
                            if torch.sum(target_mask) > 0:
                                bbox_input = bbox_prompts_i[j].unsqueeze(0).to(device)

                        with torch.cuda.amp.autocast():
                            sparse_emb, dense_emb = sam.prompt_encoder(
                                points=(prompt_point, prompt_label), boxes=bbox_input, masks=None)
                            pred_masks, pred_iou = sam.mask_decoder(
                                image_embeddings=image_embedding,
                                image_pe=sam.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_emb,
                                dense_prompt_embeddings=dense_emb,
                                multimask_output=True,
                            )
                        loss, loss_dict = loss_fn(pred_masks, target_mask, left_mask, right_mask, pred_iou)

                        # Accumulate loss
                        image_total_loss = image_total_loss + loss

                        image_num_chars += 1
                        epoch_loss += loss.item()
                        epoch_target_loss += loss_dict['target_loss']
                        epoch_left_loss += loss_dict['left_loss']
                        epoch_right_loss += loss_dict['right_loss']
                        epoch_iou_loss += loss_dict['iou_loss']

                        with torch.no_grad():
                            best_target_pred = pred_masks[:, loss_dict['best_target_idx'], :, :].unsqueeze(1)
                            pred_binary = (torch.sigmoid(best_target_pred) > 0.5).float()
                            target_binary = (target_mask > 0.5).float()
                            inter = (pred_binary * target_binary).sum().item()
                            uni = ((pred_binary + target_binary) > 0).float().sum().item()
                            epoch_iou += inter / uni if uni > 0 else 0.0

                        del pred_masks, sparse_emb, dense_emb

                    # ===== SINGLE BACKWARD PASS PER IMAGE =====
                    if image_num_chars > 0:
                        avg_image_loss = image_total_loss / image_num_chars
                        scaler.scale(avg_image_loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        
                        num_train_chars += image_num_chars
                        num_train_batches += 1

        # ============================================================
        # METRICS LOGGING
        # ============================================================
        denom_b = num_train_batches if num_train_batches > 0 else 1
        denom_c = num_train_chars if num_train_chars > 0 else 1

        if args.use_legacy_loss:
            avg_epoch_loss = epoch_loss / num_train_batches if num_train_batches > 0 else 0
            avg_iou = epoch_iou / num_train_chars if num_train_chars > 0 else 0
            
            print(f"\nEpoch {epoch+1}/{args.epochs} [LEGACY MODE]")
            print(f"  Train Loss: {avg_epoch_loss:.4f} | IoU: {avg_iou:.4f}")
            
            metrics.update({
                "epoch": epoch + 1,
                "train/loss": avg_epoch_loss,
                "train/iou": avg_iou,
                "train/learning_rate": optimizer.param_groups[0]['lr'],
            })

        else:
            avg_epoch_loss = epoch_loss / num_train_chars if num_train_chars > 0 else 0
            avg_target = epoch_target_loss / num_train_chars if num_train_chars > 0 else 0
            avg_left = epoch_left_loss / num_train_chars if num_train_chars > 0 else 0
            avg_right = epoch_right_loss / num_train_chars if num_train_chars > 0 else 0
            avg_overlap = epoch_overlap_loss / num_train_chars if num_train_chars > 0 else 0
            avg_iou_l = epoch_iou_loss / num_train_chars if num_train_chars > 0 else 0
            avg_iou = epoch_iou / num_train_chars if num_train_chars > 0 else 0

            print(f"\nEpoch {epoch+1}/{args.epochs} [CHAR TRAINING]")
            print(f"  Train Loss: {avg_epoch_loss:.4f} | IoU: {avg_iou:.4f}")
            print(f"    Target: {avg_target:.4f} | Left: {avg_left:.4f} | Right: {avg_right:.4f}")
            print(f"    Overlap: {avg_overlap:.4f} | IoU Loss: {avg_iou_l:.4f}")

            metrics.update({
                "epoch": epoch + 1,
                "train/loss": avg_epoch_loss,
                "train/target_loss": avg_target,
                "train/left_loss": avg_left,
                "train/right_loss": avg_right,
                "train/overlap_loss": avg_overlap,
                "train/iou_loss": avg_iou_l,
                "train/iou": avg_iou,
                "train/learning_rate": optimizer.param_groups[0]['lr'],
            })

        # ============================================================
        # VALIDATION
        # ============================================================
        if eval_loader is not None:
            # Always evaluate character-level metrics
            char_loss_fn = SequentialCharacterLoss(
                focal_weight=20.0, overlap_weight=1.0, iou_loss_weight=1.0,
                iou_loss_type=args.iou_loss_type, line_overlap_weight=0.0
            )
            val_loss, val_iou, val_dice = evaluate(sam, eval_loader, char_loss_fn, device)
            print(f"  Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}")
            metrics.update({"val/loss": val_loss, "val/iou": val_iou, "val/dice": val_dice})

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                best_checkpoint_path = f"{args.checkpoint_dir}/best_model.pth"
                save_checkpoint(sam, optimizer, scheduler, epoch, best_val_iou, best_checkpoint_path)
                print(f"  New best model! Val IoU: {val_iou:.4f}")
                if args.use_wandb:
                    metrics["val/best_iou"] = best_val_iou

        if args.use_wandb:
            wandb.log(metrics)

        # ============================================================
        # VISUALIZATION
        # ============================================================
        if (epoch + 1) % args.log_images_every == 0:
            vis_loader = eval_loader if eval_loader else train_loader
            log_comparison_images(epoch + 1, sam, vis_loader, device, args)

        # Save checkpoint
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        if epoch % 10 == 0:
            save_checkpoint(
                sam, optimizer, scheduler, epoch, best_val_iou,
                f"{args.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth"
            )
        scheduler.step()

    # Final eval
    if eval_loader is not None and best_val_iou > 0:
        print("\n" + "="*50)
        print("Final Evaluation on Best Model")
        sam.load_state_dict(torch.load(f"{args.checkpoint_dir}/best_model.pth"))
        char_loss_fn = SequentialCharacterLoss(focal_weight=20.0, overlap_weight=1.0,
                                                iou_loss_weight=1.0, iou_loss_type=args.iou_loss_type)
        final_val_loss, final_val_iou, final_val_dice = evaluate(sam, eval_loader, char_loss_fn, device)
        print(f"Final Val Loss: {final_val_loss:.4f} | IoU: {final_val_iou:.4f} | Dice: {final_val_dice:.4f}")
        if args.use_wandb:
            wandb.log({"final/val_loss": final_val_loss,
                       "final/val_iou": final_val_iou,
                       "final/val_dice": final_val_dice})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', '-m', type=str, default='vit_b')
    parser.add_argument('--checkpoint', '-c', type=str, required=True)
    parser.add_argument('--train_index_file', '-t', type=str, required=True)
    parser.add_argument('--val_index_file', '-v', type=str, default=None)
    parser.add_argument('--device', '-d', type=str, default='cuda:0')
    parser.add_argument('--lr', '-l', type=float, default=1e-5)
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--comparison_images_dir', '-ci', type=str, default='./Comparision/run_0')
    parser.add_argument('--checkpoint_dir', '-cd', type=str, default='./Finetuned_Models/run_0')
    parser.add_argument('--run_name', '-r', type=str, default=None)
    parser.add_argument('--use_wandb', '-w', action='store_true')
    parser.add_argument('--log_images_every', '-li', type=int, default=1)
    parser.add_argument('--use_bbox_prompt', '-mp', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--eval_results_dir', type=str, default='./eval_results')
    parser.add_argument('--iou_loss_type', type=str, default='mse', choices=['mse', 'l1'])
    parser.add_argument('--use_legacy_loss', '-legacy', action='store_true',
                        help='Use old CombinedLoss training (no neighbors, single mask output)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--wandb_run_id', type=str, default=None, help='WandB run ID to resume')
    args = parser.parse_args()

    main(args)
