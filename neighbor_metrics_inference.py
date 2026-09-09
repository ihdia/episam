"""
Inference script that evaluates:
1. Main character mask predictions (Dice, IoU)
2. Left neighbor mask predictions (Dice, IoU)
3. Right neighbor mask predictions (Dice, IoU)
"""

import torch
import argparse
import numpy as np
from tqdm import tqdm
import os
from segment_anything import sam_model_registry
from data_loader import SimplePreprocessedSamDataset
from train_centroid_plus_bbox_prompt import custom_collate_fn


def dice_score(pred, target):
    """Compute Dice score between prediction and target masks."""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection / (pred.sum() + target.sum() + 1e-6)).item()


def iou_score(pred, target):
    """Compute IoU score between prediction and target masks."""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    union = ((pred + target) > 0).sum()
    return (intersection / (union + 1e-6)).item()


def evaluate_model(sam, loader, device, args):
    """
    Evaluate model on dataset and compute metrics for:
    - Main character masks
    - Left neighbor masks  
    - Right neighbor masks
    """
    sam.eval()
    
    # Metrics storage
    char_dice_scores = []
    char_iou_scores = []
    left_dice_scores = []
    left_iou_scores = []
    right_dice_scores = []
    right_iou_scores = []
    
    print(f"Evaluating on {len(loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            images = batch['image'].to(device)
            gt_points = batch['gt_points']
            gt_masks = batch['masks'].to(device) if 'masks' in batch else None
            left_neighbor_masks = batch.get('left_neighbor_masks', None)
            right_neighbor_masks = batch.get('right_neighbor_masks', None)
            bbox_prompts = batch.get('bbox_prompts', None)
            
            if left_neighbor_masks is not None:
                left_neighbor_masks = left_neighbor_masks.to(device)
            if right_neighbor_masks is not None:
                right_neighbor_masks = right_neighbor_masks.to(device)
            
            # Get image embeddings
            image_embeddings = sam.image_encoder(images)
            
            # Process each image in batch
            for i in range(images.shape[0]):
                points_i = gt_points[i]
                bbox_prompts_i = bbox_prompts[i] if bbox_prompts is not None else None
                
                # Process each character point
                for j in range(len(points_i)):
                    # Prepare prompts
                    prompt_point = points_i[j].unsqueeze(0).unsqueeze(0).to(device)
                    prompt_label = torch.tensor([[1]], device=device)
                    
                    bbox_input = None
                    if bbox_prompts_i is not None and j < len(bbox_prompts_i):
                        bbox_input = bbox_prompts_i[j].unsqueeze(0).to(device)
                    
                    # Get sparse and dense embeddings
                    sparse_emb, dense_emb = sam.prompt_encoder(
                        points=(prompt_point, prompt_label),
                        boxes=bbox_input,
                        masks=None
                    )
                    
                    # Decode masks
                    pred_masks, pred_iou, _ = sam.mask_decoder(
                        image_embeddings=image_embeddings[i].unsqueeze(0),
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=True,
                    )
                    
                    # Extract predictions
                    # pred_masks shape: [1, 5, H, W]
                    # Indices 0-2: main character (3 masks)
                    # Index 3: left neighbor
                    # Index 4: right neighbor
                    target_preds = pred_masks[:, 0:3, :, :]  # [1, 3, H, W]
                    left_pred = pred_masks[:, 3, :, :]      # [1, H, W]
                    right_pred = pred_masks[:, 4, :, :]     # [1, H, W]
                    
                    # Select best main character mask (highest mean score)
                    dice_scores_char = []
                    for k in range(3):
                        dice_scores_char.append(torch.sigmoid(target_preds[:, k, :, :]).mean())
                    best_idx = torch.argmax(torch.tensor(dice_scores_char))
                    best_char_pred = target_preds[:, best_idx, :, :]  # [1, H, W]
                    
                    # Upsample predictions to original size
                    def upsample(mask):
                        upsampled = sam.postprocess_masks(
                            mask.unsqueeze(1),
                            input_size=images[i].shape[-2:],
                            original_size=gt_masks[i, j].shape
                        )
                        return torch.sigmoid(upsampled.squeeze())
                    
                    best_char_pred = upsample(best_char_pred)
                    left_pred = upsample(left_pred)
                    right_pred = upsample(right_pred)
                    
                    # Get ground truth masks
                    gt_char_mask = gt_masks[i, j]
                    
                    # Compute main character metrics
                    char_dice = dice_score(best_char_pred, gt_char_mask)
                    char_iou = iou_score(best_char_pred, gt_char_mask)
                    char_dice_scores.append(char_dice)
                    char_iou_scores.append(char_iou)
                    
                    # Compute left neighbor metrics (if available)
                    if left_neighbor_masks is not None:
                        gt_left_mask = left_neighbor_masks[i, j]
                        left_dice = dice_score(left_pred, gt_left_mask)
                        left_iou = iou_score(left_pred, gt_left_mask)
                        left_dice_scores.append(left_dice)
                        left_iou_scores.append(left_iou)
                    
                    # Compute right neighbor metrics (if available)
                    if right_neighbor_masks is not None:
                        gt_right_mask = right_neighbor_masks[i, j]
                        right_dice = dice_score(right_pred, gt_right_mask)
                        right_iou = iou_score(right_pred, gt_right_mask)
                        right_dice_scores.append(right_dice)
                        right_iou_scores.append(right_iou)
            
            # Optional: limit evaluation samples
            if args.max_samples and batch_idx * args.batch_size >= args.max_samples:
                break
    
    # Print results
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*70}")
    
    print(f"\n--- Main Character Masks ---")
    print(f"Samples evaluated: {len(char_dice_scores)}")
    print(f"Mean Dice: {np.mean(char_dice_scores):.4f} ± {np.std(char_dice_scores):.4f}")
    print(f"Mean IoU:  {np.mean(char_iou_scores):.4f} ± {np.std(char_iou_scores):.4f}")
    print(f"Median Dice: {np.median(char_dice_scores):.4f}")
    print(f"Median IoU:  {np.median(char_iou_scores):.4f}")
    
    if left_dice_scores:
        print(f"\n--- Left Neighbor Masks ---")
        print(f"Samples evaluated: {len(left_dice_scores)}")
        print(f"Mean Dice: {np.mean(left_dice_scores):.4f} ± {np.std(left_dice_scores):.4f}")
        print(f"Mean IoU:  {np.mean(left_iou_scores):.4f} ± {np.std(left_iou_scores):.4f}")
        print(f"Median Dice: {np.median(left_dice_scores):.4f}")
        print(f"Median IoU:  {np.median(left_iou_scores):.4f}")
    
    if right_dice_scores:
        print(f"\n--- Right Neighbor Masks ---")
        print(f"Samples evaluated: {len(right_dice_scores)}")
        print(f"Mean Dice: {np.mean(right_dice_scores):.4f} ± {np.std(right_dice_scores):.4f}")
        print(f"Mean IoU:  {np.mean(right_iou_scores):.4f} ± {np.std(right_iou_scores):.4f}")
        print(f"Median Dice: {np.median(right_dice_scores):.4f}")
        print(f"Median IoU:  {np.median(right_iou_scores):.4f}")
    
    print(f"{'='*70}\n")
    
    # Save results to file
    if args.output_file:
        results = {
            'char_dice': char_dice_scores,
            'char_iou': char_iou_scores,
            'left_dice': left_dice_scores,
            'left_iou': left_iou_scores,
            'right_dice': right_dice_scores,
            'right_iou': right_iou_scores,
        }
        np.savez(args.output_file, **results)
        print(f"Results saved to {args.output_file}")


def main(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device)
    
    # Setup dataset
    print(f"Loading dataset from {args.index_file}")
    dataset = SimplePreprocessedSamDataset(
        index_file=args.index_file,
        image_size=sam.image_encoder.img_size,
        pixel_mean=sam.pixel_mean,
        pixel_std=sam.pixel_std,
        use_line_masks=False,
        use_bbox_prompt=args.use_bbox_prompt,
        use_neighbor_masks=True,  # Enable neighbor masks
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: custom_collate_fn(x),
        num_workers=args.num_workers
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    
    # Run evaluation
    evaluate_model(sam, loader, device, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate SAM model on character and neighbor mask predictions'
    )
    parser.add_argument('--model_type', '-m', type=str, default='vit_b',
                        help='SAM model type')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Path to fine-tuned SAM checkpoint')
    parser.add_argument('--index_file', '-i', type=str, required=True,
                        help='Path to JSON index file')
    parser.add_argument('--device', '-d', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--batch_size', '-b', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--use_bbox_prompt', action='store_true',
                        help='Use bounding box prompts')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (None = all)')
    parser.add_argument('--output_file', '-o', type=str, default=None,
                        help='Path to save detailed results (NPZ format)')
    
    args = parser.parse_args()
    main(args)