import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(preds, targets, smooth=1.0):
    preds = torch.sigmoid(preds)
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return 1 - dice

def focal_loss(preds, targets, alpha=0.25, gamma=2.0):
    """Proper focal loss implementation."""
    preds = preds.view(-1)
    targets = targets.view(-1)
    bce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
    p = torch.sigmoid(preds)
    pt = p * targets + (1 - p) * (1 - targets)
    focal_term = alpha * (1 - pt) ** gamma
    return (focal_term * bce_loss).mean()

def iou_loss(pred_iou, true_iou, loss_type='mse'):
    if loss_type == 'mse':
        return F.mse_loss(pred_iou, true_iou)
    elif loss_type == 'l1':
        return F.l1_loss(pred_iou, true_iou)
    else:
        raise ValueError(f"Unknown IoU loss type: {loss_type}")

def overlap_penalty(target_pred, neighbor_pred):
    target_prob = torch.sigmoid(target_pred)
    neighbor_prob = torch.sigmoid(neighbor_pred)
    overlap = (target_prob * neighbor_prob).mean()
    return overlap

class SequentialCharacterLoss(torch.nn.Module):
    def __init__(self, focal_weight=20.0, overlap_weight=1.0, iou_loss_weight=1.0,
                 iou_loss_type='mse', line_overlap_weight=0.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.overlap_weight = overlap_weight
        self.iou_loss_weight = iou_loss_weight
        self.iou_loss_type = iou_loss_type
        self.line_overlap_weight = line_overlap_weight

    def forward(self, pred_masks, target_mask, left_neighbor_mask, right_neighbor_mask,
                pred_iou=None,
                # Optional line penalty args (used when train_from_scratch_with_lines=True)
                char_line_num=None, gt_line_masks=None, line_numbers_available=None):
        """
        Args:
            pred_masks: [B, 5, H, W] - 0-2 target, 3 left, 4 right
            target_mask, left_neighbor_mask, right_neighbor_mask: [B, 1, H, W]
            pred_iou: [B, num_masks] optional
            char_line_num: int - line number this character belongs to
            gt_line_masks: [num_lines, 256, 256] tensor - GT line masks on device
            line_numbers_available: list of int - line numbers for gt_line_masks indices
        """
        target_preds = pred_masks[:, 0:3, :, :]
        left_neighbor_pred = pred_masks[:, 3:4, :, :]
        right_neighbor_pred = pred_masks[:, 4:5, :, :]

        # ===== Step A: Target Tokens - Winner-Takes-All =====
        target_losses = []
        for i in range(3):
            target_pred_i = target_preds[:, i:i+1, :, :]
            loss_i = self.focal_weight * focal_loss(target_pred_i, target_mask) + \
                     dice_loss(target_pred_i, target_mask)
            target_losses.append(loss_i)

        target_losses_tensor = torch.stack(target_losses)
        best_idx = torch.argmin(target_losses_tensor)
        target_loss = target_losses_tensor[best_idx]
        best_target_pred = target_preds[:, best_idx:best_idx+1, :, :]

        # ===== IoU Computation =====
        with torch.no_grad():
            target_preds_binary = (torch.sigmoid(target_preds) > 0.5).float()
            target_mask_expanded = target_mask.expand(-1, 3, -1, -1)
            intersection = (target_preds_binary * target_mask_expanded).sum(dim=(2, 3))
            union = ((target_preds_binary + target_mask_expanded) > 0).float().sum(dim=(2, 3))
            target_ious = intersection / (union + 1e-6)

            left_pred_binary = (torch.sigmoid(left_neighbor_pred) > 0.5).float()
            left_intersection = (left_pred_binary * left_neighbor_mask).sum(dim=(2, 3))
            left_union = ((left_pred_binary + left_neighbor_mask) > 0).float().sum(dim=(2, 3))
            left_iou = left_intersection / (left_union + 1e-6)

            right_pred_binary = (torch.sigmoid(right_neighbor_pred) > 0.5).float()
            right_intersection = (right_pred_binary * right_neighbor_mask).sum(dim=(2, 3))
            right_union = ((right_pred_binary + right_neighbor_mask) > 0).float().sum(dim=(2, 3))
            right_iou = right_intersection / (right_union + 1e-6)

        # ===== Step B: Neighbor Tokens =====
        left_loss = self.focal_weight * focal_loss(left_neighbor_pred, left_neighbor_mask) + \
                    dice_loss(left_neighbor_pred, left_neighbor_mask)
        right_loss = self.focal_weight * focal_loss(right_neighbor_pred, right_neighbor_mask) + \
                     dice_loss(right_neighbor_pred, right_neighbor_mask)

        # ===== Step C: Overlap Penalty =====
        overlap_left = overlap_penalty(best_target_pred, left_neighbor_pred)
        overlap_right = overlap_penalty(best_target_pred, right_neighbor_pred)
        overlap_loss = self.overlap_weight * (overlap_left + overlap_right)

        # ===== Step D: IoU Prediction Loss =====
        iou_prediction_loss = torch.tensor(0.0, device=pred_masks.device)
        if pred_iou is not None:
            pred_iou_probs = torch.sigmoid(pred_iou)
            true_ious = torch.zeros_like(pred_iou_probs)
            true_ious[:, 0:3] = target_ious
            iou_prediction_loss = self.iou_loss_weight * iou_loss(pred_iou_probs, true_ious, self.iou_loss_type)



        # ===== Total Loss =====
        total_loss = target_loss + left_loss + right_loss + overlap_loss + iou_prediction_loss 

        loss_dict = {
            'total_loss': total_loss.item(),
            'target_loss': target_loss.item(),
            'left_loss': left_loss.item(),
            'right_loss': right_loss.item(),
            'overlap_loss': overlap_loss.item(),
            'iou_loss': iou_prediction_loss.item(),
            'best_target_idx': best_idx.item()
        }

        return total_loss, loss_dict



def compute_char_vs_other_lines_penalty(pred_char_mask, char_line_num, gt_line_masks,
                                         line_numbers_available, line_overlap_weight=1.0):
    """
    Match the soft neighbor overlap approach: use sigmoid probs + mean,
    instead of hard normalization.
    """
    if line_overlap_weight == 0 or gt_line_masks is None or len(gt_line_masks) < 2:
        return torch.tensor(0.0, device=pred_char_mask.device)

    pred_prob = torch.sigmoid(pred_char_mask)  # Soft probabilities
    total_penalty = torch.tensor(0.0, device=pred_char_mask.device)
    num_other_lines = 0

    for line_idx, line_num in enumerate(line_numbers_available):
        if line_num == char_line_num:
            continue
        other_line_mask = gt_line_masks[line_idx].float()  # Ensure float
        
        # Use element-wise product + mean (like neighbor overlap)
        overlap = (pred_prob * other_line_mask).mean()
        total_penalty = total_penalty + overlap
        num_other_lines += 1

    if num_other_lines > 0:
        avg_penalty = line_overlap_weight * total_penalty / num_other_lines
    else:
        avg_penalty = torch.tensor(0.0, device=pred_char_mask.device)

    return avg_penalty

class CombinedLoss(torch.nn.Module):
    """Legacy loss for backward compatibility."""
    def __init__(self, focal_weight=20.0):
        super().__init__()
        self.focal_weight = focal_weight

    def forward(self, preds, targets):
        return self.focal_weight * focal_loss(preds, targets) + dice_loss(preds, targets)
