import torch
import numpy as np
import json
import random
import os
import cv2
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

'''
This is for handling giving the bounding box
of the binary mask as a prompt along with the centroid.
'''

# We will use SAM's own resizing utility, as it's a required dependency anyway.
# This avoids adding external libraries like Albumentations.
from segment_anything.utils.transforms import ResizeLongestSide

''' Changes to be made to the original SimpleSamDataset:
Here make the processing faster load the dataset into the memory for training SAM.
1. Load the masks, data from disk (np.load) on-the-fly (efficient).
2. Perform essential preprocessing (resize, pad, normalize) only.
3. Cache the changes into memory.
4. Get_item from the memory.
'''

def get_nearest_foreground_point_in_mask(mask, centroid):
    """
    Find the nearest foreground pixel to the centroid within the mask.
    
    Args:
        mask: Binary mask (H, W) with values 0 or 1
        centroid: [x, y] coordinates of the centroid
        
    Returns:
        [x, y] coordinates of nearest foreground pixel, or None if mask is empty
    """
    if mask.sum() == 0:
        return None
    
    # Get all foreground pixel coordinates
    y_coords, x_coords = np.where(mask > 0)
    foreground_points = np.stack([x_coords, y_coords], axis=1)  # [N, 2] in (x, y) format
    
    # Calculate distances from centroid to all foreground pixels
    centroid_arr = np.array([centroid[0], centroid[1]])
    distances = np.linalg.norm(foreground_points - centroid_arr, axis=1)
    
    # Find nearest point
    nearest_idx = np.argmin(distances)
    nearest_point = foreground_points[nearest_idx]
    
    return nearest_point

def get_bbox_from_connected_component_at_point(mask, point):
    """
    Extract bounding box from the connected component that contains the given point.
    
    Args:
        mask: Binary mask (H, W) with values 0 or 1
        point: [x, y] coordinates of the centroid point
        
    Returns:
        [x_min, y_min, x_max, y_max] or None if point is not on mask
    """
    if mask.sum() == 0:
        return None
    
    # Ensure mask is binary
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Get point coordinates
    x, y = int(round(point[0])), int(round(point[1]))
    
    # Check if point is within image bounds
    if x < 0 or y < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
        return None
    
    # Check if point is on the mask
    if binary_mask[y, x] == 0:
        return None
    
    # Find all connected components
    num_labels, labels = cv2.connectedComponents(binary_mask)
    
    # Get the label at the point location
    target_label = labels[y, x]
    
    if target_label == 0:  # Background
        return None
    
    # Create mask for only the connected component containing the point
    component_mask = (labels == target_label)
    
    # Get bounding box of this specific component
    y_indices, x_indices = np.where(component_mask)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

def get_bbox_from_mask(mask):
    """
    Extract bounding box from binary mask.
    Returns [x_min, y_min, x_max, y_max] or None if mask is empty.
    """
    if mask.sum() == 0:
        return None
    
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


class SimplePreprocessedSamDataset(Dataset):
    """
    A simplified PyTorch Dataset for loading preprocessed data for SAM fine-tuning.
    Loads character masks and prompt masks from NPZ files in the master index.
    """
    def __init__(self, index_file, image_size, pixel_mean, pixel_std, use_bbox_prompt=False):
        """
        Args:
            index_file (str): Path to the preprocessed master JSON index file.
            image_size (int): The target size for the images (e.g., 1024).
            pixel_mean (torch.Tensor): The mean values for image normalization.
            pixel_std (torch.Tensor): The standard deviation values for image normalization.
            use_bbox_prompt (bool): If True, load prompt bounding boxes from prompt_masks_npz_path.
        """
        try:
            with open(index_file, 'r') as f:
                self.index = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Index file not found at: {index_file}")
            
        self.image_size = image_size
        self.pixel_mean = pixel_mean.cpu()
        self.pixel_std = pixel_std.cpu()
        self.use_bbox_prompt = use_bbox_prompt
        
        # Use SAM's built-in transform for resizing
        self.transform = ResizeLongestSide(image_size)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        try:
            record = self.index[idx]

            # --- 1. Load data from disk ---
            selected_image_path = record['image_path']
            
            # Load the selected image
            image = np.array(Image.open(selected_image_path).convert("RGB"))
            original_image_size = image.shape[:2]  # (H, W)
            
            
            # ALWAYS load character-level masks (regardless of mode)
            masks_key = 'char_masks_npz' if 'char_masks_npz' in record else 'masks_npz_path'
            
            if masks_key not in record:
                raise ValueError(f"No character masks NPZ found in record at index {idx}")
            
            with np.load(record[masks_key]) as data:
                gt_masks = data['masks']  # Shape: [Num_Chars, H, W]
            
            # Load sampled points (character centroids)
            if 'sampled_points' in record:
                gt_points_np = np.array(record['sampled_points'])
            else:
                raise ValueError(f"No sampled_points found in record at index {idx}")
            
            # Load metadata
            line_numbers = record.get('line_numbers', None)
            # print("================DEBUGGIG LINE NUMBERS FROM THE JSON=========")
            # print(f"{selected_image_path}, {line_numbers}", flush=True)
            char_numbers = record.get('char_numbers', None)

            # ============ BACKGROUND HANDLING ============
            if line_numbers is not None:
                is_background = np.array(line_numbers) == -1
                
                # For background points, ensure their masks are zero
                for idx_mask, is_bg in enumerate(is_background):
                    if is_bg and idx_mask < len(gt_masks):
                        gt_masks[idx_mask] = np.zeros_like(gt_masks[idx_mask])
            
            # Ensure masks and points have same length
            if len(gt_masks) != len(gt_points_np):
                min_len = min(len(gt_masks), len(gt_points_np))
                print(f"WARNING: Record {idx} has {len(gt_masks)} masks but {len(gt_points_np)} points! Truncating to {min_len}")
                gt_masks = gt_masks[:min_len]
                gt_points_np = gt_points_np[:min_len]
                if line_numbers is not None:
                    line_numbers = line_numbers[:min_len]
                if char_numbers is not None:
                    char_numbers = char_numbers[:min_len]

            # --- 2. Load bbox prompts if enabled ---
            bbox_prompts = None
            if self.use_bbox_prompt and 'prompt_masks_npz_path' in record:
                try:
                    with np.load(record['prompt_masks_npz_path']) as data:
                        prompt_masks = data['masks'].astype(np.float32)
                        if prompt_masks.max() > 1.0:
                            prompt_masks = prompt_masks / 255.0
                    
                    bbox_list = []
                    for i, prompt_mask in enumerate(prompt_masks):
                        if i < len(gt_points_np):
                            centroid = gt_points_np[i]
                            binary_prompt_mask = (prompt_mask > 0.5).astype(np.uint8)
                            nearest_point = get_nearest_foreground_point_in_mask(binary_prompt_mask, centroid)
                            
                            if nearest_point is not None:
                                bbox = get_bbox_from_connected_component_at_point(binary_prompt_mask, nearest_point)
                            else:
                                bbox = None
                            
                            if bbox is not None:
                                bbox_list.append(bbox)
                            else:
                                bbox_list.append(np.zeros(4, dtype=np.float32))
                        else:
                            bbox_list.append(np.zeros(4, dtype=np.float32))
                    
                    bbox_prompts = np.array(bbox_list)
                    
                    if bbox_prompts.shape[0] != gt_masks.shape[0]:
                        if bbox_prompts.shape[0] < gt_masks.shape[0]:
                            padding = np.zeros((gt_masks.shape[0] - bbox_prompts.shape[0], 4), dtype=np.float32)
                            bbox_prompts = np.vstack([bbox_prompts, padding])
                        else:
                            bbox_prompts = bbox_prompts[:gt_masks.shape[0]]

                    if line_numbers is not None and bbox_prompts is not None:
                        is_background = np.array(line_numbers) == -1
                        for idx_bbox, is_bg in enumerate(is_background):
                            if is_bg and idx_bbox < len(bbox_prompts):
                                bbox_prompts[idx_bbox] = np.zeros(4, dtype=np.float32)
                                
                except Exception as e:
                    print(f"Warning: Could not load/process prompt masks for record {idx}: {e}")
                    bbox_prompts = None

            # --- 3. Load neighbor masks ---
            left_neighbor_masks = None
            right_neighbor_masks = None

            if 'left_neighbor_masks_npz_path' in record:
                left_npz_path = record['left_neighbor_masks_npz_path']
                if os.path.exists(left_npz_path):
                    left_data = np.load(left_npz_path)
                    left_neighbor_masks = left_data['masks']
                else:
                    print(f"Warning: Left neighbor masks not found: {left_npz_path}")

            if 'right_neighbor_masks_npz_path' in record:
                right_npz_path = record['right_neighbor_masks_npz_path']
                if os.path.exists(right_npz_path):
                    right_data = np.load(right_npz_path)
                    right_neighbor_masks = right_data['masks']
                else:
                    print(f"Warning: Right neighbor masks not found: {right_npz_path}")

            original_image_size = image.shape[:2]

            # --- 5. Transform data to SAM's coordinate system ---
            gt_masks_float = gt_masks.astype(np.float32) / 255.0
            
            transformed_image = self.transform.apply_image(image)
            image_tensor = torch.as_tensor(transformed_image, dtype=torch.float).permute(2, 0, 1)

            transformed_masks_list = [self.transform.apply_image(mask) for mask in gt_masks_float]
            gt_masks_1024_tensor = torch.stack([torch.as_tensor(m, dtype=torch.float) for m in transformed_masks_list])
            
            transformed_points_tensor = torch.as_tensor(
                self.transform.apply_coords(gt_points_np, original_image_size), 
                dtype=torch.float
            )
            
            # Transform bbox prompts
            bbox_prompts_256_tensor = None
            if bbox_prompts is not None:
                transformed_bboxes = []
                for bbox in bbox_prompts:
                    if bbox.sum() > 0:
                        bbox_2pts = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
                        transformed_2pts = self.transform.apply_coords(bbox_2pts, original_image_size)
                        transformed_bbox = np.array([
                            transformed_2pts[0, 0], transformed_2pts[0, 1],
                            transformed_2pts[1, 0], transformed_2pts[1, 1]
                        ])
                        transformed_bboxes.append(transformed_bbox)
                    else:
                        transformed_bboxes.append(np.zeros(4, dtype=np.float32))

                bbox_prompts_256_tensor = torch.tensor(np.array(transformed_bboxes), dtype=torch.float32)

            # Pad to 1024x1024
            h, w = image_tensor.shape[1:]
            pad_h = self.image_size - h
            pad_w = self.image_size - w
            
            padded_image = F.pad(image_tensor, (0, pad_w, 0, pad_h))
            padded_masks = F.pad(gt_masks_1024_tensor, (0, pad_w, 0, pad_h))

            normalized_image = (padded_image - self.pixel_mean) / self.pixel_std

            # Downsample character masks to 256x256
            gt_masks_256_tensor = F.interpolate(
                padded_masks.unsqueeze(1),
                (256, 256),
                mode='bilinear',
                align_corners=False
            ).squeeze(1)
            gt_masks_256_tensor = (gt_masks_256_tensor > 0.5).float()

            # Process neighbor masks
            left_neighbor_masks_256 = None
            right_neighbor_masks_256 = None

            if left_neighbor_masks is not None:
                left_masks_float = left_neighbor_masks.astype(np.float32) / 255.0
                transformed_left_masks_list = [self.transform.apply_image(mask) for mask in left_masks_float]
                left_masks_1024_tensor = torch.stack([torch.as_tensor(m, dtype=torch.float) for m in transformed_left_masks_list])
                padded_left_masks = F.pad(left_masks_1024_tensor, (0, pad_w, 0, pad_h))
                
                left_neighbor_masks_256 = F.interpolate(
                    padded_left_masks.unsqueeze(1),
                    (256, 256),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                left_neighbor_masks_256 = (left_neighbor_masks_256 > 0.5).float()

            if right_neighbor_masks is not None:
                right_masks_float = right_neighbor_masks.astype(np.float32) / 255.0
                transformed_right_masks_list = [self.transform.apply_image(mask) for mask in right_masks_float]
                right_masks_1024_tensor = torch.stack([torch.as_tensor(m, dtype=torch.float) for m in transformed_right_masks_list])
                padded_right_masks = F.pad(right_masks_1024_tensor, (0, pad_w, 0, pad_h))
                
                right_neighbor_masks_256 = F.interpolate(
                    padded_right_masks.unsqueeze(1),
                    (256, 256),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
                right_neighbor_masks_256 = (right_neighbor_masks_256 > 0.5).float()

            

            # ============ Build result dictionary ============
            result = {
                'image': normalized_image,
                'gt_masks': gt_masks_256_tensor,  # CHARACTER masks (always)
                'gt_points': transformed_points_tensor,
                'dataset_idx': idx
            }

            # Add optional fields
            if left_neighbor_masks_256 is not None:
                result['left_neighbor_masks'] = left_neighbor_masks_256
            if right_neighbor_masks_256 is not None:
                result['right_neighbor_masks'] = right_neighbor_masks_256
            
            if bbox_prompts_256_tensor is not None:
                result['bbox_prompts'] = bbox_prompts_256_tensor
            
            # Handle line_numbers
            if line_numbers is not None:
                line_numbers_clean = [ln if ln is not None else -1 for ln in line_numbers]
                result['line_numbers'] = torch.tensor(line_numbers_clean, dtype=torch.long)
            else:
                result['line_numbers'] = None
            
            # Add metadata
            result['num_chars'] = len(gt_masks_256_tensor)
            if 'char_numbers' in record:
                result['char_numbers'] = record['char_numbers']
            

            
            return result

        except Exception as e:
            print(f"\n--- DataLoader Error ---")
            print(f"Error loading data at index {idx}")
            if idx < len(self.index):
                print(f"Record keys: {list(self.index[idx].keys())}")
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc()
            return self.__getitem__((idx + 1) % len(self))


if __name__ == '__main__':
    # SAM ViT-B
    IMG_SIZE = 1024
    PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
    PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
    
    # Update this path to your actual master index file
    PREPROCESSED_INDEX_FILE = './preprocessed_data_neighbors/master_index_from_associations_end_train.json'
    
    mask_type = "CHARACTER"
    print(f"\nCreating dataset instance with {mask_type} masks...")
    dataset = SimplePreprocessedSamDataset(
        index_file=PREPROCESSED_INDEX_FILE,
        image_size=IMG_SIZE,
        pixel_mean=PIXEL_MEAN,
        pixel_std=PIXEL_STD,
    )
    print(f"Dataset created with {len(dataset)} samples.")

    # ======================================================================
    # IMPORTANT: Use batch_size=1
    # The default collate function cannot batch items with a variable
    # number of masks/points.
    # ======================================================================
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    print("\nFetching one batch (batch_size=1) to verify...")
    try:
        batch = next(iter(data_loader))
        
        print("Batch loaded successfully!")
        # Note the shape of the image tensor now includes the batch dimension of 1
        print(f"Image tensor shape: {batch['image'].shape}") 
        print(f"Masks tensor shape: {batch['gt_masks'].shape}")
        print(f"Points tensor shape: {batch['gt_points'].shape}")
        
        # The batch dimension for masks/points is also 1, so we access with [0]
        num_masks = batch['gt_masks'].shape[1]
        print(f"Number of characters in this image: {num_masks}")
        
    except Exception as e:
        print(f"An error occurred while fetching a batch: {e}")
