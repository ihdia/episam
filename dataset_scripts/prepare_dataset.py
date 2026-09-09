"""
Create dataset from associated centroids in the format similar to sample_char_centroids_with_lines.py

This script:
1. Loads associated centroids (predicted points matched with GT)
2. Creates character masks from GT polygons
3. Creates prompt masks from binarized images within convex hull regions
4. Creates line masks from GT polygons grouped by line_number
5. Saves in NPZ format similar to sample_char_centroids_with_lines.py
6. Creates visualizations
"""

import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import cv2
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import glob
from create_line_masks_improved import create_line_mask_from_centers as improved_create_line_mask


def find_binarized_image(binarized_dir, inscription_name):
    """
    Recursively search for binarized image matching the inscription name.
    
    Args:
        binarized_dir: Root directory to search for binarized images
        inscription_name: Name of the inscription to find
    
    Returns:
        Path to binarized image or None if not found
    """
    if binarized_dir is None:
        return None
        
    binarized_path = Path(binarized_dir)
    
    # Common patterns to search for
    patterns = [
        f"**/{inscription_name}.png",
        f"**/{inscription_name}.jpg",
        f"**/{inscription_name}_binarized.png",
        f"**/{inscription_name}_binary.png",
        f"**/*{inscription_name}*.png",
    ]
    
    for pattern in patterns:
        matches = list(binarized_path.glob(pattern))
        if matches:
            return matches[0]
    
    return None



def debug_log_line_assignments(gt_polygons, inscription_name):
    """
    Log detailed information about character-to-line assignments.
    """
    print(f"\n{'='*80}")
    print(f"DEBUG: Line Assignments for {inscription_name}")
    print(f"{'='*80}")
    
    # Group characters by line
    line_groups = {}
    for img_key, char_data in gt_polygons.items():
        line_num = char_data.get('line_number', None)
        char_num = char_data.get('char_number', None)
        char_name = char_data.get('char_name', 'Unknown')
        
        if line_num is not None:
            if line_num not in line_groups:
                line_groups[line_num] = []
            line_groups[line_num].append({
                'img_key': img_key,
                'char_num': char_num,
                'char_name': char_name
            })
    
    # Print line-by-line summary
    for line_num in sorted(line_groups.keys()):
        chars = line_groups[line_num]
        char_nums = [c['char_num'] for c in chars]
        char_names = [c['char_name'] for c in chars]
        # print(f"\nLine {line_num}: {len(chars)} characters")
        # print(f"  Char Numbers: {char_nums}")
        # print(f"  Char Names: {char_names[:10]}..." if len(char_names) > 10 else f"  Char Names: {char_names}")
    
    print(f"\n{'='*80}\n")
    
    return line_groups

def create_mask_from_polygon(polygon, image_shape):
    """
    Create binary mask from polygon.
    
    Args:
        polygon: List of [x, y] points
        image_shape: (height, width) of the image
    
    Returns:
        np.ndarray: Binary mask (H, W)
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    polygon_np = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [polygon_np], 255)
    return mask


def create_prompt_mask_from_binarized(binarized_np, polygon, image_shape):
    """
    Create a prompt mask by extracting the foreground from binarized image
    within the convex hull bounding box region.
    
    Args:
        binarized_np: Numpy array of binarized image (H, W) grayscale
        polygon: List of [x, y] coordinates defining the convex hull
        image_shape: (height, width) tuple
    
    Returns:
        np.ndarray: Prompt mask (H, W)
    """
    h, w = image_shape
    
    if polygon is None or len(polygon) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    
    # Get bounding box from polygon
    polygon_np = np.array(polygon)
    x_min = max(0, int(np.min(polygon_np[:, 0])))
    x_max = min(w, int(np.max(polygon_np[:, 0])))
    y_min = max(0, int(np.min(polygon_np[:, 1])))
    y_max = min(h, int(np.max(polygon_np[:, 1])))
    
    # Create output mask (same size as original image)
    prompt_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Extract the region from binarized image
    region = binarized_np[y_min:y_max, x_min:x_max]
    
    # Threshold to get binary foreground
    # Assuming foreground is white (255) in binarized image
    # Adjust threshold if your binarization uses different convention
    foreground = region > 127
    
    # Place the foreground region in the output mask
    prompt_mask[y_min:y_max, x_min:x_max] = (foreground * 255).astype(np.uint8)
    
    return prompt_mask


def load_gt_polygons_for_image(convex_hulls_dir, inscription_name, split):
    """
    Load GT character polygons for an inscription.
    
    Args:
        convex_hulls_dir: Path to convex_hulls_data directory
        inscription_name: Name of the inscription
        split: 'train' or 'val'
    
    Returns:
        dict: GT polygons data or None if not found
    """
    gt_path = os.path.join(convex_hulls_dir, split, inscription_name, 'character_polygons.json')
    
    if not os.path.exists(gt_path):
        return None
    
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    
    return gt_data


def find_augmented_image_versions(images_dir, base_image_name):
    """
    Find all color-augmented versions of an image.
    
    Args:
        images_dir: Directory containing augmented images
        base_image_name: Original image name (e.g., 'img.png')
    
    Returns:
        List of paths to all color variants (including original)
    """
    from glob import glob
    import os
    
    stem = base_image_name.replace('.png', '')
    base_path = os.path.join(images_dir, base_image_name)
    
    # Pattern matches: img.png, img___0.png, img___1.png, etc.
    pattern = os.path.join(images_dir, f"{stem}*.png")
    all_versions = sorted(glob(pattern))
    
    # Ensure original is first (if it exists)
    if base_path in all_versions:
        all_versions.remove(base_path)
        all_versions.insert(0, base_path)
    
    return [os.path.abspath(p) for p in all_versions]



def process_association_file(association_path, convex_hulls_dir, images_dir, split,
                            output_char_masks_dir, output_line_masks_dir,
                            output_prompt_masks_dir, output_neighbor_masks_dir,  
                            save_visualizations, viz_output_dir,
                            binarized_dir=None,
                            use_color_augmented_images=False):
    """
    Process a single association JSON file and create dataset entry.
    
    Args:
        association_path: Path to association JSON file
        convex_hulls_dir: Path to convex_hulls_data directory
        images_dir: Path to whole_images directory
        split: 'train' or 'val'
        output_char_masks_dir: Directory to save character masks NPZ
        output_line_masks_dir: Directory to save line masks NPZ
        output_prompt_masks_dir: Directory to save prompt masks NPZ
        save_visualizations: Whether to save visualizations
        viz_output_dir: Directory to save visualizations
        binarized_dir: Directory containing binarized images (searched recursively)
    
    Returns:
        dict: Record with all information or None if failed
    """
    # Load association data
    with open(association_path, 'r') as f:
        association_data = json.load(f)
    
    original_image_name = association_data['original_image_name']
    inscription_name = original_image_name.replace('.png', '')
    
    # Load original image
    image_path = os.path.join(images_dir, split, original_image_name)
    if not os.path.exists(image_path):
        print(f"  Warning: Image not found: {image_path}")
        return None

    # NEW: Find all color-augmented versions
    augmented_image_paths = None
    if use_color_augmented_images:
        augmented_dir = os.path.join(images_dir, split)  # Adjust if augmented images are elsewhere
        augmented_image_paths = find_augmented_image_versions(augmented_dir, original_image_name)
        
        if len(augmented_image_paths) == 0:
            print(f"  Warning: No augmented versions found for {original_image_name}, using original only")
            augmented_image_paths = [image_path]
    else:
        augmented_image_paths = [image_path]  # Only original
    
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    image_shape = (h, w)
    
    # Load binarized image if available
    binarized_np = None
    if binarized_dir:
        binarized_path = find_binarized_image(binarized_dir, inscription_name)
        if binarized_path:
            binarized_img = Image.open(binarized_path).convert('L')
            # Resize if dimensions don't match
            if binarized_img.size != (w, h):
                binarized_img = binarized_img.resize((w, h), Image.NEAREST)
            binarized_np = np.array(binarized_img)
        else:
            print(f"  Warning: Binarized image not found for {inscription_name}")
    
    # Load GT polygons
    gt_polygons = load_gt_polygons_for_image(convex_hulls_dir, inscription_name, split)
    if gt_polygons is None:
        print(f"  Warning: GT polygons not found for {inscription_name}")
        return None

    # NEW: Debug log line assignments
    # line_groups_debug = debug_log_line_assignments(gt_polygons, inscription_name) 

    # Process predicted components (foreground)
    predicted_components = association_data['predicted_components']
    background_components = association_data.get('background_components', [])
    
    # Collect sampled points and their metadata
    sampled_points = []
    line_numbers = []
    char_names = []
    char_numbers = []
    gt_image_keys = []
    char_masks_list = []
    prompt_masks_list = []
    
    # Group GT polygons by gt_image_key for efficient lookup
    gt_polygons_by_key = {}
    for img_key, char_data in gt_polygons.items():
        # Skip if char_number is None
        if char_data.get('char_number') is None:
            print(f"  Skipping GT polygon with None char_number for image key: {img_key}")
            continue
        gt_polygons_by_key[img_key] = char_data
    
    # Process each predicted component
    for comp in predicted_components:
        # Use centroid_scaled (in original image coordinates) instead of centroid (in predicted mask coordinates)
        centroid = comp.get('centroid_scaled', comp['centroid'])
        match = comp['matched_gt']
        
        if match and match['match_quality'] == 'inside':
            # Get the GT polygon for this match
            gt_key = match['gt_image_key']
            
            if gt_key in gt_polygons_by_key:
                char_data = gt_polygons_by_key[gt_key]
                polygon = char_data.get('polygon_enhanced') or char_data.get('polygon_original')
                
                if polygon:
                    # Create GT mask from polygon
                    char_mask = create_mask_from_polygon(polygon, image_shape)
                    char_masks_list.append(char_mask)
                    
                    # Create prompt mask from binarized image
                    if binarized_np is not None:
                        prompt_mask = create_prompt_mask_from_binarized(
                            binarized_np, polygon, image_shape
                        )
                    else:
                        # If no binarized image, use empty mask
                        prompt_mask = np.zeros(image_shape, dtype=np.uint8)
                    prompt_masks_list.append(prompt_mask)
                    
                    # Store metadata
                    sampled_points.append(centroid)
                    line_numbers.append(match['line_number'])
                    char_names.append(match['char_name'])
                    char_numbers.append(match['char_number'])
                    gt_image_keys.append(gt_key)
    
    # Process background components (add as separate entries with line_number=-1)
    background_masks_list = []
    background_prompt_masks_list = []
    # Only add background points for training split
    if split == "train":
        for bg_comp in background_components:
            centroid = bg_comp.get('centroid_scaled', bg_comp['centroid'])
            match = bg_comp['matched_gt']
            bg_mask = np.zeros(image_shape, dtype=np.uint8)
            background_masks_list.append(bg_mask)
            background_prompt_masks_list.append(bg_mask.copy())
            sampled_points.append(centroid)
            line_numbers.append(-1)
            char_names.append(None)
            char_numbers.append(-1)
            gt_image_keys.append(None)

    # Combine foreground and background masks
    all_char_masks = char_masks_list + background_masks_list
    all_prompt_masks = prompt_masks_list + background_prompt_masks_list
    
    if not all_char_masks:
        print(f"  Warning: No character masks created for {inscription_name}")
        return None
    
    char_masks_array = np.stack(all_char_masks, axis=0)
    prompt_masks_array = np.stack(all_prompt_masks, axis=0)
    
    # Create neighbor masks (left and right) for each character
    left_neighbor_masks_list = []
    right_neighbor_masks_list = []

    for comp in predicted_components:
        match = comp['matched_gt']
        
        if match and match['match_quality'] == 'inside':
            gt_key = match['gt_image_key']
            
            if gt_key in gt_polygons_by_key:
                char_data = gt_polygons_by_key[gt_key]
                
                # Left neighbor mask
                left_polygon = char_data.get('left_neighbor_polygon', [])
                if left_polygon and len(left_polygon) > 0:
                    left_mask = create_mask_from_polygon(left_polygon, image_shape)
                else:
                    left_mask = np.zeros(image_shape, dtype=np.uint8)
                left_neighbor_masks_list.append(left_mask)
                
                # Right neighbor mask
                right_polygon = char_data.get('right_neighbor_polygon', [])
                if right_polygon and len(right_polygon) > 0:
                    right_mask = create_mask_from_polygon(right_polygon, image_shape)
                else:
                    right_mask = np.zeros(image_shape, dtype=np.uint8)
                right_neighbor_masks_list.append(right_mask)

    # Add empty neighbor masks for background components
    for bg_comp in background_components:
        left_neighbor_masks_list.append(np.zeros(image_shape, dtype=np.uint8))
        right_neighbor_masks_list.append(np.zeros(image_shape, dtype=np.uint8))

    print(f"\n{'='*60}")
    print(f"Sampled Points Analysis for {inscription_name}")
    print(f"{'='*60}")
    print(f"Total sampled points: {len(sampled_points)}")
    print(f"Foreground points: {sum(1 for ln in line_numbers if ln is not None and ln >= 0)}")
    print(f"Background points: {sum(1 for ln in line_numbers if ln == -1)}")
    
    # # Count points per line
    # line_point_counts = {}
    # for ln in line_numbers:
    #     if ln >= 0:
    #         line_point_counts[ln] = line_point_counts.get(ln, 0) + 1
    
    # print(f"\nPoints per line:")
    # for line_num in sorted(line_point_counts.keys()):
    #     print(f"  Line {line_num}: {line_point_counts[line_num]} points")
    
    # Show first 10 point assignments
    # print(f"\nFirst 10 point assignments:")
    # for i in range(min(10, len(sampled_points))):
    #     print(f"  Point {i}: Line={line_numbers[i]}, CharNum={char_numbers[i]}, CharName={char_names[i]}")
    # print(f"{'='*60}\n")

    # Create line masks from GT polygons grouped by line_number
    # line_masks_dict = {}
    # for img_key, char_data in gt_polygons.items():
    #     line_num = char_data.get('line_number', 0)
    #     polygon = char_data.get('polygon_enhanced') or char_data.get('polygon_original')
        
    #     # Skip if line_num is None or negative (background)
    #     if polygon and line_num is not None and line_num >= 0:
    #         if line_num not in line_masks_dict:
    #             line_masks_dict[line_num] = np.zeros(image_shape, dtype=np.uint8)
            
    #         # Add this character's polygon to the line mask
    #         polygon_np = np.array(polygon, dtype=np.int32)
    #         cv2.fillPoly(line_masks_dict[line_num], [polygon_np], 255)
    
    # # Convert line masks dict to array
    # unique_lines = sorted(line_masks_dict.keys())
    # line_masks_list = [line_masks_dict[line_num] for line_num in unique_lines]
    line_masks_dict = {}
    line_char_masks = {}
    unique_lines = []
    line_masks_list = []
    line_char_mapping = {}
    
    for img_key, char_data in gt_polygons.items():
        line_num = char_data.get('line_number', 0)
        char_num = char_data.get('char_number', None)
        char_name = char_data.get('char_name', 'Unknown')
        polygon = char_data.get('polygon_enhanced') or char_data.get('polygon_original')
        
        if polygon and line_num is not None and line_num >= 0:
            mask = create_mask_from_polygon(polygon, image_shape)
            if line_num not in line_char_masks:
                line_char_masks[line_num] = []
                line_char_mapping[line_num] = []
            line_char_masks[line_num].append(mask)
            line_char_mapping[line_num].append({
                'img_key': img_key,
                'char_num': char_num,
                'char_name': char_name
            })
            
            # Log detailed assignment for verification
            # if len(line_char_masks[line_num]) <= 5:  # Log first 5 chars per line
            #     print(f"    Adding char to Line {line_num}: img_key={img_key}, char_num={char_num}, char_name={char_name}, mask_pixels={mask.sum()}")

     # Debug: Print line mask composition
    # print(f"\n{'='*60}")
    # print(f"Line Mask Composition for {inscription_name}")
    # print(f"{'='*60}")
    
    if line_char_masks:
        unique_lines = sorted(line_char_masks.keys())
        for line_num in unique_lines:
            char_masks = line_char_masks[line_num]
            char_info = line_char_mapping[line_num]
            
            # print(f"\nLine {line_num}:")
            # print(f"  Number of character masks: {len(char_masks)}")
            # print(f"  Character numbers: {[c['char_num'] for c in char_info]}")
            # print(f"  Character names: {[c['char_name'] for c in char_info][:10]}...")
            
            # Create improved line mask
            improved_mask = improved_create_line_mask(
                char_masks, image_shape, 
                alpha=0.02, 
                size_multiplier=1.0, 
                use_shapely=True
            )
            line_masks_list.append(improved_mask)
            line_masks_dict[line_num] = improved_mask
            
            # Check mask coverage
            total_char_pixels = sum(mask.sum() for mask in char_masks)
            line_mask_pixels = improved_mask.sum()
            # print(f"  Total char pixels: {total_char_pixels}, Line mask pixels: {line_mask_pixels}")
    
    # print(f"{'='*60}\n")
    
    if line_char_masks:
        unique_lines = sorted(line_char_masks.keys())
        for line_num in unique_lines:
            char_masks = line_char_masks[line_num]
            # Use improved method from create_line_masks_improved
            improved_mask = improved_create_line_mask(
                char_masks, image_shape, 
                alpha=0.02, 
                size_multiplier=1.0, 
                use_shapely=True
            )
            line_masks_list.append(improved_mask)
            line_masks_dict[line_num] = improved_mask

    
    # Create line mask mapping
    line_mask_mapping = {line_num: idx for idx, line_num in enumerate(unique_lines)}
    
    # Map each sampled point's line_number to the line mask index
    point_to_line_mask_idx = [
        line_mask_mapping.get(ln, -1) for ln in line_numbers
    ]
    
    # Create base name for saving
    base_name = Path(association_path).stem.replace('_associations', '')
    
    # Save character masks as NPZ
    char_npz_path = os.path.join(output_char_masks_dir, f"{base_name}_char_masks.npz")
    # np.savez_compressed(char_npz_path, masks=char_masks_array)
    np.savez_compressed(char_npz_path, masks=char_masks_array, line_numbers=np.array(line_numbers))


    # Save neighbor masks as NPZ
    if left_neighbor_masks_list and right_neighbor_masks_list:
        left_neighbor_masks_array = np.stack(left_neighbor_masks_list, axis=0)
        right_neighbor_masks_array = np.stack(right_neighbor_masks_list, axis=0)
        
        left_neighbor_npz_path = os.path.join(output_neighbor_masks_dir, f"{base_name}_left_neighbor_masks.npz")
        right_neighbor_npz_path = os.path.join(output_neighbor_masks_dir, f"{base_name}_right_neighbor_masks.npz")
        
        np.savez_compressed(left_neighbor_npz_path, masks=left_neighbor_masks_array)
        np.savez_compressed(right_neighbor_npz_path, masks=right_neighbor_masks_array)
        
        left_neighbor_npz_path = os.path.abspath(left_neighbor_npz_path)
        right_neighbor_npz_path = os.path.abspath(right_neighbor_npz_path)
    
    # Save prompt masks as NPZ
    prompt_npz_path = os.path.join(output_prompt_masks_dir, f"{base_name}_prompt_masks.npz")
    np.savez_compressed(prompt_npz_path, masks=prompt_masks_array)

    char_npz_path = os.path.abspath(char_npz_path)
    prompt_npz_path = os.path.abspath(prompt_npz_path)
    image_path = os.path.abspath(image_path)
    
    # Create record
    record = {
        'image_name': original_image_name,
        'inscription_name': inscription_name,
        'split': split,
        'image_path': image_path,  # Already converted to absolute above
        'augmented_image_paths': augmented_image_paths,
        'num_characters': len(predicted_components),
        'num_background': len(background_components),
        'num_sampled_points': len(sampled_points),
        'sampled_points': sampled_points,
        'line_numbers': line_numbers,
        'char_names': char_names,
        'char_numbers': char_numbers,
        'masks_npz_path': char_npz_path,  # Already converted to absolute above
        'prompt_masks_npz_path': prompt_npz_path,  # Already converted to absolute above
        'char_masks_shape': list(char_masks_array.shape),
        'prompt_masks_shape': list(prompt_masks_array.shape),
        'has_binarized': binarized_np is not None,
        'left_neighbor_masks_npz_path': left_neighbor_npz_path,
        'right_neighbor_masks_npz_path': right_neighbor_npz_path,
        'left_neighbor_masks_shape': list(left_neighbor_masks_array.shape),
        'right_neighbor_masks_shape': list(right_neighbor_masks_array.shape),
        'line_char_mapping': {int(k): v for k, v in line_char_mapping.items()}  # Convert to serializable format
    }
    
    # Save line masks if available
    if line_masks_list:
        line_masks_array = np.stack(line_masks_list, axis=0)
        line_npz_path = os.path.join(output_line_masks_dir, f"{base_name}_line_masks.npz")
        np.savez_compressed(line_npz_path, masks=line_masks_array)

        line_npz_path = os.path.abspath(line_npz_path)
        
        record['line_masks_npz'] = line_npz_path
        record['line_masks_shape'] = list(line_masks_array.shape)
        record['line_numbers_available'] = unique_lines
        record['line_mask_mapping'] = line_mask_mapping
        record['point_to_line_mask_idx'] = point_to_line_mask_idx

    above_neighbor_masks_list = []
    below_neighbor_masks_list = []

    foreground_idx = 0
    for comp in predicted_components:
        match = comp['matched_gt']
        if match and match['match_quality'] == 'inside':
            gt_key = match['gt_image_key']
            if gt_key in gt_polygons_by_key:
                char_data = gt_polygons_by_key[gt_key]
                line_num = match['line_number']
                if line_num is None:
                    above_neighbor_masks_list.append(np.zeros(image_shape, dtype=np.uint8))
                    below_neighbor_masks_list.append(np.zeros(image_shape, dtype=np.uint8))
                    foreground_idx += 1
                    continue

                # Union of left and right neighbor masks
                left_mask = left_neighbor_masks_list[foreground_idx]
                right_mask = right_neighbor_masks_list[foreground_idx]
                union_neighbor_mask = np.logical_or(left_mask, right_mask).astype(np.uint8)

                # Get bounding box of union mask
                ys, xs = np.where(union_neighbor_mask > 0)
                if len(xs) > 0 and len(ys) > 0:
                    x_min, x_max = xs.min(), xs.max()
                    y_min, y_max = ys.min(), ys.max()
                else:
                    # If no neighbor, use full width
                    x_min, x_max = 0, image_shape[1]
                    y_min, y_max = 0, image_shape[0]

                # Above neighbor
                above_line_num = line_num - 1
                above_mask = np.zeros(image_shape, dtype=np.uint8)
                if above_line_num in line_masks_dict:
                    above_line_mask = line_masks_dict[above_line_num]
                    above_mask[:, x_min:x_max+1] = above_line_mask[:, x_min:x_max+1]
                above_neighbor_masks_list.append(above_mask)

                # Below neighbor
                below_line_num = line_num + 1
                below_mask = np.zeros(image_shape, dtype=np.uint8)
                if below_line_num in line_masks_dict:
                    below_line_mask = line_masks_dict[below_line_num]
                    below_mask[:, x_min:x_max+1] = below_line_mask[:, x_min:x_max+1]
                below_neighbor_masks_list.append(below_mask)

                foreground_idx += 1

    # Add empty above/below neighbor masks for background components
    for bg_comp in background_components:
        above_neighbor_masks_list.append(np.zeros(image_shape, dtype=np.uint8))
        below_neighbor_masks_list.append(np.zeros(image_shape, dtype=np.uint8))

    # Save above/below neighbor masks as NPZ
    above_neighbor_masks_array = np.stack(above_neighbor_masks_list, axis=0)
    below_neighbor_masks_array = np.stack(below_neighbor_masks_list, axis=0)

    above_neighbor_npz_path = os.path.join(output_neighbor_masks_dir, f"{base_name}_above_neighbor_masks.npz")
    below_neighbor_npz_path = os.path.join(output_neighbor_masks_dir, f"{base_name}_below_neighbor_masks.npz")

    np.savez_compressed(above_neighbor_npz_path, masks=above_neighbor_masks_array)
    np.savez_compressed(below_neighbor_npz_path, masks=below_neighbor_masks_array)

    above_neighbor_npz_path = os.path.abspath(above_neighbor_npz_path)
    below_neighbor_npz_path = os.path.abspath(below_neighbor_npz_path)

    # Add to record
    record['above_neighbor_masks_npz_path'] = above_neighbor_npz_path
    record['below_neighbor_masks_npz_path'] = below_neighbor_npz_path
    record['above_neighbor_masks_shape'] = list(above_neighbor_masks_array.shape)
    record['below_neighbor_masks_shape'] = list(below_neighbor_masks_array.shape)
    
    # Create visualization
    if save_visualizations and line_masks_list:
        visualize_points_and_lines(
            image_np, char_masks_array, prompt_masks_array, line_masks_array,
            sampled_points, line_numbers, background_components,
            viz_output_dir, base_name,
            left_neighbor_masks=left_neighbor_masks_array if 'left_neighbor_masks_array' in locals() else None,
            right_neighbor_masks=right_neighbor_masks_array if 'right_neighbor_masks_array' in locals() else None,
            above_neighbor_masks=above_neighbor_masks_array,
            below_neighbor_masks=below_neighbor_masks_array
        )

    
    
    return record


def visualize_points_and_lines(image, char_masks, prompt_masks, line_masks, sampled_points, 
                                line_numbers, background_components, output_path, image_name,
                                left_neighbor_masks=None, right_neighbor_masks=None, above_neighbor_masks=None, below_neighbor_masks=None):
    """
    Visualize sampled points on character masks, prompt masks, line masks, and neighbor masks.
    """
    try:
        fig, axes = plt.subplots(2, 4, figsize=(32, 16))  # 4 columns now

        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        fg_points, fg_line_nums, bg_points = [], [], []

        for point, line_num in zip(sampled_points, line_numbers):
            if point is not None:
                if line_num is None or line_num == -1:
                    bg_points.append(point)
                else:
                    fg_points.append(point)
                    fg_line_nums.append(line_num)

        # 1. Original image with centroids
        axes[0, 0].imshow(image)
        for point, line_num in zip(fg_points, fg_line_nums):
            color = colors[line_num % 20]
            axes[0, 0].scatter(point[0], point[1], c=[color], s=50, marker='o', edgecolors='white', linewidths=2)
        if bg_points:
            bg_arr = np.array(bg_points)
            axes[0, 0].scatter(bg_arr[:, 0], bg_arr[:, 1], c='red', s=60, marker='s', edgecolors='white', linewidths=2)
        axes[0, 0].set_title('Original Image with Centroids')
        axes[0, 0].axis('off')

        # 2. GT Character masks
        combined_char_mask = np.any(char_masks, axis=0).astype(np.uint8)
        axes[0, 1].imshow(combined_char_mask, cmap='gray')
        axes[0, 1].set_title('GT Masks')
        axes[0, 1].axis('off')

        # 3. Prompt masks
        combined_prompt_mask = np.any(prompt_masks, axis=0).astype(np.uint8)
        axes[0, 2].imshow(combined_prompt_mask, cmap='gray')
        axes[0, 2].set_title('Prompt Masks')
        axes[0, 2].axis('off')

        # 4. Neighbor masks (overlay left and right)
        if left_neighbor_masks is not None and right_neighbor_masks is not None:
            combined_left = np.any(left_neighbor_masks, axis=0).astype(np.uint8)
            combined_right = np.any(right_neighbor_masks, axis=0).astype(np.uint8)
            neighbor_overlay = np.zeros((*combined_left.shape, 3), dtype=np.uint8)
            neighbor_overlay[..., 0] = combined_left * 255  # Left: Red
            neighbor_overlay[..., 2] = combined_right * 255 # Right: Blue
            axes[0, 3].imshow(neighbor_overlay)
            axes[0, 3].set_title('Neighbor Masks (Left=Red, Right=Blue)')
            axes[0, 3].axis('off')
        else:
            axes[0, 3].text(0.5, 0.5, 'No neighbor masks', ha='center', va='center')
            axes[0, 3].axis('off')
        
        # 4. Line masks color-coded
        if line_masks.shape[0] > 0:
            combined_line_mask = np.zeros((line_masks.shape[1], line_masks.shape[2], 3), dtype=np.uint8)
            for idx, line_mask in enumerate(line_masks):
                color = (colors[idx % 20][:3] * 255).astype(np.uint8)
                combined_line_mask[line_mask > 0] = color[::-1]  # RGB to BGR
            axes[1, 0].imshow(cv2.cvtColor(combined_line_mask, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title(f'Line Masks ({line_masks.shape[0]} lines)')
        else:
            axes[1, 0].text(0.5, 0.5, 'No line masks', ha='center', va='center')
        axes[1, 0].axis('off')
        
        # 5. Overlay of character centroids on line masks
        if line_masks.shape[0] > 0:
            axes[1, 1].imshow(cv2.cvtColor(combined_line_mask, cv2.COLOR_BGR2RGB))
            
            for point, line_num in zip(fg_points, fg_line_nums):
                color = colors[line_num % 20]
                axes[1, 1].scatter(point[0], point[1], c=[color], s=50,
                                  marker='x', linewidths=3)
            
            if bg_points:
                bg_arr = np.array(bg_points)
                axes[1, 1].scatter(bg_arr[:, 0], bg_arr[:, 1], c='red', s=60,
                                  marker='s', edgecolors='white', linewidths=2)
            
            axes[1, 1].set_title('Character Centroids on Line Masks')
        else:
            axes[1, 1].text(0.5, 0.5, 'No line masks', ha='center', va='center')
        axes[1, 1].axis('off')

        if above_neighbor_masks is not None and below_neighbor_masks is not None:
            combined_above = np.any(above_neighbor_masks, axis=0).astype(np.uint8)
            combined_below = np.any(below_neighbor_masks, axis=0).astype(np.uint8)
            above_below_overlay = np.zeros((*combined_above.shape, 3), dtype=np.uint8)
            above_below_overlay[..., 1] = combined_above * 255  # Above: Green
            above_below_overlay[..., 2] = combined_below * 255  # Below: Blue
            axes[1, 3].imshow(above_below_overlay)
            axes[1, 3].set_title('Above (Green) / Below (Blue) Neighbor Masks')
            axes[1, 3].axis('off')
        else:
            axes[1, 3].text(0.5, 0.5, 'No above/below neighbor masks', ha='center', va='center')
            axes[1, 3].axis('off')
        
        # 6. Side-by-side comparison of GT vs Prompt masks
        comparison = np.zeros((char_masks.shape[1], char_masks.shape[2], 3), dtype=np.uint8)
        comparison[:, :, 0] = combined_char_mask * 255  # GT in red
        comparison[:, :, 1] = combined_prompt_mask * 255  # Prompt in green
        # Overlap will appear yellow
        axes[1, 2].imshow(comparison)
        axes[1, 2].set_title('GT (Red) vs Prompt (Green) Masks - Yellow=Overlap')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        viz_path = os.path.join(output_path, f"{image_name}_dataset_visualization.png")
        plt.savefig(viz_path, bbox_inches='tight', dpi=150)
        plt.close()

        # Save individual neighbor visualizations for 3 characters
        neighbor_viz_dir = os.path.join(output_path, f"{image_name}_neighbor_viz")
        os.makedirs(neighbor_viz_dir, exist_ok=True)
        num_chars = min(3, char_masks.shape[0])
        for idx in range(num_chars):
            # Overlay masks on original image
            overlay = image.copy()
            if overlay.ndim == 2 or overlay.shape[2] == 1:
                overlay = np.stack([overlay]*3, axis=-1)
            else:
                overlay = overlay.copy()

            # Normalize if needed
            if overlay.max() > 255:
                overlay = (overlay / overlay.max() * 255).astype(np.uint8)

            # Define colors: char=yellow, left=red, right=blue, above=green, below=magenta
            char_color = [255, 255, 0]
            left_color = [255, 0, 0]
            right_color = [0, 0, 255]
            above_color = [0, 255, 0]
            below_color = [255, 0, 255]

            # Overlay masks
            mask_alpha = 0.4
            mask_shapes = [
                (char_masks[idx], char_color),
                (left_neighbor_masks[idx], left_color),
                (right_neighbor_masks[idx], right_color),
                (above_neighbor_masks[idx], above_color),
                (below_neighbor_masks[idx], below_color),
            ]
            for mask, color in mask_shapes:
                mask_bool = mask > 0
                for c in range(3):
                    overlay[..., c][mask_bool] = (
                        mask_alpha * color[c] + (1 - mask_alpha) * overlay[..., c][mask_bool]
                    )

            fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
            ax2.imshow(overlay.astype(np.uint8))
            ax2.set_title('Overlay: Char (Yellow), Left (Red), Right (Blue), Above (Green), Below (Magenta)')
            ax2.axis('off')
            plt.tight_layout()
            viz_path2 = os.path.join(neighbor_viz_dir, f"{idx}_neighbors_overlay.png")
            plt.savefig(viz_path2, bbox_inches='tight', dpi=120)
            plt.close(fig2)
        
    except Exception as e:
        print(f"  Error creating visualization: {e}")
        import traceback
        traceback.print_exc()


def process_split(associations_dir, convex_hulls_dir, images_dir, split,
                 output_char_masks_dir, output_neighbor_masks_dir, output_line_masks_dir, output_prompt_masks_dir,
                 save_visualizations, viz_output_dir, binarized_dir=None, use_color_augmented_images=False):
    """
    Process all association files in a split.
    
    Args:
        associations_dir: Path to associated_centroids directory
        convex_hulls_dir: Path to convex_hulls_data directory
        images_dir: Path to whole_images directory
        split: 'train' or 'val'
        output_char_masks_dir: Directory to save character masks NPZ
        output_line_masks_dir: Directory to save line masks NPZ
        output_prompt_masks_dir: Directory to save prompt masks NPZ
        save_visualizations: Whether to save visualizations
        viz_output_dir: Directory to save visualizations
        binarized_dir: Directory containing binarized images
    
    Returns:
        list: Records for all processed images
    """
    split_associations_dir = os.path.join(associations_dir, split)
    
    if not os.path.exists(split_associations_dir):
        print(f"Error: Associations directory not found: {split_associations_dir}")
        return []
    
    # Get all association JSON files
    association_files = sorted(Path(split_associations_dir).glob('*_associations.json'))
    
    print(f"\nProcessing {split} split: {len(association_files)} files")
    
    records = []
    
    for assoc_file in tqdm(association_files, desc=f"Processing {split}"):
        try:
            record = process_association_file(
                str(assoc_file),
                convex_hulls_dir,
                images_dir,
                split,
                output_char_masks_dir,
                output_line_masks_dir,
                output_prompt_masks_dir,
                output_neighbor_masks_dir,
                save_visualizations,
                viz_output_dir,
                binarized_dir=binarized_dir,
                use_color_augmented_images=use_color_augmented_images
            )
            
            if record:
                records.append(record)
                
        except Exception as e:
            print(f"  Error processing {assoc_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return records


# Editing for better line creation
def get_mask_center(mask):
    moments = cv2.moments(mask)
    if moments['m00'] > 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return np.array([cx, cy])
    else:
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) > 0:
            return np.array([np.mean(x_coords), np.mean(y_coords)])
    return None

def get_mask_dimensions(mask):
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) > 0 and len(y_coords) > 0:
        width = x_coords.max() - x_coords.min()
        height = y_coords.max() - y_coords.min()
        return width, height
    return 0, 0

def calculate_average_mask_height(char_masks):
    heights = []
    for mask in char_masks:
        _, h = get_mask_dimensions(mask)
        if h > 0:
            heights.append(h)
    avg_height = np.mean(heights) if heights else 50
    return avg_height

def create_line_mask_from_centers(char_masks, image_shape, size_multiplier=1.0):
    if not char_masks:
        return np.zeros(image_shape, dtype=np.uint8)
    result_mask = np.zeros(image_shape, dtype=np.uint8)
    for mask in char_masks:
        result_mask = cv2.bitwise_or(result_mask, mask)
    avg_height = calculate_average_mask_height(char_masks)
    avg_height *= size_multiplier
    centers = []
    for mask in char_masks:
        center = get_mask_center(mask)
        if center is not None:
            centers.append(center)
    if len(centers) < 2:
        return result_mask
    centers = np.array(centers)
    sorted_indices = np.argsort(centers[:, 0])
    sorted_centers = centers[sorted_indices]
    thickness = max(1, int(avg_height))
    for i in range(len(sorted_centers) - 1):
        pt1 = tuple(sorted_centers[i].astype(int))
        pt2 = tuple(sorted_centers[i + 1].astype(int))
        cv2.line(result_mask, pt1, pt2, 255, thickness)
    radius = max(1, thickness // 2)
    for center in sorted_centers:
        cv2.circle(result_mask, tuple(center.astype(int)), radius, 255, -1)
    return result_mask



def main():
    parser = argparse.ArgumentParser(
        description='Create dataset from associated centroids in sample_char_centroids format'
    )
    parser.add_argument(
        '--associations-dir',
        type=str,
        default='associated_centroids_updated',
        help='Path to associated_centroids directory'
    )
    parser.add_argument(
        '--convex-hulls-dir',
        type=str,
        default='chars',
        help='Path to convex_hulls_data directory'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        default='images',
        help='Path to whole_images directory'
    )
    parser.add_argument(
        '--binarized-dir',
        type=str,
        default='binary_images',
        help='Path to directory containing binarized images (searched recursively)'
    )
    parser.add_argument(
        '--output-neighbor-masks',
        type=str,
        default='processed_data/neighbor_masks_end_npz',
        help='Directory to save neighbor masks NPZ files'
    )
    parser.add_argument(
        '--output-char-masks',
        type=str,
        default='processed_data/char_masks_end_npz',
        help='Directory to save character masks NPZ files'
    )
    parser.add_argument(
        '--output-line-masks',
        type=str,
        default='processed_data/line_masks_end_npz',
        help='Directory to save line masks NPZ files'
    )
    parser.add_argument(
        '--output-prompt-masks',
        type=str,
        default='processed_data/prompt_masks_end_npz',
        help='Directory to save prompt masks NPZ files'
    )
    parser.add_argument(
        '--output-index',
        type=str,
        default='processed_data/master_index_from_associations_end_train.json',
        help='Path to save master index JSON file'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        help='Splits to process (train, val)'
    )
    parser.add_argument(
        '--save-viz',
        action='store_true',
        default=False,
        help='Save visualization images'
    )
    parser.add_argument(
        '--viz-dir',
        type=str,
        default='preprocessed_data_neighbors_2_hop_above/visualizations_from_associations_neighbors',
        help='Directory to save visualizations'
    )
    parser.add_argument(
        '--use-color-augmented-images',
        action='store_true',
        help='Whether to use template based color augmentations or not.'
    )
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_char_masks, exist_ok=True)
    os.makedirs(args.output_neighbor_masks, exist_ok=True)
    os.makedirs(args.output_line_masks, exist_ok=True)
    os.makedirs(args.output_prompt_masks, exist_ok=True)
    if args.save_viz:
        os.makedirs(args.viz_dir, exist_ok=True)
    
    print("="*60)
    print("CREATE DATASET FROM ASSOCIATED CENTROIDS")
    print("="*60)
    print(f"Associations directory: {args.associations_dir}")
    print(f"Convex hulls directory: {args.convex_hulls_dir}")
    print(f"Images directory: {args.images_dir}")
    print(f"Binarized images directory: {args.binarized_dir}")
    print(f"Output char masks: {args.output_char_masks}")
    print(f"Output line masks: {args.output_line_masks}")
    print(f"Output prompt masks: {args.output_prompt_masks}")
    print(f"Output index: {args.output_index}")
    print(f"Splits: {args.splits}")
    print(f"Save visualizations: {args.save_viz}")
    
    all_records = []
    
    # Process each split
    for split in args.splits:
        records = process_split(
            args.associations_dir,
            args.convex_hulls_dir,
            args.images_dir,
            split,
            args.output_char_masks,
            args.output_neighbor_masks,
            args.output_line_masks,
            args.output_prompt_masks,
            args.save_viz,
            args.viz_dir,
            binarized_dir=args.binarized_dir,
            use_color_augmented_images=args.use_color_augmented_images
        )
        
        all_records.extend(records)
    
    # Save master index
    with open(args.output_index, 'w') as f:
        json.dump(all_records, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total records created: {len(all_records)}")
    print(f"Master index saved to: {args.output_index}")
    print(f"Character masks NPZ saved to: {args.output_char_masks}")
    print(f"Line masks NPZ saved to: {args.output_line_masks}")
    print(f"Prompt masks NPZ saved to: {args.output_prompt_masks}")
    if args.save_viz:
        print(f"Visualizations saved to: {args.viz_dir}")
    
    # Split summary
    train_count = sum(1 for r in all_records if r['split'] == 'train')
    val_count = sum(1 for r in all_records if r['split'] == 'val')
    total_fg = sum(r['num_characters'] for r in all_records)
    total_bg = sum(r['num_background'] for r in all_records)
    has_binarized = sum(1 for r in all_records if r.get('has_binarized', False))
    
    print(f"\nSplit Summary:")
    print(f"  Train: {train_count} images")
    print(f"  Val: {val_count} images")
    print(f"\nPoint Summary:")
    print(f"  Total foreground points: {total_fg}")
    print(f"  Total background points: {total_bg}")
    print(f"  Total points: {total_fg + total_bg}")
    print(f"\nBinarized Images:")
    print(f"  Images with binarized masks: {has_binarized}/{len(all_records)}")


if __name__ == '__main__':
    main()