import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Literal

def check_and_fix_irregular_mask(
    mask: torch.Tensor,
    strategy: Literal["pad", "crop", "fill_bbox", "nearest_square"] = "fill_bbox",
    target_aspect_ratio: Optional[Tuple[int, int]] = None,
    return_mapping: bool = False
) -> Tuple[torch.Tensor, int, int, Optional[torch.Tensor]]:
    """
    Check if a mask is irregular (can't be reshaped to w*h) and fix it if needed.
    
    Args:
        mask: Binary mask tensor of shape [H, W] or [B, H, W]
        strategy: Strategy to fix irregular masks:
            - "pad": Pad to nearest rectangle
            - "crop": Crop to nearest rectangle
            - "fill_bbox": Fill the bounding box to make it rectangular
            - "nearest_square": Make it a square
        target_aspect_ratio: Optional (h, w) ratio to maintain, e.g., (1, 1) for square
        return_mapping: If True, return index mapping for attention map reconstruction
    
    Returns:
        fixed_mask: Modified mask [H, W] or [B, H, W]
        h: Height of the masked region
        w: Width of the masked region
        index_mapping: Optional tensor mapping old indices to new indices
    """
    
    original_shape = mask.shape
    has_batch = len(mask.shape) == 3
    
    if has_batch:
        # Process first item in batch (assume all have same mask)
        mask_2d = mask[0]
    else:
        mask_2d = mask
    
    # Get non-zero indices
    non_zero_indices = torch.nonzero(mask_2d, as_tuple=False)
    num_masked_pixels = non_zero_indices.shape[0]
    
    if num_masked_pixels == 0:
        print("Warning: Empty mask detected!")
        return mask, 1, 1, None
    
    # Get bounding box
    y_min, x_min = non_zero_indices.min(dim=0)[0]
    y_max, x_max = non_zero_indices.max(dim=0)[0]
    
    bbox_h = (y_max - y_min + 1).item()
    bbox_w = (x_max - x_min + 1).item()
    
    print(f"Mask analysis:")
    print(f"  - Non-zero pixels: {num_masked_pixels}")
    print(f"  - Bounding box: [{bbox_h}, {bbox_w}] = {bbox_h * bbox_w} pixels")
    print(f"  - Coverage: {num_masked_pixels / (bbox_h * bbox_w) * 100:.1f}%")
    
    # Check if already regular (forms a perfect rectangle in bbox)
    is_regular = (num_masked_pixels == bbox_h * bbox_w)
    
    if is_regular:
        print("✓ Mask is already regular!")
        h, w = bbox_h, bbox_w
        fixed_mask = mask.clone()
    else:
        print(f"✗ Mask is irregular, applying '{strategy}' strategy...")
        fixed_mask = mask.clone()
        
        if strategy == "fill_bbox":
            # Fill the entire bounding box
            if has_batch:
                fixed_mask[:, y_min:y_max+1, x_min:x_max+1] = 1
            else:
                fixed_mask[y_min:y_max+1, x_min:x_max+1] = 1
            h, w = bbox_h, bbox_w
            
        elif strategy == "pad":
            # Pad to make the number of pixels factorable
            target_pixels = _find_nearest_factorable(num_masked_pixels, "up")
            h, w = _factorize_to_rectangle(target_pixels, target_aspect_ratio)
            
            # Expand bounding box symmetrically
            fixed_mask = _expand_mask_to_size(fixed_mask, non_zero_indices, h, w, has_batch)
            
        elif strategy == "crop":
            # Crop to make the number of pixels factorable
            target_pixels = _find_nearest_factorable(num_masked_pixels, "down")
            h, w = _factorize_to_rectangle(target_pixels, target_aspect_ratio)
            
            # Keep only the most central pixels
            fixed_mask = _crop_mask_to_size(fixed_mask, non_zero_indices, h, w, has_batch)
            
        elif strategy == "nearest_square":
            # Make it a square
            side = int(np.sqrt(num_masked_pixels))
            if side * side < num_masked_pixels:
                side += 1  # Round up
            h, w = side, side
            
            # Fill bounding box and pad/crop to square
            if has_batch:
                fixed_mask[:, y_min:y_max+1, x_min:x_max+1] = 1
            else:
                fixed_mask[y_min:y_max+1, x_min:x_max+1] = 1
            
            # If bbox is not square, expand to square
            if bbox_h != bbox_w:
                fixed_mask = _make_square(fixed_mask, y_min, y_max, x_min, x_max, has_batch)
                h = w = max(bbox_h, bbox_w)
        
        print(f"✓ Fixed mask: [{h}, {w}] = {h * w} pixels")
    
    # Create index mapping if requested
    index_mapping = None
    if return_mapping:
        index_mapping = _create_index_mapping(mask_2d, fixed_mask[0] if has_batch else fixed_mask)
    
    return fixed_mask, h, w, index_mapping


def _find_nearest_factorable(n: int, direction: Literal["up", "down"] = "up") -> int:
    """Find nearest number that can be nicely factorized into h*w."""
    if direction == "up":
        candidate = n
        while True:
            if _is_nicely_factorable(candidate):
                return candidate
            candidate += 1
            if candidate > n * 1.5:  # Safety limit
                return n
    else:
        candidate = n
        while candidate > 0:
            if _is_nicely_factorable(candidate):
                return candidate
            candidate -= 1
        return 1


def _is_nicely_factorable(n: int) -> bool:
    """Check if a number can be factorized into reasonable h*w."""
    # A number is nicely factorable if its aspect ratio is reasonable
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            h, w = i, n // i
            aspect_ratio = max(h, w) / min(h, w)
            if aspect_ratio <= 4:  # Allow up to 4:1 aspect ratio
                return True
    return False


def _factorize_to_rectangle(
    n: int, 
    target_aspect_ratio: Optional[Tuple[int, int]] = None
) -> Tuple[int, int]:
    """Factorize n into h*w with optional aspect ratio constraint."""
    if target_aspect_ratio:
        target_h, target_w = target_aspect_ratio
        target_ratio = target_h / target_w
    else:
        target_ratio = 1.0  # Default to square-ish
    
    best_h, best_w = 1, n
    best_diff = float('inf')
    
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            h, w = i, n // i
            ratio = h / w
            diff = abs(ratio - target_ratio)
            if diff < best_diff:
                best_diff = diff
                best_h, best_w = h, w
    
    return best_h, best_w


def _expand_mask_to_size(
    mask: torch.Tensor,
    indices: torch.Tensor,
    target_h: int,
    target_w: int,
    has_batch: bool
) -> torch.Tensor:
    """Expand mask by adding nearby pixels."""
    # Get center of mass
    center_y = indices[:, 0].float().mean()
    center_x = indices[:, 1].float().mean()
    
    # Create expanded region
    y_start = max(0, int(center_y - target_h // 2))
    x_start = max(0, int(center_x - target_w // 2))
    y_end = min(mask.shape[-2], y_start + target_h)
    x_end = min(mask.shape[-1], x_start + target_w)
    
    # Adjust if we hit boundaries
    if y_end - y_start < target_h:
        y_start = max(0, y_end - target_h)
    if x_end - x_start < target_w:
        x_start = max(0, x_end - target_w)
    
    new_mask = mask.clone()
    if has_batch:
        new_mask[:, y_start:y_end, x_start:x_end] = 1
    else:
        new_mask[y_start:y_end, x_start:x_end] = 1
    
    return new_mask


def _crop_mask_to_size(
    mask: torch.Tensor,
    indices: torch.Tensor,
    target_h: int,
    target_w: int,
    has_batch: bool
) -> torch.Tensor:
    """Crop mask to keep only most central pixels."""
    # Sort by distance to center of mass
    center_y = indices[:, 0].float().mean()
    center_x = indices[:, 1].float().mean()
    
    distances = torch.sqrt(
        (indices[:, 0].float() - center_y) ** 2 + 
        (indices[:, 1].float() - center_x) ** 2
    )
    
    # Keep only closest pixels
    target_pixels = target_h * target_w
    sorted_indices = torch.argsort(distances)
    keep_indices = indices[sorted_indices[:target_pixels]]
    
    new_mask = torch.zeros_like(mask)
    if has_batch:
        new_mask[:, keep_indices[:, 0], keep_indices[:, 1]] = 1
    else:
        new_mask[keep_indices[:, 0], keep_indices[:, 1]] = 1
    
    return new_mask


def _make_square(
    mask: torch.Tensor,
    y_min: torch.Tensor,
    y_max: torch.Tensor,
    x_min: torch.Tensor,
    x_max: torch.Tensor,
    has_batch: bool
) -> torch.Tensor:
    """Expand rectangle to square."""
    y_min, y_max = y_min.item(), y_max.item()
    x_min, x_max = x_min.item(), x_max.item()
    
    h = y_max - y_min + 1
    w = x_max - x_min + 1
    side = max(h, w)
    
    # Center the square
    y_center = (y_min + y_max) // 2
    x_center = (x_min + x_max) // 2
    
    new_y_min = max(0, y_center - side // 2)
    new_x_min = max(0, x_center - side // 2)
    new_y_max = min(mask.shape[-2], new_y_min + side)
    new_x_max = min(mask.shape[-1], new_x_min + side)
    
    new_mask = mask.clone()
    if has_batch:
        new_mask[:, new_y_min:new_y_max, new_x_min:new_x_max] = 1
    else:
        new_mask[new_y_min:new_y_max, new_x_min:new_x_max] = 1
    
    return new_mask


def _create_index_mapping(old_mask: torch.Tensor, new_mask: torch.Tensor) -> torch.Tensor:
    """Create mapping from old mask indices to new mask indices."""
    old_indices = torch.nonzero(old_mask, as_tuple=False)
    new_indices = torch.nonzero(new_mask, as_tuple=False)
    
    # For each old index, find closest new index
    mapping = torch.zeros(old_indices.shape[0], dtype=torch.long)
    
    for i, old_idx in enumerate(old_indices):
        distances = torch.sum((new_indices - old_idx) ** 2, dim=1)
        mapping[i] = torch.argmin(distances)
    
    return mapping


# Example usage and testing function
def test_mask_fixing():
    """Test the mask fixing function with various irregular masks."""
    
    print("=" * 60)
    print("Test 1: Irregular L-shaped mask")
    print("=" * 60)
    mask1 = torch.zeros(32, 32)
    mask1[10:20, 10:15] = 1  # Vertical part
    mask1[15:20, 15:25] = 1  # Horizontal part
    
    fixed1, h1, w1, _ = check_and_fix_irregular_mask(mask1, strategy="fill_bbox")
    print(f"Result: h={h1}, w={w1}, total={h1*w1}\n")
    
    print("=" * 60)
    print("Test 2: Scattered irregular mask")
    print("=" * 60)
    mask2 = torch.zeros(64, 64)
    mask2[20:25, 20:25] = 1
    mask2[22:27, 30:35] = 1
    mask2[30, 25] = 1  # Extra pixel
    
    fixed2, h2, w2, _ = check_and_fix_irregular_mask(mask2, strategy="pad")
    print(f"Result: h={h2}, w={w2}, total={h2*w2}\n")
    
    print("=" * 60)
    print("Test 3: Nearly square mask with gaps")
    print("=" * 60)
    mask3 = torch.zeros(128, 128)
    mask3[40:60, 40:60] = 1
    mask3[45:55, 45:55] = 0  # Gap in the middle
    
    fixed3, h3, w3, _ = check_and_fix_irregular_mask(mask3, strategy="nearest_square")
    print(f"Result: h={h3}, w={w3}, total={h3*w3}\n")
    
    return fixed1, fixed2, fixed3


if __name__ == "__main__":
    test_mask_fixing()