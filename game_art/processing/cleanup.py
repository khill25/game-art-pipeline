"""Post-generation cleanup for sprites after background removal.

Handles: largest-component extraction, character base removal, and quality metrics.
These run on the Python side after ComfyUI's GPU-side rembg has already removed
the main background.
"""

import numpy as np
from PIL import Image


def keep_largest_component(img: Image.Image, alpha_threshold: int = 32) -> Image.Image:
    """Keep only the largest connected component by alpha, zero out the rest.

    After rembg, there may be small floating fragments (dust, shadow remnants).
    This keeps only the main subject.

    Args:
        img: RGBA image with background already removed
        alpha_threshold: Minimum alpha to count as "opaque"

    Returns:
        RGBA image with only the largest connected region preserved
    """
    img = img.convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3]
    mask = alpha > alpha_threshold

    if not mask.any():
        return img

    try:
        from scipy.ndimage import label
    except ImportError:
        # scipy not available — skip component filtering
        return img

    labeled, num_features = label(mask)
    if num_features <= 1:
        return img

    # Find the largest component
    component_sizes = np.bincount(labeled.ravel())
    # Skip label 0 (background)
    component_sizes[0] = 0
    largest_label = component_sizes.argmax()

    # Zero out everything except the largest component
    remove_mask = (labeled != largest_label) & (labeled != 0)
    arr[remove_mask] = [0, 0, 0, 0]

    return Image.fromarray(arr, "RGBA")


def remove_character_base(img: Image.Image, threshold_ratio: float = 0.8) -> Image.Image:
    """Remove flat "miniature base" or pedestal from bottom of character/enemy sprites.

    Many AI-generated characters have a circular or rectangular base at the bottom.
    This detects rows in the bottom 20% where opaque pixels span > threshold_ratio
    of the subject width, and removes them.

    Args:
        img: RGBA image (background already removed)
        threshold_ratio: If a bottom row has opaque pixels spanning more than
                        this fraction of the image width, consider it a base

    Returns:
        RGBA image with base removed
    """
    img = img.convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3]
    h, w = alpha.shape

    if not alpha.any():
        return img

    # Find opaque bounding box
    rows_with_content = np.where(alpha.max(axis=1) > 32)[0]
    if len(rows_with_content) == 0:
        return img

    top_row = rows_with_content[0]
    bot_row = rows_with_content[-1]
    content_height = bot_row - top_row + 1

    if content_height < 10:
        return img

    # Scan bottom 20% of the opaque region
    scan_start = bot_row - max(int(content_height * 0.2), 5)
    scan_start = max(scan_start, top_row)

    # Find the first row from the bottom that looks like a base
    base_start = None
    for y in range(bot_row, scan_start - 1, -1):
        row_alpha = alpha[y]
        opaque_cols = np.where(row_alpha > 32)[0]
        if len(opaque_cols) == 0:
            continue
        span = opaque_cols[-1] - opaque_cols[0] + 1
        if span > w * threshold_ratio:
            base_start = y
        else:
            # Stop scanning once we hit a non-base row
            break

    if base_start is not None:
        # Remove everything from base_start to bottom
        arr[base_start:, :] = [0, 0, 0, 0]

    return Image.fromarray(arr, "RGBA")


def compute_fill_ratio(img: Image.Image, alpha_threshold: int = 32) -> float:
    """Compute the ratio of opaque pixels to total pixels.

    A fill ratio near 1.0 means background removal failed (entire image is opaque).
    A good sprite typically has 0.1-0.6 fill ratio.
    """
    arr = np.array(img.convert("RGBA"))
    alpha = arr[:, :, 3]
    total = alpha.size
    opaque = (alpha > alpha_threshold).sum()
    return float(opaque / total) if total > 0 else 0.0


def compute_aspect_ratio(img: Image.Image, alpha_threshold: int = 32) -> float:
    """Compute the aspect ratio (height/width) of the opaque bounding box.

    Useful for validating weapons (should be elongated, >1.3) vs
    characters (roughly square, 0.8-1.5).
    """
    arr = np.array(img.convert("RGBA"))
    alpha = arr[:, :, 3]
    opaque = alpha > alpha_threshold

    if not opaque.any():
        return 1.0

    rows = np.where(opaque.any(axis=1))[0]
    cols = np.where(opaque.any(axis=0))[0]

    height = rows[-1] - rows[0] + 1
    width = cols[-1] - cols[0] + 1

    if width == 0:
        return 1.0

    return float(height / width)
