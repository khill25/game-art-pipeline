"""Orientation normalization for generated sprites.

Rotates/flips sprites to canonical poses based on category:
- Weapons: long axis vertical, tip pointing up
- Characters/enemies: light upright check only
- Other (pickups, tiles): no rotation
"""

import math
from PIL import Image
import numpy as np


def normalize_orientation(img: Image.Image, category: str = "item") -> Image.Image:
    """Normalize sprite orientation based on category.

    Args:
        img: RGBA PIL Image (should already have background removed).
        category: Asset category — "weapon", "enemy", "character", etc.

    Returns:
        Rotated/flipped RGBA image with canonical orientation.
    """
    if category == "weapon":
        return _orient_weapon(img)
    # Characters, enemies, and everything else: no aggressive rotation
    return img


def _orient_weapon(img: Image.Image) -> Image.Image:
    """Orient a weapon sprite: long axis vertical, tip up.

    Uses principal component analysis on opaque pixel coordinates
    to find the dominant axis, then rotates to vertical. Determines
    tip vs handle by comparing pixel density in top/bottom halves
    (the tapered end = tip = fewer pixels).
    """
    arr = np.array(img)
    alpha = arr[:, :, 3]

    # Get coordinates of opaque pixels (alpha > 32 to ignore faint edges)
    ys, xs = np.where(alpha > 32)
    if len(xs) < 10:
        return img  # Not enough pixels to analyze

    # Check aspect ratio — skip if not elongated (shields, orbs, etc.)
    x_span = xs.max() - xs.min()
    y_span = ys.max() - ys.min()
    if min(x_span, y_span) == 0:
        return img
    aspect = max(x_span, y_span) / max(min(x_span, y_span), 1)
    if aspect < 1.5:
        return img  # Not elongated enough to orient

    # Compute principal axis via 2x2 covariance matrix eigenvector
    cx = xs.mean()
    cy = ys.mean()
    dx = xs - cx
    dy = ys - cy

    cov_xx = (dx * dx).mean()
    cov_xy = (dx * dy).mean()
    cov_yy = (dy * dy).mean()

    # Eigenvalues/vectors of 2x2 matrix [[cov_xx, cov_xy], [cov_xy, cov_yy]]
    # The eigenvector for the larger eigenvalue is the principal axis
    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    discriminant = max(trace * trace / 4 - det, 0)
    sqrt_disc = math.sqrt(discriminant)

    lambda1 = trace / 2 + sqrt_disc
    # Principal eigenvector for lambda1
    if abs(cov_xy) > 1e-6:
        evx = lambda1 - cov_yy
        evy = cov_xy
    elif cov_xx >= cov_yy:
        evx, evy = 1.0, 0.0
    else:
        evx, evy = 0.0, 1.0

    # Normalize
    mag = math.sqrt(evx * evx + evy * evy)
    if mag < 1e-6:
        return img
    evx /= mag
    evy /= mag

    # Angle of principal axis relative to vertical (0, -1)
    # We want the long axis to be vertical
    angle_rad = math.atan2(evx, -evy)  # angle from "up" direction
    angle_deg = math.degrees(angle_rad)

    # Only rotate if significantly off-vertical (> 15 degrees)
    if abs(angle_deg) < 15:
        rotation_needed = False
    else:
        rotation_needed = True

    if rotation_needed:
        # Rotate to make principal axis vertical
        # PIL rotates counter-clockwise, we need to rotate by -angle
        img = img.rotate(-angle_deg, resample=Image.BICUBIC, expand=True)

    # Now determine tip vs handle: tip should point up
    # The tip is the tapered end (fewer opaque pixels)
    arr2 = np.array(img)
    alpha2 = arr2[:, :, 3]
    ys2, xs2 = np.where(alpha2 > 32)

    if len(ys2) < 10:
        return img

    mid_y = (ys2.min() + ys2.max()) / 2
    top_count = np.sum(ys2 < mid_y)
    bottom_count = np.sum(ys2 >= mid_y)

    # If bottom has fewer pixels, it's the tip — flip vertically so tip is on top
    if bottom_count < top_count * 0.85:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    return img
