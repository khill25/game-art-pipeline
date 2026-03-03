"""Pivot point detection for game sprites.

Detects semantic rotation/anchor points based on sprite category:
- Weapons (tip-up): grip (handle), tip (topmost), center (centroid)
- Characters/enemies: feet (bottom-center), center (centroid)
- Other: center (centroid)
"""

from dataclasses import dataclass
from PIL import Image
import numpy as np


@dataclass
class PivotPoint:
    """A detected pivot/rotation point on a sprite.

    Coordinates are normalized 0.0–1.0 (origin = top-left), so they work
    at any sprite resolution.  Multiply by the sprite's pixel size to get
    pixel coordinates.
    """
    name: str           # "grip", "tip", "center", "feet"
    x: float            # normalized x, 0.0 – 1.0
    y: float            # normalized y, 0.0 – 1.0
    confidence: float   # 0.0 - 1.0


def detect_pivots(img: Image.Image, category: str = "item") -> list[PivotPoint]:
    """Detect pivot points on a sprite.

    Args:
        img: RGBA PIL Image (should be orientation-normalized already).
        category: Asset category — "weapon", "enemy", "character", etc.

    Returns:
        List of PivotPoint objects for the sprite.
    """
    arr = np.array(img)
    alpha = arr[:, :, 3]
    h, w = alpha.shape
    ys, xs = np.where(alpha > 32)

    if len(xs) < 4:
        # Fallback: center of image
        return [PivotPoint(
            name="center",
            x=0.5,
            y=0.5,
            confidence=0.1,
        )]

    # Centroid — always computed (normalized)
    cx = float(xs.mean()) / w
    cy = float(ys.mean()) / h

    if category == "weapon":
        return _weapon_pivots(img, alpha, xs, ys, cx, cy, w, h)
    elif category in ("character", "enemy", "boss", "npc"):
        return _character_pivots(img, alpha, xs, ys, cx, cy, w, h)
    else:
        return [PivotPoint(name="center", x=cx, y=cy, confidence=0.9)]


def _weapon_pivots(
    img: Image.Image,
    alpha: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    cx: float,
    cy: float,
    w: int,
    h: int,
) -> list[PivotPoint]:
    """Detect weapon pivots (assumes tip-up after orientation normalization).

    - tip: topmost opaque pixel
    - grip: narrowest horizontal cross-section in bottom 40%
    - center: geometric centroid

    All coordinates returned normalized 0.0–1.0.
    """
    pivots = []

    # Tip: topmost opaque pixel (center of topmost row)
    top_y = int(ys.min())
    top_row_xs = xs[ys == top_y]
    tip_x = float(top_row_xs.mean()) / w
    pivots.append(PivotPoint(name="tip", x=tip_x, y=top_y / h, confidence=0.9))

    # Grip: narrowest horizontal cross-section in bottom 40% of the sprite
    y_min = int(ys.min())
    y_max = int(ys.max())
    y_range = y_max - y_min
    if y_range > 0:
        bottom_40_start = y_max - int(y_range * 0.4)
        grip_y, grip_x, grip_conf = _find_narrowest_crosssection(
            alpha, bottom_40_start, y_max,
        )
        pivots.append(PivotPoint(
            name="grip", x=grip_x / w, y=grip_y / h, confidence=grip_conf,
        ))

    # Center: centroid
    pivots.append(PivotPoint(name="center", x=cx, y=cy, confidence=0.9))

    return pivots


def _character_pivots(
    img: Image.Image,
    alpha: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    cx: float,
    cy: float,
    w: int,
    h: int,
) -> list[PivotPoint]:
    """Detect character/enemy pivots.

    - feet: bottom-center of the sprite
    - center: geometric centroid

    All coordinates returned normalized 0.0–1.0.
    """
    # Feet: center of bottommost opaque row
    bottom_y = int(ys.max())
    bottom_row_xs = xs[ys == bottom_y]
    feet_x = float(bottom_row_xs.mean()) / w

    return [
        PivotPoint(name="feet", x=feet_x, y=bottom_y / h, confidence=0.9),
        PivotPoint(name="center", x=cx, y=cy, confidence=0.9),
    ]


def _find_narrowest_crosssection(
    alpha: np.ndarray,
    y_start: int,
    y_end: int,
) -> tuple[int, int, float]:
    """Find the y-row with the narrowest horizontal span of opaque pixels.

    Scans rows from y_start to y_end, measuring the horizontal extent
    (rightmost - leftmost opaque pixel) on each row. Returns the row
    with the smallest span.

    Returns:
        (y, center_x, confidence)
    """
    best_y = (y_start + y_end) // 2
    best_width = alpha.shape[1]  # start with max possible
    best_cx = alpha.shape[1] // 2

    for y in range(max(y_start, 0), min(y_end + 1, alpha.shape[0])):
        row = alpha[y]
        opaque = np.where(row > 32)[0]
        if len(opaque) < 2:
            continue
        width = int(opaque[-1] - opaque[0])
        if width < best_width and width > 0:
            best_width = width
            best_y = y
            best_cx = int(round((opaque[0] + opaque[-1]) / 2))

    # Confidence is higher when the grip is clearly narrow relative to the sprite
    confidence = min(0.95, max(0.5, 1.0 - best_width / alpha.shape[1]))
    return best_y, best_cx, confidence
