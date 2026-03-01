"""Color palette enforcement — remap image colors to a target palette."""

import math
from PIL import Image


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _color_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    """Weighted Euclidean distance in RGB space.

    Uses weights that approximate human color perception
    (red and green matter more than blue).
    """
    dr = c1[0] - c2[0]
    dg = c1[1] - c2[1]
    db = c1[2] - c2[2]
    # Weighted RGB distance (redmean approximation simplified)
    return math.sqrt(2 * dr * dr + 4 * dg * dg + 3 * db * db)


def find_nearest_color(
    color: tuple[int, int, int],
    palette: list[tuple[int, int, int]],
) -> tuple[int, int, int]:
    """Find the nearest palette color to the given color."""
    best = palette[0]
    best_dist = _color_distance(color, palette[0])
    for p in palette[1:]:
        d = _color_distance(color, p)
        if d < best_dist:
            best = p
            best_dist = d
    return best


def enforce_palette(
    img: Image.Image,
    palette_hex: list[str],
    alpha_threshold: int = 32,
) -> Image.Image:
    """Remap all pixel colors to the nearest color in the palette.

    Pixels with alpha below alpha_threshold are made fully transparent.
    Preserves alpha channel for semi-transparent pixels above threshold.

    Args:
        img: RGBA PIL Image
        palette_hex: List of hex color strings (e.g., ["#2d1b2e", "#5c3a6e"])
        alpha_threshold: Alpha values below this become fully transparent
    """
    if not palette_hex:
        return img

    palette_rgb = [hex_to_rgb(h) for h in palette_hex]
    img = img.convert("RGBA")
    pixels = img.load()
    w, h = img.size

    # Build a cache of color mappings for speed
    cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}

    for y in range(h):
        for x in range(w):
            r, g, b, a = pixels[x, y]
            if a < alpha_threshold:
                pixels[x, y] = (0, 0, 0, 0)
                continue

            key = (r, g, b)
            if key not in cache:
                cache[key] = find_nearest_color(key, palette_rgb)

            nr, ng, nb = cache[key]
            pixels[x, y] = (nr, ng, nb, a)

    return img
