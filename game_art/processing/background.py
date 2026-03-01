"""Background removal for generated sprites.

Supports multiple strategies since AI image generators can't produce
true transparent backgrounds:

1. Green screen: Generate with #00FF00 background, chroma-key it out
2. Flood fill: Remove connected background regions from edges
3. Threshold: Simple darkness threshold (fallback for black backgrounds)
"""

import math
from PIL import Image
from collections import deque


def remove_green_screen(
    img: Image.Image,
    tolerance: int = 80,
    edge_blend: int = 1,
) -> Image.Image:
    """Remove bright green (#00FF00) background via chroma-key.

    Args:
        img: RGBA or RGB image
        tolerance: How far from pure green a pixel can be and still count as bg.
                   Higher = more aggressive removal.
        edge_blend: Pixels within this many pixels of a removed pixel get
                    their alpha reduced for smoother edges.
    """
    img = img.convert("RGBA")
    pixels = img.load()
    w, h = img.size

    # First pass: mark green pixels as transparent
    mask = [[False] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            r, g, b, a = pixels[x, y]
            # Check if pixel is "green screen green":
            # High green channel, low red and blue
            if g > 100 and g > r + 40 and g > b + 40:
                greenness = g - max(r, b)
                if greenness > (130 - tolerance):
                    pixels[x, y] = (0, 0, 0, 0)
                    mask[y][x] = True

    # Edge blend: soften edges between subject and removed bg
    if edge_blend > 0:
        for y in range(h):
            for x in range(w):
                if mask[y][x]:
                    continue
                # Check if any neighbor was removed
                near_removed = False
                for dy in range(-edge_blend, edge_blend + 1):
                    for dx in range(-edge_blend, edge_blend + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny][nx]:
                            near_removed = True
                            break
                    if near_removed:
                        break
                if near_removed:
                    r, g, b, a = pixels[x, y]
                    # Reduce green spill on edge pixels
                    g_reduced = min(g, max(r, b) + 20)
                    pixels[x, y] = (r, g_reduced, b, max(0, a - 40))

    return img


def flood_fill_remove(
    img: Image.Image,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    tolerance: int = 35,
    start_from_edges: bool = True,
) -> Image.Image:
    """Remove background by flood-filling from edges.

    Only removes pixels that are connected to the image border,
    so dark pixels inside the subject are preserved.
    This solves the "dark subject on dark background" problem.

    Args:
        img: RGBA image
        bg_color: Expected background color (R, G, B)
        tolerance: How different a pixel can be from bg_color and still count
        start_from_edges: If True, only flood from image borders
    """
    img = img.convert("RGBA")
    pixels = img.load()
    w, h = img.size
    visited = [[False] * w for _ in range(h)]
    to_remove = [[False] * w for _ in range(h)]

    def is_bg(x: int, y: int) -> bool:
        r, g, b, a = pixels[x, y]
        dr = abs(r - bg_color[0])
        dg = abs(g - bg_color[1])
        db = abs(b - bg_color[2])
        return (dr + dg + db) < tolerance * 3

    # Start flood fill from all edge pixels that match bg
    queue = deque()
    for x in range(w):
        for y in [0, h - 1]:
            if is_bg(x, y):
                queue.append((x, y))
                visited[y][x] = True
    for y in range(h):
        for x in [0, w - 1]:
            if is_bg(x, y) and not visited[y][x]:
                queue.append((x, y))
                visited[y][x] = True

    # BFS flood fill
    while queue:
        x, y = queue.popleft()
        to_remove[y][x] = True
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not visited[ny][nx]:
                visited[ny][nx] = True
                if is_bg(nx, ny):
                    queue.append((nx, ny))

    # Apply removal
    for y in range(h):
        for x in range(w):
            if to_remove[y][x]:
                pixels[x, y] = (0, 0, 0, 0)

    return img


def auto_remove_background(
    img: Image.Image,
    method: str = "auto",
) -> Image.Image:
    """Automatically remove background using the best available method.

    Args:
        method: "green" for green screen, "flood" for flood fill from edges,
                "auto" to detect (checks if image has green-dominant corners)
    """
    img = img.convert("RGBA")
    pixels = img.load()
    w, h = img.size

    if method == "auto":
        # Sample corner pixels to detect bg type
        corners = [
            pixels[2, 2],
            pixels[w - 3, 2],
            pixels[2, h - 3],
            pixels[w - 3, h - 3],
        ]
        avg_r = sum(c[0] for c in corners) / 4
        avg_g = sum(c[1] for c in corners) / 4
        avg_b = sum(c[2] for c in corners) / 4

        if avg_g > 100 and avg_g > avg_r + 30 and avg_g > avg_b + 30:
            method = "green"
        else:
            method = "flood"

    if method == "green":
        return remove_green_screen(img)
    elif method == "flood":
        bg = (
            int(sum(pixels[x, y][0] for x, y in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]) / 4),
            int(sum(pixels[x, y][1] for x, y in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]) / 4),
            int(sum(pixels[x, y][2] for x, y in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]) / 4),
        )
        return flood_fill_remove(img, bg_color=bg)
    else:
        raise ValueError(f"Unknown bg removal method: {method}")
