"""Image resize utilities for game art — nearest-neighbor for pixel art."""

from PIL import Image


def nearest_neighbor_downscale(img: Image.Image, target_size: int) -> Image.Image:
    """Downscale image to target_size x target_size using nearest-neighbor.

    This preserves hard pixel edges — critical for pixel art.
    Handles non-square images by fitting to a square canvas.
    """
    img = img.convert("RGBA")

    # If already at target size, return as-is
    if img.width == target_size and img.height == target_size:
        return img

    # Resize with nearest-neighbor interpolation
    return img.resize((target_size, target_size), Image.NEAREST)


def center_on_canvas(
    img: Image.Image,
    canvas_size: int,
    background: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Image.Image:
    """Center a smaller image on a larger transparent canvas."""
    if img.width >= canvas_size and img.height >= canvas_size:
        return img

    canvas = Image.new("RGBA", (canvas_size, canvas_size), background)
    x = (canvas_size - img.width) // 2
    y = (canvas_size - img.height) // 2
    canvas.paste(img, (x, y), img if img.mode == "RGBA" else None)
    return canvas


def trim_transparent(img: Image.Image, padding: int = 0) -> Image.Image:
    """Crop transparent borders from an image.

    Useful for cleaning up generated sprites that have excess whitespace.
    """
    img = img.convert("RGBA")
    bbox = img.getbbox()
    if bbox is None:
        return img

    if padding > 0:
        x0 = max(0, bbox[0] - padding)
        y0 = max(0, bbox[1] - padding)
        x1 = min(img.width, bbox[2] + padding)
        y1 = min(img.height, bbox[3] + padding)
        bbox = (x0, y0, x1, y1)

    return img.crop(bbox)
