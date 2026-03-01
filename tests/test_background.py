"""Tests for background removal processing."""

import pytest
from PIL import Image

from game_art.processing.background import (
    remove_green_screen,
    flood_fill_remove,
    auto_remove_background,
)


def _make_image_with_bg(bg_color, subject_color, size=64):
    """Create a test image: bg_color border, subject_color center block."""
    img = Image.new("RGBA", (size, size), (*bg_color, 255))
    # Draw a 32x32 subject in the center
    for y in range(16, 48):
        for x in range(16, 48):
            img.putpixel((x, y), (*subject_color, 255))
    return img


def test_green_screen_removal():
    img = _make_image_with_bg((0, 255, 0), (200, 50, 50))
    result = remove_green_screen(img)
    px = result.load()

    # Corner should be transparent (was green)
    assert px[0, 0][3] == 0
    # Center should be opaque (was red)
    assert px[32, 32][3] > 0
    assert px[32, 32][0] > 150  # still red-ish


def test_flood_fill_black_bg():
    img = _make_image_with_bg((0, 0, 0), (100, 150, 200))
    result = flood_fill_remove(img, bg_color=(0, 0, 0), tolerance=30)
    px = result.load()

    # Corner should be transparent
    assert px[0, 0][3] == 0
    # Center should be opaque
    assert px[32, 32][3] > 0


def test_flood_fill_preserves_dark_subject_interior():
    """Dark pixels inside the subject should NOT be removed."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 255))  # black bg
    # Draw a light border with dark interior
    for y in range(16, 48):
        for x in range(16, 48):
            if x == 16 or x == 47 or y == 16 or y == 47:
                img.putpixel((x, y), (200, 200, 200, 255))  # light border
            else:
                img.putpixel((x, y), (10, 10, 10, 255))  # dark interior

    result = flood_fill_remove(img, bg_color=(0, 0, 0), tolerance=30)
    px = result.load()

    # Background corner: removed
    assert px[0, 0][3] == 0
    # Light border: preserved
    assert px[16, 16][3] > 0
    # Dark interior: preserved (not connected to edge bg)
    assert px[30, 30][3] > 0


def test_auto_detect_green():
    img = _make_image_with_bg((0, 230, 0), (200, 50, 50))
    result = auto_remove_background(img, method="auto")
    px = result.load()
    assert px[0, 0][3] == 0  # green corner removed


def test_auto_detect_black():
    img = _make_image_with_bg((5, 5, 5), (200, 50, 50))
    result = auto_remove_background(img, method="auto")
    px = result.load()
    assert px[0, 0][3] == 0  # black corner removed
    assert px[32, 32][3] > 0  # red subject preserved
