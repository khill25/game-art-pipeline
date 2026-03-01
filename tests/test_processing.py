"""Tests for image processing utilities."""

import pytest
from PIL import Image
import io

from game_art.processing.palette import hex_to_rgb, find_nearest_color, enforce_palette
from game_art.processing.resize import nearest_neighbor_downscale, center_on_canvas, trim_transparent


def test_hex_to_rgb_full():
    assert hex_to_rgb("#ff0000") == (255, 0, 0)
    assert hex_to_rgb("#00ff00") == (0, 255, 0)
    assert hex_to_rgb("#0000ff") == (0, 0, 255)
    assert hex_to_rgb("2d1b2e") == (45, 27, 46)


def test_hex_to_rgb_short():
    assert hex_to_rgb("#f00") == (255, 0, 0)
    assert hex_to_rgb("#0f0") == (0, 255, 0)


def test_find_nearest_color():
    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    # Red-ish color should match red
    assert find_nearest_color((200, 10, 10), palette) == (255, 0, 0)
    # Green-ish should match green
    assert find_nearest_color((10, 200, 10), palette) == (0, 255, 0)
    # Blue-ish should match blue
    assert find_nearest_color((10, 10, 200), palette) == (0, 0, 255)


def test_enforce_palette_remaps_colors():
    # Create a 4x4 image with known colors
    img = Image.new("RGBA", (4, 4), (200, 50, 50, 255))  # Red-ish
    palette_hex = ["#ff0000", "#00ff00", "#0000ff"]

    result = enforce_palette(img, palette_hex)
    px = result.load()

    # Should be remapped to pure red
    assert px[0, 0] == (255, 0, 0, 255)


def test_enforce_palette_transparent_threshold():
    # Create image with low alpha
    img = Image.new("RGBA", (4, 4), (200, 50, 50, 10))
    palette_hex = ["#ff0000"]

    result = enforce_palette(img, palette_hex, alpha_threshold=32)
    px = result.load()

    # Should be fully transparent
    assert px[0, 0] == (0, 0, 0, 0)


def test_enforce_palette_empty():
    img = Image.new("RGBA", (4, 4), (100, 100, 100, 255))
    result = enforce_palette(img, [])
    # Should return unchanged
    px = result.load()
    assert px[0, 0] == (100, 100, 100, 255)


def test_nearest_neighbor_downscale():
    img = Image.new("RGBA", (512, 512), (255, 0, 0, 255))
    result = nearest_neighbor_downscale(img, 32)
    assert result.size == (32, 32)
    # Color should be preserved
    assert result.load()[0, 0] == (255, 0, 0, 255)


def test_nearest_neighbor_already_target():
    img = Image.new("RGBA", (32, 32), (0, 255, 0, 255))
    result = nearest_neighbor_downscale(img, 32)
    assert result.size == (32, 32)


def test_center_on_canvas():
    small = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
    result = center_on_canvas(small, 32)
    assert result.size == (32, 32)

    # Center pixel should be red
    px = result.load()
    assert px[16, 16][:3] == (255, 0, 0)
    # Corner should be transparent
    assert px[0, 0] == (0, 0, 0, 0)


def test_trim_transparent():
    # 32x32 transparent with a 10x10 red block in the center
    img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(10, 20):
        for x in range(10, 20):
            img.putpixel((x, y), (255, 0, 0, 255))

    result = trim_transparent(img)
    assert result.size == (10, 10)
