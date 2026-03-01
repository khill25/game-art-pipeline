"""Tests for sprite generator and batch generator."""

import pytest
import tempfile
from pathlib import Path
from PIL import Image
import io

from game_art.providers.placeholder import PlaceholderProvider
from game_art.styles.pixel_art import PixelArtStyle
from game_art.generators.sprite import SpriteGenerator
from game_art.batch import BatchGenerator, SpriteRequest


@pytest.fixture
def provider():
    return PlaceholderProvider()


@pytest.fixture
def style():
    return PixelArtStyle(target_size=32)


@pytest.fixture
def generator(provider, style):
    return SpriteGenerator(provider, style)


@pytest.mark.asyncio
async def test_generate_sprite(generator):
    result = await generator.generate("fire sword", category="weapon")
    assert isinstance(result, bytes)

    img = Image.open(io.BytesIO(result))
    assert img.size == (32, 32)
    assert img.mode == "RGBA"


@pytest.mark.asyncio
async def test_generate_sprite_with_palette():
    provider = PlaceholderProvider()
    style = PixelArtStyle(palette=["#ff0000", "#00ff00", "#0000ff"], target_size=32)
    gen = SpriteGenerator(provider, style)

    result = await gen.generate("fire sword", category="weapon")
    img = Image.open(io.BytesIO(result))

    # Check that all non-transparent pixels use only palette colors
    pixels = list(img.load()[x, y] for y in range(img.height) for x in range(img.width))
    palette_colors = {(255, 0, 0), (0, 255, 0), (0, 0, 255)}
    for r, g, b, a in pixels:
        if a > 32:
            assert (r, g, b) in palette_colors, f"Color ({r},{g},{b}) not in palette"


@pytest.mark.asyncio
async def test_generate_to_file(generator):
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "weapons" / "sword.png"
        result_path = await generator.generate_to_file(
            "fire sword", output_path=out, category="weapon"
        )
        assert result_path.exists()
        img = Image.open(result_path)
        assert img.size == (32, 32)


@pytest.mark.asyncio
async def test_batch_generate(provider, style):
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        batch = BatchGenerator(provider, style, output_dir=output_dir)

        requests = [
            SpriteRequest(id="sword", prompt="fire sword", category="weapon", output_subdir="weapons"),
            SpriteRequest(id="shield", prompt="ice shield", category="weapon", output_subdir="weapons"),
            SpriteRequest(id="skeleton", prompt="skeleton warrior", category="enemy", output_subdir="enemies"),
        ]

        results = await batch.generate_batch(requests)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert (output_dir / "weapons" / "sword.png").exists()
        assert (output_dir / "weapons" / "shield.png").exists()
        assert (output_dir / "enemies" / "skeleton.png").exists()


@pytest.mark.asyncio
async def test_batch_progress_callback(provider, style):
    progress_log = []

    def on_progress(completed, total, item_id, success):
        progress_log.append((completed, total, item_id, success))

    batch = BatchGenerator(provider, style, on_progress=on_progress)
    requests = [
        SpriteRequest(id="a", prompt="item a"),
        SpriteRequest(id="b", prompt="item b"),
    ]

    await batch.generate_batch(requests)
    assert len(progress_log) == 2
    assert all(p[3] for p in progress_log)  # all succeeded


@pytest.mark.asyncio
async def test_requests_from_pack_data():
    pack_data = {
        "weapons": [
            {"id": "sword", "name": "Fire Sword", "description": "A burning blade",
             "evolution": {"evolves_into": "inferno_blade", "evolved_name": "Inferno Blade"}},
        ],
        "passives": [
            {"id": "shield", "name": "Magic Shield", "description": "Absorbs damage"},
        ],
        "enemies": [
            {"id": "skeleton", "name": "Skeleton", "description": "Undead warrior"},
        ],
        "characters": [
            {"id": "wizard", "name": "Wizard", "description": "Fire mage"},
        ],
        "meta": {"description": "dark fantasy dungeon"},
    }

    requests = BatchGenerator.requests_from_pack_data(pack_data)

    # 1 weapon + 1 evolved + 1 passive + 1 enemy + 1 character + 1 bg tile
    assert len(requests) == 6

    ids = [r.id for r in requests]
    assert "sword" in ids
    assert "inferno_blade" in ids
    assert "shield" in ids
    assert "skeleton" in ids
    assert "wizard" in ids
    assert "background" in ids
