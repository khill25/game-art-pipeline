"""Tests for image providers."""

import pytest
from PIL import Image
import io

from game_art.providers.base import GenerationParams, ProviderCapability
from game_art.providers.placeholder import PlaceholderProvider


@pytest.fixture
def placeholder():
    return PlaceholderProvider()


@pytest.mark.asyncio
async def test_placeholder_generates_png(placeholder):
    params = GenerationParams(prompt="test sword", width=64, height=64)
    result = await placeholder.txt2img(params)

    assert isinstance(result, bytes)
    assert len(result) > 0

    # Verify it's valid PNG
    img = Image.open(io.BytesIO(result))
    assert img.size == (64, 64)
    assert img.mode == "RGBA"


@pytest.mark.asyncio
async def test_placeholder_deterministic_color(placeholder):
    params1 = GenerationParams(prompt="fire sword", width=32, height=32)
    params2 = GenerationParams(prompt="fire sword", width=32, height=32)

    result1 = await placeholder.txt2img(params1)
    result2 = await placeholder.txt2img(params2)

    assert result1 == result2


@pytest.mark.asyncio
async def test_placeholder_different_prompts_different_colors(placeholder):
    params1 = GenerationParams(prompt="fire sword", width=32, height=32)
    params2 = GenerationParams(prompt="ice shield", width=32, height=32)

    result1 = await placeholder.txt2img(params1)
    result2 = await placeholder.txt2img(params2)

    assert result1 != result2


def test_placeholder_capabilities(placeholder):
    assert ProviderCapability.TXT2IMG in placeholder.capabilities


@pytest.mark.asyncio
async def test_placeholder_health(placeholder):
    assert await placeholder.check_health() is True
