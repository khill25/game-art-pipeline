"""Tests for art styles."""

import pytest
from game_art.styles.pixel_art import PixelArtStyle


def test_pixel_art_build_prompt():
    style = PixelArtStyle(target_size=32)
    prompt = style.build_prompt("fire sword", category="weapon")

    assert "pixel art" in prompt
    assert "fire sword" in prompt
    assert "weapon sprite" in prompt


def test_pixel_art_build_prompt_v1():
    style = PixelArtStyle(target_size=32, use_v2_prompts=False)
    prompt = style.build_prompt("fire sword", category="weapon")

    assert "pixel art" in prompt
    assert "fire sword" in prompt
    assert "game weapon sprite" in prompt


def test_pixel_art_build_prompt_with_palette():
    style = PixelArtStyle(palette=["#ff0000", "#00ff00"], target_size=32)
    prompt = style.build_prompt("ice shield", category="weapon")

    assert "#ff0000" in prompt
    assert "#00ff00" in prompt


def test_pixel_art_negative_prompt():
    style = PixelArtStyle(target_size=32)
    neg = style.build_negative_prompt()

    assert "blurry" in neg
    assert "realistic" in neg


def test_pixel_art_gen_params():
    style = PixelArtStyle(target_size=32)
    params = style.get_gen_params()

    assert "steps" in params
    assert "cfg_scale" in params
    assert params["steps"] > 0


def test_pixel_art_category_hints():
    style = PixelArtStyle(target_size=32)

    weapon_prompt = style.build_prompt("sword", category="weapon")
    enemy_prompt = style.build_prompt("goblin", category="enemy")
    char_prompt = style.build_prompt("knight", category="character")

    # Each category should have different hints
    assert "weapon sprite" in weapon_prompt
    assert "enemy creature" in enemy_prompt
    assert "hero character" in char_prompt
