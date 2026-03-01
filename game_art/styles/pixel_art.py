"""Pixel art style — prompt templates and post-processing for retro game sprites."""

import io
from PIL import Image

from .base import Style, StyleConfig
from game_art.processing.palette import enforce_palette
from game_art.processing.resize import nearest_neighbor_downscale


# Category-specific prompt hints
CATEGORY_HINTS = {
    "weapon": "game weapon sprite, single item, centered",
    "enemy": "enemy creature sprite, game monster, full body",
    "character": "hero character sprite, game protagonist, full body, front-facing",
    "pickup": "collectible item sprite, pickup icon, centered",
    "tile": "seamless game tile, top-down, tileable texture",
    "icon": "game icon, UI element, clean simple design",
    "effect": "visual effect sprite, magic spell, glowing energy",
    "boss": "large boss creature sprite, imposing, detailed",
    "npc": "NPC character sprite, full body, front-facing",
}

BASE_POSITIVE = (
    "pixel art, {category_hint}, {prompt}, "
    "retro game style, clean pixels, sharp edges, transparent background, "
    "16-bit era, sprite, centered composition, no anti-aliasing"
)

BASE_NEGATIVE = (
    "blurry, smooth, realistic, photo, 3d render, text, watermark, "
    "signature, frame, border, multiple objects, busy background, "
    "anti-aliased, gradient shading, noisy, jpeg artifacts, "
    "low quality, deformed, ugly"
)


class PixelArtStyle(Style):
    """Pixel art style for retro game sprites.

    Generates at 512x512, then downscales with nearest-neighbor
    interpolation and optionally enforces a color palette.
    """

    def __init__(self, palette: list[str] = None, target_size: int = 32):
        config = StyleConfig(
            target_size=target_size,
            gen_width=512,
            gen_height=512,
            palette=palette or [],
        )
        super().__init__(config)

    def build_prompt(self, base_prompt: str, category: str = "item") -> str:
        hint = CATEGORY_HINTS.get(category, f"game {category} sprite, centered")
        prompt = BASE_POSITIVE.format(category_hint=hint, prompt=base_prompt)

        # Add palette colors as prompt hints if available
        if self.config.palette:
            color_str = ", ".join(self.config.palette[:6])
            prompt += f", color palette: {color_str}"

        return prompt

    def build_negative_prompt(self) -> str:
        return BASE_NEGATIVE

    def get_gen_params(self) -> dict:
        return {
            "steps": 25,
            "cfg_scale": 7.5,
            "sampler": "DPM++ 2M Karras",
        }

    def post_process(self, image_bytes: bytes) -> bytes:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        # Step 1: Downscale to target size with nearest-neighbor
        if img.width != self.config.target_size or img.height != self.config.target_size:
            img = nearest_neighbor_downscale(img, self.config.target_size)

        # Step 2: Enforce palette if one was provided
        if self.config.palette:
            img = enforce_palette(img, self.config.palette)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
