"""Pixel art style — prompt templates and post-processing for retro game sprites."""

import io
from typing import Optional
from PIL import Image

from .base import Style, StyleConfig
from game_art.processing.palette import enforce_palette
from game_art.processing.resize import nearest_neighbor_downscale, trim_transparent, center_on_canvas
from game_art.processing.background import auto_remove_background
from game_art.processing.orientation import normalize_orientation


# Category-specific prompt hints
CATEGORY_HINTS = {
    "weapon": "game weapon sprite, single item only, centered, isolated object",
    "enemy": "enemy creature sprite, game monster, full body, single character",
    "character": "hero character sprite, game protagonist, full body, front-facing, single character",
    "pickup": "collectible item sprite, pickup icon, centered, single item only",
    "tile": "seamless game tile, top-down, tileable texture",
    "icon": "game icon, UI element, clean simple design, single icon",
    "effect": "visual effect sprite, magic spell, glowing energy",
    "boss": "large boss creature sprite, imposing, detailed, full body, single character",
    "npc": "NPC character sprite, full body, front-facing, single character",
}

# Per-category recommended output sizes (can be overridden)
CATEGORY_SIZES = {
    "weapon": 48,
    "enemy": 32,
    "character": 32,
    "pickup": 24,
    "tile": 32,
    "icon": 32,
    "effect": 48,
    "boss": 64,
    "npc": 32,
}

BASE_POSITIVE = (
    "pixel art, {category_hint}, {prompt}, "
    "retro game style, clean pixels, sharp edges, solid black background, "
    "16-bit era, sprite, centered composition, no anti-aliasing"
)

BASE_NEGATIVE = (
    "blurry, smooth, realistic, photo, 3d render, text, watermark, "
    "signature, frame, border, multiple objects, busy background, "
    "anti-aliased, gradient shading, noisy, jpeg artifacts, "
    "low quality, deformed, ugly, grid, checkerboard pattern, "
    "extra items, additional objects, multiple sprites, "
    "cropped, cut off, partial"
)


class PixelArtStyle(Style):
    """Pixel art style for retro game sprites.

    V1 mode: Generates at 512x512, Python bg removal, trim, NN downscale.
    V2 mode: Generates at 1024x1024, provider-side rembg, Python cleanup,
             orientation on hi-res, LANCZOS 256px reference, NN 32px game sprite.

    The v2 pipeline is used when use_v2_prompts=True (default). The v1 post_process()
    path remains for backward compatibility with existing callers.
    """

    def __init__(
        self,
        palette: list[str] = None,
        target_size: int = 32,
        lora: Optional[str] = None,
        lora_weight: float = 0.8,
        use_category_sizes: bool = False,
        remove_bg: bool = True,
        enforce_colors: bool = False,
        normalize_orient: bool = True,
        use_v2_prompts: bool = True,
    ):
        """
        Args:
            palette: Hex color strings for palette hints in prompts.
            target_size: Default output size in pixels.
            lora: LoRA name to inject in prompt (e.g., "pixel-art-xl-v1.1").
            lora_weight: LoRA weight (0.0-1.0).
            use_category_sizes: If True, use per-category sizes instead of target_size.
            remove_bg: If True, auto-remove background in post-processing.
            enforce_colors: If True, remap all pixels to palette colors.
                           Usually too aggressive — leave False and let the
                           palette prompt hints guide SD naturally.
            normalize_orient: If True, rotate weapons to canonical tip-up pose.
            use_v2_prompts: If True, generate at 1024px and use v2 prompt builder.
                           v2 produces stronger isolation and category-specific negatives.
        """
        gen_size = 1024 if use_v2_prompts else 512
        config = StyleConfig(
            target_size=target_size,
            gen_width=gen_size,
            gen_height=gen_size,
            palette=palette or [],
        )
        super().__init__(config)
        self.lora = lora
        self.lora_weight = lora_weight
        self.use_category_sizes = use_category_sizes
        self.remove_bg = remove_bg
        self.enforce_colors = enforce_colors
        self.normalize_orient = normalize_orient
        self.use_v2_prompts = use_v2_prompts
        self._current_category = "item"

    def _get_target_size(self, category: str) -> int:
        """Get the target output size for a category."""
        if self.use_category_sizes:
            return CATEGORY_SIZES.get(category, self.config.target_size)
        return self.config.target_size

    def build_prompt(self, base_prompt: str, category: str = "item") -> str:
        self._current_category = category

        if self.use_v2_prompts:
            from game_art.pipeline.prompts import build_prompt_v2
            positive, _ = build_prompt_v2(
                base_prompt=base_prompt,
                category=category,
                palette=self.config.palette or None,
                lora=self.lora,
                lora_weight=self.lora_weight,
            )
            return positive

        hint = CATEGORY_HINTS.get(category, f"game {category} sprite, centered")
        prompt = BASE_POSITIVE.format(category_hint=hint, prompt=base_prompt)

        # Add palette colors as prompt hints
        if self.config.palette:
            color_str = ", ".join(self.config.palette[:6])
            prompt += f", color palette: {color_str}"

        # Inject LoRA trigger
        if self.lora:
            prompt += f", <lora:{self.lora}:{self.lora_weight}>"

        return prompt

    def build_negative_prompt(self) -> str:
        if self.use_v2_prompts:
            from game_art.pipeline.prompts import build_prompt_v2
            _, negative = build_prompt_v2(
                base_prompt="",
                category=self._current_category,
            )
            return negative

        return BASE_NEGATIVE

    def get_gen_params(self) -> dict:
        params = {
            "steps": 25,
            "cfg_scale": 7.5,
            "sampler": "DPM++ 2M Karras",
        }
        if self.lora:
            params["lora"] = self.lora
            params["lora_weight"] = self.lora_weight
        return params

    def post_process(self, image_bytes: bytes) -> bytes:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

        target = self._get_target_size(self._current_category)

        # Step 1: Remove background
        if self.remove_bg:
            img = auto_remove_background(img, method="auto")

        # Step 2: Normalize orientation (weapons tip-up, etc.)
        if self.normalize_orient:
            img = normalize_orientation(img, category=self._current_category)

        # Step 3: Trim transparent borders
        img = trim_transparent(img, padding=4)

        # Step 4: Fit to square canvas (preserve aspect ratio)
        max_dim = max(img.width, img.height)
        if max_dim > 0:
            img = center_on_canvas(img, max_dim)

        # Step 5: Downscale to target size with nearest-neighbor
        if img.width != target or img.height != target:
            img = nearest_neighbor_downscale(img, target)

        # Step 6: Enforce palette only if explicitly requested
        if self.enforce_colors and self.config.palette:
            img = enforce_palette(img, self.config.palette)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
