"""Single sprite generation — the core generator."""

import io
from pathlib import Path
from typing import Optional

from game_art.providers.base import ImageProvider, GenerationParams
from game_art.styles.base import Style, StyleConfig


class SpriteGenerator:
    """Generate individual game sprites with style-aware prompt construction
    and post-processing.

    Usage:
        from game_art import SpriteGenerator, PixelArtStyle
        from game_art.providers import SDWebUIProvider

        provider = SDWebUIProvider(url="http://192.168.50.181:7860")
        style = PixelArtStyle(palette=["#2d1b2e", "#5c3a6e"], target_size=32)
        gen = SpriteGenerator(provider, style)

        png_bytes = await gen.generate("fire sword", category="weapon")
    """

    def __init__(self, provider: ImageProvider, style: Style):
        self.provider = provider
        self.style = style

    async def generate(
        self,
        prompt: str,
        category: str = "item",
        seed: int = -1,
        extra_positive: str = "",
        extra_negative: str = "",
    ) -> bytes:
        """Generate a single sprite.

        Args:
            prompt: Description of the sprite (e.g., "fire sword", "skeleton warrior")
            category: Asset type — "weapon", "enemy", "character", "pickup",
                     "tile", "icon", "effect", "boss", "npc"
            seed: Random seed (-1 for random)
            extra_positive: Additional positive prompt text
            extra_negative: Additional negative prompt text

        Returns:
            PNG bytes of the final sprite at the style's target size.
        """
        # Build styled prompt
        full_prompt = self.style.build_prompt(prompt, category)
        if extra_positive:
            full_prompt += f", {extra_positive}"

        neg_prompt = self.style.build_negative_prompt()
        if extra_negative:
            neg_prompt += f", {extra_negative}"

        # Build generation params
        style_params = self.style.get_gen_params()
        params = GenerationParams(
            prompt=full_prompt,
            negative_prompt=neg_prompt,
            width=self.style.config.gen_width,
            height=self.style.config.gen_height,
            steps=style_params.get("steps", 20),
            cfg_scale=style_params.get("cfg_scale", 7.0),
            sampler=style_params.get("sampler", "Euler a"),
            seed=seed,
        )

        # Generate raw image
        raw_bytes = await self.provider.txt2img(params)

        # Apply style post-processing (downscale, palette, etc.)
        return self.style.post_process(raw_bytes)

    async def generate_to_file(
        self,
        prompt: str,
        output_path: Path,
        category: str = "item",
        seed: int = -1,
        extra_positive: str = "",
        extra_negative: str = "",
    ) -> Path:
        """Generate a sprite and save to a file.

        Creates parent directories if needed. Returns the output path.
        """
        png_bytes = await self.generate(
            prompt=prompt,
            category=category,
            seed=seed,
            extra_positive=extra_positive,
            extra_negative=extra_negative,
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(png_bytes)
        return output_path
