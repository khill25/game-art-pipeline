"""Single sprite generation — the core generator.

Supports two paths:
- generate() — v1 pipeline (style builds prompts, style does all post-processing)
- generate_v2() — multi-stage pipeline (v2 prompts, provider-side cleanup,
                   Python cleanup + validation, orientation on hi-res, two output sizes)
"""

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from game_art.providers.base import ImageProvider, GenerationParams
from game_art.styles.base import Style, StyleConfig


@dataclass
class SpriteContext:
    """Tracks state through the multi-stage v2 pipeline."""
    prompt: str
    category: str
    seed: int
    raw_image: Optional[Image.Image] = None       # hi-res from provider (bg removed)
    cleaned_image: Optional[Image.Image] = None    # after component cleanup
    reference_image: Optional[Image.Image] = None  # 256px LANCZOS
    final_image: Optional[Image.Image] = None      # 32px NN
    pivots: Optional[list] = None
    validated: bool = False
    issues: list[str] = field(default_factory=list)
    attempt: int = 1
    caption: str = ""


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
        """Generate a single sprite (v1 pipeline).

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

        # Extract LoRA into extra dict for providers that need structured data
        # (e.g. ComfyUI injects a LoraLoader node; SDWebUI uses prompt syntax)
        extra = {}
        if style_params.get("lora"):
            extra["lora"] = style_params["lora"]
            extra["lora_weight"] = style_params.get("lora_weight", 0.8)

        params = GenerationParams(
            prompt=full_prompt,
            negative_prompt=neg_prompt,
            width=self.style.config.gen_width,
            height=self.style.config.gen_height,
            steps=style_params.get("steps", 20),
            cfg_scale=style_params.get("cfg_scale", 7.0),
            sampler=style_params.get("sampler", "Euler a"),
            seed=seed,
            extra=extra,
        )

        # Generate raw image
        raw_bytes = await self.provider.txt2img(params)

        # Apply style post-processing (downscale, palette, etc.)
        return self.style.post_process(raw_bytes)

    async def generate_v2(
        self,
        prompt: str,
        category: str = "item",
        seed: int = -1,
        extra_positive: str = "",
        extra_negative: str = "",
        max_retries: int = 3,
        reference_size: int = 256,
        target_size: int = 32,
        palette: list[str] | None = None,
        lora: str | None = None,
        lora_weight: float = 0.8,
    ) -> SpriteContext:
        """Generate a sprite using the multi-stage v2 pipeline.

        Flow:
        1. Build v2 prompts (strong isolation hints)
        2. provider.txt2img_with_cleanup() at gen resolution (bg removed by GPU)
        3. Python cleanup: largest component + base removal for chars/enemies
        4. Validate: retry with seed+1 if quality checks fail
        5. Normalize orientation on hi-res image (much better PCA than 32px)
        6. LANCZOS downscale to reference_size → reference_image
        7. Nearest-neighbor downscale to target_size → final_image

        Returns:
            SpriteContext with all intermediate images and metadata.
        """
        from game_art.pipeline.prompts import build_prompt_v2
        from game_art.pipeline.validation import validate_sprite
        from game_art.processing.cleanup import (
            compute_fill_ratio,
            keep_largest_component,
            remove_character_base,
        )
        from game_art.processing.background import auto_remove_background
        from game_art.processing.orientation import normalize_orientation
        from game_art.processing.resize import (
            trim_transparent,
            center_on_canvas,
            nearest_neighbor_downscale,
        )

        style_params = self.style.get_gen_params()

        # Build v2 prompts
        positive, negative = build_prompt_v2(
            base_prompt=prompt,
            category=category,
            palette=palette or getattr(self.style.config, "palette", None),
            lora=lora or style_params.get("lora"),
            lora_weight=lora_weight,
            extra_positive=extra_positive,
        )
        if extra_negative:
            negative += f", {extra_negative}"

        # Extract LoRA for providers that need structured data
        extra = {}
        effective_lora = lora or style_params.get("lora")
        if effective_lora:
            extra["lora"] = effective_lora
            extra["lora_weight"] = lora_weight

        ctx = SpriteContext(prompt=prompt, category=category, seed=seed)

        for attempt in range(1, max_retries + 1):
            ctx.attempt = attempt
            current_seed = seed + (attempt - 1) if seed >= 0 else seed

            params = GenerationParams(
                prompt=positive,
                negative_prompt=negative,
                width=self.style.config.gen_width,
                height=self.style.config.gen_height,
                steps=style_params.get("steps", 20),
                cfg_scale=style_params.get("cfg_scale", 7.0),
                sampler=style_params.get("sampler", "Euler a"),
                seed=current_seed,
                extra=extra,
            )

            # Step 1: Generate — tiles skip bg removal (they fill the whole frame)
            if category == "tile":
                raw_bytes = await self.provider.txt2img(params)
            else:
                raw_bytes = await self.provider.txt2img_with_cleanup(params)
            ctx.raw_image = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")

            if category == "tile":
                # Tiles don't need cleanup — they're meant to fill the frame
                ctx.cleaned_image = ctx.raw_image
            else:
                # Step 1b: If provider didn't remove bg (fill ~100%), do it Python-side
                raw_fill = compute_fill_ratio(ctx.raw_image)
                if raw_fill > 0.95:
                    ctx.raw_image = auto_remove_background(ctx.raw_image, method="auto")

                # Step 2: Python-side cleanup
                cleaned = keep_largest_component(ctx.raw_image)
                if category in ("character", "enemy", "boss", "npc"):
                    cleaned = remove_character_base(cleaned)
                ctx.cleaned_image = cleaned

            # Step 3: Validate
            passed, issues = validate_sprite(ctx.cleaned_image, category)
            ctx.issues = issues
            ctx.validated = passed

            if passed or attempt == max_retries:
                break
            # Retry with different seed

        # Step 4: Orientation normalization on hi-res image
        oriented = normalize_orientation(ctx.cleaned_image, category=category)

        # Step 5: Trim and create reference image (LANCZOS)
        trimmed = trim_transparent(oriented, padding=4)
        max_dim = max(trimmed.width, trimmed.height)
        if max_dim > 0:
            trimmed = center_on_canvas(trimmed, max_dim)

        if trimmed.width != reference_size or trimmed.height != reference_size:
            ctx.reference_image = trimmed.resize(
                (reference_size, reference_size), Image.LANCZOS
            )
        else:
            ctx.reference_image = trimmed

        # Step 6: Final game sprite (nearest-neighbor)
        if trimmed.width != target_size or trimmed.height != target_size:
            ctx.final_image = nearest_neighbor_downscale(trimmed, target_size)
        else:
            ctx.final_image = trimmed

        # Optionally enforce palette on final
        if getattr(self.style, "enforce_colors", False):
            style_palette = getattr(self.style.config, "palette", [])
            if style_palette:
                from game_art.processing.palette import enforce_palette
                ctx.final_image = enforce_palette(ctx.final_image, style_palette)

        return ctx

    async def generate_to_file(
        self,
        prompt: str,
        output_path: Path,
        category: str = "item",
        seed: int = -1,
        extra_positive: str = "",
        extra_negative: str = "",
    ) -> Path:
        """Generate a sprite and save to a file (v1 pipeline).

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
