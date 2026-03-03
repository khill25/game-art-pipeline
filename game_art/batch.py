"""Batch sprite generation with consistent style across multiple assets."""

import asyncio
import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from game_art.providers.base import ImageProvider
from game_art.styles.base import Style
from game_art.generators.sprite import SpriteGenerator


@dataclass
class SpriteRequest:
    """A single sprite to generate in a batch."""
    id: str
    prompt: str
    category: str = "item"
    output_subdir: str = ""
    seed: int = -1
    extra_positive: str = ""


@dataclass
class SpriteResult:
    """Result of generating a single sprite."""
    request: SpriteRequest
    output_path: Optional[Path] = None
    png_bytes: Optional[bytes] = None
    error: Optional[str] = None
    pivots: Optional[list] = None  # list of PivotPoint dicts if detection was run
    reference_path: Optional[Path] = None  # 256px LANCZOS reference image
    validation_issues: list[str] = field(default_factory=list)
    attempts: int = 1

    @property
    def success(self) -> bool:
        return self.error is None


class BatchGenerator:
    """Generate multiple sprites with consistent style.

    Manages concurrency, error handling, and progress reporting
    for generating an entire game's worth of sprites.

    Usage:
        gen = BatchGenerator(provider, style, output_dir=Path("sprites/"))
        requests = [
            SpriteRequest(id="sword", prompt="fire sword", category="weapon", output_subdir="weapons"),
            SpriteRequest(id="skeleton", prompt="skeleton warrior", category="enemy", output_subdir="enemies"),
        ]
        results = await gen.generate_batch(requests)
    """

    def __init__(
        self,
        provider: ImageProvider,
        style: Style,
        output_dir: Optional[Path] = None,
        concurrency: int = 2,
        on_progress: Optional[callable] = None,
        detect_pivots: bool = False,
        use_v2: bool = True,
        reference_size: int = 256,
        max_retries: int = 3,
    ):
        self.generator = SpriteGenerator(provider, style)
        self.output_dir = output_dir
        self.concurrency = concurrency
        self.on_progress = on_progress
        self.detect_pivots = detect_pivots
        self.use_v2 = use_v2
        self.reference_size = reference_size
        self.max_retries = max_retries

    async def generate_batch(
        self,
        requests: list[SpriteRequest],
    ) -> list[SpriteResult]:
        """Generate all sprites in the batch.

        Respects concurrency limit to avoid overwhelming the provider.
        Returns results in the same order as requests.
        """
        semaphore = asyncio.Semaphore(self.concurrency)
        results: list[SpriteResult] = [None] * len(requests)
        completed = 0

        async def gen_one(idx: int, req: SpriteRequest):
            nonlocal completed
            async with semaphore:
                result = await self._generate_single(req)
                results[idx] = result
                completed += 1
                if self.on_progress:
                    self.on_progress(completed, len(requests), req.id, result.success)

        tasks = [gen_one(i, req) for i, req in enumerate(requests)]
        await asyncio.gather(*tasks)
        return results

    async def _generate_single(self, req: SpriteRequest) -> SpriteResult:
        """Generate a single sprite, handling errors gracefully."""
        try:
            if self.use_v2:
                return await self._generate_single_v2(req)
            else:
                return await self._generate_single_v1(req)
        except Exception as e:
            return SpriteResult(request=req, error=str(e))

    async def _generate_single_v1(self, req: SpriteRequest) -> SpriteResult:
        """V1 pipeline: style handles all prompt building and post-processing."""
        pivots = None
        if self.output_dir:
            subdir = req.output_subdir or req.category
            out_path = self.output_dir / subdir / f"{req.id}.png"
            path = await self.generator.generate_to_file(
                prompt=req.prompt,
                output_path=out_path,
                category=req.category,
                seed=req.seed,
                extra_positive=req.extra_positive,
            )
            if self.detect_pivots:
                pivots = self._run_pivot_detection(path, req.category)
            return SpriteResult(request=req, output_path=path, pivots=pivots)
        else:
            png = await self.generator.generate(
                prompt=req.prompt,
                category=req.category,
                seed=req.seed,
                extra_positive=req.extra_positive,
            )
            if self.detect_pivots:
                pivots = self._run_pivot_detection_bytes(png, req.category)
            return SpriteResult(request=req, png_bytes=png, pivots=pivots)

    async def _generate_single_v2(self, req: SpriteRequest) -> SpriteResult:
        """V2 pipeline: multi-stage with provider-side cleanup and validation."""
        from game_art.generators.sprite import SpriteContext

        ctx = await self.generator.generate_v2(
            prompt=req.prompt,
            category=req.category,
            seed=req.seed,
            extra_positive=req.extra_positive,
            max_retries=self.max_retries,
            reference_size=self.reference_size,
            target_size=self.generator.style.config.target_size,
        )

        pivots = None
        output_path = None
        reference_path = None
        png_bytes = None

        if self.output_dir and ctx.final_image:
            # Save final game sprite
            subdir = req.output_subdir or req.category
            out_path = self.output_dir / subdir / f"{req.id}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ctx.final_image.save(out_path, format="PNG")
            output_path = out_path

            # Save reference image
            if ctx.reference_image:
                ref_subdir = subdir.replace("sprites/", "references/", 1)
                if ref_subdir == subdir:
                    ref_subdir = f"references/{subdir}"
                ref_path = self.output_dir / ref_subdir / f"{req.id}.png"
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                ctx.reference_image.save(ref_path, format="PNG")
                reference_path = ref_path

            # Pivot detection on reference (better quality than 32px)
            if self.detect_pivots and ctx.reference_image:
                pivots = self._run_pivot_detection_image(
                    ctx.reference_image, req.category
                )
        elif ctx.final_image:
            buf = io.BytesIO()
            ctx.final_image.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            if self.detect_pivots and ctx.reference_image:
                pivots = self._run_pivot_detection_image(
                    ctx.reference_image, req.category
                )

        return SpriteResult(
            request=req,
            output_path=output_path,
            png_bytes=png_bytes,
            pivots=pivots,
            reference_path=reference_path,
            validation_issues=ctx.issues,
            attempts=ctx.attempt,
        )

    @staticmethod
    def _run_pivot_detection(image_path: Path, category: str) -> list[dict]:
        """Detect pivots from a saved sprite file."""
        from PIL import Image as PILImage
        from game_art.processing.pivot import detect_pivots
        img = PILImage.open(image_path).convert("RGBA")
        pivots = detect_pivots(img, category=category)
        return [
            {"name": p.name, "x": round(p.x, 4), "y": round(p.y, 4), "confidence": round(p.confidence, 3)}
            for p in pivots
        ]

    @staticmethod
    def _run_pivot_detection_bytes(png_bytes: bytes, category: str) -> list[dict]:
        """Detect pivots from PNG bytes."""
        from PIL import Image as PILImage
        from game_art.processing.pivot import detect_pivots
        img = PILImage.open(io.BytesIO(png_bytes)).convert("RGBA")
        pivots = detect_pivots(img, category=category)
        return [
            {"name": p.name, "x": round(p.x, 4), "y": round(p.y, 4), "confidence": round(p.confidence, 3)}
            for p in pivots
        ]

    @staticmethod
    def _run_pivot_detection_image(img, category: str) -> list[dict]:
        """Detect pivots from a PIL Image directly."""
        from game_art.processing.pivot import detect_pivots
        img = img.convert("RGBA")
        pivots = detect_pivots(img, category=category)
        return [
            {"name": p.name, "x": round(p.x, 4), "y": round(p.y, 4), "confidence": round(p.confidence, 3)}
            for p in pivots
        ]

    @staticmethod
    def requests_from_pack_data(pack_data: dict) -> list[SpriteRequest]:
        """Build sprite requests from a VampireGPT-style pack data dict.

        Convenience method for game pipelines that produce structured
        content data with id/name/description fields.
        """
        requests = []

        for weapon in pack_data.get("weapons", []):
            requests.append(SpriteRequest(
                id=weapon["id"],
                prompt=f"{weapon.get('name', weapon['id'])} — {weapon.get('description', 'weapon')}",
                category="weapon",
                output_subdir="weapons",
            ))
            # Evolved variant
            evo = weapon.get("evolution", {})
            if evo and evo.get("evolves_into"):
                evo_id = evo["evolves_into"]
                evo_name = evo.get("evolved_name", evo_id)
                requests.append(SpriteRequest(
                    id=evo_id,
                    prompt=f"{evo_name} — evolved form of {weapon.get('name', '')}, more powerful",
                    category="weapon",
                    output_subdir="weapons",
                ))

        for passive in pack_data.get("passives", []):
            requests.append(SpriteRequest(
                id=passive["id"],
                prompt=f"{passive.get('name', passive['id'])} — {passive.get('description', 'passive item')}",
                category="pickup",
                output_subdir="pickups",
            ))

        for enemy in pack_data.get("enemies", []):
            requests.append(SpriteRequest(
                id=enemy["id"],
                prompt=f"{enemy.get('name', enemy['id'])} — {enemy.get('description', 'enemy creature')}",
                category="enemy",
                output_subdir="enemies",
            ))

        for char in pack_data.get("characters", []):
            requests.append(SpriteRequest(
                id=char["id"],
                prompt=f"{char.get('name', char['id'])} — {char.get('description', 'hero character')}",
                category="character",
                output_subdir="characters",
            ))

        # Background tile
        theme = pack_data.get("meta", {}).get("description", "game world")
        requests.append(SpriteRequest(
            id="background",
            prompt=f"ground texture, {theme}",
            category="tile",
            output_subdir="tiles",
        ))

        return requests
