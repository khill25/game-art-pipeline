"""Batch sprite generation with consistent style across multiple assets."""

import asyncio
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
    ):
        self.generator = SpriteGenerator(provider, style)
        self.output_dir = output_dir
        self.concurrency = concurrency
        self.on_progress = on_progress

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
                return SpriteResult(request=req, output_path=path)
            else:
                png = await self.generator.generate(
                    prompt=req.prompt,
                    category=req.category,
                    seed=req.seed,
                    extra_positive=req.extra_positive,
                )
                return SpriteResult(request=req, png_bytes=png)
        except Exception as e:
            return SpriteResult(request=req, error=str(e))

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
