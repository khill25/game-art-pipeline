"""Configuration for game-art-pipeline providers and defaults."""

import os
from dataclasses import dataclass


@dataclass
class GameArtConfig:
    """Configuration loaded from environment or passed directly."""

    # Provider selection
    provider: str = "placeholder"  # "placeholder", "sdwebui", "diffusers"

    # SD WebUI / Forge settings
    sdwebui_url: str = "http://localhost:7860"
    sdwebui_timeout: float = 120.0

    # Default generation settings
    default_style: str = "pixel_art"
    default_size: int = 32
    default_steps: int = 25
    default_cfg: float = 7.5

    # Batch settings
    batch_concurrency: int = 2

    @classmethod
    def from_env(cls) -> "GameArtConfig":
        """Load config from environment variables."""
        return cls(
            provider=os.environ.get("GAME_ART_PROVIDER", "placeholder"),
            sdwebui_url=os.environ.get("GAME_ART_SDWEBUI_URL", "http://localhost:7860"),
            sdwebui_timeout=float(os.environ.get("GAME_ART_SDWEBUI_TIMEOUT", "120")),
            default_size=int(os.environ.get("GAME_ART_SPRITE_SIZE", "32")),
            default_steps=int(os.environ.get("GAME_ART_STEPS", "25")),
            default_cfg=float(os.environ.get("GAME_ART_CFG", "7.5")),
            batch_concurrency=int(os.environ.get("GAME_ART_CONCURRENCY", "2")),
        )
