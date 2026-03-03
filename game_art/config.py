"""Configuration for game-art-pipeline providers and defaults."""

import os
from dataclasses import dataclass


@dataclass
class GameArtConfig:
    """Configuration loaded from environment or passed directly."""

    # Provider selection
    provider: str = "placeholder"  # "placeholder", "sdwebui", "comfyui", "diffusers"

    # SD WebUI / Forge settings
    sdwebui_url: str = "http://localhost:7860"
    sdwebui_timeout: float = 120.0

    # ComfyUI settings
    comfyui_url: str = "http://localhost:8088"
    comfyui_timeout: float = 300.0
    comfyui_checkpoint: str = ""  # empty = auto-detect first available

    # Default generation settings
    default_style: str = "pixel_art"
    default_size: int = 32
    default_steps: int = 25
    default_cfg: float = 7.5

    # Batch settings
    batch_concurrency: int = 2

    # V2 pipeline settings
    reference_size: int = 256       # LANCZOS reference image size
    max_retries: int = 3            # Retry count for failed validation
    use_v2_prompts: bool = True     # Use v2 prompt builder + 1024px generation

    @classmethod
    def from_env(cls) -> "GameArtConfig":
        """Load config from environment variables."""
        return cls(
            provider=os.environ.get("GAME_ART_PROVIDER", "placeholder"),
            sdwebui_url=os.environ.get("GAME_ART_SDWEBUI_URL", "http://localhost:7860"),
            sdwebui_timeout=float(os.environ.get("GAME_ART_SDWEBUI_TIMEOUT", "120")),
            comfyui_url=os.environ.get("GAME_ART_COMFYUI_URL", "http://localhost:8088"),
            comfyui_timeout=float(os.environ.get("GAME_ART_COMFYUI_TIMEOUT", "300")),
            comfyui_checkpoint=os.environ.get("GAME_ART_COMFYUI_CHECKPOINT", ""),
            default_size=int(os.environ.get("GAME_ART_SPRITE_SIZE", "32")),
            default_steps=int(os.environ.get("GAME_ART_STEPS", "25")),
            default_cfg=float(os.environ.get("GAME_ART_CFG", "7.5")),
            batch_concurrency=int(os.environ.get("GAME_ART_CONCURRENCY", "2")),
            reference_size=int(os.environ.get("GAME_ART_REFERENCE_SIZE", "256")),
            max_retries=int(os.environ.get("GAME_ART_MAX_RETRIES", "3")),
            use_v2_prompts=os.environ.get("GAME_ART_V2_PROMPTS", "true").lower() in ("true", "1", "yes"),
        )
