"""Base style interface for game art generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StyleConfig:
    """Common style configuration."""
    # Target output size (sprites get generated larger then downscaled)
    target_size: int = 32
    # Generation resolution (AI generates at this size, then post-processes)
    gen_width: int = 512
    gen_height: int = 512
    # Color palette (hex strings like "#2d1b2e")
    palette: list[str] = field(default_factory=list)


class Style(ABC):
    """A style defines how prompts are constructed and images post-processed.

    Different art styles (pixel art, hand-drawn, painterly) need different
    prompt modifiers, negative prompts, generation parameters, and
    post-processing steps.
    """

    def __init__(self, config: Optional[StyleConfig] = None):
        self.config = config or StyleConfig()

    @abstractmethod
    def build_prompt(self, base_prompt: str, category: str = "item") -> str:
        """Add style-specific modifiers to the base prompt.

        Args:
            base_prompt: The core description (e.g., "fire sword")
            category: Asset type hint — "weapon", "enemy", "character",
                     "pickup", "tile", "icon", "effect"
        """

    @abstractmethod
    def build_negative_prompt(self) -> str:
        """Return the negative prompt for this style."""

    def get_gen_params(self) -> dict:
        """Return extra generation parameters for this style.

        Override to set steps, cfg_scale, sampler, etc.
        """
        return {}

    @abstractmethod
    def post_process(self, image_bytes: bytes) -> bytes:
        """Apply style-specific post-processing to the generated image.

        Takes raw PNG bytes from the provider, returns processed PNG bytes.
        """
