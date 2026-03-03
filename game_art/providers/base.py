"""Base image provider interface for game art generation."""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class ProviderCapability(str, Enum):
    """What an image provider supports."""
    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"
    INPAINT = "inpaint"
    CONTROLNET = "controlnet"
    LORA = "lora"
    UPSCALE = "upscale"


@dataclass
class GenerationParams:
    """Parameters for image generation."""
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = -1
    sampler: str = "Euler a"
    # img2img
    init_image: Optional[bytes] = None
    denoising_strength: float = 0.75
    # Extra provider-specific params
    extra: dict = field(default_factory=dict)


class ImageProvider(ABC):
    """Abstract base for all image generation providers.

    Providers wrap different backends (Stable Diffusion WebUI, ComfyUI,
    DALL-E, local diffusers, etc.) behind a common interface.
    """

    @abstractmethod
    async def txt2img(self, params: GenerationParams) -> bytes:
        """Generate an image from a text prompt.

        Returns PNG bytes.
        """

    async def img2img(self, params: GenerationParams) -> bytes:
        """Generate an image from an input image + text prompt.

        Returns PNG bytes. Raises NotImplementedError if not supported.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support img2img")

    @property
    def capabilities(self) -> set[ProviderCapability]:
        """What this provider supports. Override in subclasses."""
        return {ProviderCapability.TXT2IMG}

    async def check_health(self) -> bool:
        """Check if the provider is reachable and ready. Returns True if healthy."""
        return True

    async def txt2img_with_cleanup(self, params: GenerationParams) -> bytes:
        """Generate an image with provider-side cleanup (e.g., background removal).

        Providers that support GPU-side post-processing (like ComfyUI with rembg
        nodes) should override this to perform cleanup in the same GPU pass.

        Default: delegates to txt2img() with no extra cleanup.
        Returns PNG bytes (RGBA if cleanup was applied).
        """
        return await self.txt2img(params)

    async def get_models(self) -> list[str]:
        """List available models. Returns empty list if not applicable."""
        return []
