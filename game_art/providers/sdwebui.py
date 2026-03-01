"""Stable Diffusion WebUI (A1111/Forge) provider via REST API.

Requires the WebUI to be running with --api flag:
    ./webui.sh --listen --api --port 7860

API docs at: http://<host>:7860/docs
"""

import base64
import io
import httpx
from typing import Optional

from .base import ImageProvider, GenerationParams, ProviderCapability


class SDWebUIProvider(ImageProvider):
    """Connects to A1111 or Forge WebUI REST API for image generation.

    Supports txt2img, img2img, and model switching.
    """

    def __init__(
        self,
        url: str = "http://localhost:7860",
        timeout: float = 120.0,
        default_sampler: str = "Euler a",
        default_steps: int = 20,
        default_cfg: float = 7.0,
    ):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.default_sampler = default_sampler
        self.default_steps = default_steps
        self.default_cfg = default_cfg

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {
            ProviderCapability.TXT2IMG,
            ProviderCapability.IMG2IMG,
            ProviderCapability.LORA,
        }

    async def txt2img(self, params: GenerationParams) -> bytes:
        payload = {
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt,
            "width": params.width,
            "height": params.height,
            "steps": params.steps or self.default_steps,
            "cfg_scale": params.cfg_scale or self.default_cfg,
            "sampler_name": params.sampler or self.default_sampler,
            "seed": params.seed,
            "batch_size": 1,
            "n_iter": 1,
        }
        # Merge any extra provider-specific params (e.g., override_settings)
        payload.update(params.extra)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.url}/sdapi/v1/txt2img", json=payload)
            resp.raise_for_status()
            data = resp.json()

        return base64.b64decode(data["images"][0])

    async def img2img(self, params: GenerationParams) -> bytes:
        if params.init_image is None:
            raise ValueError("img2img requires init_image in params")

        init_b64 = base64.b64encode(params.init_image).decode("utf-8")
        payload = {
            "init_images": [init_b64],
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt,
            "width": params.width,
            "height": params.height,
            "steps": params.steps or self.default_steps,
            "cfg_scale": params.cfg_scale or self.default_cfg,
            "denoising_strength": params.denoising_strength,
            "sampler_name": params.sampler or self.default_sampler,
            "seed": params.seed,
        }
        payload.update(params.extra)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.url}/sdapi/v1/img2img", json=payload)
            resp.raise_for_status()
            data = resp.json()

        return base64.b64decode(data["images"][0])

    async def check_health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.url}/sdapi/v1/sd-models")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def get_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.url}/sdapi/v1/sd-models")
            resp.raise_for_status()
            return [m["model_name"] for m in resp.json()]

    async def set_model(self, model_name: str) -> None:
        """Switch the active checkpoint model."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.url}/sdapi/v1/options",
                json={"sd_model_checkpoint": model_name},
            )
            resp.raise_for_status()

    async def get_samplers(self) -> list[str]:
        """List available samplers."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.url}/sdapi/v1/samplers")
            resp.raise_for_status()
            return [s["name"] for s in resp.json()]

    async def get_loras(self) -> list[str]:
        """List available LoRAs."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.url}/sdapi/v1/loras")
            resp.raise_for_status()
            return [l["name"] for l in resp.json()]
