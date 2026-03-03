"""ComfyUI provider via workflow-based API.

ComfyUI uses a different paradigm from SDWebUI:
- Submit a workflow (prompt) as JSON → POST /prompt
- Wait for completion via WebSocket at /ws
- Fetch the result image via GET /view

Requires ComfyUI running with default API enabled:
    python main.py --listen --port 8099

API docs: https://docs.comfy.org/essentials/comfyui_server
"""

import base64
import io
import json
import uuid
from typing import Optional

import httpx

from .base import ImageProvider, GenerationParams, ProviderCapability


# ── Sampler mapping ──────────────────────────────────────────────────
# SDWebUI uses combined names like "DPM++ 2M Karras".
# ComfyUI splits into sampler_name + scheduler.

SAMPLER_MAP = {
    # Combined name → (sampler, scheduler)
    "DPM++ 2M Karras": ("dpmpp_2m", "karras"),
    "DPM++ 2M SDE Karras": ("dpmpp_2m_sde", "karras"),
    "DPM++ 2S a Karras": ("dpmpp_2s_ancestral", "karras"),
    "DPM++ SDE Karras": ("dpmpp_sde", "karras"),
    "DPM++ 2M": ("dpmpp_2m", "normal"),
    "DPM++ 2M SDE": ("dpmpp_2m_sde", "normal"),
    "Euler a": ("euler_ancestral", "normal"),
    "Euler": ("euler", "normal"),
    "LMS": ("lms", "normal"),
    "Heun": ("heun", "normal"),
    "DDIM": ("ddim", "normal"),
    "UniPC": ("uni_pc", "normal"),
    # Already split — pass through
    "euler": ("euler", "normal"),
    "euler_ancestral": ("euler_ancestral", "normal"),
    "dpmpp_2m": ("dpmpp_2m", "normal"),
    "dpmpp_2m_sde": ("dpmpp_2m_sde", "normal"),
}


def _split_sampler(combined: str) -> tuple[str, str]:
    """Split a combined sampler name into (sampler, scheduler) for ComfyUI."""
    if combined in SAMPLER_MAP:
        return SAMPLER_MAP[combined]
    # If it contains "karras", split it
    lower = combined.lower()
    if "karras" in lower:
        name = lower.replace("karras", "").strip().rstrip()
        return (name, "karras")
    # Fallback: assume it's already a ComfyUI sampler name
    return (combined, "normal")


# ── Workflow templates ────────────────────────────────────────────────

def _txt2img_workflow(
    positive: str,
    negative: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    checkpoint: str,
) -> dict:
    """Build a txt2img workflow as a ComfyUI prompt dict."""
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed if seed >= 0 else _random_seed(),
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": checkpoint,
            },
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": positive,
                "clip": ["4", 1],
            },
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative,
                "clip": ["4", 1],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2],
            },
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "game_art",
                "images": ["8", 0],
            },
        },
    }


def _img2img_workflow(
    positive: str,
    negative: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    checkpoint: str,
    image_name: str,
    denoise: float,
) -> dict:
    """Build an img2img workflow."""
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed if seed >= 0 else _random_seed(),
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": denoise,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["10", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": checkpoint,
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": positive,
                "clip": ["4", 1],
            },
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative,
                "clip": ["4", 1],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2],
            },
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "game_art",
                "images": ["8", 0],
            },
        },
        "10": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["11", 0],
                "vae": ["4", 2],
            },
        },
        "11": {
            "class_type": "LoadImage",
            "inputs": {
                "image": image_name,
            },
        },
    }


def _txt2img_rembg_workflow(
    positive: str,
    negative: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler: str,
    scheduler: str,
    seed: int,
    checkpoint: str,
) -> dict:
    """Build a txt2img + bg removal workflow that generates and removes background in one GPU pass.

    Chains: CheckpointLoader → CLIP encode → KSampler → VAEDecode → BiRefNetRMBG → SaveImage.
    BiRefNet runs on GPU for high-quality background segmentation, producing a clean RGBA PNG.
    """
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed if seed >= 0 else _random_seed(),
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": checkpoint,
            },
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": positive,
                "clip": ["4", 1],
            },
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative,
                "clip": ["4", 1],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2],
            },
        },
        "12": {
            "class_type": "BiRefNetRMBG",
            "inputs": {
                "model": "BiRefNet-general",
                "image": ["8", 0],
                "mask_blur": 0,
                "mask_offset": 0,
                "invert_output": False,
                "refine_foreground": True,
                "background": "Alpha",
                "background_color": "#000000",
            },
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "game_art",
                "images": ["12", 0],
            },
        },
    }


def _inject_lora(workflow: dict, lora_name: str, lora_weight: float) -> dict:
    """Insert a LoraLoader node between checkpoint and CLIP/KSampler.

    Rewires model/clip references from node "4" (checkpoint) through
    a new node "20" (LoraLoader), then into KSampler and CLIP nodes.
    """
    # Add LoRA loader node
    workflow["20"] = {
        "class_type": "LoraLoader",
        "inputs": {
            "lora_name": lora_name,
            "strength_model": lora_weight,
            "strength_clip": lora_weight,
            "model": ["4", 0],
            "clip": ["4", 1],
        },
    }

    # Rewire: KSampler model input → LoRA output
    workflow["3"]["inputs"]["model"] = ["20", 0]

    # Rewire: CLIP text encoders → LoRA clip output
    workflow["6"]["inputs"]["clip"] = ["20", 1]
    workflow["7"]["inputs"]["clip"] = ["20", 1]

    return workflow


def _random_seed() -> int:
    """Generate a random seed in ComfyUI's valid range."""
    import random
    return random.randint(0, 2**32 - 1)


# ── Provider class ────────────────────────────────────────────────────

class ComfyUIProvider(ImageProvider):
    """Connects to ComfyUI's API for image generation.

    Unlike SDWebUI's synchronous REST API, ComfyUI uses an async
    workflow-based approach:
    1. POST /prompt with a workflow graph
    2. Poll /history/{prompt_id} for completion
    3. GET /view to fetch the output image
    """

    def __init__(
        self,
        url: str = "http://localhost:8099",
        timeout: float = 300.0,
        default_sampler: str = "euler",
        default_steps: int = 20,
        default_cfg: float = 7.0,
        default_checkpoint: str = "",
    ):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.default_sampler = default_sampler
        self.default_steps = default_steps
        self.default_cfg = default_cfg
        self.default_checkpoint = default_checkpoint
        self._client_id = str(uuid.uuid4())

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {
            ProviderCapability.TXT2IMG,
            ProviderCapability.IMG2IMG,
            ProviderCapability.LORA,
        }

    async def txt2img(self, params: GenerationParams) -> bytes:
        sampler, scheduler = _split_sampler(params.sampler or self.default_sampler)
        checkpoint = params.extra.get("checkpoint", self.default_checkpoint)
        if not checkpoint:
            checkpoint = await self._get_first_checkpoint()

        workflow = _txt2img_workflow(
            positive=params.prompt,
            negative=params.negative_prompt,
            width=params.width,
            height=params.height,
            steps=params.steps or self.default_steps,
            cfg=params.cfg_scale or self.default_cfg,
            sampler=sampler,
            scheduler=scheduler,
            seed=params.seed,
            checkpoint=checkpoint,
        )

        # Inject LoRA if specified
        lora = params.extra.get("lora")
        if lora:
            lora_weight = params.extra.get("lora_weight", 0.8)
            # Ensure .safetensors extension
            if not lora.endswith(".safetensors") and "." not in lora:
                lora = f"{lora}.safetensors"
            workflow = _inject_lora(workflow, lora, lora_weight)

        return await self._queue_and_wait(workflow)

    async def txt2img_with_cleanup(self, params: GenerationParams) -> bytes:
        """Generate + GPU-side background removal in one ComfyUI pass.

        Uses Image Rembg node if available and working. Falls back to plain
        txt2img + Python-side bg removal if:
        - The rembg node isn't registered
        - The rembg node errors at runtime (e.g., missing rembg pip package)

        Returns PNG bytes (RGBA if rembg worked, RGB otherwise).
        """
        if not await self._has_rembg_node():
            return await self.txt2img(params)

        sampler, scheduler = _split_sampler(params.sampler or self.default_sampler)
        checkpoint = params.extra.get("checkpoint", self.default_checkpoint)
        if not checkpoint:
            checkpoint = await self._get_first_checkpoint()

        workflow = _txt2img_rembg_workflow(
            positive=params.prompt,
            negative=params.negative_prompt,
            width=params.width,
            height=params.height,
            steps=params.steps or self.default_steps,
            cfg=params.cfg_scale or self.default_cfg,
            sampler=sampler,
            scheduler=scheduler,
            seed=params.seed,
            checkpoint=checkpoint,
        )

        # Inject LoRA if specified
        lora = params.extra.get("lora")
        if lora:
            lora_weight = params.extra.get("lora_weight", 0.8)
            if not lora.endswith(".safetensors") and "." not in lora:
                lora = f"{lora}.safetensors"
            workflow = _inject_lora(workflow, lora, lora_weight)

        try:
            return await self._queue_and_wait(workflow)
        except RuntimeError as e:
            error_msg = str(e)
            # If the bg removal node failed (missing deps, model download, etc.),
            # disable it for future calls and fall back to plain txt2img
            if "node_id': '12'" in error_msg or "ModuleNotFoundError" in error_msg:
                self._rembg_available = False
                import sys
                print(
                    f"Warning: ComfyUI bg removal node failed. "
                    "Falling back to txt2img without GPU bg removal.",
                    file=sys.stderr,
                )
                return await self.txt2img(params)
            raise

    async def img2img(self, params: GenerationParams) -> bytes:
        if params.init_image is None:
            raise ValueError("img2img requires init_image in params")

        sampler, scheduler = _split_sampler(params.sampler or self.default_sampler)
        checkpoint = params.extra.get("checkpoint", self.default_checkpoint)
        if not checkpoint:
            checkpoint = await self._get_first_checkpoint()

        # Upload the init image
        image_name = await self._upload_image(params.init_image)

        workflow = _img2img_workflow(
            positive=params.prompt,
            negative=params.negative_prompt,
            width=params.width,
            height=params.height,
            steps=params.steps or self.default_steps,
            cfg=params.cfg_scale or self.default_cfg,
            sampler=sampler,
            scheduler=scheduler,
            seed=params.seed,
            checkpoint=checkpoint,
            image_name=image_name,
            denoise=params.denoising_strength,
        )

        # Inject LoRA if specified
        lora = params.extra.get("lora")
        if lora:
            lora_weight = params.extra.get("lora_weight", 0.8)
            if not lora.endswith(".safetensors") and "." not in lora:
                lora = f"{lora}.safetensors"
            workflow = _inject_lora(workflow, lora, lora_weight)

        return await self._queue_and_wait(workflow)

    async def check_health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.url}/system_stats")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def get_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.url}/object_info/CheckpointLoaderSimple")
                resp.raise_for_status()
                data = resp.json()
                # Navigate the nested structure to find checkpoint names
                inputs = data.get("CheckpointLoaderSimple", {}).get("input", {})
                required = inputs.get("required", {})
                ckpt_info = required.get("ckpt_name", [])
                if ckpt_info and isinstance(ckpt_info[0], list):
                    return ckpt_info[0]
                return []
        except (httpx.ConnectError, httpx.TimeoutException):
            return []

    # ── Internal helpers ──────────────────────────────────────────────

    async def _has_rembg_node(self) -> bool:
        """Check if the ComfyUI server has a background removal node available.

        Checks for BiRefNetRMBG (preferred) or Image Rembg as fallback.
        """
        if hasattr(self, "_rembg_available"):
            return self._rembg_available
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check for BiRefNetRMBG (preferred — standalone, no pip deps)
                resp = await client.get(f"{self.url}/object_info/BiRefNetRMBG")
                if resp.status_code == 200:
                    self._rembg_available = True
                    return True
                # Fallback: check for WAS rembg node
                resp = await client.get(
                    f"{self.url}/object_info/Image Rembg (Remove Background)"
                )
                self._rembg_available = resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            self._rembg_available = False
        return self._rembg_available

    async def _get_first_checkpoint(self) -> str:
        """Get the first available checkpoint if none was specified."""
        models = await self.get_models()
        if not models:
            raise RuntimeError(
                "No checkpoints found on ComfyUI server. "
                "Place a .safetensors checkpoint in ComfyUI/models/checkpoints/"
            )
        return models[0]

    async def _upload_image(self, image_bytes: bytes) -> str:
        """Upload an image to ComfyUI's input directory. Returns the filename."""
        filename = f"game_art_input_{uuid.uuid4().hex[:8]}.png"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.url}/upload/image",
                files={"image": (filename, image_bytes, "image/png")},
                data={"overwrite": "true"},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("name", filename)

    async def _queue_and_wait(self, workflow: dict) -> bytes:
        """Submit a workflow, poll for completion, and fetch the result image."""
        prompt_id = await self._submit_prompt(workflow)
        output_data = await self._poll_history(prompt_id)
        return await self._fetch_image(output_data)

    async def _submit_prompt(self, workflow: dict) -> str:
        """POST /prompt — submit a workflow for execution. Returns prompt_id."""
        payload = {
            "prompt": workflow,
            "client_id": self._client_id,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{self.url}/prompt", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["prompt_id"]

    async def _poll_history(self, prompt_id: str) -> dict:
        """Poll /history/{prompt_id} until the job completes.

        Uses polling instead of WebSocket for simplicity and reliability.
        Returns the output data for the completed prompt.
        """
        import asyncio

        poll_interval = 0.5
        elapsed = 0.0

        async with httpx.AsyncClient(timeout=10.0) as client:
            while elapsed < self.timeout:
                resp = await client.get(f"{self.url}/history/{prompt_id}")
                resp.raise_for_status()
                data = resp.json()

                if prompt_id in data:
                    entry = data[prompt_id]
                    status = entry.get("status", {})

                    # Check for errors
                    if status.get("status_str") == "error":
                        messages = status.get("messages", [])
                        error_msg = str(messages) if messages else "Unknown ComfyUI error"
                        raise RuntimeError(f"ComfyUI generation failed: {error_msg}")

                    # Check if completed
                    outputs = entry.get("outputs", {})
                    if outputs:
                        return outputs

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                # Ramp up interval to avoid hammering the server
                if elapsed > 5:
                    poll_interval = 1.0
                if elapsed > 30:
                    poll_interval = 2.0

        raise TimeoutError(
            f"ComfyUI generation timed out after {self.timeout}s for prompt {prompt_id}"
        )

    async def _fetch_image(self, outputs: dict) -> bytes:
        """Extract image filename from outputs and fetch via GET /view."""
        # Find the SaveImage node output
        for node_id, node_output in outputs.items():
            images = node_output.get("images", [])
            if images:
                img_info = images[0]
                filename = img_info["filename"]
                subfolder = img_info.get("subfolder", "")
                img_type = img_info.get("type", "output")

                params = {
                    "filename": filename,
                    "type": img_type,
                }
                if subfolder:
                    params["subfolder"] = subfolder

                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(f"{self.url}/view", params=params)
                    resp.raise_for_status()
                    return resp.content

        raise RuntimeError("No images found in ComfyUI output")
