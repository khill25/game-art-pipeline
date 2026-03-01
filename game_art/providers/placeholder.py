"""Placeholder image provider — colored rectangles for development/testing."""

import hashlib
import io
from PIL import Image, ImageDraw, ImageFont

from .base import ImageProvider, GenerationParams, ProviderCapability


class PlaceholderProvider(ImageProvider):
    """Generates simple colored placeholder sprites.

    No AI, no API calls — just deterministic colored rectangles
    with text labels. Useful for testing the full pipeline without
    GPU or API costs.
    """

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return {ProviderCapability.TXT2IMG}

    async def txt2img(self, params: GenerationParams) -> bytes:
        r, g, b = self._color_from_text(params.prompt)
        img = Image.new("RGBA", (params.width, params.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Draw filled rectangle with margin
        margin = max(2, params.width // 16)
        draw.rectangle(
            [margin, margin, params.width - margin - 1, params.height - margin - 1],
            fill=(r, g, b, 220),
            outline=(min(r + 40, 255), min(g + 40, 255), min(b + 40, 255), 255),
        )

        # Draw first letter of prompt as label
        label = params.prompt[0].upper() if params.prompt else "?"
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        bbox = draw.textbbox((0, 0), label, font=font) if font else (0, 0, 8, 12)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = (params.width - tw) // 2
        ty = (params.height - th) // 2
        draw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _color_from_text(text: str) -> tuple[int, int, int]:
        """Generate a deterministic color from text."""
        h = hashlib.md5(text.encode()).hexdigest()
        return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
