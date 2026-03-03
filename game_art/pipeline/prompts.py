"""V2 prompt builder with strong category-specific isolation hints.

V1 prompts produced generic "game weapon sprite" text that often resulted in
icon-style renders, extra elements (hands, bases, platforms), and poor isolation.

V2 prompts are much more explicit about isolation, pose, and what to exclude.
"""

from typing import Optional


# ── Category-specific positive hints ─────────────────────────────────
# These replace the generic CATEGORY_HINTS from pixel_art.py v1

CATEGORY_POSITIVE = {
    "weapon": (
        "single weapon sprite, isolated weapon item, no hand, no arm, "
        "just the weapon, floating, side view, centered, white background"
    ),
    "character": (
        "hero character sprite, full body standing pose, single character only, "
        "front-facing, idle stance, no ground, no base, no pedestal, no platform, "
        "no shadow circle, no grass, floating, isolated, white background"
    ),
    "enemy": (
        "enemy creature sprite, full body, single character only, front-facing, "
        "no ground, no base, no platform, floating, isolated, white background"
    ),
    "boss": (
        "large boss creature sprite, full body, imposing, single character only, "
        "front-facing, no ground, no base, no platform, floating, isolated, "
        "white background"
    ),
    "pickup": (
        "collectible item sprite, small pickup icon, floating, glowing, "
        "no background elements, centered, white background"
    ),
    "tile": (
        "seamless game tile, top-down ground texture, tileable, fills entire frame, "
        "texture only"
    ),
    "npc": (
        "NPC character sprite, full body, front-facing, single character only, "
        "no ground, no base, floating, isolated, white background"
    ),
    "icon": (
        "game UI icon, clean simple design, single icon, centered, "
        "no background elements, white background"
    ),
    "effect": (
        "visual effect sprite, magic spell, glowing energy, isolated, "
        "transparent background"
    ),
}

# ── Category-specific negative additions ─────────────────────────────

CATEGORY_NEGATIVE = {
    "weapon": (
        "hand, arm, fingers, holding, grip, ability icon, spell icon, "
        "UI button, circular frame, multiple weapons, person, character, "
        "body, torso"
    ),
    "character": (
        "base, pedestal, platform, ground circle, grass, shadow circle, "
        "tabletop miniature, figurine base, floor, ground, multiple characters, "
        "crowd, extra limbs"
    ),
    "enemy": (
        "base, pedestal, platform, ground circle, grass, shadow circle, "
        "tabletop miniature, figurine base, floor, ground, multiple characters, "
        "crowd, extra limbs"
    ),
    "boss": (
        "base, pedestal, platform, ground circle, grass, shadow circle, "
        "tabletop miniature, figurine base, floor, ground, multiple characters, "
        "crowd, extra limbs"
    ),
    "pickup": (
        "hand, arm, person, character, multiple items, UI frame, border"
    ),
    "tile": (
        "objects, characters, frame, border, items, creatures, UI elements"
    ),
    "npc": (
        "base, pedestal, platform, ground circle, grass, shadow circle, "
        "tabletop miniature, figurine base, floor, ground, multiple characters"
    ),
    "icon": (
        "person, character, 3d, realistic, complex background"
    ),
    "effect": (
        "person, character, weapon, ground, solid background"
    ),
}

# ── Base prompt templates ────────────────────────────────────────────

BASE_POSITIVE_V2 = (
    "pixel art, {category_hint}, {prompt}, "
    "retro game style, clean pixels, sharp edges, "
    "16-bit era, sprite, centered composition, no anti-aliasing"
)

BASE_NEGATIVE_V2 = (
    "blurry, smooth, realistic, photo, 3d render, text, watermark, "
    "signature, frame, border, multiple objects, busy background, "
    "anti-aliased, gradient shading, noisy, jpeg artifacts, "
    "low quality, deformed, ugly, grid, checkerboard pattern, "
    "extra items, additional objects, multiple sprites, "
    "cropped, cut off, partial"
)


def build_prompt_v2(
    base_prompt: str,
    category: str,
    palette: list[str] | None = None,
    lora: str | None = None,
    lora_weight: float = 0.8,
    extra_positive: str = "",
) -> tuple[str, str]:
    """Build v2 positive and negative prompts with strong isolation hints.

    Args:
        base_prompt: Core description (e.g., "fire sword — blazing blade")
        category: Asset type — "weapon", "enemy", "character", "pickup", "tile", etc.
        palette: Hex color strings for palette hints
        lora: LoRA name to inject (e.g., "pixel-art-xl-v1.1")
        lora_weight: LoRA weight
        extra_positive: Additional positive text (e.g., theme description)

    Returns:
        (positive_prompt, negative_prompt) tuple
    """
    # Build positive
    hint = CATEGORY_POSITIVE.get(category, f"game {category} sprite, centered, isolated")
    positive = BASE_POSITIVE_V2.format(category_hint=hint, prompt=base_prompt)

    if extra_positive:
        positive += f", {extra_positive}"

    if palette:
        color_str = ", ".join(palette[:6])
        positive += f", color palette: {color_str}"

    if lora:
        positive += f", <lora:{lora}:{lora_weight}>"

    # Build negative
    negative = BASE_NEGATIVE_V2
    cat_neg = CATEGORY_NEGATIVE.get(category, "")
    if cat_neg:
        negative += f", {cat_neg}"

    return positive, negative
