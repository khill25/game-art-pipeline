"""Sprite validation — catches common generation failures before saving.

Runs after cleanup, before orientation and downscaling. Failures trigger
a retry with a different seed.
"""

from PIL import Image

from game_art.processing.cleanup import compute_fill_ratio, compute_aspect_ratio


def validate_sprite(
    img: Image.Image,
    category: str,
) -> tuple[bool, list[str]]:
    """Validate a cleaned sprite image for quality.

    Args:
        img: RGBA image after background removal and cleanup
        category: Asset type ("weapon", "character", "enemy", etc.)

    Returns:
        (passed, issues) — passed is True if no fatal issues.
        Issues list may contain warnings even when passed is True.
    """
    issues: list[str] = []
    passed = True

    # Tiles are meant to fill the whole frame — skip fill checks
    if category == "tile":
        return True, []

    fill = compute_fill_ratio(img)
    aspect = compute_aspect_ratio(img)

    # Fill > 90% — background removal likely failed
    if fill > 0.90:
        issues.append(f"bg_removal_failed (fill={fill:.2f})")
        passed = False

    # Subject < 10% but > 3% — subject is tiny, probably bad generation
    if 0.03 <= fill < 0.10:
        issues.append(f"subject_too_small (fill={fill:.2f})")
        passed = False

    # < 3% opaque — image is nearly empty
    if fill < 0.03:
        issues.append(f"image_nearly_empty (fill={fill:.2f})")
        passed = False

    # Weapon-specific: should be elongated
    if category == "weapon" and aspect < 1.3:
        issues.append(f"weapon_not_elongated (aspect={aspect:.2f})")
        # Warning only, don't trigger retry

    return passed, issues
