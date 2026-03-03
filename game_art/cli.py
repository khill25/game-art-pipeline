"""CLI interface for game-art-pipeline.

Usage:
    game-art generate sprite --prompt "fire sword" --style pixel_art --size 32 --output sword.png
    game-art generate sprite --prompt "skeleton warrior" --category enemy --provider sdwebui --url http://192.168.50.181:7860
    game-art health --url http://192.168.50.181:7860
    game-art models --url http://192.168.50.181:7860
"""

import argparse
import asyncio
import sys
from pathlib import Path

from game_art.config import GameArtConfig


def _get_provider(args, config: GameArtConfig):
    """Create the appropriate provider based on args/config."""
    provider_name = getattr(args, "provider", None) or config.provider

    if provider_name == "sdwebui":
        from game_art.providers.sdwebui import SDWebUIProvider
        url = getattr(args, "url", None) or config.sdwebui_url
        return SDWebUIProvider(url=url, timeout=config.sdwebui_timeout)
    elif provider_name == "comfyui":
        from game_art.providers.comfyui import ComfyUIProvider
        url = getattr(args, "url", None) or config.comfyui_url
        checkpoint = getattr(args, "checkpoint", None) or config.comfyui_checkpoint
        return ComfyUIProvider(
            url=url, timeout=config.comfyui_timeout,
            default_checkpoint=checkpoint,
        )
    elif provider_name == "placeholder":
        from game_art.providers.placeholder import PlaceholderProvider
        return PlaceholderProvider()
    else:
        print(f"Unknown provider: {provider_name}", file=sys.stderr)
        sys.exit(1)


def _get_style(args, config: GameArtConfig):
    """Create the appropriate style based on args/config."""
    style_name = getattr(args, "style", None) or config.default_style
    size = getattr(args, "size", None) or config.default_size
    palette = getattr(args, "palette", None)
    palette_list = palette.split(",") if palette else []

    if style_name == "pixel_art":
        from game_art.styles.pixel_art import PixelArtStyle
        return PixelArtStyle(palette=palette_list, target_size=size)
    else:
        print(f"Unknown style: {style_name}", file=sys.stderr)
        sys.exit(1)


async def cmd_generate_sprite(args):
    """Generate a single sprite."""
    config = GameArtConfig.from_env()
    provider = _get_provider(args, config)
    style = _get_style(args, config)

    from game_art.generators.sprite import SpriteGenerator
    gen = SpriteGenerator(provider, style)

    output = Path(args.output)
    await gen.generate_to_file(
        prompt=args.prompt,
        output_path=output,
        category=args.category,
        seed=args.seed,
    )
    print(f"Saved: {output}")


async def cmd_health(args):
    """Check if a provider is reachable."""
    config = GameArtConfig.from_env()
    provider = _get_provider(args, config)
    ok = await provider.check_health()
    if ok:
        print("Provider is healthy")
    else:
        print("Provider is not reachable", file=sys.stderr)
        sys.exit(1)


async def cmd_models(args):
    """List available models."""
    config = GameArtConfig.from_env()
    provider = _get_provider(args, config)
    models = await provider.get_models()
    if models:
        for m in models:
            print(f"  {m}")
    else:
        print("No models found or provider doesn't support listing.")


def main():
    parser = argparse.ArgumentParser(prog="game-art", description="AI game art generation")
    sub = parser.add_subparsers(dest="command")

    # --- generate ---
    gen_parser = sub.add_parser("generate", help="Generate game art")
    gen_sub = gen_parser.add_subparsers(dest="asset_type")

    sprite_parser = gen_sub.add_parser("sprite", help="Generate a single sprite")
    sprite_parser.add_argument("--prompt", "-p", required=True, help="Description of the sprite")
    sprite_parser.add_argument("--output", "-o", default="sprite.png", help="Output file path")
    sprite_parser.add_argument("--category", "-c", default="item", help="Asset category (weapon, enemy, character, etc.)")
    sprite_parser.add_argument("--style", "-s", default="pixel_art", help="Art style (pixel_art)")
    sprite_parser.add_argument("--size", type=int, default=32, help="Output sprite size in pixels")
    sprite_parser.add_argument("--provider", default=None, help="Image provider (placeholder, sdwebui, comfyui)")
    sprite_parser.add_argument("--url", default=None, help="Provider URL")
    sprite_parser.add_argument("--checkpoint", default=None, help="Checkpoint model name (comfyui)")
    sprite_parser.add_argument("--palette", default=None, help="Comma-separated hex colors")
    sprite_parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    sprite_parser.set_defaults(func=cmd_generate_sprite)

    # --- health ---
    health_parser = sub.add_parser("health", help="Check provider health")
    health_parser.add_argument("--provider", default="sdwebui")
    health_parser.add_argument("--url", default=None)
    health_parser.set_defaults(func=cmd_health)

    # --- models ---
    models_parser = sub.add_parser("models", help="List available models")
    models_parser.add_argument("--provider", default="sdwebui")
    models_parser.add_argument("--url", default=None)
    models_parser.set_defaults(func=cmd_models)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
