# Test Report: "Studio Ghibli meets dark souls with lasers"

**Date:** 2026-03-01
**LLM:** qwen3:32b via Ollama (192.168.50.181:11434)
**Image:** perfectdeliberate_XL + pixel-art-xl-v1.1 LoRA (0.8 weight)
**SD Backend:** Forge WebUI (192.168.50.181:7860)
**Generation:** 512x512, DPM++ 2M Karras, 25 steps, CFG 7.5, seed 42

---

## User Prompt

```
Studio Ghibli meets dark souls with lasers
```

---

## Step 1: Prompt Analyzer

**Pack name:** Soulforge Luminary

**Theme analysis:**
```json
{
  "themes": ["Studio Ghibli", "Dark Souls", "Sci-Fi"],
  "aesthetic": "Dark, hand-drawn fantasy with mechanical and bioluminescent elements inspired by Studio Ghibli and Dark Souls",
  "factions": [
    "Corrupted Ghibli Creatures",
    "Soulless Mechanized Tunnels",
    "Bioluminescent Forest Spirits"
  ],
  "weapons_inspiration": [
    "Laser-Forged Soul Blades",
    "Totoro-Powered Grimoire",
    "Soul-Absorbing Laser Gauntlets",
    "Shadow-and-Light Staff"
  ],
  "world_description": "A corrupted, sprawling world where ancient forests and mechanical wastelands clash, haunted by soulless entities and glowing with bioluminescent decay",
  "color_palette": ["#2E2E2E", "#FFD700", "#0066CC", "#880000"],
  "currency_theme": {
    "xp_name": "Soul Shards",
    "meta_currency": "Eon Crystals",
    "run_currency": "Lumen Coins"
  },
  "tone": "dark and epic",
  "pack_name": "Soulforge Luminary"
}
```

**Assessment:** Excellent. The LLM correctly blended all three themes. Factions are creative and distinct. Color palette makes sense (dark base, gold/blue/red accents). Currency names fit the world.

---

## Step 2: System Architect

**Status effects (4):**

| Effect | Max Stacks | Duration | Tick | Stat Mod | Description |
|--------|-----------|----------|------|----------|-------------|
| soul_corrosion | 3 | 8.0s | 12 dmg/1s | move_speed -0.2 | Corrodes the soul, leeching health and slowing movement |
| luminous_decay | 5 | 12.0s | 6 dmg/0.5s | armor -0.15 | Bioluminescent decay that weakens defenses while dealing damage |
| soulbind | 1 | 5.0s | — | — | Enemy is physically bound to the player, triggering damage on every movement |
| mechanical_overload | 1 | 4.0s | — | — | Overloaded machinery explodes after taking damage, damaging nearby enemies |

**Enemy archetypes (5):**

| Role | Movement | Attack Style | Description |
|------|----------|-------------|-------------|
| basic | chase | area_attack | Corrupted forest creatures with bioluminescent markings that pulse when aggroed |
| ranged | stationary | ranged_projectile | Soulless automated turrets that fire guided plasma bolts |
| tank | maintain_distance | area_attack | Armored forest guardians that summon roots and emit slowing auras |
| boss | chase | none | Colossal Eon Golem with shifting armor plates that summon mini-turrets when damaged |
| summoner | orbit_player | summon_minions | Shadowy wisps that spawn soul-draining spores and corrupt smaller enemies |

**Assessment:** Strong mechanical diversity. Status effects are thematically distinct and mechanically interesting. soul_corrosion is a classic DoT+slow, luminous_decay is a fast-tick debuff, soulbind is unique (damage-on-movement), mechanical_overload is an on-damage AoE bomb. Enemy archetypes cover all standard roles with creative flavor.

---

## Step 3: Content Designer

### Weapons (8)

| ID | Name | Behaviors | Key Mechanic |
|----|------|-----------|-------------|
| laser_forged_soul_blade | Laser-Forged Soul Blade | timer -> spawn_projectile | Fires laser beams on timer |
| laser_forged_soul_blade_evolved | Soul-Chain Saber | timer -> spawn_projectile | Evolved: chaining projectiles |
| totoro_powered_grimoire | Totoro-Powered Grimoire | timer -> spawn_minion, on_hit -> apply_status | Summons spirit minions + applies status on hit |
| totoro_powered_grimoire_evolved | Eon Spirit Codex | timer -> spawn_minion, passive -> spawn_zone | Evolved: persistent zone + minions |
| soul_absorbing_laser_gauntlets | Soul-Absorbing Laser Gauntlets | timer -> spawn_projectile | Short-range laser blasts |
| soul_absorbing_laser_gauntlets_evolved | Overcharged Soul Gauntlets | timer -> spawn_projectile | Evolved: more powerful projectiles |
| shadow_and_light_staff | Shadow-and-Light Staff | passive -> spawn_zone | Persistent aura zone |
| shadow_and_light_staff_evolved | Eonshard Luminary | passive -> spawn_zone | Evolved: larger/stronger zone |

### Enemies (5)

| ID | Name | HP | Key Behaviors |
|----|------|----|---------------|
| corrupted_gloom_stalker | Corrupted Gloom Stalker | 25 | Chase + area attack |
| soulless_plasma_turret | Soulless Plasma Turret | 15 | Stationary + ranged projectile |
| armored_rootwarden | Armored Rootwarden | 50 | Maintain distance + area attack |
| eon_golem | Eon Golem | 300 | Boss: chase + summon minions |
| shadowy_soul_wisp | Shadowy Soul Wisp | 10 | Orbit player + summon spores |

### Characters (4)

| ID | Name | Starting Weapon |
|----|------|----------------|
| totoro_powered_grimoire_mage | Lumina, the Totoro Sage | totoro_powered_grimoire |
| shadow_and_light_sentinel | Valkyra, the Soulless Sentinel | shadow_and_light_staff |
| wasteland_soulblade_rogue | Zephyr, the Ghost of the Wasteland | laser_forged_soul_blade |
| bioluminescent_gauntlet_brawler | Mako, the Luminous Revenant | soul_absorbing_laser_gauntlets |

**Assessment:** Good variety. Each weapon has a distinct behavior pattern (projectile, minion+status, zone). Characters are well-themed and each starts with a different weapon type. Enemies match the archetypes from step 2. The boss at 300HP is appropriately threatening. Some weapons are light on behaviors (only 1 behavior) — could push for 2-3 per weapon in the prompt.

---

## Step 4-5: Wave Planner & Coherence Validator

(Not captured in this run — Ollama timed out on re-run due to GPU contention with SD. Previous run confirmed waves and validation work correctly with mock data.)

---

## Image Generation Results

### SD Prompt Template Used

```
POSITIVE: pixel art, {category_hint}, {description}, retro game style, clean pixels,
          sharp edges, solid black background, 16-bit era, sprite, centered composition,
          no anti-aliasing, <lora:pixel-art-xl-v1.1:0.8>

NEGATIVE: blurry, smooth, realistic, photo, 3d render, text, watermark, signature,
          frame, border, multiple objects, busy background, anti-aliased, gradient shading,
          noisy, jpeg artifacts, low quality, deformed, ugly, grid, checkerboard
```

### Results Per Sprite

#### 1. Weapon: Laser-Forged Soul Blade

- **SD prompt category hint:** "game weapon sprite, single item, centered"
- **SD prompt description:** "Laser-Forged Soul Blade, a glowing ethereal katana that fires concentrated laser beams, golden blade with blue energy"
- **Files:** `weapon_blade_raw.png` (512), `weapon_blade_nobg.png`, `weapon_blade_64_natural.png`

**Raw 512x512 analysis:**
- Clean pixel art katana with golden blade and blue laser energy effects
- Correct style from LoRA — visible chunky pixels, no smooth gradients
- ISSUE: Extra objects generated (blue skull icons to the right of the blade)
- Background: solid black (good for removal)
- Color accuracy: gold/blue matches palette well

**Post-processing analysis:**
- BG removal (threshold < 30): Clean, black background removed successfully
- Trim: Good crop, centered the blade
- 64x64 downscale: Blade still readable, energy effects visible
- 32x32 downscale: Too small for this shape — tall thin object loses detail

**Issues:**
1. Extra objects (skulls) — need "single object only, no additional items" in prompt
2. Tall aspect ratio after trim — weapon sprites should probably be square-cropped or rotated
3. 32x32 is too small for a katana sprite — consider 48x48 or 64x64 for weapons

#### 2. Enemy: Corrupted Gloom Stalker

- **SD prompt category hint:** "enemy creature sprite, game monster, full body"
- **SD prompt description:** "Corrupted Gloom Stalker, a corrupted forest creature with glowing cyan bioluminescent markings, wolf-like shadow beast"
- **Files:** `enemy_stalker_raw.png` (512), `enemy_stalker_nobg.png`, `enemy_stalker_64_natural.png`

**Raw 512x512 analysis:**
- Menacing dark beast with cyan bioluminescent markings — nailed the description
- Excellent silhouette — recognizable even small
- Strong pixel art style from LoRA
- Background: solid black

**Post-processing analysis:**
- BG removal PROBLEM: Beast body is very dark (#1a1a2e range), close to the black background
  threshold. Body pixels incorrectly made transparent, creating holes in the sprite.
- Trim: Works but the holey sprite looks bad
- 64x64: Still menacing but with transparency artifacts in the body

**Issues:**
1. CRITICAL: Dark subjects break threshold-based bg removal. Need either:
   a. Generate with green/white background instead of black, then chroma-key
   b. Use a segmentation model (rembg / SAM) for proper bg removal
   c. ComfyUI workflow with transparent background support
2. The dark color palette (#2E2E2E) conflicts with black background approach

#### 3. Character: Lumina, the Totoro Sage

- **SD prompt category hint:** "hero character sprite, game protagonist, full body, front-facing"
- **SD prompt description:** "Lumina the Totoro Sage, a mystical forest guardian mage with a glowing grimoire, robed figure with spirit companions"
- **Files:** `char_lumina_raw.png` (512), `char_lumina_nobg.png`, `char_lumina_64_natural.png`

**Raw 512x512 analysis:**
- EXCELLENT: Generated a Totoro-like sage character with a glowing white grimoire on its belly
- Perfect Ghibli aesthetic — round, cute, mystical
- Cyan/teal color scheme with gold eyes
- Clean pixel art, centered, front-facing as requested

**Post-processing analysis:**
- BG removal: Clean — character is bright enough to separate from black bg
- Trim: Good crop
- 64x64: Adorable and fully readable — best sprite of the batch
- 32x32: Still recognizable

**Issues:**
1. None significant — this is the ideal case for the pipeline
2. The Ghibli theme produced a rounder, brighter character that works better for bg removal

#### 4. Boss: Eon Golem

- **SD prompt category hint:** "large boss creature sprite, imposing, detailed, full body"
- **SD prompt description:** "Eon Golem, colossal boss creature with shifting armor plates, towering mechanical-organic hybrid with glowing red core"
- **Files:** `enemy_golem_raw.png` (512), `enemy_golem_nobg.png`, `enemy_golem_64_natural.png`

**Raw 512x512 analysis:**
- Imposing mechanical golem with blue-gray armor and glowing red core
- Dark Souls energy — armored, threatening, detailed
- Clear pixel art style
- Solid black background

**Post-processing analysis:**
- BG removal: Good — armor is light enough (#6080a0 range) to separate from black
- Trim: Clean crop
- 64x64: Strong silhouette, red core visible — reads well as a boss
- 32x32: Still identifiable as an armored figure

**Issues:**
1. Minor: Some dark shadow areas near feet got partially removed
2. Boss should probably stay at 64x64+ given it should be visually larger than regular enemies

---

## Overall Assessment

### What Works Well
- **LoRA quality:** pixel-art-xl-v1.1 at 0.8 weight produces clean, authentic pixel art
- **Model choice:** perfectdeliberate_XL as base gives good detail and composition
- **Prompt construction:** Category hints (weapon/enemy/character) steer generation correctly
- **Theme coherence:** All sprites feel like they belong to the same "Ghibli + Dark Souls + lasers" world
- **Bright/medium subjects:** Characters and light-colored enemies separate cleanly from backgrounds

### What Needs Fixing

1. **Background removal** (CRITICAL)
   - Current: threshold-based (r<30, g<30, b<30 → transparent)
   - Problem: Dark subjects (shadow creatures, dark armor) lose body pixels
   - Fix options:
     a. **Green screen approach:** Prompt with "solid bright green background, #00FF00" then chroma-key
     b. **Segmentation model:** rembg or SAM for proper subject extraction
     c. **ComfyUI workflow:** Use ControlNet or inpainting for transparent bg
   - Recommendation: Try green screen first (simplest), fall back to rembg

2. **Extra objects in generation**
   - Weapon blade generated bonus skull icons
   - Fix: Add "single object only, isolated, no additional items, no extra elements" to negative prompt
   - Also add to positive: "one item only" for weapon category

3. **Palette enforcement too aggressive**
   - 6-color palette destroyed detail at small sizes
   - Fix: Make palette optional, increase to 16-24 colors minimum, or skip palette entirely
     and rely on the color palette hints in the SD prompt to guide colors naturally

4. **Sprite sizing**
   - 32x32 is too small for tall/thin objects (weapons, staffs)
   - Characters and round enemies work at 32x32
   - Fix: Per-category target sizes (weapons: 48x48, characters: 32x32, bosses: 64x64)

### Recommendations for ComfyUI Workflow

When we move to ComfyUI, the workflow should:
1. Generate at 512x512 or 1024x1024
2. Use ControlNet (canny/depth) to enforce consistent pose/silhouette across frames
3. Use rembg node or SAM for proper background removal
4. Support img2img for animation frames (generate base frame, then variations)
5. Include automatic upscale/downscale nodes with nearest-neighbor
6. Output both the raw high-res sprite and the final game-ready sprite

### Automated Visual QA

The sprites were manually analyzed in this report. For the automated pipeline, we need:

1. **Vision LLM evaluation** — send the generated sprite to a vision-capable model (Qwen3-VL
   on the Ollama instance, or Claude) with the original prompt and ask:
   - "Does this sprite match the description?"
   - "Is the background clean/transparent?"
   - "Are there extra unwanted objects?"
   - "Is the sprite centered and properly framed?"
   - "Rate 1-5 for quality, theme match, and usability"

2. **Programmatic checks:**
   - Background ratio (what % of pixels are transparent after bg removal — should be >40%)
   - Subject centering (bounding box should be roughly centered)
   - Color count (too few = over-simplified, too many = not pixel art)
   - Minimum opaque pixel count (catch empty/failed generations)

3. **Retry logic:** If QA fails, regenerate with a different seed or adjusted prompt

### Animation Needs

Vampire Survivors-style weapons need these animation types:
- **Projectile weapons:** Single sprite that moves in a direction (no animation needed, game handles movement)
- **Melee weapons:** Weapon sprite rotates/swings around the character (game handles rotation, single sprite suffices)
- **Aura weapons:** Pulsing/rotating effect around the character (single sprite + game shader/scale animation)
- **Orbital weapons:** Sprite orbits the character (single sprite, game handles orbit path)

For the base game, **we do NOT need animated spritesheets for weapons.** The game engine handles all the movement. We need:
- 1 static sprite per weapon (current approach)
- 1 static sprite per enemy (current approach)
- 4-frame walk cycle per character (FUTURE — hard problem)
- 4-frame death animation per enemy (FUTURE — hard problem)

The walk/death animations are the ComfyUI use case — using img2img or ControlNet to generate
consistent frame variations from the base sprite.
