# Asteroid Belt Prospector — Graphics & Audio Spec (Kenney Packs)

This document specifies how to map **sprites / PNGs / atlases / audio** to the game’s **entities, actions, UI, and events**. It is designed to be implemented by a coding agent with **no custom art creation**.

This mapping covers all game actions and gameplay elements described in the Game Design Document. fileciteturn5file0  
Action indices referenced below match the formal RL action map (Discrete 0..68). fileciteturn5file4

---

## 0) Asset sources (what to download)

### Required packs (sprites + UI)
1. **Kenney – Space Shooter Redux (2D)** (CC0, ~295 files) citeturn0search0turn2search3  
   Includes separate PNG sprites, backgrounds, spritesheet, vector files, and **bonus fonts + 7 sound effects**. citeturn2search11  
2. **Kenney – Space Shooter Extension (250+)** (CC0, ~270 files) citeturn0search9turn0search1  
   Includes separate PNG files + spritesheet(s) and more space parts. citeturn0search1  
3. **Kenney – UI Pack – Sci‑Fi** (CC0, ~130 files) citeturn0search2turn0search6  
4. **Kenney – UI Pack (2D)** (CC0, 400+ sprites) citeturn0search14  
5. **Kenney – Planets (2D)** (CC0, 50+ sprites) citeturn0search3turn0search7turn0search11  
6. **Kenney – Simple Space (2D)** (CC0, ~48 sprites + tilesheet) citeturn1search0turn1search8  

### Recommended packs (audio coverage)
These are not in your “required packs” list, but they remove guesswork for full sound coverage.

7. **Kenney – Sci‑Fi Sounds** (CC0, ~70 OGG files) citeturn1search1turn1search13  
8. **Kenney – UI Audio** (CC0, ~50 UI SFX files) citeturn1search3turn1search7  

---

## 1) Core principle: “Semantic Manifest” drives everything

The graphics system must not hardcode filenames throughout the codebase.

Instead:
- The engine/frontend emits **semantic states** (ship stats, selected asteroid, action id, events, etc.)
- The renderer looks up **semantic keys** in a single manifest:
  - `graphics_manifest.json` for sprites/fonts/backgrounds
  - `audio_manifest.json` for sound cues

This keeps implementation stable even if the underlying sprite filenames change (e.g., switching between Space Shooter Redux vs Simple Space style).

---

## 2) Asset pipeline (how to go from packs → atlases → manifest)

### 2.1 Folder layout (repo conventions)

```
/assets_raw/
  kenney_space_shooter_redux/
  kenney_space_shooter_extension/
  kenney_ui_pack_scifi/
  kenney_ui_pack/
  kenney_planets/
  kenney_simple_space/
  kenney_scifi_sounds/          (recommended)
  kenney_ui_audio/              (recommended)

/public/assets/                 (what Next.js serves)
  atlases/
    world/atlas.png
    world/atlas.json
    ui/atlas.png
    ui/atlas.json
    vfx/atlas.png
    vfx/atlas.json
  audio/
    sfx/...
    ui/...
  fonts/...
  backgrounds/...
  manifests/
    graphics_manifest.json
    audio_manifest.json
```

### 2.2 Atlas creation (spritesheets)

Use **Free Texture Packer** (open source) to pack PNGs into one or more atlases and export JSON formats compatible with your renderer. citeturn2search0turn2search4  
Free Texture Packer supports multiple export formats including Pixi/Phaser. citeturn2search4turn2search8

**Atlas strategy (recommended)**
- `world` atlas: ship, station, asteroids, satellites, miscellaneous world sprites
- `ui` atlas: panels, buttons, bars, icons
- `vfx` atlas: beams, explosions, particles (if separated; optional)

**Packer rules**
- Keep original filenames as frame names.
- Disable rotation (so beams don’t rotate unexpectedly).
- Enable trimming (reduces atlas size).
- Produce **one atlas per category** unless the atlas exceeds GPU limits (then allow multipack).

### 2.3 Sprite loading (PixiJS recommended)

Pixi’s asset system supports `Assets.load()` for JSON and uses `sheet.textures` to create Sprites by frame name. citeturn2search1turn2search5

---

## 3) Render model (what we actually draw)

### 3.1 Primary views (no bespoke “art” required)

**A) Sector View (main canvas)**
- Background starfield (Redux backgrounds) + optional planet overlay
- Center: ship sprite
- Around ship: asteroid sprites (current node)
- Station sprite when at station node
- VFX overlay: scan beam / mining beam / explosions / warning flashes

**B) Node Graph Mini‑Map (SVG or simple canvas)**
- Graph nodes as circles
- Current node highlighted
- Neighbors highlighted with edge travel time / threat indicator
(This is important because your world is a navigation graph, not a continuous 2D map.) fileciteturn5file0

**C) HUD Panels**
- Heat/Fuel/Hull/Tool/Alert bars
- Cargo panel with commodity icons (6 commodities)
- Market panel with price arrows
- Event log panel (pirate/fracture/overheat/dock/sell)

---

## 4) Graphics manifest (semantic keys → atlas frames)

### 4.1 Manifest format

`graphics_manifest.json` (example schema):

```json
{
  "atlases": {
    "world": {"image": "/assets/atlases/world/atlas.png", "data": "/assets/atlases/world/atlas.json"},
    "ui":    {"image": "/assets/atlases/ui/atlas.png",    "data": "/assets/atlases/ui/atlas.json"},
    "vfx":   {"image": "/assets/atlases/vfx/atlas.png",   "data": "/assets/atlases/vfx/atlas.json"}
  },
  "frames": {
    "entity.ship.agent": {"atlas":"world","frame":"<FILL_IN>"},
    "entity.ship.human": {"atlas":"world","frame":"<FILL_IN>"},
    "...": {}
  },
  "backgrounds": {
    "bg.starfield.0": "/assets/backgrounds/<FILL_IN>.png"
  },
  "fonts": {
    "font.hud": "/assets/fonts/<FILL_IN>.ttf"
  },
  "colors": {
    "commodity.iron": "#9aa0a6"
  }
}
```

**Rule:** `<FILL_IN>` values are chosen from available file/frame names once packs are unpacked. The coding agent must populate these based on the selection criteria below.

### 4.2 World entities

| Semantic Key | Source Pack(s) | Selection Criteria | Notes |
|---|---|---|---|
| `entity.ship.agent` | Space Shooter Redux, Simple Space | A clearly visible ship silhouette; prefer “blue” or neutral ship | Used in replays |
| `entity.ship.human` | Space Shooter Redux, Simple Space | Distinct color from agent (e.g., green) | Used in /play |
| `entity.station` | Space Shooter Extension (satellites/stations), Redux | Choose a space station / satellite sprite that reads as a “safe dock” | Shows when node_type=station |
| `entity.asteroid.small.*` | Redux, Extension | Pick 3–6 small meteor sprites for variety | Random selection per asteroid slot |
| `entity.asteroid.medium.*` | Redux, Extension | Pick 3–6 medium meteor sprites | Random selection |
| `entity.asteroid.large.*` | Redux, Extension | Pick 3–6 large meteor sprites | Random selection |
| `entity.hazard.meteoroidField` | UI Pack Sci‑Fi + simple particles | Icon + particle overlay | Hazard nodes |
| `entity.hazard.radiation` | UI Pack Sci‑Fi | Radiation-ish icon or warning symbol | Hazard nodes |
| `entity.pirate.marker` | UI Pack Sci‑Fi | “alert”, “warning”, or “skull” icon | Used on HUD + events |
| `entity.pirate.ship` (optional) | Redux | Choose enemy ship silhouette | Used for encounter animation only |
| `entity.cargoCrate` (optional) | UI Pack / Redux | Any crate/box icon if available | Used for cargo HUD garnish |

### 4.3 Backgrounds / planets

| Semantic Key | Source Pack(s) | Selection Criteria |
|---|---|---|
| `bg.starfield.0..3` | Space Shooter Redux | Use the included backgrounds for variety (rotate per episode) citeturn2search11 |
| `bg.planet.*` | Planets | Use as decorative overlays at low opacity citeturn0search7turn0search11 |

### 4.4 UI (panels, bars, icons, buttons)

Use **UI Pack – Sci‑Fi** for primary styling (consistent space vibe) and fallback to UI Pack generic if needed. citeturn0search2turn0search14

**HUD panels**
- `ui.panel.stats`, `ui.panel.cargo`, `ui.panel.market`, `ui.panel.events`, `ui.panel.minimap`
- Choose matching panel sprites with consistent border style.

**Bars**
- `ui.bar.frame`, `ui.bar.fill`
- Use for Fuel/Hull/Heat/Tool/Alert and show numeric overlay.

**Buttons**
- `ui.button.primary`, `ui.button.secondary`, `ui.button.danger`
- Used for playback controls and play-mode action palette.

**Icons**
- `icon.action.scan`, `icon.action.mine`, `icon.action.dock`, `icon.action.sell`, `icon.action.repair`, etc.
- `icon.event.pirate`, `icon.event.fracture`, `icon.event.overheat`, `icon.event.stranded`, `icon.event.destroyed`
- `icon.market.up`, `icon.market.down`, `icon.market.flat`

### 4.5 Commodity icons (6 commodities)

Commodities are: iron, nickel, water_ice, pge, rare_isotopes, volatiles. fileciteturn5file2

Because Kenney packs may not ship “iron vs nickel” icons, the spec uses **simple icons + color coding**:

| Commodity | Semantic Key | Icon Source | Color Rule |
|---|---|---|---|
| Iron | `icon.commodity.iron` | generic ore/rock icon | gray |
| Nickel | `icon.commodity.nickel` | generic ore/rock icon | silver |
| Water Ice | `icon.commodity.water_ice` | droplet / snowflake / crystal icon | cyan |
| PGE | `icon.commodity.pge` | gem/diamond icon | gold |
| Rare Isotopes | `icon.commodity.rare_isotopes` | atom / core icon | magenta |
| Volatiles | `icon.commodity.volatiles` | gas / flame icon | orange |

**Implementation rule:** if unique icons are not present, reuse 1–2 “ore” icons and differentiate via color + label.

---

## 5) VFX mapping (actions + events → visual effects)

### 5.1 VFX manifest keys

Use VFX frames from Space Shooter Redux (lasers/explosions) and/or UI overlays.

| VFX Key | Source Pack(s) | Selection Criteria |
|---|---|---|
| `vfx.scan.widePulse` | Redux | light-blue pulse or beam |
| `vfx.scan.focusBeam` | Redux | tighter beam (thin laser) |
| `vfx.scan.deepSpectral` | Redux | distinct color (purple/green) + longer duration |
| `vfx.mine.conservative` | Redux | low-intensity beam + small particles |
| `vfx.mine.standard` | Redux | medium beam |
| `vfx.mine.aggressive` | Redux | bright beam + sparks |
| `vfx.cooldown.burst` | UI Pack Sci‑Fi + particles | radial burst overlay |
| `vfx.stabilize.field` | UI Pack Sci‑Fi | “shield” overlay / ring |
| `vfx.refine.sparkle` | particles | small sparkle / twinkle overlay |
| `vfx.repair.tool` | UI Pack Sci‑Fi | wrench/gear icon flash |
| `vfx.repair.hull` | UI Pack Sci‑Fi | patch/plus icon flash |
| `vfx.travel.warp` | UI Pack Sci‑Fi | screen wipe / streak overlay |
| `vfx.emergencyBurn` | Redux | thruster flare / boost trail |
| `vfx.dock` | UI Pack Sci‑Fi | docking highlight / glow |
| `vfx.sell` | UI Pack Sci‑Fi | credit “+” flyout |
| `vfx.jettison` | particles | eject particles drifting away |
| `vfx.explosion.small/med/large` | Redux | explosion sprite(s) |
| `vfx.warning.alert` | UI Pack Sci‑Fi | red flashing border |

### 5.2 Action → VFX mapping (Discrete 0..68)

This section maps RL actions to VFX keys. Action indices must match your RL action-space definition. fileciteturn5file4

**Travel / movement**
- `0..5 TRAVEL_NEIGHBOR[k]` → `vfx.travel.warp` + thrust trail
- `6 HOLD_DRIFT` → no VFX (optional: subtle idle particles)
- `7 EMERGENCY_BURN` → `vfx.emergencyBurn` + screen shake (small)

**Sensing**
- `8 WIDE_SCAN` → `vfx.scan.widePulse`
- `9 FOCUSED_SCAN_SELECTED` → `vfx.scan.focusBeam`
- `10 DEEP_SCAN_SELECTED` → `vfx.scan.deepSpectral`
- `11 PASSIVE_THREAT_LISTEN` → subtle HUD ping (no world VFX)

**Selection**
- `12..27 SELECT_ASTEROID[a]` → selection ring around asteroid (UI overlay)

**Mining**
- `28 MINE_CONSERVATIVE_SELECTED` → `vfx.mine.conservative`
- `29 MINE_STANDARD_SELECTED` → `vfx.mine.standard`
- `30 MINE_AGGRESSIVE_SELECTED` → `vfx.mine.aggressive`
- `31 STABILIZE_SELECTED` → `vfx.stabilize.field`
- `32 REFINE_ONBOARD` → `vfx.refine.sparkle`

**Thermal / maintenance / safety**
- `33 ACTIVE_COOLDOWN` → `vfx.cooldown.burst`
- `34 TOOL_MAINTENANCE` → `vfx.repair.tool`
- `35 HULL_PATCH` → `vfx.repair.hull`
- `36..41 JETTISON_COMMODITY[c]` → `vfx.jettison`

**Station**
- `42 DOCK` → `vfx.dock`
- `43..60 SELL[c,b]` → `vfx.sell`
- `61..63 BUY_FUEL_*` → small HUD confirmation pulse
- `64 BUY_REPAIR_KIT` → HUD confirmation
- `65 BUY_STABILIZER` → HUD confirmation
- `66 BUY_DECOY` → HUD confirmation
- `67 FULL_REPAIR_OVERHAUL` → bigger repair pulse + sparkle
- `68 CASH_OUT_END_EPISODE` → end-of-run banner (UI)

### 5.3 Event → VFX mapping

Events are emitted by the backend replay frames (pirate encounter, fracture, overheat, etc.). fileciteturn5file0

- `event.pirate_encounter` → `vfx.warning.alert` + pirate icon flash
- `event.fracture` → `vfx.explosion.med/large` + debris particles
- `event.overheat_damage` → red heat shimmer overlay
- `event.stranded` → engine sputter particles + warning border
- `event.destroyed` → `vfx.explosion.large` + fade to black
- `event.price_spike` (optional) → market panel glow + up arrow highlight

---

## 6) Audio spec (sound cues for all actions/events)

### 6.1 Audio sources

- Space Shooter Redux includes **7 sound effects**. citeturn2search11  
- Sci‑Fi Sounds adds engines/lasers/explosions and more (70 OGG). citeturn1search13  
- UI Audio adds consistent UI clicks/switches (50 SFX). citeturn1search3turn1search7  

### 6.2 Audio manifest format

`audio_manifest.json`:

```json
{
  "groups": {
    "ui": {"basePath": "/assets/audio/ui/", "volume": 0.6},
    "sfx": {"basePath": "/assets/audio/sfx/", "volume": 0.8}
  },
  "cues": {
    "ui.click": ["ui_click_01.ogg"],
    "sfx.scan.wide": ["scan_01.ogg"],
    "sfx.mine.aggressive": ["laser_03.ogg"]
  }
}
```

### 6.3 Action → SFX mapping

**Core SFX cues** (choose one file per cue; allow arrays for random variation):

| Action / Event | Cue Key | Notes |
|---|---|---|
| `TRAVEL_NEIGHBOR` | `sfx.travel.warp` | short whoosh |
| `HOLD_DRIFT` | (none) | optionally ambient hum |
| `EMERGENCY_BURN` | `sfx.travel.boost` | louder thrust |
| `WIDE_SCAN` | `sfx.scan.wide` | soft ping/pulse |
| `FOCUSED_SCAN` | `sfx.scan.focus` | tighter beam |
| `DEEP_SCAN` | `sfx.scan.deep` | longer synth sweep |
| `PASSIVE_THREAT_LISTEN` | `sfx.scan.listen` | subtle sonar click |
| `MINE_CONSERVATIVE` | `sfx.mine.low` | low laser |
| `MINE_STANDARD` | `sfx.mine.mid` | medium laser |
| `MINE_AGGRESSIVE` | `sfx.mine.high` | loud laser |
| `STABILIZE` | `sfx.utility.stabilize` | shield pop |
| `REFINE_ONBOARD` | `sfx.utility.refine` | sparkle chime |
| `ACTIVE_COOLDOWN` | `sfx.utility.cooldown` | vent hiss |
| `TOOL_MAINTENANCE` | `sfx.utility.tool_repair` | wrench click |
| `HULL_PATCH` | `sfx.utility.hull_patch` | repair beep |
| `JETTISON` | `sfx.utility.jettison` | eject puff |
| `DOCK` | `sfx.station.dock` | clamp / confirm |
| `SELL` | `sfx.station.sell` | cash/confirm beep |
| `BUY_*` | `sfx.station.buy` | confirm beep |
| `FULL_REPAIR_OVERHAUL` | `sfx.station.overhaul` | longer repair sound |
| `CASH_OUT_END_EPISODE` | `sfx.ui.run_end` | success sting |

**Event SFX**
- pirate encounter → `sfx.event.pirate_alarm` (warning siren)
- fracture → `sfx.event.explosion`
- overheat damage → `sfx.event.overheat_alarm`
- stranded → `sfx.event.engine_fail`
- destroyed → `sfx.event.big_explosion`

### 6.4 UI sounds (consistent feel)
Use UI Audio pack cues for:
- button click / hover / toggles
- playback controls (play/pause/step)
- play-mode action palette clicks

---

## 7) Renderer behavior rules (how to apply assets at runtime)

### 7.1 State-driven rendering (replay + play mode)

The renderer consumes:
- current `render_state` (ship stats, asteroids, market, node)
- current `action_id` and `events` for the frame

Then:
1) Updates sprite positions/layout
2) Applies selection highlight (selected asteroid)
3) Triggers VFX for the current action
4) Triggers event VFX/SFX for any events listed
5) Updates HUD bars and text

### 7.2 Layout rules (graph world → 2D presentation)

Because your world is a **graph**, the “sector view” is a **node-local visualization**:

- Station node:
  - station sprite centered
  - ship sprite near station
- Cluster node:
  - ship centered
  - asteroids placed in a ring or spiral around ship, stable across frames:
    - asteroid index `a` at angle `2π*a/MAX_ASTEROIDS`
    - radius can encode `depletion` or `scan_conf`
- Hazard node:
  - overlay hazard field particles + warning border
  - optionally hide asteroids

Mini-map always shows:
- current node highlight
- neighbor nodes (up to 6) with threat indicator bar

---

## 8) Implementation checklist for the coding agent

### 8.1 One-time setup
- Download and unpack all required packs (and recommended audio packs).
- Create atlases with Free Texture Packer (world/ui/vfx) and copy outputs to `/public/assets/atlases/...` citeturn2search0turn2search4
- Copy audio to `/public/assets/audio/...` (OGG recommended for web).
- Copy fonts/backgrounds to `/public/assets/fonts` and `/public/assets/backgrounds`.

### 8.2 Build the manifests
- Populate `graphics_manifest.json` with:
  - atlas paths
  - frame names chosen from the atlases
  - backgrounds and fonts
- Populate `audio_manifest.json` with:
  - cue keys → list of audio files

### 8.3 Runtime loading
- Load atlases via Pixi Assets.load and construct sprites from `sheet.textures[frameName]`. citeturn2search1turn2search5
- Preload audio files into an audio cache (Web Audio or HTMLAudio), keyed by cue.

### 8.4 Validation (must pass before shipping)
- “Manifest completeness” test:
  - every semantic key resolves to an existing atlas frame or file
- “Action VFX/SFX” test:
  - for each action id 0..68, confirm a mapped VFX/SFX exists (or explicitly “none”)
- “Event playback” test:
  - pirate/fracture/overheat/dock/sell events show correct icon and sound
- “Performance” test:
  - replay at 10 fps is stable and does not leak textures/audio instances

---

## 9) Minimal deliverable set (what to commit)

1) `/public/assets/atlases/*` (png+json)  
2) `/public/assets/audio/*` (ogg)  
3) `/public/assets/backgrounds/*`  
4) `/public/assets/fonts/*` (optional)  
5) `/public/assets/manifests/graphics_manifest.json`  
6) `/public/assets/manifests/audio_manifest.json`  
7) `lib/assets.ts` that loads manifests and exposes helpers:
   - `getFrame(key)`, `getSoundCue(key)`

---

## 10) Notes on style consistency

- Use **Space Shooter Redux + Extension** for the in-world sprites and VFX to keep a consistent look. citeturn0search0turn0search1  
- Use **UI Pack – Sci‑Fi** for HUD controls and panels. citeturn0search2turn0search6  
- Use **Planets** as optional background flavor only (avoid clutter). citeturn0search7turn0search11  
- Use **Simple Space** primarily for mini-map icons or as a “minimal mode” skin. citeturn1search0turn1search8  
