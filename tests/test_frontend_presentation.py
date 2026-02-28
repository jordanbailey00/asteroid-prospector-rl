from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "frontend" / "public" / "assets" / "manifests"

CORE_GRAPHICS_KEYS = {
    "entity.ship.agent",
    "entity.ship.human",
    "entity.station",
    "entity.asteroid.small",
    "entity.asteroid.medium",
    "entity.asteroid.large",
    "entity.hazard.radiation",
    "entity.pirate.marker",
    "ui.panel.stats",
    "ui.panel.cargo",
    "ui.panel.market",
    "ui.panel.events",
    "ui.panel.minimap",
    "ui.button.primary",
    "ui.button.secondary",
    "ui.button.danger",
    "icon.market.up",
    "icon.market.down",
    "icon.market.flat",
    "icon.event.pirate",
    "icon.event.fracture",
    "icon.event.overheat",
    "icon.event.stranded",
    "icon.event.destroyed",
    "icon.event.invalid",
}

CORE_BACKGROUND_KEYS = {
    "bg.starfield.0",
    "bg.starfield.1",
    "bg.planet.0",
    "bg.planet.1",
}

SEMANTIC_PATH_HINTS = {
    "entity.ship.agent": "/sprites/world/ship",
    "entity.ship.human": "/sprites/world/ship",
    "entity.station": "/sprites/world/station",
    "entity.asteroid.small": "/sprites/world/asteroid",
    "entity.asteroid.medium": "/sprites/world/asteroid",
    "entity.asteroid.large": "/sprites/world/asteroid",
    "entity.hazard.radiation": "/sprites/world/hazard",
    "entity.pirate.marker": "/sprites/world/pirate",
    "ui.panel.stats": "/sprites/ui/panel",
    "ui.panel.cargo": "/sprites/ui/panel",
    "ui.panel.market": "/sprites/ui/panel",
    "ui.panel.events": "/sprites/ui/panel",
    "ui.panel.minimap": "/sprites/ui/panel",
    "ui.button.primary": "/sprites/ui/button",
    "ui.button.secondary": "/sprites/ui/button",
    "ui.button.danger": "/sprites/ui/button",
    "icon.market.up": "/sprites/ui/icon_market",
    "icon.market.down": "/sprites/ui/icon_market",
    "icon.market.flat": "/sprites/ui/icon_market",
    "icon.event.pirate": "/sprites/ui/icon_event",
    "icon.event.fracture": "/sprites/ui/icon_event",
    "icon.event.overheat": "/sprites/ui/icon_event",
    "icon.event.stranded": "/sprites/ui/icon_event",
    "icon.event.destroyed": "/sprites/ui/icon_event",
    "icon.event.invalid": "/sprites/ui/icon_event",
}


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise AssertionError(f"Missing manifest file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_public_asset(path_str: str) -> Path:
    assert path_str.startswith("/assets/"), f"Asset path must be /assets/...: {path_str}"
    relative = path_str.lstrip("/")
    return REPO_ROOT / "frontend" / "public" / relative


@pytest.fixture(scope="module")
def graphics_manifest() -> dict:
    return _load_json(MANIFEST_DIR / "graphics_manifest.json")


@pytest.fixture(scope="module")
def audio_manifest() -> dict:
    return _load_json(MANIFEST_DIR / "audio_manifest.json")


@pytest.fixture(scope="module")
def effects_manifest() -> dict:
    return _load_json(REPO_ROOT / "frontend" / "lib" / "action_effects_manifest.json")


def test_manifest_files_exist_and_parse(
    graphics_manifest: dict,
    audio_manifest: dict,
    effects_manifest: dict,
) -> None:
    assert graphics_manifest["version"].startswith("m6.5")
    assert audio_manifest["version"].startswith("m6.5")
    assert effects_manifest["version"].startswith("m6.5")


def test_graphics_manifest_has_core_semantic_keys(graphics_manifest: dict) -> None:
    frames = graphics_manifest.get("frames")
    assert isinstance(frames, dict)

    missing = sorted(key for key in CORE_GRAPHICS_KEYS if key not in frames)
    assert not missing, f"Missing core graphics keys: {missing}"

    for key in sorted(CORE_GRAPHICS_KEYS):
        frame_def = frames[key]
        assert isinstance(frame_def, dict), f"Frame value must be an object for {key}"
        path_str = frame_def.get("path")
        assert isinstance(path_str, str) and path_str.startswith(
            "/assets/"
        ), f"Frame must have a file-backed /assets path: {key}"
        asset_path = _resolve_public_asset(path_str)
        assert asset_path.exists(), f"Frame asset file missing for {key}: {asset_path}"

        expected_hint = SEMANTIC_PATH_HINTS.get(key)
        if expected_hint is not None:
            assert expected_hint in path_str, (
                f"Frame path for {key} does not match semantic class hint "
                f"'{expected_hint}': {path_str}"
            )


def test_action_effect_mappings_cover_all_actions(effects_manifest: dict) -> None:
    actions = effects_manifest.get("actions")
    assert isinstance(actions, list)
    assert len(actions) == 69

    ids = sorted(int(row["id"]) for row in actions)
    assert ids == list(range(69))


def test_action_and_event_vfx_resolve_to_graphics_manifest(
    graphics_manifest: dict,
    effects_manifest: dict,
) -> None:
    frames = graphics_manifest["frames"]

    action_vfx = {str(row["vfx"]) for row in effects_manifest["actions"]}
    event_vfx = {str(row["vfx"]) for row in effects_manifest["events"].values()}
    required_vfx = sorted(action_vfx | event_vfx)

    missing = [key for key in required_vfx if key not in frames]
    assert not missing, f"VFX keys missing from graphics manifest: {missing}"

    for key in required_vfx:
        frame_def = frames[key]
        assert isinstance(frame_def, dict), f"VFX frame definition must be an object: {key}"
        path_str = frame_def.get("path")
        assert isinstance(path_str, str) and path_str.startswith(
            "/assets/"
        ), f"VFX key must map to /assets path: {key}"
        asset_path = _resolve_public_asset(path_str)
        assert asset_path.exists(), f"VFX asset path missing for {key}: {asset_path}"

        if key.startswith("vfx."):
            assert (
                "/sprites/vfx/" in path_str
            ), f"VFX key must resolve to vfx sprite path for {key}: {path_str}"


def test_action_and_event_cues_resolve_to_audio_manifest(
    audio_manifest: dict,
    effects_manifest: dict,
) -> None:
    cues = audio_manifest.get("cues")
    groups = audio_manifest.get("groups")

    assert isinstance(cues, dict)
    assert isinstance(groups, dict)

    required_cues = {str(row["cue"]) for row in effects_manifest["actions"]} | {
        str(row["cue"]) for row in effects_manifest["events"].values()
    }

    missing = sorted(cue for cue in required_cues if cue not in cues)
    assert not missing, f"Cue keys missing from audio manifest: {missing}"

    for cue_key, cue_def in cues.items():
        assert isinstance(cue_def, dict), f"Cue must be an object: {cue_key}"
        group_key = cue_def.get("group")
        assert group_key in groups, f"Cue group is not declared: {cue_key}"
        files = cue_def.get("files")
        assert isinstance(files, list), f"Cue files must be a list: {cue_key}"

        base_path = groups[group_key].get("basePath")
        assert isinstance(base_path, str) and base_path.startswith(
            "/assets/"
        ), f"Cue group basePath must be /assets/... for {cue_key}"

        if cue_key.endswith(".none"):
            continue

        assert files, f"Non-none cue must have at least one file: {cue_key}"
        for filename in files:
            assert (
                isinstance(filename, str) and filename.strip()
            ), f"Cue file names must be non-empty strings: {cue_key}"
            full_path = f"{base_path.rstrip('/')}/{filename}"
            asset_path = _resolve_public_asset(full_path)
            assert asset_path.exists(), f"Cue audio file missing for {cue_key}: {asset_path}"


def test_background_manifest_paths_exist(graphics_manifest: dict) -> None:
    backgrounds = graphics_manifest.get("backgrounds", {})
    assert isinstance(backgrounds, dict)

    missing = sorted(key for key in CORE_BACKGROUND_KEYS if key not in backgrounds)
    assert not missing, f"Missing background keys: {missing}"

    for key in sorted(CORE_BACKGROUND_KEYS):
        value = backgrounds[key]
        assert isinstance(value, dict), f"Background value must be an object: {key}"
        path_str = value.get("path")
        assert isinstance(path_str, str) and path_str.startswith("/assets/")

        if key.startswith("bg.planet"):
            assert (
                "/planets/" in path_str
            ), f"Planet key must map to planet asset path: {key} -> {path_str}"
        if key.startswith("bg.starfield"):
            assert (
                "/backgrounds/" in path_str
            ), f"Starfield key must map to background asset path: {key} -> {path_str}"

        path = _resolve_public_asset(path_str)
        assert path.exists(), f"Background path missing for {key}: {path}"
