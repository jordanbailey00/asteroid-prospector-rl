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


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise AssertionError(f"Missing manifest file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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


def test_background_manifest_paths_exist(graphics_manifest: dict) -> None:
    backgrounds = graphics_manifest.get("backgrounds", {})
    assert isinstance(backgrounds, dict)
    for key, value in backgrounds.items():
        path_str = value.get("path")
        assert isinstance(path_str, str) and path_str.startswith("/assets/")
        relative = path_str.lstrip("/")
        path = REPO_ROOT / "frontend" / "public" / relative
        assert path.exists(), f"Background path missing for {key}: {path}"
