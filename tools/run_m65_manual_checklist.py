#!/usr/bin/env python3
"""Generate reproducible M6.5 replay/play checklist evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PYTHON_SRC = REPO_ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from asteroid_prospector import ProspectorReferenceEnv, ReferenceEnvConfig  # noqa: E402
from asteroid_prospector.constants import MAX_ASTEROIDS, MAX_NEIGHBORS  # noqa: E402

from replay.schema import frame_from_step, validate_replay_frame  # noqa: E402

WARNING_EVENTS = {"invalid_action", "pirate_encounter", "overheat_tick"}


@dataclass
class StepRecord:
    step_index: int
    purpose: str
    action: int
    reward: float
    dt: int
    invalid_action: bool
    terminated: bool
    truncated: bool
    node_context: str
    credits_before: float
    credits_after: float
    cargo_before: float
    cargo_after: float
    events: list[str]
    action_vfx_key: str = ""
    action_vfx_path: str = ""
    action_cue_key: str = ""
    action_cue_files: list[str] = field(default_factory=list)
    event_vfx_keys: list[str] = field(default_factory=list)
    event_cue_keys: list[str] = field(default_factory=list)


@dataclass
class ChecklistResult:
    passed: bool
    seed: int
    frames: list[dict[str, Any]]
    steps: list[StepRecord]
    checks: dict[str, bool]
    reason: str = ""


def _action_label(action: int) -> str:
    if 0 <= action <= 5:
        return f"TRAVEL_SLOT_{action}"
    labels = {
        6: "HOLD",
        7: "EMERGENCY_BURN",
        8: "SCAN_WIDE",
        9: "SCAN_FOCUSED",
        10: "SCAN_DEEP",
        11: "THREAT_LISTEN",
        28: "MINE_CONSERVATIVE",
        29: "MINE_STANDARD",
        30: "MINE_AGGRESSIVE",
        42: "DOCK",
        68: "END_RUN",
    }
    if action in labels:
        return labels[action]
    if 12 <= action <= 27:
        return f"SELECT_ASTEROID_{action - 12}"
    if 43 <= action <= 60:
        commodity = (action - 43) // 3
        bucket = (action - 43) % 3
        bucket_label = ("25%", "50%", "100%")[bucket]
        return f"SELL_COMMODITY_{commodity}_{bucket_label}"
    return f"ACTION_{action}"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_public_asset(path_str: str) -> Path:
    if not path_str.startswith("/assets/"):
        raise ValueError(f"Asset path must begin with /assets/: {path_str}")
    return REPO_ROOT / "frontend" / "public" / path_str.lstrip("/")


def _find_shortest_slot_path(
    env: ProspectorReferenceEnv, start_node: int, target_node: int
) -> list[int]:
    queue: deque[tuple[int, list[int]]] = deque([(int(start_node), [])])
    visited = {int(start_node)}

    while queue:
        node, path = queue.popleft()
        if node == int(target_node):
            return path

        for slot in range(MAX_NEIGHBORS):
            neighbor = int(env.neighbors[node, slot])
            if neighbor < 0 or neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, [*path, slot]))

    raise RuntimeError(f"No route found from node {start_node} to node {target_node}")


def _find_first_valid_asteroid(env: ProspectorReferenceEnv) -> int:
    for asteroid in range(MAX_ASTEROIDS):
        if int(env.ast_valid[env.current_node, asteroid]) <= 0:
            continue
        if float(env.depletion[env.current_node, asteroid]) >= 1.0:
            continue
        return asteroid
    raise RuntimeError(f"No valid asteroid at node {env.current_node}")


def _derive_events(
    info: dict[str, Any],
    prev_info: dict[str, Any] | None,
    terminated: bool,
    truncated: bool,
) -> list[str]:
    events: list[str] = []

    if bool(info.get("invalid_action", False)):
        events.append("invalid_action")

    pirates_now = float(info.get("pirate_encounters", 0.0))
    pirates_prev = float((prev_info or {}).get("pirate_encounters", 0.0))
    if pirates_now > pirates_prev:
        events.append("pirate_encounter")

    overheat_now = float(info.get("overheat_ticks", 0.0))
    overheat_prev = float((prev_info or {}).get("overheat_ticks", 0.0))
    if overheat_now > overheat_prev:
        events.append("overheat_tick")

    if terminated:
        events.append("terminated")
    if truncated:
        events.append("truncated")

    return events


def _attempt_seed(seed: int) -> ChecklistResult:
    env = ProspectorReferenceEnv(config=ReferenceEnvConfig(), seed=seed)
    obs, info = env.reset(seed=seed)

    frames: list[dict[str, Any]] = []
    steps: list[StepRecord] = []
    prev_info: dict[str, Any] | None = info
    t_cumulative = 0
    done = False

    def run_step(action: int, purpose: str) -> tuple[bool, bool]:
        nonlocal obs, info, prev_info, t_cumulative, done
        if done:
            raise RuntimeError("Attempted to step after terminal state")

        credits_before = float(info.get("credits", 0.0))
        cargo_before = float(np.sum(env.cargo))

        obs, reward, terminated, truncated, next_info = env.step(action)

        credits_after = float(next_info.get("credits", credits_before))
        cargo_after = float(np.sum(env.cargo))
        dt = int(next_info.get("dt", 1))
        if dt <= 0:
            dt = 1
        t_cumulative += dt

        events = _derive_events(
            info=next_info,
            prev_info=prev_info,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        frame = frame_from_step(
            frame_index=len(frames),
            t=t_cumulative,
            dt=dt,
            action=int(action),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            render_state={
                "credits": float(next_info.get("credits", 0.0)),
                "net_profit": float(next_info.get("net_profit", 0.0)),
                "time_remaining": float(next_info.get("time_remaining", 0.0)),
                "node_context": str(next_info.get("node_context", "unknown")),
                "survival": float(next_info.get("survival", 0.0)),
            },
            events=events,
            info=next_info,
            include_info=True,
        )
        validate_replay_frame(frame)
        frames.append(frame)

        steps.append(
            StepRecord(
                step_index=len(steps) + 1,
                purpose=purpose,
                action=int(action),
                reward=float(reward),
                dt=dt,
                invalid_action=bool(next_info.get("invalid_action", False)),
                terminated=bool(terminated),
                truncated=bool(truncated),
                node_context=str(next_info.get("node_context", "unknown")),
                credits_before=credits_before,
                credits_after=credits_after,
                cargo_before=cargo_before,
                cargo_after=cargo_after,
                events=events,
            )
        )

        info = next_info
        prev_info = next_info
        done = bool(terminated or truncated)
        return bool(terminated), bool(truncated)

    try:
        travel_slot = next(
            slot for slot in range(MAX_NEIGHBORS) if int(env.neighbors[env.current_node, slot]) >= 0
        )
    except StopIteration:
        return ChecklistResult(
            passed=False,
            seed=seed,
            frames=[],
            steps=[],
            checks={},
            reason="station node has no valid travel slots",
        )

    try:
        run_step(travel_slot, "Travel from station to field node")
        run_step(43, "Intentional invalid sell outside station to trigger warning cue")

        asteroid_index = _find_first_valid_asteroid(env)
        run_step(12 + asteroid_index, f"Select asteroid {asteroid_index}")
        run_step(9, "Focused scan on selected asteroid")
        run_step(29, "Standard mining action")

        path_back = _find_shortest_slot_path(env, start_node=int(env.current_node), target_node=0)
        for hop, slot in enumerate(path_back, start=1):
            run_step(slot, f"Travel hop {hop} back to station")

        run_step(42, "Dock at station")

        commodity = int(np.argmax(env.cargo))
        if float(env.cargo[commodity]) <= 0.0:
            return ChecklistResult(
                passed=False,
                seed=seed,
                frames=frames,
                steps=steps,
                checks={},
                reason="cargo remained empty after mining; no sale demonstration",
            )
        sell_action = 43 + commodity * 3 + 2
        run_step(sell_action, f"Sell mined commodity {commodity} at station")

        run_step(68, "Trigger terminal state via END_RUN")
    except RuntimeError as exc:
        return ChecklistResult(
            passed=False,
            seed=seed,
            frames=frames,
            steps=steps,
            checks={},
            reason=str(exc),
        )

    checks = {
        "travel_action": any(0 <= row.action <= 5 and not row.invalid_action for row in steps),
        "scan_action": any(8 <= row.action <= 11 and not row.invalid_action for row in steps),
        "mining_action": any(28 <= row.action <= 30 and not row.invalid_action for row in steps),
        "dock_action": any(row.action == 42 and not row.invalid_action for row in steps),
        "sell_action": any(
            43 <= row.action <= 60
            and not row.invalid_action
            and row.credits_after > row.credits_before
            and row.cargo_after < row.cargo_before
            for row in steps
        ),
        "warning_event": any(bool(WARNING_EVENTS.intersection(set(row.events))) for row in steps),
        "terminal_event": bool(
            steps and ("terminated" in steps[-1].events or "truncated" in steps[-1].events)
        ),
    }

    passed = all(checks.values())
    reason = "" if passed else "scenario checks failed"
    return ChecklistResult(
        passed=passed,
        seed=seed,
        frames=frames,
        steps=steps,
        checks=checks,
        reason=reason,
    )


def _attach_presentation_mappings(steps: list[StepRecord]) -> None:
    effects = _load_json(REPO_ROOT / "frontend" / "lib" / "action_effects_manifest.json")
    graphics = _load_json(
        REPO_ROOT / "frontend" / "public" / "assets" / "manifests" / "graphics_manifest.json"
    )
    audio = _load_json(
        REPO_ROOT / "frontend" / "public" / "assets" / "manifests" / "audio_manifest.json"
    )

    action_rows = effects.get("actions", [])
    if not isinstance(action_rows, list) or len(action_rows) != 69:
        raise ValueError("action_effects_manifest must provide 69 action mappings")
    action_map = {int(row["id"]): row for row in action_rows}
    event_map = effects.get("events", {})
    if not isinstance(event_map, dict):
        raise ValueError("events mapping must be an object in action_effects_manifest")

    frames_map = graphics.get("frames", {})
    cues_map = audio.get("cues", {})
    groups_map = audio.get("groups", {})
    if (
        not isinstance(frames_map, dict)
        or not isinstance(cues_map, dict)
        or not isinstance(groups_map, dict)
    ):
        raise ValueError("graphics/audio manifests are malformed")

    def resolve_vfx(vfx_key: str) -> str:
        frame_def = frames_map.get(vfx_key)
        if not isinstance(frame_def, dict):
            raise ValueError(f"Missing graphics frame for key: {vfx_key}")
        path_str = frame_def.get("path")
        if not isinstance(path_str, str):
            raise ValueError(f"Graphics frame path must be string for key: {vfx_key}")
        if not _resolve_public_asset(path_str).exists():
            raise ValueError(f"Graphics asset missing for key {vfx_key}: {path_str}")
        return path_str

    def resolve_cue(cue_key: str) -> list[str]:
        cue_def = cues_map.get(cue_key)
        if not isinstance(cue_def, dict):
            raise ValueError(f"Missing audio cue mapping for key: {cue_key}")
        group_key = cue_def.get("group")
        if not isinstance(group_key, str) or group_key not in groups_map:
            raise ValueError(f"Audio cue group missing for key: {cue_key}")
        group_def = groups_map[group_key]
        if not isinstance(group_def, dict):
            raise ValueError(f"Audio group must be object for group: {group_key}")
        base_path = group_def.get("basePath")
        if not isinstance(base_path, str):
            raise ValueError(f"Audio basePath missing for group: {group_key}")
        raw_files = cue_def.get("files", [])
        if not isinstance(raw_files, list):
            raise ValueError(f"Audio cue files must be list for key: {cue_key}")

        full_paths: list[str] = []
        for file_name in raw_files:
            if not isinstance(file_name, str):
                raise ValueError(f"Audio file entry must be a string for cue: {cue_key}")
            full = f"{base_path.rstrip('/')}/{file_name.lstrip('/')}"
            if not _resolve_public_asset(full).exists():
                raise ValueError(f"Audio asset missing for cue {cue_key}: {full}")
            full_paths.append(full)
        return full_paths

    for row in steps:
        action_row = action_map.get(row.action)
        if action_row is None:
            raise ValueError(f"Missing action mapping for action id: {row.action}")
        action_vfx = str(action_row["vfx"])
        action_cue = str(action_row["cue"])

        row.action_vfx_key = action_vfx
        row.action_vfx_path = resolve_vfx(action_vfx)
        row.action_cue_key = action_cue
        row.action_cue_files = resolve_cue(action_cue)

        event_vfx: list[str] = []
        event_cues: list[str] = []
        for event_name in row.events:
            mapping = event_map.get(event_name)
            if mapping is None:
                continue
            if not isinstance(mapping, dict):
                raise ValueError(f"Event mapping must be object for event: {event_name}")
            vfx_key = str(mapping["vfx"])
            cue_key = str(mapping["cue"])
            resolve_vfx(vfx_key)
            resolve_cue(cue_key)
            event_vfx.append(vfx_key)
            event_cues.append(cue_key)

        row.event_vfx_keys = event_vfx
        row.event_cue_keys = event_cues


def _write_replay_jsonl(path: Path, frames: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for frame in frames:
            handle.write(json.dumps(frame, separators=(",", ":")))
            handle.write("\n")


def _build_markdown(result: ChecklistResult, replay_path: Path, command_hint: str) -> str:
    generated_at = datetime.now(UTC).isoformat()
    relative_replay = replay_path.relative_to(REPO_ROOT).as_posix()

    lines: list[str] = []
    lines.append("# M6.5 Manual Replay/Play Verification")
    lines.append("")
    lines.append(f"- Generated: `{generated_at}`")
    lines.append(f"- Seed: `{result.seed}`")
    lines.append(f"- Replay sample: `{relative_replay}`")
    lines.append(f"- Reproduce: `{command_hint}`")
    lines.append("")
    lines.append("## Checklist Outcome")
    lines.append("")
    for key, value in result.checks.items():
        status = "PASS" if value else "FAIL"
        lines.append(f"- [{status}] `{key}`")
    lines.append("")
    lines.append("## Sample Step Trace")
    lines.append("")
    lines.append("| Step | Purpose | Action | Invalid | Events | Action VFX | Action Cue |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in result.steps:
        action_with_label = f"`{row.action}` `{_action_label(row.action)}`"
        events = ", ".join(row.events) if row.events else "none"
        lines.append(
            f"| {row.step_index} | {row.purpose} | {action_with_label} | "
            f"{str(row.invalid_action).lower()} | {events} | "
            f"`{row.action_vfx_key}` | `{row.action_cue_key}` |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Replay and play mode share the same action/event VFX+cues via "
        "`action_effects_manifest.json`."
    )
    lines.append(
        "- Action and event cue keys in this sample were validated against "
        "file-backed graphics/audio manifests."
    )
    return "\n".join(lines) + "\n"


def _run(seed_start: int, seed_limit: int) -> ChecklistResult:
    last_result: ChecklistResult | None = None
    for seed in range(seed_start, seed_start + seed_limit):
        result = _attempt_seed(seed)
        if result.passed:
            return result
        last_result = result
    if last_result is None:
        raise RuntimeError("No seed attempts were executed")
    end_seed = seed_start + seed_limit - 1
    raise RuntimeError(
        f"Unable to find passing seed in range {seed_start}..{end_seed}: {last_result.reason}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=11, help="first seed to attempt")
    parser.add_argument("--seed-limit", type=int, default=300, help="number of seeds to attempt")
    parser.add_argument(
        "--output-doc",
        type=Path,
        default=REPO_ROOT / "docs" / "M65_MANUAL_VERIFICATION.md",
        help="path to generated markdown checklist report",
    )
    parser.add_argument(
        "--output-replay",
        type=Path,
        default=REPO_ROOT / "docs" / "verification" / "m65_sample_replay.jsonl",
        help="path to generated sampled replay jsonl",
    )
    args = parser.parse_args()

    result = _run(seed_start=args.seed_start, seed_limit=args.seed_limit)
    _attach_presentation_mappings(result.steps)

    _write_replay_jsonl(args.output_replay, result.frames)
    command_hint = "python tools/run_m65_manual_checklist.py"
    markdown = _build_markdown(result, args.output_replay, command_hint)
    args.output_doc.parent.mkdir(parents=True, exist_ok=True)
    args.output_doc.write_text(markdown, encoding="utf-8", newline="\n")

    print(
        json.dumps(
            {
                "status": "ok",
                "seed": result.seed,
                "output_doc": str(args.output_doc),
                "output_replay": str(args.output_replay),
                "checks": result.checks,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
