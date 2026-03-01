# M6.5 Manual Replay/Play Verification

- Generated: `2026-03-01T00:04:35.589616+00:00`
- Seed: `11`
- Replay sample: `docs/verification/m65_sample_replay.jsonl`
- Reproduce: `python tools/run_m65_manual_checklist.py`

## Checklist Outcome

- [PASS] `travel_action`
- [PASS] `scan_action`
- [PASS] `mining_action`
- [PASS] `dock_action`
- [PASS] `sell_action`
- [PASS] `warning_event`
- [PASS] `terminal_event`

## Sample Step Trace

| Step | Purpose | Action | Invalid | Events | Action VFX | Action Cue |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Travel from station to field node | `0` `TRAVEL_SLOT_0` | false | none | `vfx.travel.warp` | `sfx.travel.warp` |
| 2 | Intentional invalid sell outside station to trigger warning cue | `43` `SELL_COMMODITY_0_25%` | true | invalid_action | `vfx.sell` | `sfx.station.sell` |
| 3 | Select asteroid 0 | `12` `SELECT_ASTEROID_0` | false | none | `vfx.selection.ring` | `ui.select` |
| 4 | Focused scan on selected asteroid | `9` `SCAN_FOCUSED` | false | none | `vfx.scan.focusBeam` | `sfx.scan.focus` |
| 5 | Standard mining action | `29` `MINE_STANDARD` | false | none | `vfx.mine.standard` | `sfx.mine.mid` |
| 6 | Travel hop 1 back to station | `0` `TRAVEL_SLOT_0` | false | none | `vfx.travel.warp` | `sfx.travel.warp` |
| 7 | Dock at station | `42` `DOCK` | false | none | `vfx.dock` | `sfx.station.dock` |
| 8 | Sell mined commodity 1 at station | `48` `SELL_COMMODITY_1_100%` | false | none | `vfx.sell` | `sfx.station.sell` |
| 9 | Trigger terminal state via END_RUN | `68` `END_RUN` | false | terminated | `vfx.ui.run_end` | `sfx.ui.run_end` |

## Notes

- Replay and play mode share the same action/event VFX+cues via `action_effects_manifest.json`.
- Action and event cue keys in this sample were validated against file-backed graphics/audio manifests.
