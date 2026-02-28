# Project Status

Last updated: 2026-02-28
Current focus: M6.5 real-asset frontend wiring

## Current state

- Frozen interface status: unchanged (`OBS_DIM=260`, `N_ACTIONS=69`, action indexing `0..68`, reward definition unchanged).
- Build health: local checks passing via `tools/run_checks.ps1` (`pytest -q` currently 52 passed, 2 skipped).
- Frontend build health: `npm --prefix frontend run build` passes for replay/play/analytics routes.
- Completed milestones: M0, M1, M2, M2.5, M3, M4, M5, and M6 complete.
- Active milestone: M6.5 graphics/audio real-asset integration.
- Trainer runtime baseline updated to latest published Puffer core line (`pufferlib-core==3.0.17`) with verified Docker PPO smoke run on the new dependency stack.

## Milestone board

| Milestone | Status | What is done | Evidence |
| --- | --- | --- | --- |
| M0 - Scaffold and hello env | Complete | Repo skeleton, tooling, CI, contract stub env | `001a004` |
| M1 - Python reference env | Complete | Reference environment, tier tests, determinism and reward checks | `b009f53`, `6e33091`, `b4e31bd` |
| M2 - Native core + bindings | Complete | Full C world/step dynamics, reward + obs packing, expanded native parity metrics surfaced through ctypes | `5aa1058` |
| M2.5 - Parity harness | Complete | Fixed-suite parity harness, mismatch bundles, and deterministic RNG alignment across Python/C | `1cd2dfc`, `cb2efbf` |
| M3 - Training + window metrics | Complete | Windowed trainer loop, checkpoint cadence, `windows.jsonl`, optional W&B logging, live `run_metadata.json`, and Dockerized Linux `puffer_ppo` backend with PufferLib PPO smoke validation (3 windows) | `1a90101`, `6dd1f89`, `4b3e684` |
| M4 - Eval + replay generation | Complete | Policy-driven PPO eval replays from serialized checkpoints, replay schema/index validation, `every_window` + `best_so_far` + `milestone:*` tagging, replay index filtering helpers, and checkpoint format tests | `b8a1880`, `452754c` |
| M5 - API server | Complete | FastAPI run/replay/metrics endpoints, replay frame pagination endpoint, in-memory play-session lifecycle endpoints, and CORS configuration with endpoint tests | `e1fe165`, `98149f2` |
| M6 - Frontend integration | Complete | Next.js replay page (`/`), human play mode (`/play`), analytics page (`/analytics`) wired to M5 APIs with playback controls, run/window/replay selection, and historical trend visualizations | `27ab411` |
| M6.5 - Graphics + audio integration | In progress | Real Kenney assets are now staged in `frontend/public/assets`, manifests are file-backed, sector/minimap rendering consumes mapped world/background/VFX assets, and tests enforce semantic path + file existence coverage. Remaining work is final manual replay/play verification checklist and any residual HUD/icon polish. | `1a77f36`, pending (this commit) |
| M7+ - Perf and stability | Not started | Throughput targets, soak checks, benchmark automation | pending |

## Next work (ordered)

1. Execute final M6.5 manual replay/play verification checklist and patch any residual visual/audio mismatches.
2. Add websocket replay streaming endpoint and optional frontend transport switch (HTTP vs WS).
3. Implement M7 benchmark harness for trainer throughput, replay API latency, and memory soak checks.

## Active risks and blockers

- M6.5 is not complete until required semantic keys and cues resolve to real Kenney files.
- Frontend replay playback currently uses full-frame HTTP fetch; very large replays may need WS streaming or chunked loading.

## Decision pointers

- See `docs/DECISION_LOG.md` for accepted architecture and process decisions.

## Commit ledger (latest first)

| Date | Commit | Type | Summary |
| --- | --- | --- | --- |
| 2026-02-28 | pending (this commit) | chore | Upgrade trainer deps/image to latest published Puffer core stack and patch PPO runtime compatibility |
| 2026-02-28 | `f606846` | feat | Wire sector rendering to file-backed Kenney assets and enforce manifest asset validation in tests |
| 2026-02-28 | `b2b98cf` | docs | Tighten M6.5 completion criteria to require real Kenney asset wiring |
| 2026-02-28 | `1a77f36` | feat | Implement M6.5 graphics/audio presentation scaffolding |
| 2026-02-28 | `27ab411` | feat | Implement M6 frontend replay/play/analytics UI |
| 2026-02-28 | `98149f2` | feat | Complete M5 metrics and play API endpoints |
| 2026-02-28 | `18ae6b0` | chore | Configure shareable trainer base image tag |
| 2026-02-28 | `e1fe165` | feat | Add M5 run and replay API endpoints |
| 2026-02-28 | `452754c` | feat | Add policy-driven PPO replay checkpoint eval |
| 2026-02-28 | `b8a1880` | feat | Begin M4 with eval runner, replay schema/index, and replay tests |
| 2026-02-28 | `4b3e684` | feat | Add dockerized pufferlib ppo training backend |
| 2026-02-28 | `6dd1f89` | feat | Persist live M3 run metadata contract |
| 2026-02-28 | `cb2efbf` | fix | Align reference RNG with native core for parity |
| 2026-02-28 | `1cd2dfc` | feat | Parity harness with mismatch bundle output |
| 2026-02-28 | `5aa1058` | feat | Native core full step semantics and parity metric surface |
| 2026-02-27 | `25c87e7` | docs | Public GitHub README and MIT license |
| 2026-02-27 | `87c2fc0` | feat | Native build runner and ctypes wrapper |
| 2026-02-27 | `287a79f` | feat | Native core scaffold with RNG and API |
| 2026-02-27 | `b4e31bd` | test | M1 determinism, reward, observation contract hardening |
| 2026-02-27 | `09f4852` | docs | M1 status and changelog updates |
| 2026-02-27 | `6e33091` | test | M1 tier1 and tier2 coverage |
| 2026-02-27 | `b009f53` | feat | M1 Python reference environment baseline |
| 2026-02-27 | `1be752b` | docs | Commit-per-change agent instruction |
| 2026-02-27 | `001a004` | feat | M0 scaffold and hello env contract stub |

## Update policy (required)

- Every commit must update at least one tracking file:
  - `docs/PROJECT_STATUS.md`, or
  - `docs/DECISION_LOG.md`, or
  - `CHANGELOG.md`.
- Any non-trivial technical decision must include a new entry in `docs/DECISION_LOG.md`.
- Keep the "Next work (ordered)" section current so remaining direction is explicit.
