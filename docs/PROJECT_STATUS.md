# Project Status

Last updated: 2026-02-28
Current focus: M4 eval runner + replay generation

## Current state

- Frozen interface status: unchanged (`OBS_DIM=260`, `N_ACTIONS=69`, action indexing `0..68`, reward definition unchanged).
- Build health: local checks passing via `tools/run_checks.ps1` (`pytest -q` currently 37 passed).
- Completed milestones: M0, M1, M2, M2.5, and M3 complete.
- Active milestone: M4 eval + replay generation (in progress).

## Milestone board

| Milestone | Status | What is done | Evidence |
| --- | --- | --- | --- |
| M0 - Scaffold and hello env | Complete | Repo skeleton, tooling, CI, contract stub env | `001a004` |
| M1 - Python reference env | Complete | Reference environment, tier tests, determinism and reward checks | `b009f53`, `6e33091`, `b4e31bd` |
| M2 - Native core + bindings | Complete | Full C world/step dynamics, reward + obs packing, expanded native parity metrics surfaced through ctypes | `5aa1058` |
| M2.5 - Parity harness | Complete | Fixed-suite parity harness, mismatch bundles, and deterministic RNG alignment across Python/C | `1cd2dfc`, `cb2efbf` |
| M3 - Training + window metrics | Complete | Windowed trainer loop, checkpoint cadence, `windows.jsonl`, optional W&B logging, live `run_metadata.json`, and Dockerized Linux `puffer_ppo` backend with PufferLib PPO smoke validation (3 windows) | `1a90101`, `6dd1f89`, `4b3e684` |
| M4 - Eval + replay generation | In progress | Eval runner integrated in trainer, replay schema/index modules, per-window replay generation, `every_window` + `best_so_far` tagging, replay metadata wiring, replay artifact logging hook, and replay schema/integration tests | pending (this commit) |
| M5 - API server | Not started | Run/replay/play endpoints | pending |
| M6 - Frontend integration | Not started | Replay player, play mode, analytics | pending |
| M7+ - Perf and stability | Not started | Throughput targets, soak checks, benchmark automation | pending |

## Next work (ordered)

1. Complete remaining M4 gaps: policy-driven eval from real PPO checkpoints and optional milestone replay tags.
2. Start M5 API server for run/metrics/replay/play-session endpoints.
3. Begin M6 frontend integration against live run/replay/metrics endpoints.
4. Add M7 profiling pass for trainer throughput and memory stability.

## Active risks and blockers

- Main risk: Docker trainer build is heavy on first run because `pufferlib==2.0.6` installs from source and torch pulls large binaries.
- Main blocker for MVP: M5-M6 are not implemented yet; M4 still needs policy-driven eval for PPO checkpoints.

## Decision pointers

- See `docs/DECISION_LOG.md` for accepted architecture and process decisions.

## Commit ledger (latest first)

| Date | Commit | Type | Summary |
| --- | --- | --- | --- |
| 2026-02-28 | pending (this commit) | feat | Begin M4 with eval runner, replay schema/index, and replay tests |
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
