# Project Status

Last updated: 2026-02-28
Current focus: M3 training stack start after parity gating

## Current state

- Frozen interface status: unchanged (`OBS_DIM=260`, `N_ACTIONS=69`, action indexing `0..68`, reward definition unchanged).
- Build health: local checks passing via `tools/run_checks.ps1` (`pytest -q` currently 24 passed).
- Completed milestones: M0, M1, M2, and M2.5 implementation complete.
- Active milestone: M3 training + window metrics.

## Milestone board

| Milestone | Status | What is done | Evidence |
| --- | --- | --- | --- |
| M0 - Scaffold and hello env | Complete | Repo skeleton, tooling, CI, contract stub env | `001a004` |
| M1 - Python reference env | Complete | Reference environment, tier tests, determinism and reward checks | `b009f53`, `6e33091`, `b4e31bd` |
| M2 - Native core + bindings | Complete | Full C world/step dynamics, reward + obs packing, expanded native parity metrics surfaced through ctypes | `5aa1058` |
| M2.5 - Parity harness | Complete | `tools/run_parity.py` fixed-suite harness with seed matrix, tolerance compares, mismatch bundles; sampled suites now pass after RNG alignment | `1cd2dfc` |
| M3 - Training + window metrics | Not started | Puffer training loop and W&B window logging | pending |
| M4 - Eval + replay generation | Not started | Eval runner, replay recorder, replay indexing | pending |
| M5 - API server | Not started | Run/replay/play endpoints | pending |
| M6 - Frontend integration | Not started | Replay player, play mode, analytics | pending |
| M7+ - Perf and stability | Not started | Throughput targets, soak checks, benchmark automation | pending |

## Next work (ordered)

1. Run the full parity matrix (all configured seeds/suites/time budgets) and archive a passing report.
2. Start M3 training pipeline with windowed metrics and W&B logging.
3. Add replay generation/indexing for M4 and expose via API for frontend consumption.
4. Implement backend/frontend play loop and analytics surfaces for MVP completion.

## Active risks and blockers

- Main risk: remaining untested parity matrix combinations could still surface edge-case drift.
- Main blocker for MVP: training/eval/replay/API/frontend milestones (M3+) are not yet implemented.

## Decision pointers

- See `docs/DECISION_LOG.md` for accepted architecture and process decisions.

## Commit ledger (latest first)

| Date | Commit | Type | Summary |
| --- | --- | --- | --- |
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
