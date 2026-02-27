# Project Status

Last updated: 2026-02-27
Current focus: M2 native core parity path

## Current state

- Frozen interface status: unchanged (`OBS_DIM=260`, `N_ACTIONS=69`, action indexing `0..68`, reward definition unchanged).
- Build health: local checks passing via `tools/run_checks.ps1` (`pytest -q` currently 24 passed).
- Completed milestones: M0 and M1 complete, M2 scaffold complete.
- Active milestone: M2 implementation and M2.5 parity harness.

## Milestone board

| Milestone | Status | What is done | Evidence |
| --- | --- | --- | --- |
| M0 - Scaffold and hello env | Complete | Repo skeleton, tooling, CI, contract stub env | `001a004` |
| M1 - Python reference env | Complete | Reference environment, tier tests, determinism and reward checks | `b009f53`, `6e33091`, `b4e31bd` |
| M2 - Native core + bindings | In progress | Native API scaffold, deterministic RNG, build script, ctypes wrapper smoke path | `287a79f`, `87c2fc0` |
| M2.5 - Parity harness | Not started | C/Python trace parity runner and mismatch bundles | pending |
| M3 - Training + window metrics | Not started | Puffer training loop and W&B window logging | pending |
| M4 - Eval + replay generation | Not started | Eval runner, replay recorder, replay indexing | pending |
| M5 - API server | Not started | Run/replay/play endpoints | pending |
| M6 - Frontend integration | Not started | Replay player, play mode, analytics | pending |
| M7+ - Perf and stability | Not started | Throughput targets, soak checks, benchmark automation | pending |

## Next work (ordered)

1. Complete M2 core step semantics in C to match the Python reference environment.
2. Implement M2.5 parity harness (`tools/run_parity.py`) with fixed seed and fixed action suite comparisons.
3. Gate training start on parity pass criteria from `docs/ACCEPTANCE_TESTS_PARITY_HARNESS.md`.
4. Start M3 windowed trainer and W&B artifact logging after parity is stable.

## Active risks and blockers

- Main risk: semantic drift between Python reference and C core before parity harness is in place.
- Main blocker for training scale-up: parity suite is not yet implemented.

## Decision pointers

- See `docs/DECISION_LOG.md` for accepted architecture and process decisions.

## Commit ledger (latest first)

| Date | Commit | Type | Summary |
| --- | --- | --- | --- |
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
