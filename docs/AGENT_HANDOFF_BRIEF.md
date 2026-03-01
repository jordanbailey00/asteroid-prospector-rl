# Asteroid Belt Prospector - Agent Handoff Brief

## Purpose

This brief is a quick orientation for any coding agent starting work in this repo.
It explains what is already built, what remains, and which documents are authoritative.

## Current snapshot (2026-03-01)

- Frozen RL interface remains unchanged: `OBS_DIM=260`, `N_ACTIONS=69`, action IDs `0..68`.
- Completed milestones: `M0`, `M1`, `M2`, `M2.5`, `M3`, `M4`, `M5`, `M6`, `M6.5`, `M8`.
- In progress milestone: `M9` (throughput program + W&B-backed analytics + Vercel deployment alignment).
- Remaining milestone not complete: `M7` (baseline bots + benchmark protocol automation).

## Canonical milestone map

- `M0`: Scaffold and CI/bootstrap
- `M1`: Python reference env
- `M2`: Native core + bindings
- `M2.5`: Parity harness
- `M3`: Training + window metrics + W&B logging
- `M4`: Eval runner + replay generation
- `M5`: Backend API
- `M6`: Frontend integration
- `M6.5`: Graphics/audio integration
- `M7`: Baselines + benchmarking
- `M8`: Performance/stability hardening
- `M9`: Throughput + W&B analytics dashboard + Vercel alignment

Use `docs/BUILD_CHECKLIST.md` as the execution sequence source of truth.

## Active priorities

1. Throughput: maximize training speed with native-first runtime path and repeatable profiling evidence.
2. W&B analytics integration: backend proxy endpoints plus frontend iteration views.
3. Deployment alignment: Vercel frontend with websocket-capable backend hosting.

## Required shared identifiers

- `run_id`: unique training run identifier.
- `window_env_steps`: step budget per metrics/checkpoint/eval window.
- `window_id`: monotonic window index in a run.
- `replay_id`: unique replay artifact ID.
- Replay tags:
  - `every_window`
  - `best_so_far`
  - `milestone:*`

## System boundaries

- `engine_core/`: authoritative simulation in C.
- `python/`: wrappers and reference env for parity/debug.
- `training/`: training loop, checkpointing, eval/replay generation.
- `server/`: API endpoints (runs/metrics/replays/play sessions, WS replay stream).
- `frontend/`: replay/play/analytics UI only.

Do not move authoritative game logic into frontend code.

## Docs to read first

1. `docs/DOCS_INDEX.md`
2. `docs/RL spec/01_core_constants.md` through `06_gym_puffer_compat.md`
3. `docs/BUILD_CHECKLIST.md`
4. `docs/PROJECT_STATUS.md`
5. `docs/DECISION_LOG.md`
6. `docs/AGENT_HYGIENE_GUARDRAILS.md`

If documents conflict, frozen RL interface docs win.

## Tracking artifacts (must stay current)

- `docs/PROJECT_STATUS.md`: current state, ordered next work, risks.
- `docs/DECISION_LOG.md`: ADR decisions and consequences.
- `CHANGELOG.md`: user-visible change history.

Repository policy requires each commit to update at least one of the tracking artifacts above.
