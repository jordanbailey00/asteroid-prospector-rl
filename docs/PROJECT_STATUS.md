# Project Status

Last updated: 2026-03-01
Current focus: M9 execution (throughput evidence, W&B-backed analytics integration, Vercel deployment alignment)

## Current state

- Frozen interface unchanged: `OBS_DIM=260`, `N_ACTIONS=69`, action indexing `0..68`.
- Native runtime path for PPO now supports:
  - env implementation selection (`reference|native|auto`),
  - native load-aware probing in `auto` mode,
  - in-process native batched stepping path (`step_many/reset_many`).
- Throughput tooling now includes:
  - profiler (`tools/profile_training_throughput.py`),
  - matrix runner (`tools/run_throughput_matrix.py`),
  - floor gate (`tools/gate_throughput_floors.py`),
  - Linux PPO matrix artifact (`artifacts/throughput/throughput-matrix-ppo-20260301-m9p2d.json`).
- Replay transport supports both:
  - HTTP frame pagination (`GET /api/runs/{run_id}/replays/{replay_id}/frames`),
  - websocket chunk stream (`WS /ws/runs/{run_id}/replays/{replay_id}/frames`).
- Frontend routes are live for replay (`/`), play (`/play`), and analytics (`/analytics`).
- M6.5 manual verification artifacts remain captured:
  - `docs/M65_MANUAL_VERIFICATION.md`
  - `docs/verification/m65_sample_replay.jsonl`

## Milestone board

| Milestone | Status | Summary |
| --- | --- | --- |
| M0 - Scaffold and hello env | Complete | Repo structure, baseline tooling, contract stub env |
| M1 - Python reference env | Complete | Deterministic reference implementation with contract tests |
| M2 - Native core + bindings | Complete | C simulation core with Python bridge |
| M2.5 - Parity harness | Complete | Deterministic parity runner and mismatch artifacts |
| M3 - Training + window metrics | Complete | Windowed training loop, checkpoints, metrics, optional W&B logging |
| M4 - Eval + replay generation | Complete | Policy-driven eval replays and replay indexing |
| M5 - API server | Complete | Run/metrics/replay/play-session API surface |
| M6 - Frontend integration | Complete | Replay/play/analytics pages wired to backend APIs |
| M6.5 - Graphics + audio | Complete | File-backed Kenney asset wiring plus validation checks |
| M7 - Baselines + benchmarking | Pending | Baseline bots and benchmark protocol automation are not complete yet |
| M8 - Performance + stability hardening | Complete | Replay transport tuning, benchmark/stability runners, native batch runtime path |
| M9 - Throughput + W&B dashboard + Vercel alignment | In Progress | Throughput matrix/floor artifacts are in place; W&B proxy + analytics/deployment work remains |

## Latest recorded validation health (2026-03-01)

- `python -m pytest -q` -> 80 passed, 2 skipped.
- `python -m pytest -q tests/test_native_core_wrapper.py tests/test_puffer_backend_env_impl.py` -> 17 passed.
- `npm --prefix frontend run lint` -> pass.
- `npm --prefix frontend run build` -> pass (`/`, `/play`, `/analytics`).
- `python tools/run_parity.py --seeds 2 --steps 512 --native-library engine_core/build/abp_core.dll` -> 12/12 cases passed.
- Linux PPO matrix run published with best observed candidate near `1210.8` steps/sec and recommended floor near `1089.7`.

## Next work (ordered)

1. Implement backend W&B proxy endpoints (runs, summary, history, iteration-scoped views) with bounded queries and cache TTL.
2. Extend frontend analytics UI to show:
   - selected iteration metrics,
   - full historical trend,
   - last-10 iteration dropdown drilldown,
   - KPI snapshot cards.
3. Complete deployment path:
   - frontend on Vercel,
   - backend on websocket-capable host,
   - production CORS/env/secret wiring.
4. Implement baseline bots (`greedy miner`, `cautious scanner`, `market timer`) and reproducible CLI runs.
5. Automate PPO-vs-baseline benchmark protocol across seeds and publish summary artifacts.

## Active risks and blockers

- 100,000 steps/sec remains aspirational; current measured trainer throughput is far below target and requires further bottleneck reduction.
- W&B API latency/rate limits can degrade analytics UX without backend caching and bounded history windows.
- Split frontend/backend hosting (Vercel + external API) can fail from CORS/WS configuration drift.
- M7 baseline bots/benchmark automation remains a functional gap for comparative performance reporting.

## Decision pointers

- See `docs/DECISION_LOG.md` for accepted architecture/process decisions and milestone taxonomy decisions.

## Commit ledger (latest first)

| Date | Commit | Type | Summary |
| --- | --- | --- | --- |
| 2026-03-01 | `9dbdedc` | feat | Publish Linux PPO throughput matrix and harden native auto probe behavior |
| 2026-03-01 | `c1fef2a` | feat | Add matrix-driven throughput floor gate |
| 2026-03-01 | `f96e3b5` | docs | Standardize build checklist milestone identifiers |
| 2026-03-01 | `9ff3d1c` | feat | Add throughput matrix runner and floor calibration |
| 2026-03-01 | `bca65c4` | feat | Optimize native core hot loop and throughput profiling path |
| 2026-02-28 | `84cb3db` | feat | Route native PPO runtime through batched step/reset bridge |
| 2026-02-28 | `d3328eb` | feat | Add native batch step/reset APIs and Python bridge support |
| 2026-02-28 | `bf492c3` | feat | Batch PPO callback path to reduce hot-loop overhead |
| 2026-02-28 | `1023729` | feat | Add PPO env implementation selection with native auto fallback |
| 2026-02-28 | `6714d56` | feat | Add throughput profiler with target/floor gate capability |
| 2026-02-28 | `855c1a3` | docs | Add priority plan for 100k throughput + W&B + Vercel |
| 2026-02-28 | `a433997` | feat | Complete transport tuning and nightly regression gates |
| 2026-02-28 | `6404834` | feat | Add long-run replay stability job |
| 2026-02-28 | `d537be3` | feat | Add benchmark harness |
| 2026-02-28 | `81a8bad` | feat | Add websocket replay frame streaming transport |
| 2026-02-28 | `b7efc22` | feat | Add reproducible M6.5 manual replay/play verification runner |

## Update policy (required)

- Every commit must update at least one tracking file:
  - `docs/PROJECT_STATUS.md`, or
  - `docs/DECISION_LOG.md`, or
  - `CHANGELOG.md`.
- Any non-trivial technical decision must include a same-commit ADR entry in `docs/DECISION_LOG.md`.
- Keep the "Next work (ordered)" section current so remaining direction is explicit.
