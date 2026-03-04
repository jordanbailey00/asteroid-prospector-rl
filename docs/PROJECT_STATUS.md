# Project Status

Last updated: 2026-03-04
Current focus: M7 execution (W&B benchmark logging + lineage) with M9 release evidence upkeep

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
- Replay websocket hardening now includes a defensive internal-error envelope, prelude flush yield, and explicit success-path close handshake in the backend route, plus bounded retry support in deployment smoke checks (`--ws-check-attempts`, default `3`).
- Railway production redeploy of websocket close-handshake hardening is complete, and strict smoke now passes reliably at the default websocket retry budget (`3`) based on a 12/12 production rerun loop.
- Frontend routes are live for replay (`/`), play (`/play`), and analytics (`/analytics`).
- Public Replay/Play UX is now viewport-first with compact right-side HUD rails, grouped pilot actions, and collapsed advanced controls aligned to M9.4 observer/player goals.
- Operator training management is now aligned to PufferLib-native tooling (trainer CLI + terminal dashboard output), with W&B as the persisted analytics/artifact system and optional Constellation visibility when configured.
- M7.1 baseline bots are now implemented with a reproducible seeded runner (`training/baseline_bots.py`, `tools/run_baseline_bots.py`) and execution evidence in `docs/M7_BASELINE_BOTS_EXECUTION_20260304.md`.
- M7.2 benchmark protocol automation is now implemented in `tools/run_m7_benchmark_protocol.py` with regression coverage in `tests/test_run_m7_benchmark_protocol.py` and execution evidence in `docs/M7_BENCHMARK_PROTOCOL_EXECUTION_20260304.md`.
- Backend W&B proxy endpoints are now available for iteration analytics:
  - `GET /api/wandb/runs/latest`
  - `GET /api/wandb/runs/{wandb_run_id}/summary`
  - `GET /api/wandb/runs/{wandb_run_id}/history`
  - `GET /api/wandb/status`
  - `GET /api/wandb/runs/{wandb_run_id}/iteration-view`
- W&B diagnostics endpoint now exposes proxy availability + cache telemetry for ops tuning (`ttl_seconds`, hits/misses/expired/sets, `hit_rate`) plus per-operation telemetry (`calls`, `errors`, `latency_ms_avg`, `latency_ms_total`).
- Run-scoped analytics completeness endpoint is now available for public dashboard contract checks:
  - `GET /api/runs/{run_id}/analytics/completeness`
- Analytics UI now includes W&B-backed last-10 iteration drilldown, KPI snapshot cards, trend sparklines, and an explicit completeness contract surface (coverage table + lineage + stale/missing/error signaling).
- Python quality gates now cover `python/`, `training/`, `replay/`, `server/`, `tests/`, and `tools/` in local checks, pre-commit, and CI.
- Deployment runbook and smoke tooling are now in-repo:
  - `docs/M9_DEPLOYMENT_RUNBOOK.md`
  - `tools/smoke_m9_deployment.py`
  - `.github/workflows/m9-deployment-smoke.yml` supports strict W&B status gating (`require_clean_wandb_status`) with W&B run-detail coverage (`latest`, `summary`, `history`, `iteration-view`) and post-operation status checks.
- Deployment execution templates are now staged for M9.3 release operations:
  - `server/.env.production.example`
  - `frontend/.env.production.example`
  - `docs/M9_DEPLOYMENT_EXECUTION_CHECKLIST.md`
- Live split-host deployment is now active for M9.3 dry-run validation:
  - backend: `https://abp-backend-production.up.railway.app` (`wss://abp-backend-production.up.railway.app`)
  - frontend: `https://frontend-nine-sandy-47.vercel.app`
  - evidence: `docs/M9_DEPLOYMENT_EVIDENCE_20260303.md` + `artifacts/deploy/m9-smoke-live-20260303.json`
- Production backend run root is now seeded with a validated smoke bundle at `runs/ws-profile-smoke`, enabling strict smoke runs without `--allow-empty-runs`.
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
| M7 - Baselines + benchmarking | In Progress | M7.1 baseline bots and M7.2 protocol automation are complete; M7.3 W&B benchmark logging remains |
| M8 - Performance + stability hardening | Complete | Replay transport tuning, benchmark/stability runners, native batch runtime path |
| M9 - Throughput + W&B dashboard + Vercel alignment | In Progress | Throughput matrix/floor artifacts and W&B proxy are in place, public UX and operator-tooling alignment are complete, and release evidence upkeep remains part of ongoing operations |

## Latest recorded validation health (2026-03-04)

- `python -m pytest -q` -> 121 passed, 2 skipped.
- `python -m pytest -q tests/test_native_core_wrapper.py tests/test_puffer_backend_env_impl.py` -> 17 passed.
- `python -m pytest -q tests/test_server_api.py tests/test_smoke_m9_deployment.py` -> 34 passed.
- `python -m pytest -q tests/test_baseline_bots.py tests/test_run_baseline_bots.py` -> 7 passed.
- `python -m pytest -q tests/test_run_m7_benchmark_protocol.py` -> 3 passed.
- `python tools/smoke_m9_deployment.py --backend-http-base https://abp-backend-production.up.railway.app --backend-ws-base wss://abp-backend-production.up.railway.app --frontend-base https://frontend-nine-sandy-47.vercel.app --require-clean-wandb-status --output-path artifacts/deploy/m9-smoke-strict-20260303-post-wandb-attempt1.json` -> pass (13/13) after backend W&B key + scope activation and production run-root seeding.
- `python tools/smoke_m9_deployment.py --backend-http-base https://abp-backend-production.up.railway.app --backend-ws-base wss://abp-backend-production.up.railway.app --frontend-base https://frontend-nine-sandy-47.vercel.app --require-clean-wandb-status --output-path artifacts/deploy/m9-smoke-strict-20260304-post-ws-close-run1.json` -> pass (13/13) after Railway deploy of websocket close-handshake hardening.
- `for i in 1..12: python tools/smoke_m9_deployment.py ... --require-clean-wandb-status --output-path artifacts/deploy/m9-smoke-strict-20260304-post-ws-close-run{i}.json` -> pass 12/12 at default websocket retries (`3`).

- `python tools/run_baseline_bots.py --episodes 3 --base-seed 7 --env-time-max 4000 --max-steps-per-episode 12000 --run-id m7-baseline-smoke --output-path artifacts/benchmarks/m7-baseline-smoke.json` -> pass (3 bots; non-zero net-profit summaries emitted).
- `python tools/run_m7_benchmark_protocol.py --trainer-backend random --seed-matrix 7,9 --episodes-per-seed 2 --trainer-total-env-steps 80 --trainer-window-env-steps 40 --env-time-max 4000 --max-steps-per-episode 12000 --run-id m7-protocol-smoke --output-path artifacts/benchmarks/m7-protocol-smoke.json` -> pass (seeded contender comparison report emitted).

- `npm --prefix frontend run lint` -> pass.
- `npm --prefix frontend run build` -> pass (`/`, `/play`, `/analytics`).
- `python tools/run_parity.py --seeds 2 --steps 512 --native-library engine_core/build/abp_core.dll` -> 12/12 cases passed.
- Linux PPO matrix run published with best observed candidate near `1210.8` steps/sec and recommended floor near `1089.7`.

## Next work (ordered)

1. Log benchmark summaries to W&B as benchmark/eval jobs with reproducible lineage (`M7.3`).
2. Keep deployment evidence current per release cut:
   - `docs/M9_DEPLOYMENT_EVIDENCE_20260303.md`
   - local smoke artifact under `artifacts/deploy/`
   - manual CI run artifact from `.github/workflows/m9-deployment-smoke.yml`

## Active risks and blockers

- 100,000 steps/sec remains aspirational; current measured trainer throughput is far below target and requires further bottleneck reduction.
- W&B backend proxy now authenticates with production credentials and passes strict smoke checks under default scope.
- Replay websocket transport recovered after close-handshake hardening, but should be revalidated during each release cut to catch infrastructure regressions early.
- Split frontend/backend hosting (Vercel + external API) is now live, but remains sensitive to env/CORS drift across redeploys.
- M7.3 W&B benchmark logging and artifact lineage remain the functional gap for comparative performance reporting.

## Decision pointers

- See `docs/DECISION_LOG.md` for accepted architecture/process decisions and milestone taxonomy decisions.

## Commit ledger (latest first)

| Date | Commit | Type | Summary |
| --- | --- | --- | --- |
| 2026-03-03 | `394e7af` | feat | Harden replay websocket smoke resilience |
| 2026-03-03 | `b0dc2d5` | docs | Record live M9 deployment evidence and update remaining release-gate work |
| 2026-03-03 | `ceddf8a` | fix | Unblock M9 deployment smoke serialization, add Railway backend bootstrap, and upgrade frontend Next.js for Vercel security gates |
| 2026-03-02 | `8d1e7d4` | docs | Align MVP roadmap and acceptance references with current repo state |
| 2026-03-02 | `d394e3a` | feat | Expand M9 smoke checks across W&B analytics routes |
| 2026-03-02 | `eb5bb73` | feat | Add W&B operation telemetry diagnostics to status endpoint |
| 2026-03-02 | `6816b10` | feat | Gate M9 deployment smoke checks on W&B status endpoint |
| 2026-03-02 | `140eadf` | feat | Add W&B proxy diagnostics endpoint with cache telemetry and ops guidance |
| 2026-03-02 | `4bf31ac` | feat | Add manual GitHub Actions workflow for M9 deployment smoke checks |
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
