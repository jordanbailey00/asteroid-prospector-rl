# Project Status

Last updated: 2026-03-01
Current focus: Performance-first runtime optimization (game bottleneck) for maximum training throughput

## Current state

- Frozen interface status: unchanged (`OBS_DIM=260`, `N_ACTIONS=69`, action indexing `0..68`, reward definition unchanged).
- Replay transport now supports both HTTP and websocket paths:
  - HTTP pagination: `GET /api/runs/{run_id}/replays/{replay_id}/frames`
  - WS chunk stream: `WS /ws/runs/{run_id}/replays/{replay_id}/frames`
  - WS tuning controls: `batch_size`, `max_chunk_bytes`, `yield_every_batches`
  - Frontend replay UI exposes transport selection (`HTTP /frames` vs `WebSocket stream`).
- M7 benchmark/stability/profiling automation remains active:
  - benchmark harness: `tools/bench_m7.py` (+ thresholds and non-zero exit on regression)
  - replay long-run stability job: `tools/stability_replay_long_run.py`
  - websocket transport profiler: `tools/profile_ws_replay_transport.py`
  - nightly regression workflow: `.github/workflows/m7-nightly-regression.yml`
- P1 throughput step is implemented in trainer config/runtime:
  - PPO env selection supports `ppo_env_impl` = `reference|native|auto` (default `auto`).
  - `auto` mode probes native availability and falls back to reference when native core is unavailable.
  - PPO summary metadata records requested vs selected env implementation.
- P2 throughput step is implemented in trainer runtime:
  - PPO loop supports a batch callback boundary (`on_step_batch`) per vector step.
  - Trainer window aggregation now uses `record_step_batch(...)` for PPO runs.
  - Per-env callback invocations from PPO runtime to trainer were removed from the hot path.
- P3 throughput step is now implemented in native bridge:
  - C API exposes `abp_core_reset_many(...)` and `abp_core_step_many(...)` for batched handle processing.
  - Python wrapper exposes `NativeProspectorCore.reset_many(...)` and `NativeProspectorCore.step_many(...)`.
  - Bridge keeps scalar fallback when batch symbols are unavailable or cores are mixed.
- Validation health (2026-03-01):
  - `python -m pytest -q tests/test_server_api.py` -> 9 passed.
  - `python -m pytest -q tests/test_bench_m7.py` -> 2 passed.
  - `python -m pytest -q tests/test_stability_replay_long_run.py` -> 1 passed.
  - `python -m pytest -q tests/test_profile_ws_replay_transport.py` -> 1 passed.
  - `python -m pytest -q` -> 77 passed, 2 skipped.
  - `npm --prefix frontend run lint` -> no ESLint warnings/errors.
  - `npm --prefix frontend run build` -> success for `/`, `/play`, `/analytics`.
- M6.5 manual replay/play checklist remains captured with deterministic evidence:
  - checklist report: `docs/M65_MANUAL_VERIFICATION.md`
  - sampled replay trace: `docs/verification/m65_sample_replay.jsonl`
  - reproducible generator: `tools/run_m65_manual_checklist.py`
- Trainer runtime baseline remains:
  - dependency pin: `pufferlib-core==3.0.17`
  - module runtime: `pufferlib 3.0.3`
  - image tag: `jordanbailey00/rl-puffer-base:py311-puffercore3.0.17`
  - published digest: `sha256:723c58843d9ed563fa66c0927da975bdbab5355c913ec965dbea25a2af67bb71`
- Completed milestones: M0, M1, M2, M2.5, M3, M4, M5, M6, M6.5, and M7+ performance/stability hardening.
- Priority execution plan documents:
  - `docs/PRIORITY_PLAN_100K_WANDB_VERCEL.md`
  - `docs/PERFORMANCE_BOTTLENECK_PLAN.md`

## Milestone board

| Milestone | Status | What is done | Evidence |
| --- | --- | --- | --- |
| M0 - Scaffold and hello env | Complete | Repo skeleton, tooling, CI, contract stub env | `001a004` |
| M1 - Python reference env | Complete | Reference environment, tier tests, determinism and reward checks | `b009f53`, `6e33091`, `b4e31bd` |
| M2 - Native core + bindings | Complete | Full C world/step dynamics, reward + obs packing, expanded native parity metrics surfaced through ctypes | `5aa1058` |
| M2.5 - Parity harness | Complete | Fixed-suite parity harness, mismatch bundles, and deterministic RNG alignment across Python/C | `1cd2dfc`, `cb2efbf` |
| M3 - Training + window metrics | Complete | Windowed trainer loop, checkpoint cadence, `windows.jsonl`, optional W&B logging, live `run_metadata.json`, and Dockerized Linux PPO backend validation | `1a90101`, `6dd1f89`, `4b3e684`, `dda8545` |
| M4 - Eval + replay generation | Complete | Policy-driven PPO eval replays from serialized checkpoints, replay schema/index validation, `every_window` + `best_so_far` + `milestone:*` tagging, replay index filtering helpers, and checkpoint format tests | `b8a1880`, `452754c` |
| M5 - API server | Complete | FastAPI run/replay/metrics endpoints, HTTP replay frame pagination, websocket replay frame streaming, in-memory play-session lifecycle endpoints, and CORS configuration with endpoint tests | `e1fe165`, `98149f2`, `81a8bad` |
| M6 - Frontend integration | Complete | Next.js replay page (`/`), human play mode (`/play`), analytics page (`/analytics`) wired to M5 APIs with playback controls, run/window/replay selection, and historical trend visualizations | `27ab411` |
| M6.5 - Graphics + audio integration | Complete | Real Kenney assets wired to replay/play rendering, manifests file-backed, semantic asset tests passing, and final manual replay/play checklist evidence captured | `1a77f36`, `f606846`, `b7efc22` |
| M7+ - Perf and stability hardening | Complete | HTTP+WS replay transport, websocket chunk tuning, benchmark harness, replay stability soak, websocket profiling sweep, and nightly regression workflow gates | `81a8bad`, `d537be3`, `6404834`, pending (this commit) |

## Next work (ordered)

1. Run native-core hot-path optimization pass (obs packing + critical update loops) with deterministic parity checks.
2. Execute throughput tuning matrix after native-path changes and publish updated baseline/floor artifacts.
3. Reassess enforcement threshold: keep 100,000 as aspirational target, calibrate stable floor if still unattainable.
4. Integrate native batch bridge into trainer hot path (replace per-core scalar native calls with `step_many/reset_many`).
5. Implement backend W&B proxy endpoints (runs, history, summary, iteration views) with cache/TTL behavior.
6. Extend frontend analytics UI to show:
   - current selected iteration metrics
   - full historical trends across all prior iterations
   - last-10 iteration dropdown drilldown
   - quick KPI snapshot cards
7. Complete production deployment path:
   - frontend on Vercel
   - backend on websocket-capable host
   - production CORS/env/secret wiring
8. Implement baseline bots (`greedy miner`, `cautious scanner`, `market timer`) and add reproducible CLI runs.
9. Automate PPO-vs-baseline benchmark protocol across seeds and aggregate summary metrics.
10. Publish benchmark summaries to W&B as eval job artifacts and expose them in run metadata/API.

## Active risks and blockers

- Replay UI still buffers the full selected replay in client memory after load; very large artifacts may still need incremental playback virtualization.
- Nightly threshold values may need calibration over time as CI runner performance characteristics drift.
- There is no published `pufferlib 4.0` package on PyPI as of 2026-03-01; latest published line used here is `pufferlib-core 3.0.17`.
- 100,000 steps/sec may be above current hardware/runtime ceiling; native-core hot-path integration and profiling evidence are needed before locking hard gates.
- W&B API rate limits/latency can degrade dashboard responsiveness without backend caching and bounded history queries.
- Split frontend/backend hosting (Vercel + external API) can fail due to CORS/WS misconfiguration if not validated with deployment smoke checks.
- Batch bridge is implemented but not yet wired into trainer vector stepping path; scalar native core calls still limit maximal throughput gains.

## Decision pointers

- See `docs/DECISION_LOG.md` for accepted architecture and process decisions.

## Commit ledger (latest first)

| Date | Commit | Type | Summary |
| --- | --- | --- | --- |
| 2026-03-01 | pending (this commit) | feat | Add native `step_many/reset_many` C APIs and Python bridge methods with fallback support |
| 2026-03-01 | pending (prior commit) | feat | Complete M7+ with websocket tuning, transport profiling, and nightly regression gates |
| 2026-03-01 | `6404834` | feat | Add long-run replay stability job for index consistency and leak/regression detection |
| 2026-03-01 | `d537be3` | feat | Add M7 benchmark harness for trainer throughput, replay API latency, and memory soak checks |
| 2026-03-01 | `81a8bad` | feat | Add websocket replay frame streaming endpoint and frontend HTTP/WS transport switch |
| 2026-03-01 | `b7efc22` | feat | Add reproducible M6.5 manual replay/play checklist runner and evidence artifacts |
| 2026-02-28 | `55852e2` | docs | Refresh day-end status and handoff documentation |
| 2026-02-28 | `dda8545` | chore | Upgrade trainer deps/image to latest published Puffer core stack and patch PPO runtime compatibility |
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
