# Decision Log

Use this file for non-trivial project decisions.

## Entry format

- ID: `ADR-XXXX`
- Date: `YYYY-MM-DD`
- Status: `Accepted`, `Superseded`, or `Proposed`
- Context
- Decision
- Consequences
- Related commits/docs

## Entries

### ADR-0001 - Freeze RL interface v1

- Date: 2026-02-27
- Status: Accepted
- Context: Training and replay compatibility require stable contracts across milestones.
- Decision: Keep `OBS_DIM=260`, `N_ACTIONS=69`, action indexing `0..68`, observation field layout, and reward definition frozen unless a versioned `v2` is explicitly created with docs/tests updates.
- Consequences: Feature work must fit inside the existing interface; incompatible changes require explicit versioning.
- Related commits/docs: `docs/RL spec/01_core_constants.md` through `docs/RL spec/06_gym_puffer_compat.md`

### ADR-0002 - Build order follows reference-first parity flow

- Date: 2026-02-27
- Status: Accepted
- Context: Native implementation without a correctness oracle increases risk of hard-to-debug divergence.
- Decision: Implement milestones in this order: M0 scaffold, M1 Python reference environment, M2 native C core, then M2.5 parity harness gating further training work.
- Consequences: Native progress is blocked from advancing to training integration until parity requirements pass.
- Related commits/docs: `001a004`, `b009f53`, `287a79f`, `docs/BUILD_CHECKLIST.md`, `docs/ACCEPTANCE_TESTS_PARITY_HARNESS.md`

### ADR-0003 - Use handle-based C API with Python ctypes wrapper for initial bridge

- Date: 2026-02-27
- Status: Accepted
- Context: Need a minimal, testable path from Python to native core early in M2.
- Decision: Expose `abp_core_create/destroy/init/reset/step` in C and bind with `ctypes` in Python first; defer heavier binding stacks until needed.
- Consequences: Fast bootstrap for parity work with low dependency cost; future binding migrations must preserve API semantics.
- Related commits/docs: `287a79f`, `87c2fc0`, `python/asteroid_prospector/native_core.py`

### ADR-0004 - Enforce commit-level project tracking updates

- Date: 2026-02-27
- Status: Accepted
- Context: Existing status information was split across docs and not guaranteed to stay synchronized with commits.
- Decision: Require each commit to update at least one tracking artifact (`docs/PROJECT_STATUS.md`, `docs/DECISION_LOG.md`, or `CHANGELOG.md`) and record significant decisions in this log.
- Consequences: Higher documentation discipline and clearer current-state visibility; minor overhead added to each commit.
- Related commits/docs: `AGENTS.md`, `.pre-commit-config.yaml`, `tools/check_project_tracking.py`

### ADR-0005 - Expose parity metrics directly from native step results

- Date: 2026-02-28
- Status: Accepted
- Context: Parity comparisons require the same per-step metrics (`credits`, `profit_per_tick`, `pirate_encounters`, etc.) from both Python and native runs, but deriving them externally risks drift.
- Decision: Expand `AbpCoreStepResult` with a fixed metric set and map it 1:1 into the `NativeProspectorCore.step()` info payload.
- Consequences: Parity harnesses can compare metrics without reconstructing internal state, and the C runner can emit traces with stable metric ordering.
- Related commits/docs: `engine_core/include/abp_core.h`, `engine_core/src/abp_core.c`, `python/asteroid_prospector/native_core.py`, `engine_core/core_test_runner.c`

### ADR-0006 - Parity harness emits first-mismatch bundles as the default debug artifact

- Date: 2026-02-28
- Status: Accepted
- Context: Cross-runtime parity failures are expensive to debug without reproducible traces and field-level mismatch context.
- Decision: Implement `tools/run_parity.py` to compare fixed seed/action suites and emit per-case mismatch bundles (`actions`, Python trace, native trace, mismatch metadata) on first failing field.
- Consequences: Faster triage loops and deterministic repro for parity bugs; artifact directories are excluded from git tracking.
- Related commits/docs: `tools/run_parity.py`, `.gitignore`, `docs/ACCEPTANCE_TESTS_PARITY_HARNESS.md`

### ADR-0007 - Align Python reference RNG with native PCG32 for deterministic parity

- Date: 2026-02-28
- Status: Accepted
- Context: Parity harness mismatches were occurring at step 0 because Python used NumPy RNG while native used PCG32, causing hidden-state divergence even when interface logic matched.
- Decision: Introduce `Pcg32Rng` in Python and route `ProspectorReferenceEnv` stochastic draws through this RNG with distribution helpers matching native core usage.
- Consequences: Seeded Python/native rollouts now share stochastic streams and parity harness convergence is practical under strict tolerances.
- Related commits/docs: `python/asteroid_prospector/pcg32_rng.py`, `python/asteroid_prospector/reference_env.py`, `tools/run_parity.py`

### ADR-0008 - Use window-first training telemetry with JSONL as canonical local sink

- Date: 2026-02-28
- Status: Accepted
- Context: M3 requires deterministic `window_env_steps` aggregation and offline-friendly telemetry before full Puffer/W&B infrastructure is complete.
- Decision: Introduce a `WindowMetricsAggregator` that emits one metrics row per closed window, persist rows to `runs/{run_id}/metrics/windows.jsonl`, and mirror rows to W&B when enabled via an adapter.
- Consequences: Window metrics are available immediately for backend/frontend development and tests; trainer backend can evolve from random-policy scaffold to PPO without changing metrics schema.
- Related commits/docs: `training/windowing.py`, `training/logging.py`, `training/train_puffer.py`, `tests/test_windowing.py`, `tests/test_wandb_offline.py`, `tests/test_training_loop.py`

### ADR-0009 - Treat `run_metadata.json` as the live run-state contract for M5/M6

- Date: 2026-02-28
- Status: Accepted
- Context: Upcoming API/frontend milestones need stable run-level pointers (`latest_window`, checkpoint location, run status, and URLs) while training is still running, not only after the process exits.
- Decision: Persist `runs/{run_id}/run_metadata.json` as a live-updated document with lifecycle status (`running`/`completed`/`failed`), progress counters, and latest pointers (`latest_window`, `latest_checkpoint`, replay placeholders, and observability URLs).
- Consequences: M5 endpoints can serve current run state without tailing metrics files directly; trainer now performs metadata writes at run start, per emitted window, and on completion/failure.
- Related commits/docs: `training/train_puffer.py`, `tests/test_training_loop.py`, `training/README.md`, `docs/PROJECT_STATUS.md`

### ADR-0010 - Standardize Linux trainer runtime via Docker compose for PufferLib PPO

- Date: 2026-02-28
- Status: Accepted
- Context: Native Windows runs are unsuitable for `pufferlib` in this project flow, and repeated ephemeral container probes caused redundant downloads/builds and unstable setup times.
- Decision: Add `infra/trainer/Dockerfile` + `infra/docker-compose.yml` as the single installation path for trainer dependencies, with BuildKit pip cache mounts and a build-time `import pufferlib` smoke test. Wire `training/train_puffer.py` `puffer_ppo` backend to a real PPO loop implemented in `training/puffer_backend.py` and run it in the Linux container runtime.
- Consequences: First build is still heavy due source build + torch dependency footprint, but subsequent builds are deterministic and cache-backed; M3 is unblocked for true PPO execution via Docker.
- Related commits/docs: `infra/trainer/Dockerfile`, `infra/trainer/requirements.txt`, `infra/docker-compose.yml`, `training/puffer_backend.py`, `training/train_puffer.py`, `training/README.md`

### ADR-0011 - Standardize M4 replay persistence on JSONL.GZ frames plus JSON index with monotonic best tagging

- Date: 2026-02-28
- Status: Accepted
- Context: M4 needs per-window replay generation that is easy to inspect, stream, and validate in tests while remaining lightweight to integrate before API/frontend milestones.
- Decision: Store each replay as `runs/{run_id}/replays/{replay_id}.jsonl.gz` with a stable frame schema (`t`, `dt`, `action`, `reward`, `terminated`, `truncated`, `render_state`, `events`, optional `info`) and maintain `runs/{run_id}/replay_index.json` as the run-scoped catalog. Tag every replay with `every_window`; additionally tag `best_so_far` when `return_total` is strictly greater than all previous entries in that run index.
- Consequences: Replay artifacts are human-readable after decompression, easy to validate with lightweight tests, and directly consumable by upcoming API endpoints; strict greater-than best tagging avoids repeated `best_so_far` labels on ties.
- Related commits/docs: `training/eval_runner.py`, `replay/schema.py`, `replay/index.py`, `tests/test_eval_runner.py`, `tests/test_replay_schema.py`, `training/README.md`, `replay/README.md`

### ADR-0012 - Serialize PPO policy state in window checkpoints and use it as the eval replay authority

- Date: 2026-02-28
- Status: Accepted
- Context: M4 required policy-driven eval replays from actual trainer checkpoints; JSON-only checkpoint payloads could not carry PPO policy parameters, and eval used random actions even for PPO windows.
- Decision: Extend checkpoint writing to support backend-specific formats: `json_v1` for random backend and `ppo_torch_v1` for PPO backend. Capture PPO model state (`policy_arch`, `obs_shape`, `n_actions`, `model_state_dict`) at each checkpoint window and load that state in eval runner to select actions from the saved policy.
- Consequences: Eval replays now reflect checkpointed PPO behavior instead of random rollouts, with a stable checkpoint contract that supports future resume/eval tooling.
- Related commits/docs: `training/train_puffer.py`, `training/puffer_backend.py`, `training/policy.py`, `training/eval_runner.py`, `tests/test_checkpoint_io.py`, `tests/test_eval_runner.py`, `training/README.md`

### ADR-0013 - Bootstrap M5 API from filesystem artifacts (`run_metadata.json` + `replay_index.json` + replay files)

- Date: 2026-02-28
- Status: Accepted
- Context: M5 needs immediate run/replay catalog APIs without introducing a separate database before replay/storage contracts settle.
- Decision: Implement initial FastAPI endpoints backed directly by `runs/{run_id}` artifacts: run catalog from `run_metadata.json`, replay catalog/detail from `replay_index.json`, and replay frame fetch by reading replay files (`jsonl.gz`). Include query-time filtering (`tag`, `tags_any`, `tags_all`, `window_id`, `limit`) and frame pagination (`offset`, `limit`).
- Consequences: API bootstrap is fast and aligned with current trainer outputs; future storage backends can preserve endpoint contracts while swapping internal persistence.
- Related commits/docs: `server/app.py`, `server/main.py`, `server/README.md`, `tests/test_server_api.py`, `replay/index.py`, `docs/PROJECT_STATUS.md`

### ADR-0014 - Make trainer compose service publishable as a reusable RL base image

- Date: 2026-02-28
- Status: Accepted
- Context: Reusing the same PufferLib/torch build across multiple RL repos requires a stable image tag and push path; compose services without an explicit `image` tag are local-only by default.
- Decision: Add explicit `image` naming to `infra/docker-compose.yml` for the `trainer` service with env override support (`TRAINER_IMAGE`), and document build/push/consume workflow in `infra/trainer/README.md`.
- Consequences: The trainer runtime can be shared via Docker Hub/registry and reused as `FROM ...` in other projects, avoiding repeated PufferLib source tarball downloads/builds across repos.
- Related commits/docs: `infra/docker-compose.yml`, `infra/trainer/README.md`

### ADR-0015 - Use in-memory process-local play sessions for initial M5 human-play endpoints

- Date: 2026-02-28
- Status: Accepted
- Context: M5 requires immediate play-session APIs, but persistent session storage adds complexity and coupling before frontend interaction patterns are validated.
- Decision: Implement `POST/DELETE` play-session lifecycle endpoints using a process-local in-memory store keyed by `session_id`, each holding a `ProspectorReferenceEnv` instance. Session state is ephemeral and non-persistent by design. Add CORS middleware defaults for localhost plus Vercel regex to make API/frontend integration workable by default.
- Consequences: M5 play mode is operational with minimal infrastructure, but sessions do not survive process restarts and are not horizontally sharable; future scaling can swap storage while preserving endpoint contracts.
- Related commits/docs: `server/app.py`, `server/main.py`, `server/README.md`, `tests/test_server_api.py`

### ADR-0016 - Implement M6 frontend as API-driven App Router UI with HTTP replay frame playback

- Date: 2026-02-28
- Status: Accepted
- Context: M6 required a production-lean frontend for replay inspection, human play sessions, and historical analytics without adding backend websocket complexity in the same milestone.
- Decision: Build `frontend/` as a Next.js App Router TypeScript app with three pages (`/`, `/play`, `/analytics`) that consume existing M5 HTTP endpoints. Use HTTP replay frame fetch (`/frames`) plus client-side playback timing controls for replay mode, and keep websocket replay streaming as a follow-up item.
- Consequences: M6 is unblocked and fully wired to current API contracts with minimal moving parts; replay UX on very large artifacts may require WS/chunked transport optimization in a later milestone.
- Related commits/docs: `frontend/app/page.tsx`, `frontend/app/play/page.tsx`, `frontend/app/analytics/page.tsx`, `frontend/components/replay-dashboard.tsx`, `frontend/components/play-console.tsx`, `frontend/components/analytics-dashboard.tsx`, `frontend/README.md`, `docs/PROJECT_STATUS.md`

### ADR-0017 - Deliver M6.5 with semantic manifests plus procedural rendering/audio fallback

- Date: 2026-02-28
- Status: Superseded
- Context: M6.5 required graphics/audio integration, but the repo does not yet carry final external art/audio packs. We still need deterministic, testable presentation wiring that can be upgraded without changing gameplay/UI logic.
- Decision: Implement a manifest-driven presentation layer now (`graphics_manifest.json`, `audio_manifest.json`, `action_effects_manifest.json`) and render sector/minimap visuals procedurally from observation state while playing cue sounds through a WebAudio synth fallback when audio files are absent. Add tests to enforce action/event mapping completeness and manifest key resolution.
- Consequences: M6.5 is unblocked with stable runtime contracts and validation gates; swapping in real atlases/OGG assets becomes a manifest update task rather than a code rewrite.
- Related commits/docs: `frontend/public/assets/manifests/graphics_manifest.json`, `frontend/public/assets/manifests/audio_manifest.json`, `frontend/lib/action_effects_manifest.json`, `frontend/lib/assets.ts`, `frontend/lib/presentation.ts`, `frontend/lib/audio.ts`, `frontend/components/sector-view.tsx`, `tests/test_frontend_presentation.py`, `docs/PROJECT_STATUS.md`

### ADR-0018 - M6.5 completion requires file-backed Kenney asset wiring for all core gameplay semantics

- Date: 2026-02-28
- Status: Accepted
- Context: M6.5 was initially interpreted as runtime presentation scaffolding, but project acceptance criteria require full semantic asset correctness using provided Kenney packs (planet assets for planets, asteroid assets for asteroids, ship assets for ships, etc.).
- Decision: Redefine M6.5 completion as requiring real file-backed mappings in frontend manifests and runtime rendering/audio playback for core gameplay semantics. Procedural placeholders may remain only as explicit non-required fallback behavior, not as the primary path for required keys.
- Consequences: M6.5 status is in-progress until all required semantic keys and cues are wired to real files in `frontend/public/assets/...` and validated by tests.
- Related commits/docs: `docs/BUILD_CHECKLIST.md`, `docs/PROJECT_STATUS.md`, `frontend/public/assets/manifests/graphics_manifest.json`, `frontend/public/assets/manifests/audio_manifest.json`, `tests/test_frontend_presentation.py`

### ADR-0019 - Upgrade trainer runtime to latest published Puffer core line and adapt PPO wiring for 3.x vector semantics

- Date: 2026-02-28
- Status: Accepted
- Context: The trainer image was pinned to `pufferlib==2.0.6`, and upstream package metadata now publishes newer 3.x builds (`pufferlib==3.0.0`, `pufferlib-core==3.0.17`) with updated vector runtime behavior. There is currently no published `pufferlib 4.0` release on PyPI.
- Decision: Move trainer dependency pins to `pufferlib-core==3.0.17`, `gymnasium==1.2.3`, `torch==2.10.0`, `wandb==0.25.0`, and `numpy==2.4.2`. Update compose default image tag to `py311-puffercore3.0.17`. Harden PPO backend compatibility by accepting vector-provided `seed` in env factories and normalizing per-env info extraction for dict/list/array payload forms. Fix script-mode path handling in `training/train_puffer.py` to avoid stdlib `logging` shadowing by `training/logging.py`.
- Consequences: Dockerized PPO training remains operational on the updated dependency stack and no longer relies on legacy 2.x behavior. Docs and image tags now match current dependency reality.
- Related commits/docs: `infra/trainer/requirements.txt`, `infra/docker-compose.yml`, `infra/trainer/README.md`, `training/README.md`, `training/puffer_backend.py`, `training/train_puffer.py`
### ADR-0020 - Add chunked websocket replay frame streaming while preserving HTTP pagination fallback

- Date: 2026-03-01
- Status: Accepted
- Context: Replay playback on very large artifacts can be inefficient with full-frame HTTP fetch alone, but existing frontend and API contracts already depend on `/frames` pagination semantics.
- Decision: Add websocket replay streaming endpoint `WS /ws/runs/{run_id}/replays/{replay_id}/frames` that emits chunked frame batches (`type=frames`) followed by terminal metadata (`type=complete`) and explicit error payloads (`type=error`). Keep `GET /frames` unchanged and add frontend transport selection (`HTTP` vs `WS`) with HTTP still available as fallback.
- Consequences: Frontend can opt into lower-latency chunked replay loading without breaking existing HTTP clients; backend now maintains two replay transport paths sharing the same replay artifacts.
- Related commits/docs: `server/app.py`, `tests/test_server_api.py`, `frontend/lib/api.ts`, `frontend/components/replay-dashboard.tsx`, `server/README.md`, `frontend/README.md`

### ADR-0021 - Use a single in-process M7 benchmark harness for trainer throughput, replay API latency, and memory soak checks

- Date: 2026-03-01
- Status: Accepted
- Context: M7 required actionable perf/stability measurement without introducing external load-test infrastructure before baseline automation was available.
- Decision: Add `tools/bench_m7.py` as a single benchmark harness that runs a deterministic short training job, measures trainer throughput (`steps/sec`), benchmarks replay API latency through in-process FastAPI `TestClient` calls, and executes a replay endpoint memory soak check using `tracemalloc` growth thresholds. Store benchmark reports as JSON artifacts for reproducible comparisons.
- Consequences: The project now has a reproducible local benchmark path with low setup overhead; reported memory metrics cover Python heap growth specifically and CI/nightly threshold enforcement remains a follow-up.
- Related commits/docs: `tools/bench_m7.py`, `tests/test_bench_m7.py`, `docs/PROJECT_STATUS.md`, `CHANGELOG.md`

### ADR-0022 - Add cycle-based replay stability runner with index invariants and API drift/leak checks

- Date: 2026-03-01
- Status: Accepted
- Context: After landing initial M7 benchmarks, the remaining gap was long-run stability validation for replay index integrity and replay API behavior under repeated access.
- Decision: Add `tools/stability_replay_long_run.py` to execute repeated training/eval cycles, validate replay index invariants (count, ordering, uniqueness, replay-file schema integrity), and run repeated replay catalog/frame/index reload loops with memory growth thresholds for leak/regression detection. Capture structured JSON reports for auditability.
- Consequences: The repo now has a reproducible local long-run stability workflow for replay artifacts and API access paths; enforcement in scheduled CI/nightly remains a follow-up integration step.
- Related commits/docs: `tools/stability_replay_long_run.py`, `tests/test_stability_replay_long_run.py`, `docs/PROJECT_STATUS.md`, `CHANGELOG.md`

### ADR-0023 - Complete M7 hardening with websocket chunk tuning, profiling sweeps, and nightly regression gates

- Date: 2026-03-01
- Status: Accepted
- Context: Remaining M7 work required validating websocket replay behavior on larger artifacts and turning benchmark/stability checks into automated regression gates.
- Decision: Extend replay websocket streaming with `max_chunk_bytes` and `yield_every_batches` tuning controls, add transport profiling sweeps (`tools/profile_ws_replay_transport.py`), and add scheduled nightly regression workflow (`.github/workflows/m7-nightly-regression.yml`) that runs benchmark thresholds plus replay stability gates and publishes artifacts.
- Consequences: Replay transport now has explicit chunk/backpressure controls and reproducible tuning evidence; benchmark/stability regressions can fail nightly automatically with artifact traces for diagnosis.
- Related commits/docs: `server/app.py`, `frontend/lib/api.ts`, `tests/test_server_api.py`, `tools/profile_ws_replay_transport.py`, `tests/test_profile_ws_replay_transport.py`, `tools/bench_m7.py`, `.github/workflows/m7-nightly-regression.yml`, `docs/PROJECT_STATUS.md`, `CHANGELOG.md`

### ADR-0024 - Adopt a 100k steps/sec target with native-core-first optimization and evidence-based fallback floor

- Date: 2026-03-01
- Status: Accepted
- Context: The next cycle prioritizes maximizing training throughput with a desired target of 100,000 env steps/sec, but current PPO hot paths still include Python reference-environment overhead that can cap throughput.
- Decision: Set 100,000 env steps/sec as the primary throughput target, implement a dedicated throughput profiler/report flow, and prioritize native-core integration in PPO hot loops before lower-ROI tuning. If 100,000 is not reachable on target hardware after these optimizations, record the highest stable measured floor and use that calibrated threshold for gates while continuing to track delta-to-target.
- Consequences: Throughput work is now explicitly ordered around bottleneck elimination rather than ad hoc tuning. Nightly/local thresholds become evidence-backed and can fail on regressions relative to calibrated baselines.
- Related commits/docs: `docs/PRIORITY_PLAN_100K_WANDB_VERCEL.md`, `docs/PROJECT_STATUS.md`

### ADR-0025 - Use backend W&B proxy as analytics source of truth and keep frontend hosted on Vercel

- Date: 2026-03-01
- Status: Accepted
- Context: The website must show iteration-aware analytics (current iteration, historical progress, and last-10 iteration drilldown) while keeping W&B credentials off the client and maintaining replay websocket support.
- Decision: Implement backend W&B proxy endpoints as the analytics source for the frontend dashboard, including run list, summary, history, and iteration-scoped views. Keep the frontend deployed on Vercel, and host the API separately on websocket-capable infrastructure with production CORS and secret-managed `WANDB_API_KEY`.
- Consequences: Analytics data access is centralized and secure, frontend can remain static/edge-friendly on Vercel, and backend deployment owns websocket transport plus W&B integration concerns.
- Related commits/docs: `docs/PRIORITY_PLAN_100K_WANDB_VERCEL.md`, `docs/PROJECT_STATUS.md`, `server/app.py`, `frontend/components/analytics-dashboard.tsx`

### ADR-0026 - Prioritize game-loop bottleneck removal via native PPO env path, callback batching, and C step-many APIs

- Date: 2026-03-01
- Status: Accepted
- Context: Throughput profiling shows training far below target, with the simulation path (Python reference env + per-env callback loop) as the dominant bottleneck. Reaching materially higher throughput requires reducing Python overhead before PPO hyperparameter tuning.
- Decision: Execute performance work in this strict order: (1) native-backed PPO env path (`ppo_env_impl`), (2) batch/aggregated callback processing in trainer hot loop, (3) native C batch stepping/reset interfaces with Python bridge support, then (4) micro-optimizations in C hot functions. Keep 100,000 steps/sec as aspirational target while using evidence-based floor calibration if hardware/runtime ceilings persist.
- Consequences: Performance workstream now prioritizes architecture-level bottleneck removal over incremental tuning; implementation risk shifts toward native-path parity, requiring explicit batch-vs-single parity and determinism validation gates.
- Related commits/docs: `tools/profile_training_throughput.py`, `docs/PERFORMANCE_BOTTLENECK_PLAN.md`, `docs/BUILD_CHECKLIST.md`, `docs/PROJECT_STATUS.md`

### ADR-0027 - Add selectable PPO env implementation path with native-first auto fallback

- Date: 2026-03-01
- Status: Accepted
- Context: Throughput workstream P1 requires moving PPO stepping off the Python reference env path when native runtime is available, while preserving trainer usability on setups where native binaries are absent.
- Decision: Extend PPO runtime config with `ppo_env_impl` (`reference|native|auto`), default to `auto`, add native-availability probing, and route PPO env creation to either reference or native Gym wrapper accordingly. For `auto`, prefer native and fallback to reference; for `native`, fail fast with a clear runtime error when native core is unavailable.
- Consequences: PPO training can opt into native stepping without changing external trainer APIs, and environments without native binaries remain functional via deterministic fallback behavior. Run metadata now records requested/selected env implementation for observability.
- Related commits/docs: `training/puffer_backend.py`, `training/train_puffer.py`, `training/README.md`, `tests/test_puffer_backend_env_impl.py`, `tests/test_train_puffer_args.py`, `docs/PERFORMANCE_BOTTLENECK_PLAN.md`

### ADR-0028 - Use batch PPO step callback boundary between runtime and window aggregation

- Date: 2026-03-01
- Status: Accepted
- Context: PPO runtime still invoked training callbacks once per env per step, adding Python call overhead in the hot path even after native env selection work landed.
- Decision: Introduce `on_step_batch` in `run_puffer_ppo_training` and dispatch one callback per vector step with reward/done arrays plus raw `infos`. Update trainer aggregation to consume this boundary via `WindowMetricsAggregator.record_step_batch(...)`, while preserving existing scalar `record_step(...)` behavior for non-batch paths.
- Consequences: Cross-module callback overhead is reduced in the PPO hot loop without changing window metric semantics. Further throughput gains remain tied to reducing per-env processing inside aggregation and moving stepping to C batch APIs.
- Related commits/docs: `training/puffer_backend.py`, `training/train_puffer.py`, `training/windowing.py`, `tests/test_puffer_backend_env_impl.py`, `tests/test_windowing.py`, `docs/PERFORMANCE_BOTTLENECK_PLAN.md`

### ADR-0029 - Add native batch reset/step APIs with Python bridge fast path and scalar fallback

- Date: 2026-03-01
- Status: Accepted
- Context: P3 requires reducing Python<->C call overhead beyond callback batching by allowing one C call to process many env instances per vector step.
- Decision: Extend C API with `abp_core_reset_many(...)` and `abp_core_step_many(...)` that operate over arrays of state handles and contiguous output buffers. Extend Python wrapper with `NativeProspectorCore.reset_many(...)` and `NativeProspectorCore.step_many(...)` that use batch symbols when available and fall back to per-core scalar calls when unavailable/mixed.
- Consequences: Native bridge now supports batched stepping/reset semantics needed for future trainer integration while preserving backward-compatible operation on scalar-only runtimes. Remaining performance work can focus on routing trainer hot path to these APIs.
- Related commits/docs: `engine_core/include/abp_core.h`, `engine_core/src/abp_core.c`, `python/asteroid_prospector/native_core.py`, `tests/test_native_core_wrapper.py`, `docs/PERFORMANCE_BOTTLENECK_PLAN.md`

### ADR-0030 - Route native PPO runtime through in-process batched vector stepping

- Date: 2026-03-01
- Status: Accepted
- Context: P3 added `step_many/reset_many` in C and Python bridge, but PPO training still stepped native envs through per-env wrappers in vector runtime, leaving Python<->C call overhead in the hot loop.
- Decision: For PPO runs where `ppo_env_impl` resolves to `native`, instantiate an in-process `_NativeBatchVectorEnv` that owns one `NativeProspectorCore` per env and performs vector-step/reset with `NativeProspectorCore.step_many(...)` and `reset_many(...)`. Preserve vector-env autoreset behavior for done envs and retain existing Puffer vector path for reference env runs.
- Consequences: Native PPO hot path now amortizes stepping/reset overhead into batched bridge calls and records selected runtime backend via `ppo_vector_backend_selected` (`native_batch` for native path). Native path no longer depends on per-env Gym wrapper stepping in the PPO inner loop.
- Related commits/docs: `training/puffer_backend.py`, `training/train_puffer.py`, `tests/test_puffer_backend_env_impl.py`, `training/README.md`, `docs/PROJECT_STATUS.md`, `CHANGELOG.md`

### ADR-0031 - Cache station graph distance and streamline observation packing in native core hot path

- Date: 2026-03-01
- Status: Accepted
- Context: After routing PPO to native batch stepping, native env-only throughput still spent avoidable CPU in per-step observation packing, including repeated BFS distance-to-station computation on a static episode graph.
- Decision: Add per-node station-distance cache recomputed during world generation and switch `abp_steps_to_station(...)` to O(1) lookup. Tighten `abp_pack_obs(...)` by reducing transient normalization/indexing overhead and replacing repeated divide operations with precomputed inverse multipliers where behavior remains equivalent.
- Consequences: Native hot-loop CPU overhead is reduced without changing frozen interface semantics. Parity remains required as the correctness gate after these micro-optimizations.
- Related commits/docs: `engine_core/include/abp_core.h`, `engine_core/src/abp_core.c`, `tools/profile_training_throughput.py`, `tests/test_profile_training_throughput.py`, `docs/PROJECT_STATUS.md`, `CHANGELOG.md`

### ADR-0032 - Standardize throughput matrix floor calibration as best-candidate minimum with safety margin

- Date: 2026-03-01
- Status: Accepted
- Context: After landing P1-P4 runtime optimizations, throughput tuning requires reproducible candidate sweeps and a deterministic rule for selecting enforceable local floor thresholds before the 100k aspirational target is reachable.
- Decision: Add `tools/run_throughput_matrix.py` to orchestrate env/trainer candidate sweeps and publish per-mode summaries. For each mode, select the best candidate by highest mean steps/sec and compute the recommended floor as `best_min_steps_per_sec * floor_safety_factor` (default `0.90`). Keep `100000` as the aspirational target and track delta-to-target in status artifacts.
- Consequences: Throughput threshold calibration is now evidence-based and reproducible across commits, with candidate-level artifacts and an explicit safety margin to reduce flakiness.
- Related commits/docs: `tools/run_throughput_matrix.py`, `tools/profile_training_throughput.py`, `tests/test_run_throughput_matrix.py`, `tests/test_profile_training_throughput.py`, `artifacts/throughput/throughput-matrix-20260301-p6.json`, `docs/PROJECT_STATUS.md`

### ADR-0033 - Enforce local throughput floors by replaying matrix-selected best candidates

- Date: 2026-03-01
- Status: Accepted
- Context: After calibrating per-mode floors from matrix artifacts, the project needed a deterministic local gate to verify current performance remains above those calibrated floors while keeping 100k as an aspirational reference target.
- Decision: Add `tools/gate_throughput_floors.py` to read `mode_summaries` + best candidates from a matrix artifact and run per-mode throughput profile checks with `enforce_target=True` against each mode's `recommended_floor_steps_per_sec`.
- Consequences: Local throughput enforcement is now reproducible and source-of-truth-driven by matrix artifacts; floor thresholds can be recalibrated by publishing a new matrix and rerunning the floor gate without hardcoding mode-specific constants.
- Related commits/docs: `tools/gate_throughput_floors.py`, `tests/test_gate_throughput_floors.py`, `artifacts/throughput/throughput-floor-gate-20260301-p1.json`, `docs/PROJECT_STATUS.md`

### ADR-0034 - Make native core discovery load-aware and platform-aware for PPO auto mode in Linux containers

- Date: 2026-03-01
- Status: Accepted
- Context: PPO matrix execution on Linux/Docker failed when `ppo_env_impl=auto` attempted to load a mounted Windows `abp_core.dll` (`invalid ELF header`). Path-only native availability checks treated this as available and prevented fallback.
- Decision: Update native core path selection to prefer platform-compatible library names (`.so`/`.dylib`/`.dll`) and update PPO native probing to validate actual loadability by instantiating `NativeProspectorCore`. If load fails under `auto`, fallback to reference and expose probe detail in metadata.
- Consequences: Throughput sweeps and training runs no longer fail hard on cross-OS native binary mismatches; `auto` behavior is resilient and diagnosable. Native path can be used in Linux once a valid `abp_core.so` is present.
- Related commits/docs: `python/asteroid_prospector/native_core.py`, `training/puffer_backend.py`, `tests/test_native_core_wrapper.py`, `tests/test_puffer_backend_env_impl.py`, `artifacts/throughput/throughput-matrix-ppo-20260301-m9p2d.json`, `docs/PROJECT_STATUS.md`

### ADR-0035 - Standardize milestone taxonomy across checklist/status/readme and treat M7 baseline automation as distinct from M8 hardening

- Date: 2026-03-01
- Status: Accepted
- Context: Documentation drift had created conflicting milestone references (for example "M7+ performance" in status docs while checklist defined M7 as baselines and M8 as performance/stability hardening), plus handoff/hygiene docs with encoding corruption that made execution guidance noisy.
- Decision: Treat `docs/BUILD_CHECKLIST.md` as canonical milestone taxonomy (`M7` baselines/benchmarking, `M8` performance/stability, `M9` throughput+W&B+Vercel), then align `README.md`, `docs/PROJECT_STATUS.md`, and agent-facing docs to that taxonomy. Rewrite corrupted handoff/hygiene docs into clean ASCII to remove ambiguity and redundant stale guidance.
- Consequences: Current and remaining work is now explicit and non-conflicting across planning/status docs; future updates must keep milestone naming synchronized with the checklist to avoid roadmap drift.
- Related commits/docs: `docs/BUILD_CHECKLIST.md`, `docs/PROJECT_STATUS.md`, `docs/AGENT_HANDOFF_BRIEF.md`, `docs/AGENT_HYGIENE_GUARDRAILS.md`, `README.md`, `CHANGELOG.md`
