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
