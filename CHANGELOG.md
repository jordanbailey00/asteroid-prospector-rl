# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- M0 repository scaffold (`engine_core`, `python`, `server`, `frontend`, `training`, `replay`, `tests`, `tools`).
- Pre-commit hooks for Black, Ruff, clang-format, and basic hygiene checks.
- CI workflow and local `tools/run_checks.ps1` gate for format/lint/tests.
- `HelloProspectorEnv` contract-only stub with frozen interface shape/action checks.
- `ProspectorReferenceEnv` M1 pure-Python baseline with deterministic reset, full action decode, obs packing, reward computation, and required info metrics.
- Tier-1/Tier-2 pytest coverage for resource bounds, station rules, dt/time behavior, scan normalization, market stability, rollout stability, and info key presence.
- Additional M1 hardening tests for determinism, reward sanity, and frozen observation layout contract checks.
- M2 native scaffold for `engine_core` with deterministic PCG32 RNG and C API skeleton (`abp_core_init/reset/step`).
- Handle-based native API (`abp_core_create/destroy`) for Python bindings.
- `engine_core/core_test_runner.c` smoke CLI for action-trace execution.
- `tools/build_native_core.ps1` for building `abp_core.dll` and `core_test_runner.exe`.
- Python ctypes native wrapper `NativeProspectorCore` and wrapper tests.
- MIT License (`LICENSE`).
- `docs/PROJECT_STATUS.md` as the single current-state board (status, progress, next work).
- `docs/DECISION_LOG.md` for ADR-style technical/process decisions.
- `tools/check_project_tracking.py` pre-commit guard requiring tracking updates on each commit.
- M2 native core implementation now covers full world generation, action decode, global dynamics, reward computation, and frozen observation packing in C.
- Native step result payload now includes parity-critical metrics consumed by Python wrapper and native trace runner.
- `tools/run_parity.py` parity harness with fixed suites/seeds, tolerance comparisons, and mismatch-bundle output.
- `.gitignore` now excludes `artifacts/` parity output directories.
- Python-side `Pcg32Rng` implementation aligned to native stochastic primitives.
- `ProspectorReferenceEnv` now uses `Pcg32Rng`, enabling parity convergence under fixed seed/action traces.
- M3 training modules: `training/windowing.py`, `training/logging.py`, and `training/train_puffer.py` for windowed metrics, checkpoint cadence, and optional W&B logging.
- New M3 tests: `tests/test_windowing.py`, `tests/test_wandb_offline.py`, and `tests/test_training_loop.py` (window logic, offline logger behavior, trainer artifact emission).
- `training/README.md` now documents M3 run commands and output artifact layout.
- `training/train_puffer.py` now persists live `run_metadata.json` updates (`status`, `latest_window`, `latest_checkpoint`, replay placeholders, and observability URLs).
- Trainer backend validation now surfaces explicit `puffer_ppo` blocker errors for unsupported environments (including Windows).
- Expanded M3 training-loop coverage to assert metadata contract fields and backend-blocker behavior.
- Added Dockerized Linux trainer runtime under `infra/` (`infra/trainer/Dockerfile`, `infra/trainer/requirements.txt`, `infra/docker-compose.yml`) with BuildKit pip cache mounts and build-time `import pufferlib` smoke check.
- Added `training/puffer_backend.py` implementing vectorized PufferLib PPO training with a PyTorch actor-critic policy.
- `training/train_puffer.py` now executes true `puffer_ppo` training on Linux/Docker, while preserving window metrics/checkpoint cadence and live metadata updates.
- Added Docker usage documentation for PPO backend execution in `training/README.md` and `infra/trainer/README.md`.

- Added M4 replay modules (`replay/schema.py`, `replay/index.py`, `replay/__init__.py`) with schema/index versioning and frame validation helpers.
- Added `training/eval_runner.py` and integrated per-window eval replay generation into `training/train_puffer.py`.
- Replay index entries now include `every_window` tags and `best_so_far` promotion when a replay sets a new run-level best return.
- Added replay artifact logging support in `training/logging.py` (`WandbWindowLogger.log_replay`).
- Added M4 tests: `tests/test_replay_schema.py`, `tests/test_eval_runner.py`, and training-loop replay integration coverage.
- Updated training/replay docs to document replay flags, artifact layout, and schema/index contracts.
- Added `training/policy.py` to centralize PPO actor-critic architecture and checkpoint state export/load helpers.
- `training/train_puffer.py` checkpoints are now backend-aware: `json_v1` for random backend and `ppo_torch_v1` (serialized model state) for PPO backend.
- `training/eval_runner.py` now loads serialized PPO checkpoint policy state and generates policy-driven eval replays (deterministic or stochastic mode).
- Added `milestone:*` replay tagging support with configurable return/profit/survival thresholds.
- Added replay index query helpers `filter_replay_entries(...)` and `get_replay_entry_by_id(...)` in `replay/index.py`.
- Added tests for checkpoint formats and replay index filtering (`tests/test_checkpoint_io.py`, `tests/test_replay_index.py`).
- Added initial M5 FastAPI server implementation (`server/app.py`, `server/main.py`) for run/replay catalog and replay frame fetch endpoints.
- Added replay query/filter support in API (`tag`, `tags_any`, `tags_all`, `window_id`, `limit`) and frame pagination (`offset`, `limit`).
- Added API endpoint test coverage with temp run artifacts in `tests/test_server_api.py`.
- Updated `server/README.md` with local startup instructions and endpoint list.
- Configured `infra/docker-compose.yml` trainer service with an explicit publishable image tag and `TRAINER_IMAGE` override for cross-repo reuse.
- Updated `infra/trainer/README.md` with share/push/consume workflow for using this trainer image as a reusable base in other RL projects.
- Added remaining M5 API endpoints: `GET /api/runs/{run_id}/metrics/windows` and play session lifecycle (`POST /api/play/session`, `reset`, `step`, `DELETE`).
- Added default CORS middleware configuration for localhost and Vercel-compatible origins, with env overrides (`ABP_CORS_ORIGINS`, `ABP_CORS_ORIGIN_REGEX`).
- Expanded server API tests to cover metrics endpoint behavior, play-session lifecycle, and CORS preflight handling.
- Updated `server/main.py` and `server/README.md` for runtime env configuration and endpoint documentation.
- Added M6 Next.js App Router frontend implementation under `frontend/app` with route pages for replay (`/`), human play (`/play`), and historical analytics (`/analytics`).
- Added replay dashboard wiring to M5 APIs with run/window/replay selectors, client-side playback controls, frame inspection, and window trend sparklines.
- Added human play console with play-session lifecycle controls, live HUD stats, auto-step mode, and full action palette for action indices `0..68`.
- Added analytics dashboard with run selection, optional run comparison, checkpoint/replay timeline tags, and historical metric trend panels.
- Added frontend local run/build documentation in `frontend/README.md`.
- Added M6.5 sector/minimap presentation component (`frontend/components/sector-view.tsx`) and integrated it into replay (`/`) and human play (`/play`) flows.
- Added manifest-driven graphics/audio layer with runtime loaders (`frontend/lib/assets.ts`) and action/event mapping contract (`frontend/lib/action_effects_manifest.json`, `frontend/lib/presentation.ts`).
- Added frontend audio cue player (`frontend/lib/audio.ts`) with WebAudio synth fallback and UI toggles for cue playback.
- Added presentation manifests and baseline background asset (`frontend/public/assets/manifests/graphics_manifest.json`, `frontend/public/assets/manifests/audio_manifest.json`, `frontend/public/assets/backgrounds/starfield.svg`).
- Added M6.5 validation coverage (`tests/test_frontend_presentation.py`) for action/event mapping completeness and manifest key/path resolution.
- M6.5 root Kenney asset bundles are now staged into `frontend/public/assets` (world sprites, UI sprites, VFX, backgrounds, planets, fonts, and audio).
- `frontend/public/assets/manifests/graphics_manifest.json` and `audio_manifest.json` now resolve core semantics to file-backed asset paths.
- Replay/play sector rendering now uses mapped sprite/background/planet/VFX paths from graphics manifest (ship/station/asteroid/hazard/pirate + action/event effects).
- Frontend shell styling now uses Kenney UI panel/button textures for cards, scene containers, and button variants.
- Frontend presentation tests now enforce file existence and semantic asset-class mapping for graphics keys, VFX keys, background keys, and non-`none` audio cues.

### Environment
- Installed missing development dependencies and toolchains:
  - `pre-commit` (Python package)
  - `hypothesis` (Python package)
  - `Kitware.CMake`
  - `LLVM.LLVM`
  - `BrechtSanders.WinLibs.MCF.UCRT` (GCC/MinGW toolchain)
- Configured user PATH to include CMake, LLVM, and WinLibs binaries.

### Docs
- Clarified Phase 8 (M6.5) completion criteria in docs/BUILD_CHECKLIST.md to require full file-backed Kenney asset wiring for core gameplay semantics.
- Corrected project status to mark M6.5 as in-progress until real assets are fully wired and validated.
- Added authoritative root `AGENTS.md` instructions requiring commit/push after each completed change.
- Updated README files to reflect M2 scaffold status and native build commands.
- Rewrote root `README.md` as a public GitHub-facing project overview (purpose, goals, stack, quick start, and roadmap context).
- Updated `docs/DOCS_INDEX.md` to include hygiene/parity/checklist/status/decision tracking docs in the authoritative read order.
