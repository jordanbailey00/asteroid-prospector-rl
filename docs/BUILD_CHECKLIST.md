# Asteroid Belt Prospector â€” Ordered Build Checklist (Agent Work Plan)

This document is a **chunked, ordered checklist** of all work needed to implement the project end-to-end, without attempting everything at once.

**Read order / context**
- Follow `DOCS_INDEX.md` for authoritative precedence.
- Treat the RL interface as **frozen** (OBS_DIM, N_ACTIONS, obs layout, action indexing, reward).
- Follow `AGENT_HYGIENE_GUARDRAILS.md` for repo discipline.
- Follow `ACCEPTANCE_TESTS_PARITY_HARNESS.md` for gating tests and parity requirements.

---

## Phase 0 â€” Pre-flight / Repo Bootstrap (M0)

### 0.1 Repo structure (create skeleton only)
- Create top-level structure:
  - `engine_core/` (C core)
  - `python/` (Gymnasium/Puffer wrapper + reference env)
  - `training/` (PufferLib trainer + eval runner)
  - `replay/` (recorder + serializers + index)
  - `server/` (FastAPI API server)
  - `frontend/` (Next.js/Vercel web app)
  - `tests/` and `tools/`
  - `docs/` (optional) + ensure docs index remains accurate

### 0.2 Tooling / hygiene
- Add pre-commit hooks:
  - Python: Black + Ruff
  - C: clang-format
  - basic hygiene checks (EOF newline, trailing whitespace, large files)
- Add minimal CI that runs:
  - lint/format checks
  - `pytest -q` (even if only a stub test exists)

### 0.3 â€œHello envâ€ stub (contract-only)
- Implement a minimal env that:
  - returns obs shape `(260,)` float32
  - accepts actions in `0..68` via `Discrete(69)`
  - returns proper `(obs, reward, terminated, truncated, info)`
- Add Tier 0 tests for the stub:
  - shape/dtype
  - action bounds
  - deterministic reset under seed (even if trivial)

**Exit criteria (Phase 0)**
- CI passes on main.
- Stub env contract tests pass.

---

## Phase 1 â€” Python Reference Environment (Correctness Baseline) (M1)

Purpose: Build a readable â€œgoldenâ€ Python env matching the RL spec exactly. This is used for debugging and parity comparisons.

### 1.1 Implement Python reference engine
- Implement full environment per RL spec modules:
  - obs packing (exact indices)
  - action decoding (0..68)
  - transition ordering (dt macro-actions)
  - reward function (code-ready)
  - info metrics (required keys)
- Implement procedural generation and dynamics at the spec level.

### 1.2 Unit tests (Tier 1)
Add tests for:
- resource clamps (fuel/hull/heat/tool/cargo bounds)
- station-only action gating
- dt/time_remaining correctness
- scan updates preserve normalization
- market update stability

### 1.3 Integration tests (Tier 2)
- short rollouts across seeds do not crash
- info keys always present
- invalid action handling: treated as HOLD + penalty
- window counter logic (if implemented here) or at least a stub helper

**Exit criteria (Phase 1)**
- Tier 0/1/2 tests pass consistently.
- Reference env produces sane metrics for baseline bots (no NaN/Inf).

---

## Phase 2 â€” Native C Core (Performance) + Python Bindings (M2)

Purpose: Move the simulation hot loop to C while preserving semantics.

### 2.1 Native core scaffolding
- Define C structs for:
  - ShipState, EpisodeState, Graph, AsteroidField, MarketState
- Implement authoritative RNG (single algorithm used everywhere).

### 2.2 Implement C core reset + step
- `reset(seed)`:
  - procedural generation (graph + asteroids + market)
  - initial state
  - obs pack in-place
- `step(action)`:
  - decode + validate
  - apply primary action (dt)
  - apply passive dynamics, thresholds, hazards/pirates, market tick
  - compute reward
  - pack obs in-place
  - update info-selected metrics

### 2.3 Python bindings
Choose one binding approach and stick to it:
- CPython extension module (C API) OR pybind11
- Expose:
  - `core_create(config)`
  - `core_reset(handle, seed) -> obs`
  - `core_step(handle, action) -> (obs, reward, terminated, truncated, info_selected)`
- Implement optional batching:
  - `reset_many`, `step_many` to reduce call overhead

### 2.4 Native testing harness
- Add `engine_core/core_test_runner` CLI to dump traces given:
  - seed
  - action sequence file
  - output trace file

**Exit criteria (Phase 2)**
- Native core compiles locally.
- Python wrapper can step an episode end-to-end.
- No per-step heap allocations in core hot loop (inspect / profile).

---

## Phase 3 â€” Parity Harness + Acceptance Test Completion (M2.5)

Purpose: Prove C core matches Python reference under fixed seeds and action sequences.

### 3.1 Parity runner tooling
- Implement `tools/run_parity.py`:
  - generates action suites (random/adversarial/scenario)
  - runs Python ref and C core
  - compares traces with defined tolerances
  - dumps mismatch bundle on failure

### 3.2 Mark and organize test tiers
- `pytest -q` runs Tier 0/1/2 by default
- `pytest -q -m parity` runs Tier 3 parity suite
- `pytest -q -m perf` runs perf tests

### 3.3 Determinism hardening
- Ensure both Python ref and C core use the same authoritative RNG.
- Confirm: same seed + same actions => same done flags + near-identical float outputs.

**Exit criteria (Phase 3)**
- Parity suite passes required matrix (10 seeds Ã— suites Ã— steps).
- â€œStop-the-lineâ€ failures are all absent (NaN/Inf, nondeterminism, bounds violations).

---

## Phase 4 â€” Training Loop (PufferLib) + W&B Observability (M3)

Purpose: Train locally at high throughput; log windowed metrics and artifacts.

### 4.1 PufferLib training integration
- Create `training/train_puffer.py`:
  - vectorized env creation
  - PPO config
  - run_id creation + config snapshot
  - checkpoint save cadence based on `window_env_steps`

### 4.2 Windowing implementation
- Implement `window_env_steps` logic:
  - maintain `env_steps_total`
  - compute `window_id = floor(env_steps_total / window_env_steps)`
  - aggregate metrics within the window
  - emit one metrics record per window

### 4.3 W&B logging
- `wandb.init(config=...)` with run metadata
- Log per-window metrics using `wandb.log(..., step=env_steps_total)`
- Log checkpoint artifacts every window (or every K windows)
  - use deterministic naming: `model-{run_id}:latest` and versioned artifacts
- Persist essential URLs (wandb_run_url, optionally constellation_url) into run metadata so the API can expose them

### 4.4 Puffer dashboards
- Enable PufferLib dashboard output (terminal) during training
- If Constellation is configured, ensure the run metadata captures its URL

**Exit criteria (Phase 4)**
- Training runs for â‰¥ 3 windows without crash.
- Each window produces: checkpoint + window metrics row + W&B log event.

---

## Phase 5 â€” Eval Runner + Replay Generation (Option A) (M4)

Purpose: Generate watchable replays without slowing training.

### 5.1 Eval runner
- Implement `training/eval_runner.py`:
  - triggered once per window (or every K windows)
  - loads latest checkpoint
  - runs 1..M eval episodes in a clean single-env process
  - records frame-by-frame replay output

### 5.2 Replay recorder + formats
- Implement replay frame schema (stable):
  - `t`, `dt`, `action`, `reward`, `terminated/truncated`
  - `render_state` snapshot
  - `events` list
  - optional `info`
- Store replay files (jsonl.gz or msgpack+zstd)
- Maintain replay index (SQLite or JSON index) with:
  - run_id, window_id, replay_id, tags, return_total, profit, survival, etc.

### 5.3 Replay selection & tagging
- Every window: tag one replay as `every_window`
- Track `best_so_far` by return/profit metrics
- Optional `milestone:*` triggers

### 5.4 W&B artifacts for replays
- Log replay bundles as W&B artifacts (or at minimum log them as files and store URLs)

**Exit criteria (Phase 5)**
- Each training window produces at least one replay file + index entry.
- Replay schema validation tests pass.

---

## Phase 6 â€” Backend API Server (M5)

Purpose: Provide stable endpoints for frontend replay playback, play sessions, and analytics.

### 6.1 Run/metrics endpoints
Implement:
- `GET /api/runs`
- `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/metrics/windows?limit=...`

Run metadata should include:
- latest_window
- wandb_run_url
- constellation_url (if available)

### 6.2 Replay endpoints
Implement:
- `GET /api/runs/{run_id}/replays?tag=...`
- `GET /api/runs/{run_id}/replays/{replay_id}`
- Replay frames delivery:
  - WS stream OR HTTP download endpoint

### 6.3 Human-play session endpoints
Implement:
- `POST /api/play/session`
- `POST /api/play/session/{session_id}/reset`
- `POST /api/play/session/{session_id}/step`
- `DELETE /api/play/session/{session_id}`

Sessions are ephemeral (TTL) and do not persist state.

### 6.4 CORS + deployment config
- Enable CORS for localhost + Vercel origin.
- Ensure WS runs on backend host (not Vercel).

**Exit criteria (Phase 6)**
- API smoke tests pass.
- Frontend can retrieve run/replay/metrics from local backend.

---

## Phase 7 â€” Frontend (Vercel) (M6)

Purpose: Provide replay viewing, play mode, and historical analytics.

### 7.1 Next.js app scaffold
- Pages:
  - `/` replay + window analytics
  - `/play` human pilot
  - `/analytics` historical charts
- Environment variables:
  - `NEXT_PUBLIC_BACKEND_HTTP_BASE`
  - `NEXT_PUBLIC_BACKEND_WS_BASE`

### 7.2 Replay player
- Load replay catalog + select replay
- Playback:
  - play/pause/step
  - fps control
  - stride control
  - jump-to frame/time
- Display:
  - current action + decoded name
  - ship stats + cargo + market + event log
  - window summary panel for the replayâ€™s window_id
- Add â€œOpen in W&Bâ€ and â€œOpen Constellationâ€ links (if URLs provided by backend)

### 7.3 Human pilot mode
- Create/reset session
- Render same HUD
- Provide action palette/hotkeys (0..68)
- Show invalid action feedback

### 7.4 Historical analytics page
- Fetch `metrics/windows` and render charts (return/profit/survival/risk/efficiency)
- Optional run compare overlay

**Exit criteria (Phase 7)**
- Deployed frontend can:
  - play latest replay
  - show window summary + historical charts
  - allow human play session stepping

---

## Phase 8 - Graphics + Audio Integration (M6.5)

Purpose: Deliver a fully wired game presentation layer using the provided Kenney assets (no placeholder/procedural stand-ins for required game entities).

### 8.1 Source asset ingestion (authoritative)
- Use the existing root `assets/` packs as the source of truth:
  - `assets/kenney_space-shooter-redux`
  - `assets/kenney_space-shooter-extension`
  - `assets/kenney_simpleSpace`
  - `assets/kenney_planets`
  - `assets/kenney_ui-pack`
- Build a semantic mapping table for all required game items before wiring:
  - ships (agent + human)
  - station(s)
  - asteroids/meteors (small/medium/large)
  - planets/background layers
  - hazard/pirate markers
  - HUD panels/buttons/icons
  - action/event VFX assets
  - UI + gameplay audio cues
- Rule: if an item is represented visually in gameplay, it must map to a real Kenney file path.

### 8.2 Runtime asset packaging for frontend
- Copy selected source assets into `frontend/public/assets/...` with stable web paths:
  - `frontend/public/assets/sprites/world/...`
  - `frontend/public/assets/sprites/ui/...`
  - `frontend/public/assets/sprites/vfx/...`
  - `frontend/public/assets/backgrounds/...`
  - `frontend/public/assets/audio/ui/...`
  - `frontend/public/assets/audio/sfx/...`
- Atlases are allowed but optional; direct file paths are acceptable if runtime performance remains stable.
- Do not mark M6.5 complete while required semantic keys still point to procedural placeholders.

### 8.3 Manifest completion (hard requirement)
- Populate `graphics_manifest.json` with real file-backed entries for required semantic keys.
- Populate `audio_manifest.json` with real `.ogg` file mappings for UI/action/event cues.
- Every action id `0..68` must map to VFX + SFX cue keys; every mapped key must resolve to an existing file path (except explicit `none` keys).
- Keep `action_effects_manifest.json` as the semantic action/event mapping contract.

### 8.4 Frontend wiring (replay + play)
- Replay page and play page must render the same core game objects using mapped assets:
  - ship sprite(s), station sprite(s), asteroid sprites, background/planet layer, hazard and pirate markers.
- HUD and minimap elements must use mapped UI assets (panels/icons/buttons).
- Action/event VFX must display the correct asset family for the triggered action/event.
- Audio cues must play from mapped Kenney `.ogg` files for controls + gameplay events.

### 8.5 Validation and regression checks
- Add/maintain tests for:
  - manifest key completeness (required semantic keys present)
  - file existence for referenced graphics/audio paths
  - full action mapping coverage (`0..68`) and event mapping coverage
  - replay/play rendering smoke checks (no missing asset references)
- Add a manual replay verification checklist with at least one sampled replay demonstrating:
  - travel, scan, mining, docking/sell, warning events, terminal state cues.

**Exit criteria (Phase 8)**
- Replay and play mode are fully asset-backed from Kenney packs for all core game components.
- Correct asset type is used for correct item class (planet->planet, asteroid/meteor->asteroid, ship->ship, station->station, etc.).
- No required semantic key depends on procedural placeholder rendering.
- Audio cues for UI + core gameplay are file-backed and trigger correctly.
- Repeated playback sessions show no texture/audio leak behavior.

---

## Phase 9 - Baselines + Benchmarking (M7)

Purpose: Establish non-learning benchmarks and track improvements.

### 9.1 Baseline bots
- Implement/reference:
  - greedy miner
  - cautious scanner
  - market timer
- Add CLI to run bots for N episodes and log summary metrics.

### 9.2 Benchmark protocol automation
- Run bots for 1000 episodes (or smaller for CI)
- Compare PPO vs bots on:
  - net profit, survival, profit/tick, overheat ticks, pirate encounters

### 9.3 Log benchmark results
- Log benchmark summaries to W&B as a separate job type (e.g., `eval`)

**Exit criteria (Phase 9)**
- Baselines run reproducibly across seeds.
- Benchmark report produced and stored (local + W&B).

---

## Phase 10 â€” Performance + Stability Hardening (M8)

Purpose: Ensure the system can run long training jobs reliably and fast.

### 10.1 Steps/sec benchmarks
- Add `tools/bench_steps_per_sec.py`:
  - fixed config/seed
  - report env steps/sec and CPU usage
- Track before/after for performance PRs.

### 10.2 Native safety checks
- Run sanitizer builds (ASan/UBSan) where feasible.
- Add leak checks or long-run stability tests (nightly).

### 10.3 Replay + API robustness
- Validate replay schema versioning strategy.
- Ensure replay index remains consistent under concurrent run updates.

**Exit criteria (Phase 10)**
- Meets target steps/sec on your hardware.
- Can run multi-hour training without memory growth or crashes.

---

## â€œDo not proceedâ€ blockers (stop-the-line)
- Parity harness mismatch beyond tolerance
- Nondeterminism under fixed seed/action sequence
- NaN/Inf in obs/reward
- Violated resource bounds
- window_env_steps windowing miscounts
- Replay schema incompatibility with frontend playback

---

## Deliverables summary (end state)
- C core env + Python bindings + parity suite
- PufferLib training + eval runner + W&B artifacts/logging
- Replay files + replay index + API server
- Vercel frontend with replay + play + analytics
- Graphics/audio pipeline + manifests + validation tests
- Baselines + benchmarking automation
