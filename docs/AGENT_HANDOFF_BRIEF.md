# Asteroid Belt Prospector — Agent Handoff Brief

## Purpose

Build a high-throughput RL game environment (“Asteroid Belt Prospector”) with:
- **Local training** via PufferLib (fast CPU stepping).
- **Eval-run replays** produced every **window_env_steps** and served for **record-then-play** viewing on a website.
- A **human-play mode** (no accounts, no persistence) so humans can learn the game.
- **Observability** via **Weights & Biases** (metrics + artifacts) and PufferLib dashboards (including “Constellation” if enabled).

This brief explains **what we’re building, why we’re building it, and how to use the accompanying specs**.

---

## Why this exists

The project is designed as a **strategic, long-horizon, partially observable RL environment** where progress emerges over millions of steps:
- Information gathering (scan) vs exploitation (mine)
- Risk management (pirates, hazards, fractures)
- Resource constraints (fuel, heat, tool wear, cargo)
- Market timing + slippage

The website exists to make training progress understandable to humans through:
- Watchable “best” / “every-window” replays
- Window summaries + historical training charts
- A playable version of the same environment

---

## Non-goals

- No user accounts, no persistent player state
- No live streaming from the trainer (record-then-play only)
- No custom art creation (Kenney packs only)
- No multiplayer / economy persistence across runs

---

## Definitions (must be consistent across code)

- **run_id**: unique identifier for a training run (timestamp + git SHA recommended)
- **window_env_steps**: number of environment steps in a training window; used to:
  - aggregate metrics
  - trigger checkpoint saves
  - trigger eval-run replay generation
- **window_id**: monotonically increasing window index within a run
- **replay_id**: unique ID for a recorded eval episode
- **replay tags**:
  - `every_window` (or `every_n`) — the canonical replay produced each window
  - `best_so_far` — replay beats prior best by margin
  - `milestone:*` — threshold-based replays (profit, survival, etc.)

---

## System components (backend-only + frontend)

### A) Simulation Engine (authoritative)
- Implements the environment step loop and all dynamics.
- Must keep the RL interface stable:
  - fixed obs layout (OBS_DIM=260)
  - fixed action indexing (0..68)
  - deterministic reward computation
- **Performance requirement:** core step loop should run in C where it materially improves throughput.

### B) Python RL Wrapper (Gymnasium/PufferLib-compatible)
- Thin wrapper around native core.
- Provides `reset()` / `step()` and emits:
  - `obs` (float32 vector)
  - `reward`
  - `terminated/truncated`
  - `info` metrics

### C) Trainer (PufferLib)
- Runs vectorized training, saves checkpoints each window.
- Logs window metrics to W&B (and uses Puffer dashboards).
- Does NOT generate replays in the hot path.

### D) Evaluator + Replay Generator (Option A)
- Triggered once per window:
  - loads latest checkpoint
  - runs 1..M eval episodes in a single env process
  - records frames for replay
  - tags and stores replays
  - logs replays as W&B artifacts (or at minimum links them)

### E) API Server
- Provides endpoints for:
  - listing runs/windows/replays
  - serving replay frames (WS or downloadable file)
  - human-play sessions (create/reset/step)
  - serving historical window metrics (for the website)

### F) Frontend (Vercel)
- Main page: latest replay + window summary
- Play page: human pilot mode
- Analytics page: historical window charts

---

## Build order (milestones / gates)

### M0 — Repo scaffold + CI shape
- repo structure with `engine_core/` (C), `python/` wrapper, `server/`, `frontend/`
- runnable “hello env” stub with fixed obs/action sizes

### M1 — Python reference env (correctness baseline)
- Implement a pure-Python reference that matches the RL spec.
- This is used ONLY to validate C parity.

### M2 — Native core v1 (C) + parity harness
- Implement core state + reset + step in C.
- Build Python bindings.
- Create a parity test harness:
  - same seed + same action sequence
  - compare obs/reward/done (within tolerances)

### M3 — Trainer + windowing + W&B logging
- Train loop produces window metrics every window_env_steps
- Saves checkpoints
- W&B logs:
  - window metrics
  - config
  - checkpoint artifacts

### M4 — Eval runner + replay recording
- Every window:
  - load checkpoint
  - run eval episode(s)
  - produce replay files and replay index

### M5 — API server endpoints
- run + replay catalogs
- replay playback endpoint (WS or download)
- human-play session endpoints
- metrics/windows endpoint

### M6 — Frontend integration
- replay player + window analytics
- human-play mode
- historical analytics charts

### M7 — Performance + stability pass
- target steps/sec reached
- memory leak checks
- deterministic seeds confirmed
- W&B artifacts + dashboards reliable

---

## Acceptance criteria (definition of “done enough to iterate”)

- Training can run for at least 3 windows without crashing.
- Each window produces:
  - checkpoint
  - window metrics row
  - at least one replay (every_window)
- Frontend can:
  - play latest replay at a controlled FPS
  - show window summary metrics for that replay’s window
  - allow a human to play (reset/step) and see the same HUD
  - show historical metrics across windows

---

## Artifact naming conventions (W&B + filesystem)

### W&B
- project: `asteroid-prospector`
- run name: `{run_id}`
- groups: optional `{experiment_name}`
- artifacts:
  - `model-{run_id}:latest` (checkpoint)
  - `replay-{run_id}-{window_id}-{replay_id}` (replay frames)
  - `replay_index-{run_id}` (index DB or JSON)

### Filesystem (local)
- `runs/{run_id}/checkpoints/ckpt_{window_id}.pt`
- `runs/{run_id}/metrics/windows.parquet` (or jsonl)
- `runs/{run_id}/replays/{replay_id}.<format>`
- `runs/{run_id}/replay_index.sqlite`

---

## Required specs to follow (don’t improvise)

- RL Game Design Document (GDD)
- RL spec modules (constants, obs layout, actions, transition, reward, compatibility, baselines, benchmarking)
- Backend engine spec (C/W&B updated)
- Frontend spec (W&B updated)
- Graphics + audio spec

---

## Operational notes

- The website should default to the **latest run** and **latest window replay**.
- Replay selection and analytics must always be keyed by:
  - `run_id`
  - `window_id`
  - `replay_id`

---

## Deliverables checklist (what must exist in git)

- Native core + Python binding package
- Reference env + parity tests
- Training script + eval script
- API server
- Frontend
- Docs index (so an agent can find everything quickly)
