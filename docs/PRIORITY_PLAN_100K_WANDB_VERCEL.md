# Priority Implementation Plan: 100k Throughput + W&B Dashboard + Vercel

Last updated: 2026-03-03
Status: Approved planning baseline with public UX realignment extension

## Scope

This plan covers the following priorities:

1. Training throughput target of 100,000 env steps/sec (or highest stable throughput if hardware ceiling is lower).
2. W&B-backed analytics integration into the website for current iteration, full historical trend, and last-10 iteration drilldown.
3. Frontend hosting on Vercel.
4. Public UX realignment for Replay/Play/Analytics so gameplay is viewport-first and beginner-readable.
5. PufferLib-native operator tooling for training operations (CLI/terminal dashboard + W&B + Constellation) outside the public website.

Detailed runtime performance plan: `docs/PERFORMANCE_BOTTLENECK_PLAN.md`

## Non-goals

- No RL interface changes (`OBS_DIM=260`, `N_ACTIONS=69`, reward/action/obs contracts remain frozen).
- No migration of training-control UX into public Vercel routes.
- No requirement to expose backend secrets or mutable training controls to browser clients.

## Architecture impacts

### Training and runtime

- Current PPO path uses the Python reference env in vectorized training loops; this is the largest known barrier to 100k steps/sec.
- Throughput work will require prioritizing native-core stepping in trainer paths and minimizing per-step Python overhead.
- Throughput instrumentation must distinguish:
  - raw env stepping throughput,
  - trainer end-to-end throughput,
  - replay/eval overhead impact.

### Backend API

- Website analytics needs a backend W&B proxy layer so no browser uses `WANDB_API_KEY`.
- API contract must expose:
  - last 10 runs,
  - run summary,
  - iteration-window history,
  - current iteration snapshot payload.
- Server-side caching/rate limiting is required to control W&B API usage.

### Frontend

- Existing analytics page uses local run metrics; it must be extended to consume W&B-backed endpoints for iteration-aware analytics.
- Dashboard must include:
  - current selected iteration analytics,
  - full historical trend across prior iterations,
  - last-10 iteration dropdown and iteration-specific drilldown,
  - compact KPI snapshot cards.
- Replay and Play routes need a new viewport-first layout with a compact side HUD and fewer operator-facing controls.
- Human pilot UX needs explicit onboarding and grouped action controls so first-time users can play without reading source docs.

### Product boundary and operations tooling

- Public website routes (`/`, `/play`, `/analytics`) remain observer/player only.
- Training launch/tuning/ops controls use PufferLib-native tooling and are not deployed to Vercel client routes.
- Pixel rendering remains presentation-only; training remains on non-pixel RL state space.

### Hosting and deployment

- Frontend remains on Vercel.
- Backend must be hosted separately (non-Vercel) to support websocket replay transport and secure W&B proxying.
- CORS and env wiring must be updated for production origins.

## Workstream A: Throughput to 100k env steps/sec

### A1. Benchmark and profiling baseline

- Add `tools/profile_training_throughput.py` with modes:
  - `env-only` (core step loop),
  - `trainer` (PPO end-to-end),
  - `trainer+eval` (with replay/eval enabled).
- Emit JSON report with:
  - mean/p50/p95 steps/sec,
  - CPU and memory summary,
  - config used (`num_envs`, workers, rollout, backend).
- Add gate flags:
  - `--target-steps-per-sec` (default `100000`),
  - non-zero exit on miss when `--enforce-target` is set.

### A2. Remove known throughput blockers (highest ROI first)

- Integrate native core stepping path into training backend (no Python reference env in hot loop for PPO).
- Add or expose batch stepping APIs so trainer callbacks process vectorized outputs, not per-step Python loops.
- Make high-cost info fields optional in high-throughput mode (keep required metrics path intact via aggregate counters).
- Reduce metadata write frequency during hot training loops (window checkpoints remain authoritative).

### A3. Tuning and optimization passes

- Tune PPO/vector config matrix (`num_envs`, `num_workers`, `rollout_steps`, minibatch count).
- Profile and reduce observation packing/serialization overhead on hot path.
- Verify native build flags are optimized for release benchmarks.

### A4. Target policy and fallback handling

- Primary goal: `>= 100,000` steps/sec on target local benchmark machine.
- If unattainable after A2/A3:
  - record max stable throughput with profiler evidence,
  - set calibrated threshold in nightly gates based on measured stable floor,
  - track delta-to-target in status docs until closed.

### A5. Acceptance criteria

- Profiler report artifact committed under `artifacts/throughput/`.
- Reproducible command documented in README/docs.
- Nightly/local gate includes throughput target enforcement.

## Workstream B: W&B-backed website analytics

### B1. Data contract and run identity

- Standardize `run_id` mapping between local run metadata and W&B run IDs.
- Ensure iteration identity is first-class in logs/artifacts:
  - use `window_id` as canonical iteration key,
  - include iteration in W&B logged rows and eval summaries.

### B2. Backend W&B proxy endpoints

- Implement server endpoints (under `/api/wandb/...`):
  - `GET /api/wandb/runs?limit=10`
  - `GET /api/wandb/runs/{run_id}/summary`
  - `GET /api/wandb/runs/{run_id}/history`
  - `GET /api/wandb/runs/{run_id}/iterations?limit=10`
  - `GET /api/wandb/runs/{run_id}/iterations/{iteration}`
- Add backend cache with TTL and explicit error payloads for W&B failures.
- Keep existing local metrics endpoints as fallback source during migration.

### B3. Frontend dashboard implementation

- Extend `/analytics` with clear sections:
  - KPI snapshot cards: latest return, profit, survival, invalid action rate, steps/sec.
  - Current iteration panel: detailed metrics for selected iteration.
  - Historical trend panel: full run trajectory across all iterations.
  - Last-10 iteration selector/dropdown with per-iteration chart updates.
- Use consistent chart scales and color coding for readability.
- Add loading, stale-data, and proxy-error UI states.

### B4. Validation and tests

- API tests for W&B proxy endpoint contracts and cache behavior.
- Frontend integration tests for:
  - default latest run load,
  - iteration dropdown limiting to last 10,
  - chart/KPI updates on iteration change.
- Manual verification checklist with screenshot evidence for analytics views.

### B5. Acceptance criteria

- Website analytics data can be driven entirely through backend W&B proxy.
- Latest run and last-10 iteration controls work without exposing secrets client-side.
- KPI snapshot + iteration detail + historical trend all render correctly.

## Workstream C: Vercel-first frontend deployment

### C1. Frontend deployment baseline

- Configure Vercel project for `frontend/` root.
- Set required env vars:
  - `NEXT_PUBLIC_BACKEND_HTTP_BASE`
  - `NEXT_PUBLIC_BACKEND_WS_BASE`
- Add production and preview environment documentation.

### C2. Backend production hosting alignment

- Host API separately (container platform supporting websockets).
- Configure:
  - HTTPS + WSS endpoints,
  - production CORS for Vercel domain(s),
  - secure `WANDB_API_KEY` in backend environment only.

### C3. Release and smoke checks

- Add deployment runbook:
  - deploy backend,
  - deploy frontend on Vercel,
  - validate replay WS path and analytics proxy path.
- Add post-deploy smoke script/checklist for:
  - `/`,
  - `/play`,
  - `/analytics`,
  - W&B proxy endpoint health.

### C4. Acceptance criteria

- Frontend served from Vercel with working API and WS connectivity.
- Analytics dashboard loads from backend proxy in production.
- No W&B secret exposed in frontend bundles/env.

## Workstream D: Public UX realignment for Replay and Play

### D1. Shared gameplay shell and responsive viewport

- Implement a shared viewport-first layout for Replay and Play.
- Desktop: large primary viewport with compact right-side gameplay HUD.
- Mobile: stacked layout preserving gameplay visibility and control clarity.

### D2. Replay simplification

- Default to latest run/replay selection.
- Keep advanced transport/filter controls behind a collapsed advanced section.
- Keep playback controls visible and near the viewport.

### D3. Human pilot clarity

- Add explicit `How to play` flow and grouped action categories.
- Keep advanced play-session knobs behind an advanced section.
- Add visible hotkey/action legend.

### D4. Acceptance criteria

- New users can understand and perform core pilot loop from in-page guidance.
- Replay and Play viewport occupies most of desktop screen real estate.
- Critical pilot metrics remain visible in compact side HUD throughout gameplay.

## Workstream E: PufferLib-native operator tooling

### E1. Decommission bespoke dashboard path

- Remove the in-repo custom training dashboard code and docs references.
- Keep training control workflows out of public frontend routes.

### E2. Standardize operator workflow on PufferLib + W&B

- Use trainer CLI commands (`training/train_puffer.py`) as the canonical launch path.
- Use PufferLib terminal dashboard output for live progress during runs.
- Use W&B for persistent run metrics, artifacts, and comparison.
- Use Constellation when enabled for live orchestration visibility.

### E3. Acceptance criteria

- No bespoke in-repo dashboard remains for training management.
- Training and experiment mutation controls are run through operator tooling (CLI/terminal + W&B/Constellation), not public web routes.

## Ordered execution checklist

1. Add throughput profiler + baseline report generation.
2. Integrate native-core hot path into PPO training loop.
3. Apply vector and callback overhead reductions; rerun throughput report.
4. Add/adjust throughput gate and document measured floor vs 100k target.
5. Implement W&B proxy endpoints with caching and tests.
6. Extend analytics UI for current iteration, full history, and last-10 iteration selector.
7. Add frontend integration tests and manual verification artifacts for analytics UI.
8. Execute Replay/Play public UX realignment (shared shell, large viewport, compact side HUD).
9. Add explicit human pilot onboarding and grouped action controls.
10. Remove or collapse operator-facing controls from public routes.
11. Decommission bespoke dashboard path and standardize on PufferLib-native operator tooling.
12. Validate public/private boundary (no training mutation from public website).
13. Deploy frontend to Vercel and backend to websocket-capable host with production CORS/env.
14. Run production smoke checks and publish deployment runbook.

## Risks and mitigations

- Risk: 100k target may exceed current hardware/runtime path.
  - Mitigation: native-core first, hard profiling evidence, calibrated floor gate if ceiling is lower.
- Risk: W&B API latency/rate limits can degrade dashboard UX.
  - Mitigation: backend caching, pagination/windowed queries, explicit stale-data UI state.
- Risk: split hosting (Vercel frontend + separate backend) can break WS/CORS.
  - Mitigation: pre-production smoke checklist and explicit origin/WS config tests.
