# Priority Implementation Plan: 100k Throughput + W&B Dashboard + Vercel

Last updated: 2026-03-01
Status: Approved planning baseline for next execution cycle

## Scope

This plan covers only the following priorities:

1. Training throughput target of 100,000 env steps/sec (or highest stable throughput if hardware ceiling is lower).
2. W&B-backed analytics integration into the website for current iteration, full historical trend, and last-10 iteration drilldown.
3. Frontend hosting on Vercel.

## Non-goals

- No RL interface changes (`OBS_DIM=260`, `N_ACTIONS=69`, reward/action/obs contracts remain frozen).
- No redesign of replay/play APIs unrelated to throughput or W&B analytics integration.
- No broad spec-gap closure outside the three priorities above.

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

## Ordered execution checklist

1. Add throughput profiler + baseline report generation.
2. Integrate native-core hot path into PPO training loop.
3. Apply vector and callback overhead reductions; rerun throughput report.
4. Add/adjust throughput gate and document measured floor vs 100k target.
5. Implement W&B proxy endpoints with caching and tests.
6. Extend analytics UI for current iteration, full history, and last-10 iteration selector.
7. Add frontend integration tests and manual verification artifacts for analytics UI.
8. Deploy frontend to Vercel and backend to websocket-capable host with production CORS/env.
9. Run production smoke checks and publish deployment runbook.

## Risks and mitigations

- Risk: 100k target may exceed current hardware/runtime path.
  - Mitigation: native-core first, hard profiling evidence, calibrated floor gate if ceiling is lower.
- Risk: W&B API latency/rate limits can degrade dashboard UX.
  - Mitigation: backend caching, pagination/windowed queries, explicit stale-data UI state.
- Risk: split hosting (Vercel frontend + separate backend) can break WS/CORS.
  - Mitigation: pre-production smoke checklist and explicit origin/WS config tests.
