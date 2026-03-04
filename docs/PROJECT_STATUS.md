# Project Status

Last updated: 2026-03-04
Current focus: Post-MVP validation campaign preparation (extensive testing on 2026-03-05)

## Current state

- Frozen interface unchanged: `OBS_DIM=260`, `N_ACTIONS=69`, action indexing `0..68`.
- Native PPO runtime path is implemented (`reference|native|auto`) with load-aware probing and in-process batch stepping (`step_many/reset_many`).
- Replay transport supports both HTTP pagination and websocket chunk streaming, with close-handshake hardening and bounded smoke retries (`--ws-check-attempts`, default `3`).
- Public frontend routes are live and aligned to observer/player scope:
  - replay (`/`), play (`/play`), analytics (`/analytics`),
  - viewport-first Replay/Play shell,
  - grouped Play action controls and explicit onboarding guidance,
  - no public training-mutation controls.
- Operator workflow is aligned to PufferLib-native tooling (trainer CLI + terminal dashboard output) with W&B persistence and optional Constellation visibility.
- W&B proxy + analytics completeness backend contracts are in place and exercised by deployment smoke (`latest`, `summary`, `history`, `iteration-view`, pre/post `status`).
- Deployment smoke guardrails include backend CORS simple/preflight checks for split-host drift detection.
- M9 Chunk 4 final closeout sweep is complete and documented in `docs/M9_CHUNK4_MVP_CLOSEOUT_EXECUTION_20260304.md`.
- Extensive testing plan for 2026-03-05 is documented in `docs/MVP_EXTENSIVE_TEST_PLAN_20260305.md`.

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
| M7 - Baselines + benchmarking | Complete | Baseline bots, seeded protocol automation, and W&B benchmark logging/lineage |
| M8 - Performance + stability hardening | Complete | Replay transport tuning, benchmark/stability runners, native batch runtime path |
| M9 - Throughput + W&B dashboard + Vercel alignment | Complete | Throughput matrix/floor artifacts, W&B proxy analytics, deployment smoke guardrails, public UX realignment, and PufferLib-native ops workflow |

## Latest recorded validation health (2026-03-04)

- `python -m pytest -q` -> `129 passed, 2 skipped`.
- `npm --prefix frontend run lint` -> pass (no warnings/errors).
- `npm --prefix frontend run build` -> pass (`/`, `/play`, `/analytics`).
- `python tools/run_parity.py --seeds 2 --steps 512 --native-library engine_core/build/abp_core.dll` -> `12/12` parity cases passed.
- `python tools/smoke_m9_deployment.py --backend-http-base https://abp-backend-production.up.railway.app --backend-ws-base wss://abp-backend-production.up.railway.app --frontend-base https://frontend-nine-sandy-47.vercel.app --require-clean-wandb-status --output-path artifacts/deploy/m9-smoke-strict-20260304-final.json` -> pass (`15/15`, includes CORS checks).
- `gh workflow run m9-deployment-smoke.yml ... require_clean_wandb_status=true` -> success (run `22655777348`), artifact `artifacts/deploy/ci-m9-smoke-22655777348/m9-deployment-smoke-22655777348/m9-smoke-22655777348.json` (`15/15`).

## Next work (ordered)

1. Execute `docs/MVP_EXTENSIVE_TEST_PLAN_20260305.md` end-to-end and capture artifacts under `artifacts/` plus a dated report doc.
2. Run one fresh manual CI smoke workflow against production URLs after tomorrow's validation cycle and archive the artifact.
3. Prioritize post-MVP hardening backlog (throughput uplift, stability soak automation, and release operations drift controls).

## Active risks and blockers

- 100,000 steps/sec remains aspirational; current observed trainer throughput is well below target and still needs bottleneck reduction.
- Split-host deployment (Vercel + external backend) remains sensitive to environment/CORS drift across redeploys.
- W&B integration is currently healthy in strict smoke, but scope/auth/cache drift remains an operational risk and should be guarded each release cut.

## Decision pointers

- See `docs/DECISION_LOG.md` for accepted architecture/process decisions and milestone taxonomy decisions.

## Update policy (required)

- Every commit must update at least one tracking file:
  - `docs/PROJECT_STATUS.md`, or
  - `docs/DECISION_LOG.md`, or
  - `CHANGELOG.md`.
- Any non-trivial technical decision must include a same-commit ADR entry in `docs/DECISION_LOG.md`.
- Keep the "Next work (ordered)" section current so remaining direction is explicit.
