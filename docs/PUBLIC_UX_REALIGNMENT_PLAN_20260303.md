# Public UX Realignment Plan (Replay, Play, Analytics)

Last updated: 2026-03-03
Owner: Product + engineering execution
Status: Approved implementation plan

## Objective

Realign the public web experience so users can:

1. Watch agent replays in a large, legible pixel presentation.
2. Play the game as a human pilot with obvious controls and guidance.
3. Review full training analytics without touching training controls.

At the same time, move all training-control capabilities to a private local dashboard for operator use only.

## Product boundary (public vs private)

Public website (Vercel-hosted):

- `/` Replay: observe agent playthroughs only.
- `/play`: human pilot gameplay only.
- `/analytics`: training analytics and run history only.
- No controls for launching, tuning, or mutating training jobs.

Private local dashboard (operator-only):

- Run/stop training jobs.
- Edit training specs and hyperparameters.
- Monitor throughput, logs, checkpoints, and replay generation.
- Remains local to the operator machine; not deployed to public hosting.

## Requirement mapping and gap analysis

| Requirement | Current behavior | Gap | Impacted code/docs |
| --- | --- | --- | --- |
| Large game viewport on Replay and Play | Scene uses fixed `270px` canvas height and sits inside multi-card layouts | Viewport does not dominate screen; gameplay appears too small | `frontend/app/globals.css`, `frontend/components/sector-view.tsx`, `frontend/components/replay-dashboard.tsx`, `frontend/components/play-console.tsx` |
| Small side dashboard with critical gameplay data | Data is split across multiple cards and positions; dashboard density is low | Important state is fragmented and not always near viewport | `frontend/components/replay-dashboard.tsx`, `frontend/components/play-console.tsx` |
| Controls should make "how to play" obvious | Actions exist but are presented as dense palette + technical controls | No guided pilot flow, role-based grouping, or quick-start language | `frontend/components/play-console.tsx`, `frontend/lib/actions.ts` |
| Analytics tab should show full training analytics/data | Many metrics shown, but data catalog is not exhaustive or clearly segmented by source | Missing explicit completeness contract and run-config lineage surface | `frontend/components/analytics-dashboard.tsx`, `frontend/lib/types.ts`, `frontend/lib/api.ts` |
| RL is trained in non-pixel env; pixel is presentation/replay layer | Architecture already follows this technically, but messaging is implicit | Needs explicit documentation and UX copy to avoid user confusion | `docs/RL spec/*`, `training/*`, `frontend/components/sector-view.tsx` |
| No training customization controls on public site | Public site currently exposes replay/play tuning controls, but no true training controls | Need stricter "public observer/player only" IA and copy | `frontend/components/replay-dashboard.tsx`, `frontend/components/play-console.tsx`, `frontend/README.md` |
| Separate private local training dashboard | No dedicated local dashboard exists today | New local-only operator interface required | new `ops_console/` workstream + training orchestration docs |

Assumption used for conflicting wording in request (left vs right side dashboard):

- Gameplay viewport stays primary and centered/left.
- Compact gameplay dashboard is anchored on the right on desktop.
- On mobile, dashboard stacks below viewport.

## Workstreams

### Workstream A - Replay and Play layout overhaul

- Introduce a shared `GameViewportShell` layout used by `/` and `/play`.
- Desktop: viewport takes most width, sticky right-side HUD rail for key stats.
- Replace fixed canvas height with responsive viewport sizing (`min-height` tied to viewport).
- Consolidate HUD fields into one high-signal panel (fuel, hull, heat, tool, cargo, alert, credits, profit, survival, node context, time).

### Workstream B - Human pilot clarity and onboarding

- Replace dense action presentation with grouped command families:
  - navigation, scan/select, mining/refine, maintenance/survival, dock/trade.
- Add `How To Play` quick-start panel on `/play` with recommended loop:
  - scan -> select -> mine -> manage heat/tool -> dock/sell -> repeat.
- Add hotkey legend and explicit mapping from action names to pilot intent.
- Move low-level session knobs (`seed`, `env_time_max`) behind an advanced collapsible section.

### Workstream C - Replay UX simplification

- Default Replay to latest run and latest replay entry automatically.
- Keep advanced transport/filter controls, but move them behind `Advanced Replay Controls`.
- Promote key playback controls (play/pause, step, speed, timeline) to first-class controls near viewport.
- Hide raw JSON payload by default; show only when explicitly expanded.

### Workstream D - Analytics completeness contract

- Define a canonical analytics coverage table for `/analytics`:
  - run metadata, window metrics, W&B summary, W&B history, replay timeline.
- Surface run config lineage and training context fields in dedicated cards.
- Add consistency checks for missing/empty metrics and explicit stale-data states.
- Keep analytics read-only; no training mutation controls.

### Workstream E - Private local operator dashboard

- Add a local-only operator dashboard (initially simple) for training management.
- Proposed minimum feature set:
  - run profile selection and launch,
  - stop/resume controls,
  - live log tail,
  - throughput + window metrics stream,
  - checkpoint/replay artifact listing.
- Keep deployment mode local-only (`localhost`), with no public routing.

## Dependencies and impacted areas

Frontend code and styling:

- `frontend/components/replay-dashboard.tsx`
- `frontend/components/play-console.tsx`
- `frontend/components/sector-view.tsx`
- `frontend/components/analytics-dashboard.tsx`
- `frontend/app/globals.css`
- `frontend/lib/api.ts`, `frontend/lib/types.ts`, `frontend/lib/actions.ts`

Backend/API:

- Existing endpoints remain valid for core flows.
- May add one read-only analytics aggregation endpoint if current payload fan-out becomes too expensive.
- No public training-control endpoints will be added.

Training/runtime:

- No changes to frozen RL interface or non-pixel training loop.
- Replay remains post-training (or per-window eval) visualization path.

Operational tooling:

- New local operator dashboard will depend on training scripts (`training/train_puffer.py`, profiling/bench tools) and run artifacts under `runs/`.

## Documentation updates required (this plan)

- `docs/DECISION_LOG.md`: public-vs-private product boundary decision.
- `docs/PROJECT_STATUS.md`: update current focus and next-work order.
- `docs/BUILD_CHECKLIST.md`: add M9 sub-chunks for public UX and private ops console.
- `docs/PRIORITY_PLAN_100K_WANDB_VERCEL.md`: include UX realignment and private dashboard as active priority workstreams.
- `frontend/README.md`: document public read-only/product behavior and local operator split.

## Acceptance criteria

Public website:

- Replay and Play viewport is visually primary on desktop and mobile.
- Right-side gameplay HUD contains all critical pilot metrics.
- A new user can understand basic play loop from on-screen guidance without source-code knowledge.
- Analytics page exposes complete documented metric catalog for a selected run/iteration.
- No training mutation controls are available in public routes.

Private dashboard:

- Local dashboard can launch/stop training and expose active training telemetry.
- Dashboard is local-only and not wired into public deployment.

## Validation plan

- Frontend visual regression snapshots for `/`, `/play`, `/analytics` at desktop + mobile widths.
- Updated manual checklist for "first-time pilot usability" and "observer replay clarity".
- API/contract checks for analytics completeness and empty-state handling.
- Keep existing smoke gates and add route-level UI assertions where practical.
