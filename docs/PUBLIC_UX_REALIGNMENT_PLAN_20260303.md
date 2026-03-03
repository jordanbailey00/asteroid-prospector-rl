# Public UX Realignment Plan (Replay, Play, Analytics)

Last updated: 2026-03-03
Owner: Product + engineering execution
Status: Approved implementation plan

## Objective

Realign the public web experience so users can:

1. Watch agent replays in a large, legible pixel presentation.
2. Play the game as a human pilot with obvious controls and guidance.
3. Review full training analytics without touching training controls.

At the same time, keep all training-control capabilities in private operator tooling (not in public routes).

## Product boundary (public vs private)

Public website (Vercel-hosted):

- `/` Replay: observe agent playthroughs only.
- `/play`: human pilot gameplay only.
- `/analytics`: training analytics and run history only.
- No controls for launching, tuning, or mutating training jobs.

Private operator tooling (local/secure):

- Launch/stop training jobs through trainer CLI workflows.
- Manage experiment settings through run configs/CLI flags.
- Monitor throughput/logs/artifacts through PufferLib terminal dashboard output, W&B, and optional Constellation.
- Stays outside public web routing and client bundles.

## Requirement mapping and gap analysis

| Requirement | Current behavior | Gap | Impacted code/docs |
| --- | --- | --- | --- |
| Large game viewport on Replay and Play | Scene uses fixed `270px` canvas height and sits inside multi-card layouts | Viewport does not dominate screen; gameplay appears too small | `frontend/app/globals.css`, `frontend/components/sector-view.tsx`, `frontend/components/replay-dashboard.tsx`, `frontend/components/play-console.tsx` |
| Small side dashboard with critical gameplay data | Data is split across multiple cards and positions; dashboard density is low | Important state is fragmented and not always near viewport | `frontend/components/replay-dashboard.tsx`, `frontend/components/play-console.tsx` |
| Controls should make "how to play" obvious | Actions exist but are presented as dense palette + technical controls | No guided pilot flow, role-based grouping, or quick-start language | `frontend/components/play-console.tsx`, `frontend/lib/actions.ts` |
| Analytics tab should show full training analytics/data | Many metrics shown, but data catalog is not exhaustive or clearly segmented by source | Missing explicit completeness contract and run-config lineage surface | `frontend/components/analytics-dashboard.tsx`, `frontend/lib/types.ts`, `frontend/lib/api.ts` |
| RL is trained in non-pixel env; pixel is presentation/replay layer | Architecture already follows this technically, but messaging is implicit | Needs explicit documentation and UX copy to avoid user confusion | `docs/RL spec/*`, `training/*`, `frontend/components/sector-view.tsx` |
| No training customization controls on public site | Public site currently exposes replay/play tuning controls, but no true training controls | Need stricter "public observer/player only" IA and copy | `frontend/components/replay-dashboard.tsx`, `frontend/components/play-console.tsx`, `frontend/README.md` |
| Private operator workflow should use recommended tooling | Bespoke `ops_console/` prototype was introduced for local ops | Remove bespoke dashboard and standardize on PufferLib CLI/terminal + W&B + optional Constellation | `README.md`, `training/README.md`, `docs/BUILD_CHECKLIST.md`, `docs/PRIORITY_PLAN_100K_WANDB_VERCEL.md`, `docs/DECISION_LOG.md` |

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
- Implementation status (2026-03-03): complete via `GET /api/runs/{run_id}/analytics/completeness` plus frontend coverage table and lineage cards in `analytics-dashboard.tsx`.

### Workstream E - Operator tooling alignment (no bespoke dashboard)

- Remove the custom `ops_console/` implementation path.
- Document CLI-first operator workflows through `training/train_puffer.py`.
- Standardize training monitoring on:
  - PufferLib terminal dashboard output,
  - W&B metrics/artifacts,
  - Constellation when enabled.

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
- Added read-only analytics aggregation endpoint `GET /api/runs/{run_id}/analytics/completeness` for coverage/lineage/staleness contract checks.
- No public training-control endpoints will be added.

Training/runtime:

- No changes to frozen RL interface or non-pixel training loop.
- Replay remains post-training (or per-window eval) visualization path.

Operational tooling:

- Operator workflows run through trainer CLI plus W&B/Constellation; no custom local dashboard service is required.

## Documentation updates required (this plan)

- `docs/DECISION_LOG.md`: record decommission of bespoke ops dashboard in favor of PufferLib-native tooling.
- `docs/PROJECT_STATUS.md`: update current focus and next-work order.
- `docs/BUILD_CHECKLIST.md`: redefine M9.5 as PufferLib-native ops workflow.
- `docs/PRIORITY_PLAN_100K_WANDB_VERCEL.md`: align workstream E to PufferLib tooling.
- `frontend/README.md`: document public read-only boundary and operator tooling split.

## Acceptance criteria

Public website:

- Replay and Play viewport is visually primary on desktop and mobile.
- Right-side gameplay HUD contains all critical pilot metrics.
- A new user can understand basic play loop from on-screen guidance without source-code knowledge.
- Analytics page exposes complete documented metric catalog for a selected run/iteration.
- No training mutation controls are available in public routes.

Operator workflow:

- No bespoke in-repo local dashboard is required for training management.
- Training launch/mutation/monitoring is covered by CLI + W&B + optional Constellation.

## Validation plan

- Frontend visual regression snapshots for `/`, `/play`, `/analytics` at desktop + mobile widths.
- Updated manual checklist for "first-time pilot usability" and "observer replay clarity".
- API/contract checks for analytics completeness and empty-state handling.
- Keep existing smoke gates and add route-level UI assertions where practical.
