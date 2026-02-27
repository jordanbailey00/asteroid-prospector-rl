Below is a **frontend architecture + build plan** for a Vercel-hosted web app with your 3 deliverables:

1) **Replay produced every window_env_steps environment steps + analytics associated with that window**  
2) **Playable “human pilot” mode** (ephemeral, no saved state)  
3) **Historical analytics** across training windows and runs (where each window is defined by window_env_steps environment steps)

This is designed to integrate cleanly with your existing game/RL spec (obs/action/reward) fileciteturn5file3 fileciteturn5file4 fileciteturn5file6 and the game loop/metrics in the GDD fileciteturn5file0, plus the backend’s record-then-play replay approach.

---

## 0) Key hosting constraint (Vercel + realtime)

- Your **Next.js frontend on Vercel should NOT host WebSocket servers**.
- It **can** connect *as a client* to a WebSocket server running elsewhere (your backend host).  
Vercel’s own KB + community threads indicate WebSockets aren’t supported as long-lived server connections in Vercel Functions. (https://vercel.com/kb/guide/do-vercel-serverless-functions-support-websocket-connections?utm_source=chatgpt.com)

So the architecture is:
- **Frontend (Vercel)**: UI + client-side WS/HTTP calls
- **Backend (your FastAPI host)**: replay streaming endpoint + play-session step/reset endpoints + metrics endpoints

---

## 1) Frontend information architecture (pages)

### A) `/` Main: “Agent Replay + Window Analytics”
**Purpose:** Show the most recent (or selected) replay from the latest training window (each window covering window_env_steps environment steps) and the analytics for that window.

Components:
- **ReplayPlayer**
  - playback controls: play/pause, step, speed (fps), stride (skip frames), jump-to-t
  - action display: action index + decoded action name
- **StatsPanel**
  - ship stats (fuel/hull/heat/tool/alert/credits)
  - cargo + market prices
  - event log (pirates, fracture, overheat, dock/sell)
- **WindowSummaryPanel**
  - aggregated metrics for the window that produced this replay (window = window_env_steps environment steps):
    - mean/median return, survival rate, profit/tick, overheat ticks, pirate encounters, etc. (these are the same “measures of success” you defined) fileciteturn5file0
- **ReplayPicker**
  - select: run_id → window_id → replay_id
  - quick filters: `every_n`, `best_so_far`, `milestone:*`
- **ObservabilityLinks**
  - “Open in W&B” (run + selected window)
  - “Open Constellation” (if available)

### B) `/play` Human Pilot Mode
**Purpose:** Let a human learn the game by piloting the same env (no persistence, no accounts).

Components:
- **PlaySessionHUD**
  - same StatsPanel layout as replay mode
- **ActionPalette**
  - buttons/hotkeys mapped to action indices 0..68 (per your action indexing) fileciteturn5file4
  - show disabled states when actions invalid (client can pre-check, backend is authoritative)
- **StepControls**
  - single-step
  - optional “hold-to-run” (e.g., run 5 steps/sec) so humans can play at a reasonable pace

### C) `/analytics` Historical Training Analytics
**Purpose:** Show time-series trends across windows and runs (window index corresponds to contiguous blocks of window_env_steps environment steps).

Components:
- **RunSelector**
- **MetricCharts**
  - line charts over window index/time:
    - return_mean, profit_mean, survival_rate
    - overheat_ticks_mean, pirate_encounters_mean, value_lost_to_pirates_mean
    - scan_count_mean, mining_ticks_mean, profit_per_tick_mean  
  (all align with your game performance metrics) fileciteturn5file0
- **CompareRuns**
  - overlay charts for two runs (optional but high value)
- **CheckpointTimeline**
  - show where checkpoints/replays were produced
- **ObservabilityLinks**
  - “Open in W&B” (run)
  - “Open Constellation” (if available)

---

## 2) Data contracts (what frontend needs from backend)

You’ll likely already have most of this from your backend design. The frontend assumes these endpoints exist:

### A) Runs + windows
- `GET /api/runs`
  - returns list of runs with metadata (run_id, started_at, env config hash, latest_window, wandb_run_url, constellation_url)
- `GET /api/runs/{run_id}`
  - returns run details + latest status (including wandb_run_url, constellation_url)

### B) Replay catalog + playback
- `GET /api/runs/{run_id}/replays?tag=every_n&limit=50`
  - list replay metadata (replay_id, window_id, return_total, profit, survival, tags)
- `GET /api/runs/{run_id}/replays/{replay_id}`
  - replay metadata + available streams

**Replay frames (choose one):**

**Option 1 (recommended): WebSocket stream from backend**
- `WS /ws/replay/{run_id}/{replay_id}?fps=4&stride=1`
  - server throttles frames to fps for human viewing (record-then-play)
  - messages: `init`, `frame`, `done`

**Option 2: HTTP download then client playback**
- `GET /api/runs/{run_id}/replays/{replay_id}/frames`
  - returns `jsonl.gz` or `msgpack+zstd` payload
  - frontend downloads once, then plays locally with a timer  
This avoids WS entirely, but can be heavier on initial load.

Given Vercel constraints, either is fine as long as the WS server is **not** on Vercel. (https://vercel.com/kb/guide/do-vercel-serverless-functions-support-websocket-connections?utm_source=chatgpt.com)

### C) Human play sessions (ephemeral)
- `POST /api/play/session` → `{ session_id }`
- `POST /api/play/session/{session_id}/reset` → `{ render_state, info }`
- `POST /api/play/session/{session_id}/step` body `{ action:int }` → `{ render_state, reward, done, info }`
- `DELETE /api/play/session/{session_id}`

The action integer must match your fixed action indexing (0..68). fileciteturn5file4

### D) Historical metrics
- `GET /api/runs/{run_id}/metrics/windows?limit=5000`
  - returns an array of window rows (each row summarizes a window_env_steps environment-steps block):
    - window_id, env_steps, episodes, return_mean, profit_mean, survival_rate, etc.

This keeps the frontend simple: it doesn’t need to talk to Prometheus directly.

---

## 3) Frontend “render_state” (no graphics required)

Since you said graphics will be designed separately, the UI should treat the game as a **stateful text/diagram dashboard**:

`render_state` should contain:
- ship stats, cargo, market prices
- current node + neighbors
- asteroid list (comp_est, stability_est, depletion, scan_conf)
- events list (pirate/fracture/overheat/dock/sell)

These map directly to what’s meaningful in your environment loop and strategic depth (scan → mine → manage heat/wear → sell into mark## 4) Telemetry + analytics integration (what the frontend actually uses)

You want telemetry and dashboards updated after each **window_env_steps** block (not per-iteration).

### Recommended split (W&B + PufferLib dashboards + in-app charts)

1) **Weights & Biases (authoritative training history)**
- PufferLib supports tracking via CLI (`puffer train ... --wandb`) and has a `PuffeRL.WandbLogger` path. (https://puffer.ai/docs.html)
- The backend should attach `wandb_run_url` to each `run_id` so the frontend can provide “Open in W&B” links.

2) **PufferLib dashboards (live view)**
- Use `PuffeRL.print_dashboard()` for terminal monitoring during training. (https://puffer.ai/docs.html)
- If you are using Puffer’s **Constellation** dashboard, expose a `constellation_url` per run so the frontend can link to it.

3) **Product analytics inside your website**
- The website’s `/analytics` page should remain backed by `GET /api/runs/{run_id}/metrics/windows` so you can display public-friendly charts without depending on W&B embed/auth flows.

This keeps the Vercel site lightweight and stable, while W&B remains the deep-dive tool for RL diagnostics and artifact lineage. (https://docs.wandb.ai/models/track/log, https://docs.wandb.ai/models/artifacts)
 not worth it early. (https://community.grafana.com/t/embedding-grafana-into-webpage-with-security/132980?utm_source=chatgpt.com)

So the Vercel site will show:
- “core” historical metrics via its own charts
- optional “Open Grafana” link for deep debugging (internal)

---

## 5) Frontend tech stack (Vercel-friendly)

- **Next.js (App Router) + TypeScript**
- **Tailwind + shadcn/ui** for layout and controls
- **TanStack Query** for HTTP data fetching/caching
- **Zustand** (or React Context) for playback state
- **Recharts** (or similar) for charts on `/analytics`

---

## 6) Frontend project structure (suggested)

```text
app/
  page.tsx                 # main replay page
  play/page.tsx            # human pilot
  analytics/page.tsx       # historical charts
components/
  ReplayPlayer/
    ReplayPlayer.tsx
    PlaybackControls.tsx
    ActionTimeline.tsx
  GameHUD/
    StatsPanel.tsx
    CargoPanel.tsx
    MarketPanel.tsx
    EventsPanel.tsx
    NodePanel.tsx
    AsteroidPanel.tsx
  PlayMode/
    ActionPalette.tsx
    Hotkeys.tsx
lib/
  api.ts                   # typed HTTP client
  ws.ts                    # ws helper + reconnection
  types.ts                 # Run, ReplayMeta, Frame, RenderState, WindowMetric
  actions.ts               # action index -> label (0..68) fileciteturn5file4
  metrics.ts               # chart metric definitions
```

---

## 7) Step-by-step build plan (end-to-end integration)

### Step 1 — Lock backend endpoints & CORS
- Ensure backend exposes the endpoints above
- Enable CORS for your Vercel domain (and localhost)

### Step 2 — Build Replay UI (main deliverable #1)
- Implement:
  - `GET /api/runs`
  - `GET /api/runs/{run_id}/replays?...`
  - Replay playback via WS *or* HTTP download
- Map action indices to human-readable action names using the RL spec action map. fileciteturn5file4
- Display per-frame:
  - action, reward, dt, events
  - ship stats, cargo, market

### Step 3 — Attach “window analytics” to replay
- When replay selected, also fetch:
  - `GET /api/runs/{run_id}/metrics/windows`
  - filter by the replay’s `window_id` (where window_id corresponds to the window_env_steps environment-steps block that produced the replay)
- Render a “Window Summary” card + mini-sparklines (return_mean, survival_rate, profit_mean)

### Step 4 — Build Play Mode (deliverable #2)
- Create session
- Reset
- Step with action buttons/hotkeys
- Render the same HUD used in replay mode
- Provide “invalid action feedback” (backend authoritative; client can gray out obvious invalids)

### Step 5 — Build Historical Analytics (deliverable #3)
- Fetch `metrics/windows` for selected run
- Plot the core metrics defined in your GDD (profit, survival, risk, efficiency) fileciteturn5file0
- Add run comparison (optional)

### Step 6 — Vercel deployment wiring
- Add env vars in Vercel:
  - `NEXT_PUBLIC_BACKEND_HTTP_BASE`
  - `NEXT_PUBLIC_BACKEND_WS_BASE`
- Ensure WS connects to backend host (not Vercel). (https://vercel.com/kb/guide/do-vercel-serverless-functions-support-websocket-connections?utm_source=chatgpt.com)

### Step 7 — Operational loop (how it all ties together)
- Local training runs via PufferLib
- After each window_env_steps environment-steps window:
  - eval runner generates replay files
  - backend indexes replays + updates metrics windows
- Frontend:
  - polls `/api/runs/{run_id}` every ~10–30s for “latest_window”
  - when new replay exists, it appears in catalog and can be played
