"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { allActionLabels, actionLabel } from "@/lib/actions";
import { useCuePlayer } from "@/lib/audio";
import {
  backendBaseUrl,
  createPlaySession,
  deletePlaySession,
  resetPlaySession,
  stepPlaySession,
} from "@/lib/api";
import { formatNumber } from "@/lib/format";
import { actionCueKey, eventCueKey } from "@/lib/presentation";
import { PlaySessionState } from "@/lib/types";

import { SectorView } from "@/components/sector-view";

type ObsSummary = {
  fuelPct: number;
  hullPct: number;
  heatPct: number;
  toolPct: number;
  cargoPct: number;
  alertPct: number;
  timePct: number;
  cargoByCommodity: number[];
};

type ActionGroup = {
  id: string;
  title: string;
  description: string;
  actionIndexes: number[];
};

type HotkeyBinding = {
  key: string;
  action: number;
  hint: string;
};

const ACTION_LABELS = allActionLabels();
const HOLD_ACTION = 6;
const CARGO_MAX = 200;
const CREDITS_CAP = 1.0e7;

function parseOptionalNumber(value: string): number | undefined {
  const trimmed = value.trim();
  if (trimmed === "") {
    return undefined;
  }

  const numeric = Number(trimmed);
  if (!Number.isFinite(numeric)) {
    return undefined;
  }
  return numeric;
}

function parseOptionalInt(value: string): number | undefined {
  const numeric = parseOptionalNumber(value);
  if (numeric === undefined) {
    return undefined;
  }
  return Math.trunc(numeric);
}

function decodeObsSummary(obs: number[] | undefined): ObsSummary | null {
  if (!Array.isArray(obs) || obs.length < 14) {
    return null;
  }

  const values = obs.map((item) => Number(item));
  const isInvalid = values.some((item) => !Number.isFinite(item));
  if (isInvalid) {
    return null;
  }

  return {
    fuelPct: values[0] * 100,
    hullPct: values[1] * 100,
    heatPct: values[2] * 100,
    toolPct: values[3] * 100,
    cargoPct: values[4] * 100,
    alertPct: values[5] * 100,
    timePct: values[6] * 100,
    cargoByCommodity: values.slice(8, 14).map((fraction) => fraction * CARGO_MAX),
  };
}

function inferCreditsFromObs(obs: number[] | undefined): number | null {
  if (!Array.isArray(obs) || obs.length < 8) {
    return null;
  }

  const norm = Number(obs[7]);
  if (!Number.isFinite(norm) || norm < 0) {
    return null;
  }

  return Math.expm1(norm * Math.log1p(CREDITS_CAP));
}

function deriveEvents(
  currentInfo: Record<string, unknown>,
  previousInfo: Record<string, unknown> | null,
  session: PlaySessionState,
): string[] {
  const events: string[] = [];

  if (Boolean(currentInfo.invalid_action)) {
    events.push("invalid_action");
  }

  const currentPirates = Number(currentInfo.pirate_encounters ?? 0);
  const previousPirates = Number(previousInfo?.pirate_encounters ?? 0);
  if (Number.isFinite(currentPirates) && Number.isFinite(previousPirates) && currentPirates > previousPirates) {
    events.push("pirate_encounter");
  }

  const currentOverheat = Number(currentInfo.overheat_ticks ?? 0);
  const previousOverheat = Number(previousInfo?.overheat_ticks ?? 0);
  if (
    Number.isFinite(currentOverheat) &&
    Number.isFinite(previousOverheat) &&
    currentOverheat > previousOverheat
  ) {
    events.push("overheat_tick");
  }

  if (session.terminated) {
    events.push("terminated");
  }
  if (session.truncated) {
    events.push("truncated");
  }

  return events;
}

function classifyAction(label: string): string {
  if (label.startsWith("TRAVEL_") || label === "HOLD_DRIFT" || label === "EMERGENCY_BURN") {
    return "navigation";
  }

  if (
    label.includes("SCAN") ||
    label === "PASSIVE_THREAT_LISTEN" ||
    label.startsWith("SELECT_ASTEROID_")
  ) {
    return "scan";
  }

  if (label.startsWith("MINE_") || label === "STABILIZE_SELECTED" || label === "REFINE_ONBOARD") {
    return "mining";
  }

  if (
    label === "ACTIVE_COOLDOWN" ||
    label === "TOOL_MAINTENANCE" ||
    label === "HULL_PATCH" ||
    label.startsWith("JETTISON_COMMODITY_")
  ) {
    return "survival";
  }

  return "dock_trade";
}

function buildActionGroups(labels: string[]): ActionGroup[] {
  const groups: ActionGroup[] = [
    {
      id: "navigation",
      title: "Navigation",
      description: "Move through the belt, hold drift, or burn out of danger.",
      actionIndexes: [],
    },
    {
      id: "scan",
      title: "Scan + Target",
      description: "Read the field, identify targets, and lock asteroid selections.",
      actionIndexes: [],
    },
    {
      id: "mining",
      title: "Mining + Processing",
      description: "Extract ore, stabilize yield, and refine payload on board.",
      actionIndexes: [],
    },
    {
      id: "survival",
      title: "Survival + Maintenance",
      description: "Manage heat/tools/hull and dump cargo when risk spikes.",
      actionIndexes: [],
    },
    {
      id: "dock_trade",
      title: "Dock + Trade",
      description: "Dock, sell cargo, buy supplies, and close the run.",
      actionIndexes: [],
    },
  ];

  const byId = new Map(groups.map((group) => [group.id, group]));

  labels.forEach((label, index) => {
    const groupId = classifyAction(label);
    const group = byId.get(groupId);
    if (group) {
      group.actionIndexes.push(index);
    }
  });

  return groups;
}

function findActionIndex(label: string, fallback: number): number {
  const index = ACTION_LABELS.indexOf(label);
  if (index >= 0) {
    return index;
  }
  return fallback;
}

const ACTION_GROUPS = buildActionGroups(ACTION_LABELS);
const HOTKEY_BINDINGS: HotkeyBinding[] = [
  { key: "1", action: 0, hint: "Travel lane 0" },
  { key: "2", action: 1, hint: "Travel lane 1" },
  { key: "3", action: 2, hint: "Travel lane 2" },
  { key: "q", action: findActionIndex("WIDE_SCAN", 8), hint: "Wide scan" },
  { key: "e", action: findActionIndex("FOCUSED_SCAN_SELECTED", 9), hint: "Focused scan" },
  { key: "m", action: findActionIndex("MINE_STANDARD_SELECTED", 29), hint: "Mine standard" },
  { key: "r", action: findActionIndex("REFINE_ONBOARD", 32), hint: "Refine onboard" },
  { key: "h", action: HOLD_ACTION, hint: "Hold drift" },
  { key: "d", action: findActionIndex("DOCK", 42), hint: "Dock" },
];

const HOTKEY_ACTION_MAP = new Map(HOTKEY_BINDINGS.map((binding) => [binding.key, binding.action]));

export function PlayConsole() {
  const [session, setSession] = useState<PlaySessionState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [seedInput, setSeedInput] = useState("");
  const [envTimeMaxInput, setEnvTimeMaxInput] = useState("");

  const [selectedAction, setSelectedAction] = useState<number>(HOLD_ACTION);
  const [autoRun, setAutoRun] = useState(false);
  const [stepsPerSecond, setStepsPerSecond] = useState(5);

  const [derivedEvents, setDerivedEvents] = useState<string[]>([]);

  const sessionIdRef = useRef<string | null>(null);
  const steppingRef = useRef(false);
  const prevInfoRef = useRef<Record<string, unknown> | null>(null);

  const { enabled: audioEnabled, setEnabled: setAudioEnabled, playCue } = useCuePlayer(false);

  const sessionDone = Boolean(session?.terminated || session?.truncated);

  useEffect(() => {
    sessionIdRef.current = session?.session_id ?? null;
  }, [session]);

  useEffect(() => {
    if (!session) {
      prevInfoRef.current = null;
      setDerivedEvents([]);
      return;
    }

    const info = (session.info ?? {}) as Record<string, unknown>;
    const events = deriveEvents(info, prevInfoRef.current, session);
    prevInfoRef.current = info;
    setDerivedEvents(events);
  }, [session]);

  useEffect(() => {
    for (const eventName of derivedEvents) {
      void playCue(eventCueKey(eventName));
    }
  }, [derivedEvents, playCue]);

  const startSession = useCallback(async () => {
    setBusy(true);
    setError(null);
    setAutoRun(false);

    try {
      const seed = parseOptionalInt(seedInput);
      const envTimeMax = parseOptionalNumber(envTimeMaxInput);
      const payload = await createPlaySession({ seed, env_time_max: envTimeMax });
      setSession(payload);
      void playCue("ui.click");
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      setSession(null);
    } finally {
      setBusy(false);
    }
  }, [seedInput, envTimeMaxInput, playCue]);

  const resetSession = useCallback(async () => {
    if (!session?.session_id) {
      return;
    }

    setBusy(true);
    setError(null);
    setAutoRun(false);

    try {
      const seed = parseOptionalInt(seedInput);
      const payload = await resetPlaySession(session.session_id, { seed });
      setSession(payload);
      prevInfoRef.current = null;
      void playCue("ui.click");
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
    } finally {
      setBusy(false);
    }
  }, [session?.session_id, seedInput, playCue]);

  const endSession = useCallback(async () => {
    const activeSessionId = session?.session_id;
    if (!activeSessionId) {
      return;
    }

    setBusy(true);
    setError(null);
    setAutoRun(false);

    try {
      await deletePlaySession(activeSessionId);
      setSession(null);
      void playCue("ui.click");
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
    } finally {
      setBusy(false);
    }
  }, [session?.session_id, playCue]);

  const performStep = useCallback(
    async (action: number) => {
      const activeSessionId = session?.session_id;
      if (!activeSessionId || steppingRef.current) {
        return;
      }

      steppingRef.current = true;
      setError(null);

      try {
        void playCue(actionCueKey(action));
        const payload = await stepPlaySession(activeSessionId, { action });
        setSession(payload);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        setError(message);
      } finally {
        steppingRef.current = false;
      }
    },
    [session?.session_id, playCue],
  );

  useEffect(() => {
    if (!session) {
      void startSession();
    }
  }, [session, startSession]);

  useEffect(() => {
    if (!autoRun || !session?.session_id || sessionDone) {
      return;
    }

    const intervalMs = Math.max(100, Math.floor(1000 / Math.max(stepsPerSecond, 1)));
    const timer = window.setInterval(() => {
      void performStep(selectedAction);
    }, intervalMs);

    return () => window.clearInterval(timer);
  }, [autoRun, session?.session_id, sessionDone, stepsPerSecond, selectedAction, performStep]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.repeat) {
        return;
      }

      const target = event.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag && ["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(tag)) {
        return;
      }

      const key = event.key.toLowerCase();
      const action = HOTKEY_ACTION_MAP.get(key);
      if (action === undefined || !session || busy || sessionDone) {
        return;
      }

      event.preventDefault();
      setSelectedAction(action);
      void performStep(action);
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [session, busy, sessionDone, performStep]);

  useEffect(() => {
    return () => {
      const activeSessionId = sessionIdRef.current;
      if (!activeSessionId) {
        return;
      }
      void deletePlaySession(activeSessionId).catch(() => {
        // Session cleanup is best-effort on unmount.
      });
    };
  }, []);

  const obsSummary = useMemo(() => decodeObsSummary(session?.obs), [session?.obs]);
  const info = (session?.info ?? {}) as Record<string, unknown>;

  const credits = Number(
    info.credits ?? info.net_profit ?? inferCreditsFromObs(session?.obs) ?? Number.NaN,
  );
  const netProfit = Number(info.net_profit ?? Number.NaN);
  const invalidAction = Boolean(info.invalid_action);

  return (
    <section className="gameplay-shell">
      <section className="gameplay-main">
        <article className="card gameplay-viewport-card stack">
          <div className="row">
            <h2>Human Pilot Mode</h2>
            <div className="list-inline">
              <span className="badge">session {session?.session_id?.slice(0, 8) ?? "-"}</span>
              <span className="badge">steps {session?.steps ?? 0}</span>
            </div>
          </div>
          <p className="muted">
            Fly the pixel simulation directly. RL policy training runs in the non-pixel state environment and
            replay output appears on the Replay tab.
          </p>

          <div className="player-toolbar">
            <button onClick={() => void startSession()} disabled={busy}>
              New Session
            </button>
            <button className="alt" onClick={() => void resetSession()} disabled={!session || busy}>
              Reset
            </button>
            <button className="warn" onClick={() => void endSession()} disabled={!session || busy}>
              End
            </button>
            <span className="spacer" />
            <button
              className={audioEnabled ? "warn" : "alt"}
              onClick={() => {
                setAudioEnabled((prev) => !prev);
                void playCue("ui.click");
              }}
            >
              {audioEnabled ? "Mute" : "Enable Audio"}
            </button>
          </div>

          <div className="metric-grid">
            <label>
              Selected Action
              <select
                value={selectedAction}
                onChange={(event) => {
                  setSelectedAction(Number(event.target.value));
                  void playCue("ui.select");
                }}
              >
                {ACTION_LABELS.map((label, index) => (
                  <option key={label} value={index}>
                    {index}: {label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Auto-run speed ({stepsPerSecond} steps/s)
              <input
                type="range"
                min={1}
                max={10}
                value={stepsPerSecond}
                onChange={(event) => setStepsPerSecond(Number(event.target.value))}
              />
            </label>
          </div>

          <div className="player-toolbar">
            <button onClick={() => void performStep(selectedAction)} disabled={!session || busy || sessionDone}>
              Step Selected
            </button>
            <button onClick={() => void performStep(HOLD_ACTION)} disabled={!session || busy || sessionDone}>
              Hold Drift
            </button>
            <button
              className={autoRun ? "warn" : "alt"}
              onClick={() => {
                setAutoRun((previous) => !previous);
                void playCue(autoRun ? "ui.pause" : "ui.play");
              }}
              disabled={!session || sessionDone}
            >
              {autoRun ? "Stop Auto" : "Start Auto"}
            </button>
          </div>

          {sessionDone ? <p className="notice">Episode finished. Reset or create a new session.</p> : null}
          {error ? <p className="notice error">{error}</p> : null}

          <SectorView
            mode="human"
            observation={session?.obs ?? null}
            action={session?.action ?? selectedAction}
            events={derivedEvents}
          />

          <div className="row">
            <span className="badge">
              last action {session?.action ?? "-"} ({session?.action !== undefined ? actionLabel(session.action) : "-"})
            </span>
            {invalidAction ? <span className="badge warn">Invalid action on last step</span> : null}
          </div>
          <div className="list-inline">
            {derivedEvents.map((eventName) => (
              <span key={eventName} className="badge warn">
                {eventName}
              </span>
            ))}
          </div>
        </article>

        <article className="card stack">
          <h3>How To Play</h3>
          <ol className="pilot-guide">
            <li>Scan first (`Wide` or `Focused`) and select a promising asteroid target.</li>
            <li>Mine (`Conservative`, `Standard`, or `Aggressive`) and watch heat/tool levels.</li>
            <li>Use cooldown/maintenance actions when risk rises to protect hull and cargo.</li>
            <li>Dock and sell cargo, then restock fuel/repair supplies before the next cycle.</li>
          </ol>

          <div className="hotkey-grid">
            {HOTKEY_BINDINGS.map((binding) => (
              <div key={binding.key} className="hotkey-item">
                <kbd>{binding.key.toUpperCase()}</kbd>
                {binding.hint}
              </div>
            ))}
          </div>

          <details className="advanced-panel">
            <summary>Advanced Session Controls</summary>
            <div className="stack">
              <p className="muted">Backend API: {backendBaseUrl()}</p>
              <label>
                Seed (optional)
                <input
                  value={seedInput}
                  onChange={(event) => setSeedInput(event.target.value)}
                  placeholder="12345"
                />
              </label>
              <label>
                Env time max (optional)
                <input
                  value={envTimeMaxInput}
                  onChange={(event) => setEnvTimeMaxInput(event.target.value)}
                  placeholder="20000"
                />
              </label>
            </div>
          </details>
        </article>

        <article className="card stack">
          <h3>Action Deck (All 69 Actions)</h3>
          <p className="muted">
            Actions are grouped by pilot intent. Clicking any action selects it and sends one immediate step.
          </p>
          <div className="action-group-grid">
            {ACTION_GROUPS.map((group) => (
              <div key={group.id} className="details-card action-group">
                <h3>{group.title}</h3>
                <p className="muted">{group.description}</p>
                <div className="action-buttons">
                  {group.actionIndexes.map((index) => {
                    const label = ACTION_LABELS[index];
                    return (
                      <button
                        key={label}
                        className={selectedAction === index ? "selected" : undefined}
                        onClick={() => {
                          setSelectedAction(index);
                          void performStep(index);
                        }}
                        disabled={!session || busy || sessionDone}
                      >
                        {index}: {label}
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </article>
      </section>

      <aside className="hud-rail">
        <article className="card">
          <h3>Gameplay HUD</h3>
          <dl className="kv">
            <dt>Fuel</dt>
            <dd>{formatNumber(obsSummary?.fuelPct, 1)}%</dd>
            <dt>Hull</dt>
            <dd>{formatNumber(obsSummary?.hullPct, 1)}%</dd>
            <dt>Heat</dt>
            <dd>{formatNumber(obsSummary?.heatPct, 1)}%</dd>
            <dt>Tool</dt>
            <dd>{formatNumber(obsSummary?.toolPct, 1)}%</dd>
            <dt>Cargo</dt>
            <dd>{formatNumber(obsSummary?.cargoPct, 1)}%</dd>
            <dt>Alert</dt>
            <dd>{formatNumber(obsSummary?.alertPct, 1)}%</dd>
            <dt>Credits</dt>
            <dd>{formatNumber(credits, 2)}</dd>
            <dt>Net Profit</dt>
            <dd>{formatNumber(netProfit, 2)}</dd>
          </dl>
        </article>

        <article className="card">
          <h3>Session State</h3>
          <dl className="kv">
            <dt>Terminated</dt>
            <dd>{String(session?.terminated ?? false)}</dd>
            <dt>Truncated</dt>
            <dd>{String(session?.truncated ?? false)}</dd>
            <dt>Time Remaining</dt>
            <dd>{formatNumber(Number(info.time_remaining ?? Number.NaN), 2)}</dd>
            <dt>Node Context</dt>
            <dd>{String(info.node_context ?? "-")}</dd>
            <dt>Pirate Encounters</dt>
            <dd>{formatNumber(Number(info.pirate_encounters ?? Number.NaN), 0)}</dd>
            <dt>Overheat Ticks</dt>
            <dd>{formatNumber(Number(info.overheat_ticks ?? Number.NaN), 0)}</dd>
            <dt>Survival</dt>
            <dd>{formatNumber(Number(info.survival ?? Number.NaN), 3)}</dd>
          </dl>
        </article>

        <article className="card">
          <h3>Cargo By Commodity</h3>
          {obsSummary ? (
            <dl className="kv">
              <dt>Iron</dt>
              <dd>{formatNumber(obsSummary.cargoByCommodity[0], 2)}</dd>
              <dt>Nickel</dt>
              <dd>{formatNumber(obsSummary.cargoByCommodity[1], 2)}</dd>
              <dt>Water Ice</dt>
              <dd>{formatNumber(obsSummary.cargoByCommodity[2], 2)}</dd>
              <dt>PGE</dt>
              <dd>{formatNumber(obsSummary.cargoByCommodity[3], 2)}</dd>
              <dt>Rare Isotopes</dt>
              <dd>{formatNumber(obsSummary.cargoByCommodity[4], 2)}</dd>
              <dt>Volatiles</dt>
              <dd>{formatNumber(obsSummary.cargoByCommodity[5], 2)}</dd>
              <dt>Time Budget</dt>
              <dd>{formatNumber(obsSummary.timePct, 2)}%</dd>
            </dl>
          ) : (
            <p className="muted">No observation payload yet.</p>
          )}
        </article>

        <article className="card">
          <h3>Live Info Payload</h3>
          <details className="advanced-panel">
            <summary>Expand JSON</summary>
            <pre className="frame-panel">{JSON.stringify(info, null, 2)}</pre>
          </details>
        </article>
      </aside>
    </section>
  );
}
