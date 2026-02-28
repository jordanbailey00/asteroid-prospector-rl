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
    <section className="panel-grid">
      <aside className="stack">
        <article className="card stack">
          <h2>Play Session</h2>
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
          <div className="row">
            <button onClick={() => void startSession()} disabled={busy}>
              New Session
            </button>
            <button className="alt" onClick={() => void resetSession()} disabled={!session || busy}>
              Reset
            </button>
            <button className="warn" onClick={() => void endSession()} disabled={!session || busy}>
              End
            </button>
          </div>
          <div className="row">
            <span className="badge">session: {session?.session_id?.slice(0, 8) ?? "-"}</span>
            <span className="badge">steps: {session?.steps ?? 0}</span>
          </div>
          {sessionDone ? <p className="notice">Episode finished. Reset or create a new session.</p> : null}
          {error ? <p className="notice error">{error}</p> : null}
        </article>

        <article className="card stack">
          <h2>Step Controls</h2>
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
          <div className="row">
            <button
              onClick={() => void performStep(selectedAction)}
              disabled={!session || busy || sessionDone}
            >
              Step Action
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
            <button
              onClick={() => void performStep(HOLD_ACTION)}
              disabled={!session || busy || sessionDone}
            >
              Hold
            </button>
          </div>
          <div className="row">
            <span className="muted">Audio cues</span>
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
          <p className="muted">
            Last action: {session?.action ?? "-"} (
            {session?.action !== undefined ? actionLabel(session.action) : "-"})
          </p>
          {invalidAction ? <span className="badge warn">Invalid action on last step</span> : null}
          <div className="list-inline">
            {derivedEvents.map((eventName) => (
              <span key={eventName} className="badge warn">
                {eventName}
              </span>
            ))}
          </div>
        </article>
      </aside>

      <section className="stack">
        <article className="card stack">
          <h2>Sector View</h2>
          <SectorView
            mode="human"
            observation={session?.obs ?? null}
            action={session?.action ?? selectedAction}
            events={derivedEvents}
          />
        </article>

        <article className="card stack">
          <h2>Live HUD</h2>
          <div className="metric-grid">
            <div className="metric-chip">
              Fuel
              <strong>{formatNumber(obsSummary?.fuelPct, 1)}%</strong>
            </div>
            <div className="metric-chip">
              Hull
              <strong>{formatNumber(obsSummary?.hullPct, 1)}%</strong>
            </div>
            <div className="metric-chip">
              Heat
              <strong>{formatNumber(obsSummary?.heatPct, 1)}%</strong>
            </div>
            <div className="metric-chip">
              Tool
              <strong>{formatNumber(obsSummary?.toolPct, 1)}%</strong>
            </div>
            <div className="metric-chip">
              Cargo
              <strong>{formatNumber(obsSummary?.cargoPct, 1)}%</strong>
            </div>
            <div className="metric-chip">
              Alert
              <strong>{formatNumber(obsSummary?.alertPct, 1)}%</strong>
            </div>
            <div className="metric-chip">
              Credits
              <strong>{formatNumber(credits, 2)}</strong>
            </div>
            <div className="metric-chip">
              Net Profit
              <strong>{formatNumber(netProfit, 2)}</strong>
            </div>
          </div>
        </article>

        <div className="data-board">
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

          <article className="card full">
            <h3>Info Payload</h3>
            <pre className="frame-panel">{JSON.stringify(info, null, 2)}</pre>
          </article>

          <article className="card full">
            <h3>Action Palette (0..68)</h3>
            <p className="muted">
              Click any action to select it and send one environment step immediately.
            </p>
            <div className="action-grid">
              {ACTION_LABELS.map((label, index) => (
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
              ))}
            </div>
          </article>
        </div>
      </section>
    </section>
  );
}
