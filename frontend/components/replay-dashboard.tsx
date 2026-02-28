"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { actionLabel } from "@/lib/actions";
import {
  backendBaseUrl,
  getMetricsWindows,
  getReplayFrames,
  getRun,
  listReplays,
  listRuns,
} from "@/lib/api";
import { formatNumber, formatTimestamp, toNumericSeries } from "@/lib/format";
import { ReplayEntry, ReplayFrame, RunDetailResponse, RunSummary, WindowMetric } from "@/lib/types";

import { Sparkline } from "@/components/sparkline";

type FrameHud = {
  fuelPct: number;
  hullPct: number;
  heatPct: number;
  toolPct: number;
  cargoPct: number;
  alertPct: number;
  nodeContext: string;
  timeRemaining: number;
  credits: number;
  netProfit: number;
  survival: number;
};

const CREDITS_CAP = 1.0e7;

function renderMetadataLinks(detail: RunDetailResponse | null) {
  if (!detail) {
    return null;
  }

  const wandbUrl = detail.metadata.wandb_run_url;
  const constellationUrl = detail.metadata.constellation_url;

  const hasWandb = typeof wandbUrl === "string" && wandbUrl.trim() !== "";
  const hasConstellation = typeof constellationUrl === "string" && constellationUrl.trim() !== "";

  if (!hasWandb && !hasConstellation) {
    return null;
  }

  return (
    <div className="list-inline">
      {hasWandb ? (
        <a className="external" href={wandbUrl} target="_blank" rel="noreferrer">
          Open W&B
        </a>
      ) : null}
      {hasConstellation ? (
        <a className="external" href={constellationUrl} target="_blank" rel="noreferrer">
          Open Constellation
        </a>
      ) : null}
    </div>
  );
}

function decodeFrameHud(frame: ReplayFrame | null): FrameHud | null {
  if (!frame || typeof frame.render_state !== "object" || frame.render_state === null) {
    return null;
  }

  const renderState = frame.render_state as Record<string, unknown>;
  const rawObs = renderState.observation;
  if (!Array.isArray(rawObs) || rawObs.length < 8) {
    return null;
  }

  const obs = rawObs.map((value) => Number(value));
  const invalidObs = obs.some((value) => !Number.isFinite(value));
  if (invalidObs) {
    return null;
  }

  const creditsEstimate = Math.expm1(obs[7] * Math.log1p(CREDITS_CAP));

  return {
    fuelPct: obs[0] * 100,
    hullPct: obs[1] * 100,
    heatPct: obs[2] * 100,
    toolPct: obs[3] * 100,
    cargoPct: obs[4] * 100,
    alertPct: obs[5] * 100,
    nodeContext: String(renderState.node_context ?? "unknown"),
    timeRemaining: Number(renderState.time_remaining ?? Number.NaN),
    credits: Number(renderState.credits ?? creditsEstimate),
    netProfit: Number(renderState.net_profit ?? Number.NaN),
    survival: Number(renderState.survival ?? Number.NaN),
  };
}

export function ReplayDashboard() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string>("");

  const [runDetail, setRunDetail] = useState<RunDetailResponse | null>(null);
  const [replays, setReplays] = useState<ReplayEntry[]>([]);
  const [metrics, setMetrics] = useState<WindowMetric[]>([]);

  const [selectedWindowId, setSelectedWindowId] = useState<string>("all");
  const [selectedReplayId, setSelectedReplayId] = useState<string>("");

  const [frames, setFrames] = useState<ReplayFrame[]>([]);
  const [frameError, setFrameError] = useState<string | null>(null);

  const [loadingRuns, setLoadingRuns] = useState(false);
  const [loadingRunData, setLoadingRunData] = useState(false);
  const [loadingFrames, setLoadingFrames] = useState(false);

  const [tagFilter, setTagFilter] = useState("every_window");
  const [fps, setFps] = useState(6);
  const [stride, setStride] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [cursor, setCursor] = useState(0);
  const [jumpTarget, setJumpTarget] = useState("1");

  const refreshRuns = useCallback(async () => {
    setLoadingRuns(true);
    try {
      const payload = await listRuns(80);
      setRuns(payload.runs);
      setRunsError(null);
      setSelectedRunId((current) => current || payload.runs[0]?.run_id || "");
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setRunsError(message);
    } finally {
      setLoadingRuns(false);
    }
  }, []);

  useEffect(() => {
    void refreshRuns();
    const timer = window.setInterval(() => {
      void refreshRuns();
    }, 20000);
    return () => window.clearInterval(timer);
  }, [refreshRuns]);

  useEffect(() => {
    if (!selectedRunId) {
      return;
    }

    setLoadingRunData(true);
    Promise.all([
      getRun(selectedRunId),
      listReplays(selectedRunId, {
        tag: tagFilter.trim() === "" ? undefined : tagFilter.trim(),
        limit: 4000,
      }),
      getMetricsWindows(selectedRunId, { limit: 5000, order: "desc" }),
    ])
      .then(([detail, replayPayload, metricsPayload]) => {
        setRunDetail(detail);
        setReplays(replayPayload.replays);
        setMetrics(metricsPayload.windows);

        const firstWindow = replayPayload.replays[0]?.window_id;
        setSelectedWindowId(firstWindow !== undefined ? String(firstWindow) : "all");
      })
      .catch((err: Error) => {
        setRunsError(err.message);
        setRunDetail(null);
        setReplays([]);
        setMetrics([]);
        setSelectedWindowId("all");
        setSelectedReplayId("");
      })
      .finally(() => setLoadingRunData(false));
  }, [selectedRunId, tagFilter]);

  const windowOptions = useMemo(() => {
    const values = new Set<number>();
    for (const replay of replays) {
      values.add(Number(replay.window_id));
    }
    return Array.from(values.values()).sort((a, b) => b - a);
  }, [replays]);

  const filteredReplays = useMemo(() => {
    if (selectedWindowId === "all") {
      return replays;
    }
    const windowId = Number(selectedWindowId);
    return replays.filter((entry) => Number(entry.window_id) === windowId);
  }, [replays, selectedWindowId]);

  useEffect(() => {
    if (filteredReplays.length === 0) {
      setSelectedReplayId("");
      return;
    }

    const exists = filteredReplays.some((entry) => entry.replay_id === selectedReplayId);
    if (!exists) {
      setSelectedReplayId(filteredReplays[0].replay_id);
    }
  }, [filteredReplays, selectedReplayId]);

  useEffect(() => {
    if (!selectedRunId || !selectedReplayId) {
      setFrames([]);
      setCursor(0);
      setJumpTarget("1");
      return;
    }

    setLoadingFrames(true);
    getReplayFrames(selectedRunId, selectedReplayId, { offset: 0, limit: 5000 })
      .then((payload) => {
        setFrames(payload.frames);
        setFrameError(null);
        setCursor(0);
        setJumpTarget("1");
      })
      .catch((err: Error) => {
        setFrameError(err.message);
        setFrames([]);
        setCursor(0);
      })
      .finally(() => setLoadingFrames(false));
  }, [selectedRunId, selectedReplayId]);

  useEffect(() => {
    if (!isPlaying || frames.length < 2) {
      return;
    }

    const delayMs = Math.max(40, Math.floor(1000 / Math.max(fps, 1)));
    const timer = window.setInterval(() => {
      setCursor((prev) => {
        const next = prev + Math.max(stride, 1);
        if (next >= frames.length - 1) {
          setIsPlaying(false);
          return frames.length - 1;
        }
        return next;
      });
    }, delayMs);

    return () => window.clearInterval(timer);
  }, [isPlaying, frames.length, fps, stride]);

  const selectedReplay = useMemo(
    () => filteredReplays.find((entry) => entry.replay_id === selectedReplayId) ?? null,
    [filteredReplays, selectedReplayId],
  );

  const currentFrame = frames[cursor] ?? null;
  const frameHud = useMemo(() => decodeFrameHud(currentFrame), [currentFrame]);

  const replayWindow = useMemo(() => {
    if (!selectedReplay) {
      return null;
    }
    return metrics.find((row) => Number(row.window_id) === Number(selectedReplay.window_id)) ?? null;
  }, [metrics, selectedReplay]);

  const rewardTrend = useMemo(
    () => toNumericSeries(metrics as Array<Record<string, unknown>>, "reward_mean").reverse(),
    [metrics],
  );

  const profitTrend = useMemo(
    () => toNumericSeries(metrics as Array<Record<string, unknown>>, "profit_mean").reverse(),
    [metrics],
  );

  const survivalTrend = useMemo(
    () => toNumericSeries(metrics as Array<Record<string, unknown>>, "survival_rate").reverse(),
    [metrics],
  );

  const jumpToFrame = useCallback(() => {
    const raw = Number(jumpTarget);
    if (!Number.isFinite(raw)) {
      return;
    }
    const frameIndex = Math.max(1, Math.min(Math.trunc(raw), frames.length)) - 1;
    setCursor(frameIndex);
  }, [jumpTarget, frames.length]);

  return (
    <section className="panel-grid">
      <aside className="stack">
        <article className="card stack">
          <h2>Replay Source</h2>
          <p className="muted">Backend API: {backendBaseUrl()}</p>
          <label>
            Run ID
            <select
              value={selectedRunId}
              onChange={(event) => setSelectedRunId(event.target.value)}
              disabled={loadingRuns || runs.length === 0}
            >
              {runs.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {run.run_id} ({run.status ?? "unknown"})
                </option>
              ))}
            </select>
          </label>
          <label>
            Replay tag filter
            <input
              value={tagFilter}
              onChange={(event) => setTagFilter(event.target.value)}
              placeholder="every_window"
            />
          </label>
          <label>
            Window ID
            <select
              value={selectedWindowId}
              onChange={(event) => setSelectedWindowId(event.target.value)}
              disabled={windowOptions.length === 0}
            >
              <option value="all">All windows</option>
              {windowOptions.map((windowId) => (
                <option key={windowId} value={String(windowId)}>
                  {windowId}
                </option>
              ))}
            </select>
          </label>
          <label>
            Replay ID
            <select
              value={selectedReplayId}
              onChange={(event) => setSelectedReplayId(event.target.value)}
              disabled={filteredReplays.length === 0}
            >
              {filteredReplays.map((replay) => (
                <option key={replay.replay_id} value={replay.replay_id}>
                  {replay.replay_id} (window {replay.window_id})
                </option>
              ))}
            </select>
          </label>
          <div className="row">
            <span className="badge">runs: {runs.length}</span>
            <span className="badge">replays: {filteredReplays.length}</span>
          </div>
          {loadingRuns || loadingRunData ? <p className="muted">Loading run data...</p> : null}
          {runsError ? <p className="notice error">{runsError}</p> : null}
          {renderMetadataLinks(runDetail)}
        </article>

        <article className="card stack">
          <h2>Playback Controls</h2>
          <label>
            FPS ({fps})
            <input
              type="range"
              min={1}
              max={20}
              value={fps}
              onChange={(event) => setFps(Number(event.target.value))}
            />
          </label>
          <label>
            Stride ({stride})
            <input
              type="range"
              min={1}
              max={15}
              value={stride}
              onChange={(event) => setStride(Number(event.target.value))}
            />
          </label>
          <label>
            Jump to frame
            <div className="row">
              <input
                type="number"
                min={1}
                max={Math.max(frames.length, 1)}
                value={jumpTarget}
                onChange={(event) => setJumpTarget(event.target.value)}
              />
              <button className="alt" onClick={jumpToFrame} disabled={frames.length === 0}>
                Jump
              </button>
            </div>
          </label>
          <div className="row">
            <button onClick={() => setIsPlaying((prev) => !prev)} disabled={frames.length <= 1}>
              {isPlaying ? "Pause" : "Play"}
            </button>
            <button
              className="alt"
              onClick={() => setCursor((prev) => Math.max(prev - 1, 0))}
              disabled={frames.length <= 1}
            >
              Back
            </button>
            <button
              className="alt"
              onClick={() => setCursor((prev) => Math.min(prev + 1, Math.max(frames.length - 1, 0)))}
              disabled={frames.length <= 1}
            >
              Step
            </button>
            <button className="alt" onClick={() => setCursor(0)} disabled={frames.length === 0}>
              Reset
            </button>
          </div>
          <div className="row">
            <span className="badge">frame {frames.length === 0 ? 0 : cursor + 1}</span>
            <span className="badge">total {frames.length}</span>
          </div>
          {loadingFrames ? <p className="muted">Loading frames...</p> : null}
          {frameError ? <p className="notice error">{frameError}</p> : null}
        </article>
      </aside>

      <section className="stack">
        <article className="card stack">
          <h2>Current Frame</h2>
          {currentFrame ? (
            <>
              <div className="row">
                <span className="badge">t={String(currentFrame.t ?? "-")}</span>
                <span className="badge">dt={String(currentFrame.dt ?? "-")}</span>
                <span className="badge">reward={formatNumber(currentFrame.reward, 3)}</span>
              </div>
              <p>
                <strong>Action:</strong> {actionLabel(Number(currentFrame.action))} ({String(currentFrame.action)})
              </p>
              <div className="list-inline">
                {(currentFrame.events ?? []).map((eventName) => (
                  <span key={eventName} className="badge warn">
                    {eventName}
                  </span>
                ))}
              </div>
              <pre className="frame-panel">{JSON.stringify(currentFrame.render_state ?? {}, null, 2)}</pre>
            </>
          ) : (
            <p className="muted">No frame selected.</p>
          )}
        </article>

        <div className="data-board">
          <article className="card">
            <h3>Frame HUD</h3>
            {frameHud ? (
              <dl className="kv">
                <dt>Fuel</dt>
                <dd>{formatNumber(frameHud.fuelPct, 1)}%</dd>
                <dt>Hull</dt>
                <dd>{formatNumber(frameHud.hullPct, 1)}%</dd>
                <dt>Heat</dt>
                <dd>{formatNumber(frameHud.heatPct, 1)}%</dd>
                <dt>Tool</dt>
                <dd>{formatNumber(frameHud.toolPct, 1)}%</dd>
                <dt>Cargo</dt>
                <dd>{formatNumber(frameHud.cargoPct, 1)}%</dd>
                <dt>Alert</dt>
                <dd>{formatNumber(frameHud.alertPct, 1)}%</dd>
                <dt>Credits</dt>
                <dd>{formatNumber(frameHud.credits, 2)}</dd>
                <dt>Net Profit</dt>
                <dd>{formatNumber(frameHud.netProfit, 2)}</dd>
                <dt>Survival</dt>
                <dd>{formatNumber(frameHud.survival, 3)}</dd>
                <dt>Node Context</dt>
                <dd>{frameHud.nodeContext}</dd>
                <dt>Time Remaining</dt>
                <dd>{formatNumber(frameHud.timeRemaining, 2)}</dd>
              </dl>
            ) : (
              <p className="muted">Replay frame has no decodable render_state payload.</p>
            )}
          </article>

          <article className="card">
            <h3>Window Summary</h3>
            {selectedReplay && replayWindow ? (
              <dl className="kv">
                <dt>Window ID</dt>
                <dd>{String(replayWindow.window_id)}</dd>
                <dt>Return</dt>
                <dd>{formatNumber(selectedReplay.return_total, 3)}</dd>
                <dt>Profit</dt>
                <dd>{formatNumber(selectedReplay.profit, 3)}</dd>
                <dt>Survival</dt>
                <dd>{formatNumber(selectedReplay.survival, 3)}</dd>
                <dt>Reward Mean</dt>
                <dd>{formatNumber(replayWindow.reward_mean)}</dd>
                <dt>Profit Mean</dt>
                <dd>{formatNumber(replayWindow.profit_mean)}</dd>
                <dt>Survival Rate</dt>
                <dd>{formatNumber(replayWindow.survival_rate)}</dd>
              </dl>
            ) : (
              <p className="muted">Choose a replay to inspect its window metrics.</p>
            )}
          </article>

          <article className="card">
            <h3>Replay Metadata</h3>
            {selectedReplay ? (
              <>
                <dl className="kv">
                  <dt>Replay ID</dt>
                  <dd>{selectedReplay.replay_id}</dd>
                  <dt>Window</dt>
                  <dd>{selectedReplay.window_id}</dd>
                  <dt>Steps</dt>
                  <dd>{selectedReplay.steps}</dd>
                  <dt>Created</dt>
                  <dd>{formatTimestamp(selectedReplay.created_at)}</dd>
                </dl>
                <div className="list-inline">
                  {selectedReplay.tags.map((tag) => (
                    <span key={tag} className="badge">
                      {tag}
                    </span>
                  ))}
                </div>
              </>
            ) : (
              <p className="muted">No replay selected.</p>
            )}
          </article>

          <article className="card full">
            <h3>Recent Trends</h3>
            <div className="metric-grid">
              <div className="metric-chip">
                Reward Mean
                <strong>{formatNumber(rewardTrend[rewardTrend.length - 1])}</strong>
                <Sparkline values={rewardTrend} stroke="#1f8a70" />
              </div>
              <div className="metric-chip">
                Profit Mean
                <strong>{formatNumber(profitTrend[profitTrend.length - 1])}</strong>
                <Sparkline values={profitTrend} stroke="#db3e00" />
              </div>
              <div className="metric-chip">
                Survival Rate
                <strong>{formatNumber(survivalTrend[survivalTrend.length - 1], 3)}</strong>
                <Sparkline values={survivalTrend} stroke="#13547a" />
              </div>
            </div>
          </article>
        </div>
      </section>
    </section>
  );
}
