"use client";

import { useEffect, useMemo, useState } from "react";

import { backendBaseUrl, getMetricsWindows, getRun, listReplays, listRuns } from "@/lib/api";
import { formatNumber, formatTimestamp, toNumericSeries } from "@/lib/format";
import { ReplayEntry, RunDetailResponse, RunSummary, WindowMetric } from "@/lib/types";

import { Sparkline } from "@/components/sparkline";

type TimelineRow = {
  windowId: number;
  replayCount: number;
  hasBest: boolean;
  milestoneTags: string[];
};

function renderLinks(detail: RunDetailResponse | null) {
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

function buildReplayTimeline(entries: ReplayEntry[]): TimelineRow[] {
  const byWindow = new Map<number, TimelineRow>();

  for (const replay of entries) {
    const windowId = Number(replay.window_id);
    const tags = Array.isArray(replay.tags) ? replay.tags : [];

    const existing = byWindow.get(windowId);
    if (!existing) {
      byWindow.set(windowId, {
        windowId,
        replayCount: 1,
        hasBest: tags.includes("best_so_far"),
        milestoneTags: tags.filter((tag) => tag.startsWith("milestone:")),
      });
      continue;
    }

    existing.replayCount += 1;
    existing.hasBest = existing.hasBest || tags.includes("best_so_far");
    for (const tag of tags) {
      if (tag.startsWith("milestone:") && !existing.milestoneTags.includes(tag)) {
        existing.milestoneTags.push(tag);
      }
    }
  }

  return Array.from(byWindow.values()).sort((a, b) => b.windowId - a.windowId);
}

export function AnalyticsDashboard() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [compareRunId, setCompareRunId] = useState("");

  const [runDetail, setRunDetail] = useState<RunDetailResponse | null>(null);
  const [compareRunDetail, setCompareRunDetail] = useState<RunDetailResponse | null>(null);

  const [metrics, setMetrics] = useState<WindowMetric[]>([]);
  const [compareMetrics, setCompareMetrics] = useState<WindowMetric[]>([]);
  const [replays, setReplays] = useState<ReplayEntry[]>([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listRuns(100)
      .then((payload) => {
        setRuns(payload.runs);
        if (payload.runs.length > 0 && selectedRunId === "") {
          setSelectedRunId(payload.runs[0].run_id);
        }
      })
      .catch((err: Error) => setError(err.message));
  }, [selectedRunId]);

  useEffect(() => {
    if (!selectedRunId) {
      return;
    }

    setLoading(true);
    setError(null);

    Promise.all([
      getRun(selectedRunId),
      getMetricsWindows(selectedRunId, { limit: 5000, order: "asc" }),
      listReplays(selectedRunId, { limit: 4000 }),
    ])
      .then(([detail, metricsPayload, replayPayload]) => {
        setRunDetail(detail);
        setMetrics(metricsPayload.windows);
        setReplays(replayPayload.replays);
      })
      .catch((err: Error) => {
        setError(err.message);
        setRunDetail(null);
        setMetrics([]);
        setReplays([]);
      })
      .finally(() => setLoading(false));
  }, [selectedRunId]);

  useEffect(() => {
    if (!compareRunId) {
      setCompareMetrics([]);
      setCompareRunDetail(null);
      return;
    }

    Promise.all([
      getRun(compareRunId),
      getMetricsWindows(compareRunId, { limit: 5000, order: "asc" }),
    ])
      .then(([detail, payload]) => {
        setCompareRunDetail(detail);
        setCompareMetrics(payload.windows);
      })
      .catch((err: Error) => setError(err.message));
  }, [compareRunId]);

  const rewardTrend = useMemo(
    () => toNumericSeries(metrics as Array<Record<string, unknown>>, "reward_mean"),
    [metrics],
  );
  const returnTrend = useMemo(
    () => toNumericSeries(metrics as Array<Record<string, unknown>>, "return_mean"),
    [metrics],
  );
  const profitTrend = useMemo(
    () => toNumericSeries(metrics as Array<Record<string, unknown>>, "profit_mean"),
    [metrics],
  );
  const survivalTrend = useMemo(
    () => toNumericSeries(metrics as Array<Record<string, unknown>>, "survival_rate"),
    [metrics],
  );
  const overheatTrend = useMemo(
    () => toNumericSeries(metrics as Array<Record<string, unknown>>, "overheat_ticks_mean"),
    [metrics],
  );
  const pirateTrend = useMemo(
    () => toNumericSeries(metrics as Array<Record<string, unknown>>, "pirate_encounters_mean"),
    [metrics],
  );
  const valueLostTrend = useMemo(
    () => toNumericSeries(metrics as Array<Record<string, unknown>>, "value_lost_to_pirates_mean"),
    [metrics],
  );
  const miningTrend = useMemo(
    () => toNumericSeries(metrics as Array<Record<string, unknown>>, "mining_ticks_mean"),
    [metrics],
  );
  const scanTrend = useMemo(
    () => toNumericSeries(metrics as Array<Record<string, unknown>>, "scan_count_mean"),
    [metrics],
  );

  const compareReturnTrend = useMemo(
    () => toNumericSeries(compareMetrics as Array<Record<string, unknown>>, "return_mean"),
    [compareMetrics],
  );
  const compareProfitTrend = useMemo(
    () => toNumericSeries(compareMetrics as Array<Record<string, unknown>>, "profit_mean"),
    [compareMetrics],
  );
  const compareSurvivalTrend = useMemo(
    () => toNumericSeries(compareMetrics as Array<Record<string, unknown>>, "survival_rate"),
    [compareMetrics],
  );

  const timeline = useMemo(() => buildReplayTimeline(replays), [replays]);
  const latestWindow = metrics[metrics.length - 1] ?? null;
  const recentRows = metrics.slice(Math.max(metrics.length - 10, 0)).reverse();

  return (
    <section className="stack">
      <article className="card stack">
        <h2>Historical Analytics</h2>
        <div className="row">
          <p className="muted">Backend API: {backendBaseUrl()}</p>
          <span className="badge">windows: {metrics.length}</span>
        </div>
        <div className="metric-grid">
          <label>
            Primary Run
            <select value={selectedRunId} onChange={(event) => setSelectedRunId(event.target.value)}>
              {runs.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {run.run_id} ({run.status ?? "unknown"})
                </option>
              ))}
            </select>
          </label>
          <label>
            Compare Run (optional)
            <select value={compareRunId} onChange={(event) => setCompareRunId(event.target.value)}>
              <option value="">None</option>
              {runs
                .filter((run) => run.run_id !== selectedRunId)
                .map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    {run.run_id}
                  </option>
                ))}
            </select>
          </label>
        </div>
        {renderLinks(runDetail)}
        {compareRunDetail ? <p className="muted">Comparing against: {compareRunDetail.run_id}</p> : null}
        {loading ? <p className="muted">Loading metrics...</p> : null}
        {error ? <p className="notice error">{error}</p> : null}
      </article>

      <div className="data-board">
        <article className="card">
          <h3>Primary Trends</h3>
          <div className="metric-grid">
            <div className="metric-chip">
              Return Mean
              <strong>{formatNumber(returnTrend[returnTrend.length - 1])}</strong>
              <Sparkline values={returnTrend} stroke="#db3e00" />
            </div>
            <div className="metric-chip">
              Profit Mean
              <strong>{formatNumber(profitTrend[profitTrend.length - 1])}</strong>
              <Sparkline values={profitTrend} stroke="#1f8a70" />
            </div>
            <div className="metric-chip">
              Survival Rate
              <strong>{formatNumber(survivalTrend[survivalTrend.length - 1], 3)}</strong>
              <Sparkline values={survivalTrend} stroke="#13547a" />
            </div>
            <div className="metric-chip">
              Reward Mean
              <strong>{formatNumber(rewardTrend[rewardTrend.length - 1])}</strong>
              <Sparkline values={rewardTrend} stroke="#5c3c92" />
            </div>
          </div>
        </article>

        <article className="card">
          <h3>Risk and Efficiency</h3>
          <div className="metric-grid">
            <div className="metric-chip">
              Overheat Ticks Mean
              <strong>{formatNumber(overheatTrend[overheatTrend.length - 1])}</strong>
              <Sparkline values={overheatTrend} stroke="#b00020" />
            </div>
            <div className="metric-chip">
              Pirate Encounters Mean
              <strong>{formatNumber(pirateTrend[pirateTrend.length - 1])}</strong>
              <Sparkline values={pirateTrend} stroke="#8a4f07" />
            </div>
            <div className="metric-chip">
              Value Lost to Pirates Mean
              <strong>{formatNumber(valueLostTrend[valueLostTrend.length - 1])}</strong>
              <Sparkline values={valueLostTrend} stroke="#7f5539" />
            </div>
            <div className="metric-chip">
              Mining Ticks Mean
              <strong>{formatNumber(miningTrend[miningTrend.length - 1])}</strong>
              <Sparkline values={miningTrend} stroke="#216869" />
            </div>
            <div className="metric-chip">
              Scan Count Mean
              <strong>{formatNumber(scanTrend[scanTrend.length - 1])}</strong>
              <Sparkline values={scanTrend} stroke="#3f84e5" />
            </div>
          </div>
        </article>

        <article className="card full">
          <h3>Run Comparison</h3>
          {compareRunId ? (
            <div className="metric-grid">
              <div className="metric-chip">
                Return Mean ({selectedRunId})
                <strong>{formatNumber(returnTrend[returnTrend.length - 1])}</strong>
                <Sparkline values={returnTrend} stroke="#db3e00" />
              </div>
              <div className="metric-chip">
                Return Mean ({compareRunId})
                <strong>{formatNumber(compareReturnTrend[compareReturnTrend.length - 1])}</strong>
                <Sparkline values={compareReturnTrend} stroke="#5b8c5a" />
              </div>
              <div className="metric-chip">
                Profit Mean ({selectedRunId})
                <strong>{formatNumber(profitTrend[profitTrend.length - 1])}</strong>
                <Sparkline values={profitTrend} stroke="#1f8a70" />
              </div>
              <div className="metric-chip">
                Profit Mean ({compareRunId})
                <strong>{formatNumber(compareProfitTrend[compareProfitTrend.length - 1])}</strong>
                <Sparkline values={compareProfitTrend} stroke="#216869" />
              </div>
              <div className="metric-chip">
                Survival Rate ({selectedRunId})
                <strong>{formatNumber(survivalTrend[survivalTrend.length - 1], 3)}</strong>
                <Sparkline values={survivalTrend} stroke="#13547a" />
              </div>
              <div className="metric-chip">
                Survival Rate ({compareRunId})
                <strong>{formatNumber(compareSurvivalTrend[compareSurvivalTrend.length - 1], 3)}</strong>
                <Sparkline values={compareSurvivalTrend} stroke="#4e79a7" />
              </div>
            </div>
          ) : (
            <p className="muted">Select a comparison run to overlay key metrics.</p>
          )}
        </article>

        <article className="card">
          <h3>Checkpoint and Replay Timeline</h3>
          {timeline.length > 0 ? (
            <div className="stack">
              {timeline.slice(0, 18).map((row) => (
                <div key={row.windowId} className="metric-chip">
                  <div className="row">
                    <span>Window {row.windowId}</span>
                    <span className="badge">replays {row.replayCount}</span>
                  </div>
                  <div className="list-inline">
                    {row.hasBest ? <span className="badge">best_so_far</span> : null}
                    {row.milestoneTags.map((tag) => (
                      <span key={tag} className="badge warn">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">No replay index rows available for this run.</p>
          )}
        </article>

        <article className="card">
          <h3>Latest Window Snapshot</h3>
          {latestWindow ? (
            <dl className="kv">
              <dt>Window ID</dt>
              <dd>{String(latestWindow.window_id)}</dd>
              <dt>Env Steps Total</dt>
              <dd>{formatNumber(latestWindow.env_steps_total, 0)}</dd>
              <dt>Return Mean</dt>
              <dd>{formatNumber(latestWindow.return_mean)}</dd>
              <dt>Profit Mean</dt>
              <dd>{formatNumber(latestWindow.profit_mean)}</dd>
              <dt>Survival Rate</dt>
              <dd>{formatNumber(latestWindow.survival_rate, 3)}</dd>
            </dl>
          ) : (
            <p className="muted">No windows logged yet.</p>
          )}
          <p className="muted">Updated: {formatTimestamp(runDetail?.metadata.updated_at)}</p>
        </article>

        <article className="card full">
          <h3>Recent Windows</h3>
          {recentRows.length > 0 ? (
            <div className="stack">
              {recentRows.map((row) => (
                <div key={String(row.window_id)} className="metric-chip">
                  <div className="row">
                    <strong>Window {String(row.window_id)}</strong>
                    <span className="badge">env {formatNumber(row.env_steps_total, 0)}</span>
                  </div>
                  <div className="row">
                    <span>return {formatNumber(row.return_mean)}</span>
                    <span>profit {formatNumber(row.profit_mean)}</span>
                    <span>survival {formatNumber(row.survival_rate, 3)}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">No rows to display.</p>
          )}
        </article>
      </div>
    </section>
  );
}
