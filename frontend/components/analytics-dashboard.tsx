"use client";

import { useEffect, useMemo, useState } from "react";

import {
  backendBaseUrl,
  getAnalyticsCompleteness,
  getMetricsWindows,
  getRun,
  getWandbIterationView,
  listReplays,
  listRuns,
  listWandbLatestRuns,
} from "@/lib/api";
import { formatNumber, formatTimestamp, toNumericSeries } from "@/lib/format";
import {
  AnalyticsCompletenessResponse,
  AnalyticsCoverageRow,
  ReplayEntry,
  RunDetailResponse,
  RunSummary,
  WandbIterationViewResponse,
  WandbRunLite,
  WindowMetric,
} from "@/lib/types";

import { Sparkline } from "@/components/sparkline";

type TimelineRow = {
  windowId: number;
  replayCount: number;
  hasBest: boolean;
  milestoneTags: string[];
};

function asFiniteNumber(value: unknown): number | undefined {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : undefined;
}

function coverageBadgeClass(status: string): string {
  if (status === "ok") {
    return "badge";
  }
  if (status === "stale") {
    return "badge stale";
  }
  if (status === "missing") {
    return "badge warn";
  }
  return "badge error";
}

function coverageSummaryMessage(status: string): string {
  if (status === "ok") {
    return "All required analytics sources are complete and current.";
  }
  if (status === "stale") {
    return "Analytics sources are complete but stale against the configured freshness window.";
  }
  if (status === "missing") {
    return "Some analytics sources or required fields are missing.";
  }
  return "One or more analytics sources returned errors.";
}

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

function latestWandbValue(kpis: Record<string, unknown>, key: string): number | undefined {
  return asFiniteNumber(kpis[key]);
}

function renderMissingFields(row: AnalyticsCoverageRow): string {
  if (row.missing_fields.length === 0) {
    return "none";
  }
  return row.missing_fields.join(", ");
}

function renderCoverageNotes(row: AnalyticsCoverageRow): string {
  if (row.notes.length === 0) {
    return "none";
  }
  return row.notes.join(" | ");
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

  const [wandbRuns, setWandbRuns] = useState<WandbRunLite[]>([]);
  const [selectedWandbRunId, setSelectedWandbRunId] = useState("");
  const [wandbIterationView, setWandbIterationView] = useState<WandbIterationViewResponse | null>(null);
  const [wandbLoading, setWandbLoading] = useState(false);
  const [wandbError, setWandbError] = useState<string | null>(null);

  const [completeness, setCompleteness] = useState<AnalyticsCompletenessResponse | null>(null);
  const [completenessLoading, setCompletenessLoading] = useState(false);
  const [completenessError, setCompletenessError] = useState<string | null>(null);

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
    listWandbLatestRuns({ limit: 10 })
      .then((payload) => {
        setWandbRuns(payload.runs);
        setWandbError(null);
        setSelectedWandbRunId((current) => {
          if (current !== "") {
            return current;
          }
          const first = payload.runs[0];
          return first ? first.run_id : "";
        });
      })
      .catch((err: Error) => {
        setWandbRuns([]);
        setWandbIterationView(null);
        setWandbError(err.message);
      });
  }, []);

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

  useEffect(() => {
    if (!selectedWandbRunId) {
      setWandbIterationView(null);
      return;
    }

    setWandbLoading(true);
    getWandbIterationView(selectedWandbRunId, {
      keys: [
        "_step",
        "window_id",
        "env_steps_total",
        "reward_mean",
        "return_mean",
        "profit_mean",
        "survival_rate",
      ],
      maxPoints: 1000,
    })
      .then((payload) => {
        setWandbIterationView(payload);
        setWandbError(null);
      })
      .catch((err: Error) => {
        setWandbIterationView(null);
        setWandbError(err.message);
      })
      .finally(() => setWandbLoading(false));
  }, [selectedWandbRunId]);

  useEffect(() => {
    if (!selectedRunId) {
      setCompleteness(null);
      setCompletenessError(null);
      return;
    }

    setCompletenessLoading(true);
    getAnalyticsCompleteness(selectedRunId, {
      staleAfterSeconds: 21600,
      wandbRunId: selectedWandbRunId || undefined,
      wandbHistoryMaxPoints: 1000,
    })
      .then((payload) => {
        setCompleteness(payload);
        setCompletenessError(null);
      })
      .catch((err: Error) => {
        setCompleteness(null);
        setCompletenessError(err.message);
      })
      .finally(() => setCompletenessLoading(false));
  }, [selectedRunId, selectedWandbRunId]);

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

  const wandbHistoryRows = useMemo(
    () => (wandbIterationView ? wandbIterationView.history.rows : []),
    [wandbIterationView],
  );
  const wandbReturnTrend = useMemo(
    () => toNumericSeries(wandbHistoryRows, "return_mean"),
    [wandbHistoryRows],
  );
  const wandbProfitTrend = useMemo(
    () => toNumericSeries(wandbHistoryRows, "profit_mean"),
    [wandbHistoryRows],
  );
  const wandbSurvivalTrend = useMemo(
    () => toNumericSeries(wandbHistoryRows, "survival_rate"),
    [wandbHistoryRows],
  );

  const timeline = useMemo(() => buildReplayTimeline(replays), [replays]);
  const latestWindow = metrics[metrics.length - 1] ?? null;
  const recentRows = metrics.slice(Math.max(metrics.length - 10, 0)).reverse();
  const wandbKpis = wandbIterationView?.kpis ?? {};
  const coverageRows = completeness?.coverage ?? [];
  const coverageCounts = completeness?.status_counts ?? {
    ok: 0,
    stale: 0,
    missing: 0,
    error: 0,
  };

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

      <article className="card stack">
        <h3>W&B Iteration Drilldown</h3>
        <div className="metric-grid">
          <label>
            Last 10 Iterations
            <select
              value={selectedWandbRunId}
              onChange={(event) => setSelectedWandbRunId(event.target.value)}
              disabled={wandbRuns.length === 0}
            >
              {wandbRuns.length === 0 ? <option value="">No iterations</option> : null}
              {wandbRuns.map((run) => (
                <option key={run.run_id} value={run.run_id}>
                  {run.run_id}
                </option>
              ))}
            </select>
          </label>
          <div className="metric-chip">
            Selected
            <strong>{selectedWandbRunId || "none"}</strong>
            {wandbIterationView?.run?.url ? (
              <a className="external" href={String(wandbIterationView.run.url)} target="_blank" rel="noreferrer">
                Open W&B Run
              </a>
            ) : null}
          </div>
        </div>
        {wandbLoading ? <p className="muted">Loading W&B iteration view...</p> : null}
        {wandbError ? <p className="notice error">{wandbError}</p> : null}
      </article>

      <div className="data-board">
        <article className="card full stack">
          <h3>Analytics Coverage Contract</h3>
          {completeness ? (
            <>
              <div className="list-inline">
                <span className={coverageBadgeClass(completeness.overall_status)}>
                  overall: {completeness.overall_status}
                </span>
                <span className="badge">ok {coverageCounts.ok}</span>
                <span className="badge stale">stale {coverageCounts.stale}</span>
                <span className="badge warn">missing {coverageCounts.missing}</span>
                <span className="badge error">error {coverageCounts.error}</span>
              </div>
              {completeness.overall_status === "ok" ? (
                <p className="muted">{coverageSummaryMessage(completeness.overall_status)}</p>
              ) : (
                <p className={`notice ${completeness.overall_status === "error" ? "error" : ""}`}>
                  {coverageSummaryMessage(completeness.overall_status)}
                </p>
              )}
              <div className="coverage-table-wrap">
                <table className="coverage-table">
                  <thead>
                    <tr>
                      <th>Source</th>
                      <th>Status</th>
                      <th>Observed</th>
                      <th>Missing Fields</th>
                      <th>Lineage</th>
                      <th>Notes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {coverageRows.map((row) => (
                      <tr key={row.key}>
                        <td>
                          <strong>{row.label}</strong>
                          <div className="muted">{row.key}</div>
                        </td>
                        <td>
                          <span className={coverageBadgeClass(row.status)}>{row.status}</span>
                        </td>
                        <td>{String(row.observed_count)}</td>
                        <td>{renderMissingFields(row)}</td>
                        <td>
                          <div className="muted">{row.lineage.source}</div>
                          <div className="mono-small">{row.lineage.path ?? "n/a"}</div>
                          <div className="muted">{formatTimestamp(row.lineage.updated_at)}</div>
                        </td>
                        <td>{renderCoverageNotes(row)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <p className="muted">No completeness payload loaded.</p>
          )}
          {completenessLoading ? <p className="muted">Loading completeness contract...</p> : null}
          {completenessError ? <p className="notice error">{completenessError}</p> : null}
        </article>

        <article className="card full stack">
          <h3>Run Lineage and Training Context</h3>
          {completeness ? (
            <div className="metric-grid">
              <dl className="kv">
                <dt>Trainer Backend</dt>
                <dd>{String(completeness.run_context.trainer_backend ?? "n/a")}</dd>
                <dt>Run Status</dt>
                <dd>{String(completeness.run_context.status ?? "n/a")}</dd>
                <dt>Started</dt>
                <dd>{formatTimestamp(completeness.run_context.started_at)}</dd>
                <dt>Updated</dt>
                <dd>{formatTimestamp(completeness.run_context.updated_at)}</dd>
                <dt>Finished</dt>
                <dd>{formatTimestamp(completeness.run_context.finished_at)}</dd>
                <dt>Run Config Path</dt>
                <dd>{String(completeness.run_context.run_config_path ?? "n/a")}</dd>
                <dt>Metrics Path</dt>
                <dd>{String(completeness.run_context.metrics_windows_path ?? "n/a")}</dd>
                <dt>Replay Index Path</dt>
                <dd>{String(completeness.run_context.replay_index_path ?? "n/a")}</dd>
              </dl>
              <dl className="kv">
                <dt>W&B Entity</dt>
                <dd>{String(completeness.wandb_scope.entity ?? "n/a")}</dd>
                <dt>W&B Project</dt>
                <dd>{String(completeness.wandb_scope.project ?? "n/a")}</dd>
                <dt>W&B Run ID</dt>
                <dd>{String(completeness.wandb_scope.run_id ?? "n/a")}</dd>
                <dt>Scope Error</dt>
                <dd>{String(completeness.wandb_scope.scope_error ?? "none")}</dd>
                <dt>W&B URL</dt>
                <dd>{String(completeness.run_context.wandb_run_url ?? "n/a")}</dd>
                <dt>Constellation URL</dt>
                <dd>{String(completeness.run_context.constellation_url ?? "n/a")}</dd>
              </dl>
            </div>
          ) : (
            <p className="muted">Lineage context unavailable.</p>
          )}
        </article>

        <article className="card full">
          <h3>Iteration KPI Snapshot</h3>
          {wandbIterationView ? (
            <div className="metric-grid">
              <div className="metric-chip">
                Return Mean
                <strong>{formatNumber(latestWandbValue(wandbKpis, "return_mean"))}</strong>
                <Sparkline values={wandbReturnTrend} stroke="#db3e00" />
              </div>
              <div className="metric-chip">
                Profit Mean
                <strong>{formatNumber(latestWandbValue(wandbKpis, "profit_mean"))}</strong>
                <Sparkline values={wandbProfitTrend} stroke="#1f8a70" />
              </div>
              <div className="metric-chip">
                Survival Rate
                <strong>{formatNumber(latestWandbValue(wandbKpis, "survival_rate"), 3)}</strong>
                <Sparkline values={wandbSurvivalTrend} stroke="#13547a" />
              </div>
              <div className="metric-chip">
                Env Steps Total
                <strong>{formatNumber(latestWandbValue(wandbKpis, "env_steps_total"), 0)}</strong>
              </div>
            </div>
          ) : (
            <p className="muted">W&B iteration data unavailable.</p>
          )}
        </article>

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
