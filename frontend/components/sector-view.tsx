"use client";

import { useEffect, useMemo, useState } from "react";

import { GraphicsManifest, loadGraphicsManifest } from "@/lib/assets";
import { parseObservation } from "@/lib/observation";
import { actionVfxKey, eventVfxKey } from "@/lib/presentation";

type SectorViewProps = {
  mode: "agent" | "human";
  observation: number[] | null;
  action: number | null;
  events: string[];
};

function frameColor(manifest: GraphicsManifest | null, key: string, fallback: string): string {
  return manifest?.frames[key]?.color ?? fallback;
}

function unique<T>(values: T[]): T[] {
  return Array.from(new Set(values));
}

export function SectorView({ mode, observation, action, events }: SectorViewProps) {
  const [manifest, setManifest] = useState<GraphicsManifest | null>(null);

  useEffect(() => {
    void loadGraphicsManifest().then((payload) => setManifest(payload));
  }, []);

  const parsed = useMemo(() => parseObservation(observation), [observation]);
  const actionVfx = useMemo(() => actionVfxKey(action ?? -1), [action]);
  const eventVfx = useMemo(() => unique(events.map((name) => eventVfxKey(name))), [events]);

  if (!parsed) {
    return <p className="muted">No observation available for sector rendering.</p>;
  }

  const asteroidColorSmall = frameColor(manifest, "entity.asteroid.small", "#9ca3af");
  const asteroidColorMedium = frameColor(manifest, "entity.asteroid.medium", "#78716c");
  const asteroidColorLarge = frameColor(manifest, "entity.asteroid.large", "#57534e");
  const shipKey = mode === "human" ? "entity.ship.human" : "entity.ship.agent";

  const validAsteroids = parsed.asteroids.filter((asteroid) => asteroid.valid);
  const selectedAsteroid = validAsteroids.find((asteroid) => asteroid.selected) ?? null;
  const selectedAngle = selectedAsteroid ? (Math.PI * 2 * selectedAsteroid.index) / 16 : null;
  const selectedX = selectedAngle === null ? 248 : 180 + Math.cos(selectedAngle) * 76;
  const selectedY = selectedAngle === null ? 126 : 130 + Math.sin(selectedAngle) * 76;

  return (
    <div className="scene-shell">
      <div className="scene-grid">
        <div className="scene-viewport">
          <svg viewBox="0 0 360 260" className="scene-canvas" role="img" aria-label="sector view">
            <defs>
              <radialGradient id="sectorGlow" cx="50%" cy="50%" r="65%">
                <stop offset="0%" stopColor="#1e293b" stopOpacity="0.4" />
                <stop offset="100%" stopColor="#020617" stopOpacity="0.95" />
              </radialGradient>
            </defs>

            <rect x="0" y="0" width="360" height="260" rx="14" fill="url(#sectorGlow)" />

            {parsed.nodeType === "hazard" ? (
              <rect
                x="4"
                y="4"
                width="352"
                height="252"
                rx="10"
                className="scene-alert-border"
              />
            ) : null}

            {validAsteroids.map((asteroid) => {
              const angle = (Math.PI * 2 * asteroid.index) / 16;
              const radius = 72 + (1 - asteroid.scanConf) * 20;
              const x = 180 + Math.cos(angle) * radius;
              const y = 130 + Math.sin(angle) * radius;
              const size = 5 + (1 - asteroid.depletion) * 8;
              const fill =
                size > 10
                  ? asteroidColorLarge
                  : size > 7
                    ? asteroidColorMedium
                    : asteroidColorSmall;

              return (
                <g key={`ast-${asteroid.index}`}>
                  <circle cx={x} cy={y} r={size} fill={fill} opacity={0.94} />
                  {asteroid.selected ? (
                    <circle cx={x} cy={y} r={size + 4} className="scene-selection-ring" />
                  ) : null}
                </g>
              );
            })}

            {parsed.atStation ? (
              <g>
                <circle
                  cx="180"
                  cy="130"
                  r="36"
                  fill={frameColor(manifest, "entity.station", "#f59e0b")}
                  opacity="0.22"
                />
                <circle
                  cx="180"
                  cy="130"
                  r="22"
                  fill="none"
                  stroke={frameColor(manifest, "entity.station", "#f59e0b")}
                  strokeWidth="2"
                />
              </g>
            ) : null}

            <polygon
              points="180,112 168,146 192,146"
              fill={frameColor(manifest, shipKey, mode === "human" ? "#34d399" : "#22d3ee")}
              stroke="#e2e8f0"
              strokeWidth="1"
            />

            {actionVfx === "vfx.travel.warp" || actionVfx === "vfx.emergencyBurn" ? (
              <g className="scene-warp-lines">
                <line x1="180" y1="130" x2="110" y2="130" />
                <line x1="180" y1="120" x2="118" y2="108" />
                <line x1="180" y1="140" x2="118" y2="152" />
              </g>
            ) : null}

            {actionVfx.startsWith("vfx.scan") ? (
              <circle cx="180" cy="130" r="48" className="scene-scan-ring" />
            ) : null}

            {actionVfx.startsWith("vfx.mine") ? (
              <g className="scene-mining-beam">
                <line x1="180" y1="130" x2={selectedX} y2={selectedY} />
              </g>
            ) : null}

            {actionVfx === "vfx.cooldown.burst" ? (
              <circle cx="180" cy="130" r="30" className="scene-cooldown-ring" />
            ) : null}

            {eventVfx.includes("vfx.warning.alert") ? (
              <rect x="6" y="6" width="348" height="248" rx="10" className="scene-alert-border" />
            ) : null}
          </svg>

          <div className="scene-legend">
            <span className="scene-pill">node: {parsed.nodeType}</span>
            <span className="scene-pill">asteroids: {validAsteroids.length}</span>
            <span className="scene-pill">selected: {parsed.selectedAsteroidValid ? "yes" : "no"}</span>
          </div>
        </div>

        <div className="scene-minimap">
          <svg viewBox="0 0 180 180" role="img" aria-label="node mini-map">
            <circle cx="90" cy="90" r="80" fill="#0f172a" />

            {parsed.neighbors.map((neighbor) => {
              const angle = (Math.PI * 2 * neighbor.slot) / 6;
              const x = 90 + Math.cos(angle) * 56;
              const y = 90 + Math.sin(angle) * 56;
              const stroke =
                neighbor.threat >= 0.75
                  ? "#ef4444"
                  : neighbor.threat >= 0.45
                    ? "#f59e0b"
                    : "#22c55e";

              const fill =
                neighbor.nodeType === "station"
                  ? "#fbbf24"
                  : neighbor.nodeType === "hazard"
                    ? "#fb7185"
                    : "#38bdf8";

              return (
                <g key={`neigh-${neighbor.slot}`}>
                  <line x1="90" y1="90" x2={x} y2={y} stroke={stroke} strokeWidth="2" opacity="0.9" />
                  <circle cx={x} cy={y} r="9" fill={fill} stroke="#e2e8f0" strokeWidth="1" />
                  <text x={x} y={y + 3} textAnchor="middle" className="scene-map-label">
                    {neighbor.slot}
                  </text>
                </g>
              );
            })}

            <circle
              cx="90"
              cy="90"
              r="12"
              fill={parsed.atStation ? "#fbbf24" : "#22d3ee"}
              stroke="#f8fafc"
              strokeWidth="1.5"
            />
            <text x="90" y="95" textAnchor="middle" className="scene-map-label">
              C
            </text>
          </svg>

          <dl className="kv">
            <dt>Current Node</dt>
            <dd>{parsed.currentNodeNorm.toFixed(3)}</dd>
            <dt>Steps to Station</dt>
            <dd>{parsed.stepsToStationNorm.toFixed(3)}</dd>
            <dt>Neighbors</dt>
            <dd>{parsed.neighbors.length}</dd>
          </dl>
        </div>
      </div>
    </div>
  );
}
