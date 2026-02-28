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

const DEFAULT_FRAME_PATHS = {
  shipAgent: "/assets/sprites/world/ship_agent.png",
  shipHuman: "/assets/sprites/world/ship_human.png",
  station: "/assets/sprites/world/station_main.png",
  asteroidSmall: "/assets/sprites/world/asteroid_small.png",
  asteroidMedium: "/assets/sprites/world/asteroid_medium.png",
  asteroidLarge: "/assets/sprites/world/asteroid_large.png",
  hazard: "/assets/sprites/world/hazard_marker.png",
  pirate: "/assets/sprites/world/pirate_marker.png",
  selectionRing: "/assets/sprites/vfx/selection_ring.png",
  noneVfx: "/assets/sprites/vfx/none.svg",
};

const DEFAULT_BACKGROUND_PATHS = {
  starfieldMain: "/assets/backgrounds/starfield_main.png",
  starfieldAlt: "/assets/backgrounds/starfield_alt.png",
  planetPrimary: "/assets/planets/planet_primary.png",
  planetSecondary: "/assets/planets/planet_secondary.png",
};

function unique<T>(values: T[]): T[] {
  return Array.from(new Set(values));
}

function framePath(manifest: GraphicsManifest | null, key: string, fallback: string): string {
  const path = manifest?.frames[key]?.path;
  if (typeof path === "string" && path.trim() !== "") {
    return path;
  }
  return fallback;
}

function backgroundPath(manifest: GraphicsManifest | null, key: string, fallback: string): string {
  const path = manifest?.backgrounds[key]?.path;
  if (typeof path === "string" && path.trim() !== "") {
    return path;
  }
  return fallback;
}

function asteroidFrameKey(
  depletion: number,
): "entity.asteroid.small" | "entity.asteroid.medium" | "entity.asteroid.large" {
  if (depletion <= 0.33) {
    return "entity.asteroid.large";
  }
  if (depletion <= 0.66) {
    return "entity.asteroid.medium";
  }
  return "entity.asteroid.small";
}

function nodeSpriteFallback(frameKey: string): string {
  switch (frameKey) {
    case "entity.station":
      return DEFAULT_FRAME_PATHS.station;
    case "entity.hazard.radiation":
      return DEFAULT_FRAME_PATHS.hazard;
    default:
      return DEFAULT_FRAME_PATHS.asteroidMedium;
  }
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

  const shipKey = mode === "human" ? "entity.ship.human" : "entity.ship.agent";
  const shipPath = framePath(
    manifest,
    shipKey,
    mode === "human" ? DEFAULT_FRAME_PATHS.shipHuman : DEFAULT_FRAME_PATHS.shipAgent,
  );
  const stationPath = framePath(manifest, "entity.station", DEFAULT_FRAME_PATHS.station);
  const hazardPath = framePath(manifest, "entity.hazard.radiation", DEFAULT_FRAME_PATHS.hazard);
  const piratePath = framePath(manifest, "entity.pirate.marker", DEFAULT_FRAME_PATHS.pirate);
  const selectionRingPath = framePath(manifest, "vfx.selection.ring", DEFAULT_FRAME_PATHS.selectionRing);

  const backgroundMain = backgroundPath(manifest, "bg.starfield.0", DEFAULT_BACKGROUND_PATHS.starfieldMain);
  const backgroundAlt = backgroundPath(manifest, "bg.starfield.1", DEFAULT_BACKGROUND_PATHS.starfieldAlt);
  const planetPrimary = backgroundPath(manifest, "bg.planet.0", DEFAULT_BACKGROUND_PATHS.planetPrimary);
  const planetSecondary = backgroundPath(manifest, "bg.planet.1", DEFAULT_BACKGROUND_PATHS.planetSecondary);

  const validAsteroids = parsed.asteroids.filter((asteroid) => asteroid.valid);
  const selectedAsteroid = validAsteroids.find((asteroid) => asteroid.selected) ?? null;

  const selectedAngle = selectedAsteroid ? (Math.PI * 2 * selectedAsteroid.index) / 16 : null;
  const selectedX = selectedAngle === null ? 248 : 180 + Math.cos(selectedAngle) * 76;
  const selectedY = selectedAngle === null ? 126 : 130 + Math.sin(selectedAngle) * 76;

  const actionVfxPath = framePath(manifest, actionVfx, DEFAULT_FRAME_PATHS.noneVfx);
  const actionableVfx = actionVfx !== "vfx.none";
  const mineVfx = actionVfx.startsWith("vfx.mine");

  const eventVfxVisible = eventVfx.filter((key) => key !== "vfx.none" && key !== "vfx.warning.alert");
  const warningActive = eventVfx.includes("vfx.warning.alert") || parsed.nodeType === "hazard";
  const pirateActive = events.includes("pirate_encounter") || parsed.ship.alertPct >= 70;

  return (
    <div className="scene-shell">
      <div className="scene-grid">
        <div className="scene-viewport">
          <svg viewBox="0 0 360 260" className="scene-canvas" role="img" aria-label="sector view">
            <image href={backgroundMain} x="0" y="0" width="360" height="260" preserveAspectRatio="xMidYMid slice" />
            <image
              href={backgroundAlt}
              x="0"
              y="0"
              width="360"
              height="260"
              preserveAspectRatio="xMidYMid slice"
              opacity="0.34"
            />
            <image
              href={planetPrimary}
              x="18"
              y="138"
              width="118"
              height="118"
              preserveAspectRatio="xMidYMid meet"
              opacity="0.9"
            />
            <image
              href={planetSecondary}
              x="268"
              y="4"
              width="84"
              height="84"
              preserveAspectRatio="xMidYMid meet"
              opacity="0.85"
            />

            {warningActive ? (
              <rect x="4" y="4" width="352" height="252" rx="10" className="scene-alert-border" />
            ) : null}

            {validAsteroids.map((asteroid) => {
              const angle = (Math.PI * 2 * asteroid.index) / 16;
              const radius = 72 + (1 - asteroid.scanConf) * 20;
              const x = 180 + Math.cos(angle) * radius;
              const y = 130 + Math.sin(angle) * radius;

              const frameKey = asteroidFrameKey(asteroid.depletion);
              const spritePath =
                frameKey === "entity.asteroid.large"
                  ? framePath(manifest, frameKey, DEFAULT_FRAME_PATHS.asteroidLarge)
                  : frameKey === "entity.asteroid.medium"
                    ? framePath(manifest, frameKey, DEFAULT_FRAME_PATHS.asteroidMedium)
                    : framePath(manifest, frameKey, DEFAULT_FRAME_PATHS.asteroidSmall);

              const diameter =
                frameKey === "entity.asteroid.large" ? 34 : frameKey === "entity.asteroid.medium" ? 28 : 22;

              return (
                <g key={`ast-${asteroid.index}`}>
                  <image
                    href={spritePath}
                    x={x - diameter / 2}
                    y={y - diameter / 2}
                    width={diameter}
                    height={diameter}
                    preserveAspectRatio="xMidYMid meet"
                    opacity="0.95"
                  />
                  {asteroid.selected ? (
                    <image
                      href={selectionRingPath}
                      x={x - diameter * 0.78}
                      y={y - diameter * 0.78}
                      width={diameter * 1.56}
                      height={diameter * 1.56}
                      preserveAspectRatio="xMidYMid meet"
                      className="scene-selection-ring-image"
                    />
                  ) : null}
                </g>
              );
            })}

            {parsed.atStation ? (
              <image
                href={stationPath}
                x="146"
                y="94"
                width="68"
                height="68"
                preserveAspectRatio="xMidYMid meet"
              />
            ) : null}

            <image href={shipPath} x="158" y="108" width="44" height="44" preserveAspectRatio="xMidYMid meet" />

            {parsed.nodeType === "hazard" ? (
              <image href={hazardPath} x="12" y="10" width="30" height="30" preserveAspectRatio="xMidYMid meet" />
            ) : null}
            {pirateActive ? (
              <image href={piratePath} x="318" y="10" width="30" height="30" preserveAspectRatio="xMidYMid meet" />
            ) : null}

            {actionableVfx ? (
              <image
                href={actionVfxPath}
                x={mineVfx ? selectedX - 24 : 180 - 34}
                y={mineVfx ? selectedY - 24 : 130 - 34}
                width={mineVfx ? 48 : 68}
                height={mineVfx ? 48 : 68}
                preserveAspectRatio="xMidYMid meet"
                className="scene-vfx-action"
              />
            ) : null}

            {eventVfxVisible.map((key, index) => (
              <image
                key={key}
                href={framePath(manifest, key, DEFAULT_FRAME_PATHS.noneVfx)}
                x={14 + index * 42}
                y="206"
                width="36"
                height="36"
                preserveAspectRatio="xMidYMid meet"
                className="scene-vfx-event"
              />
            ))}
          </svg>

          <div className="scene-legend">
            <span className="scene-pill">node: {parsed.nodeType}</span>
            <span className="scene-pill">asteroids: {validAsteroids.length}</span>
            <span className="scene-pill">selected: {parsed.selectedAsteroidValid ? "yes" : "no"}</span>
          </div>
        </div>

        <div className="scene-minimap">
          <svg viewBox="0 0 180 180" role="img" aria-label="node mini-map">
            <image href={backgroundMain} x="0" y="0" width="180" height="180" preserveAspectRatio="xMidYMid slice" />
            <circle cx="90" cy="90" r="81" fill="none" stroke="rgba(226, 232, 240, 0.5)" strokeWidth="2" />

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

              const nodeFrameKey =
                neighbor.nodeType === "station"
                  ? "entity.station"
                  : neighbor.nodeType === "hazard"
                    ? "entity.hazard.radiation"
                    : "entity.asteroid.medium";

              return (
                <g key={`neigh-${neighbor.slot}`}>
                  <line x1="90" y1="90" x2={x} y2={y} stroke={stroke} strokeWidth="2" opacity="0.9" />
                  <image
                    href={framePath(manifest, nodeFrameKey, nodeSpriteFallback(nodeFrameKey))}
                    x={x - 12}
                    y={y - 12}
                    width="24"
                    height="24"
                    preserveAspectRatio="xMidYMid meet"
                  />
                  {neighbor.threat >= 0.75 ? (
                    <image
                      href={piratePath}
                      x={x + 6}
                      y={y - 18}
                      width="13"
                      height="13"
                      preserveAspectRatio="xMidYMid meet"
                    />
                  ) : null}
                  <text x={x} y={y + 16} textAnchor="middle" className="scene-map-label">
                    {neighbor.slot}
                  </text>
                </g>
              );
            })}

            <image
              href={parsed.atStation ? stationPath : shipPath}
              x="76"
              y="76"
              width="28"
              height="28"
              preserveAspectRatio="xMidYMid meet"
            />
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
