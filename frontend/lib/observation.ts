export type NodeType = "station" | "cluster" | "hazard" | "unknown";

export interface ShipObservation {
  fuelPct: number;
  hullPct: number;
  heatPct: number;
  toolPct: number;
  cargoPct: number;
  alertPct: number;
  timePct: number;
  creditsNorm: number;
  cargoByCommodityPct: number[];
}

export interface NeighborObservation {
  slot: number;
  nodeType: NodeType;
  travelTimeNorm: number;
  travelFuelNorm: number;
  threat: number;
}

export interface AsteroidObservation {
  index: number;
  valid: boolean;
  composition: number[];
  stability: number;
  depletion: number;
  scanConf: number;
  selected: boolean;
}

export interface MarketObservation {
  priceNorm: number[];
  deltaPriceNorm: number[];
  stationInventoryNorm: number[];
}

export interface ParsedObservation {
  ship: ShipObservation;
  atStation: boolean;
  selectedAsteroidValid: boolean;
  nodeType: NodeType;
  currentNodeNorm: number;
  stepsToStationNorm: number;
  neighbors: NeighborObservation[];
  asteroids: AsteroidObservation[];
  market: MarketObservation;
}

const OBS_DIM = 260;
const N_NEIGHBORS = 6;
const N_ASTEROIDS = 16;

function toFinite(value: unknown, fallback = 0): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function toNodeType(oneHot: number[]): NodeType {
  const [station, cluster, hazard] = oneHot;
  if (station >= cluster && station >= hazard) {
    return "station";
  }
  if (cluster >= station && cluster >= hazard) {
    return "cluster";
  }
  if (hazard >= station && hazard >= cluster) {
    return "hazard";
  }
  return "unknown";
}

export function parseObservation(raw: number[] | null | undefined): ParsedObservation | null {
  if (!Array.isArray(raw) || raw.length < OBS_DIM) {
    return null;
  }

  const obs = raw.map((value) => toFinite(value));

  const ship: ShipObservation = {
    fuelPct: obs[0] * 100,
    hullPct: obs[1] * 100,
    heatPct: obs[2] * 100,
    toolPct: obs[3] * 100,
    cargoPct: obs[4] * 100,
    alertPct: obs[5] * 100,
    timePct: obs[6] * 100,
    creditsNorm: obs[7],
    cargoByCommodityPct: obs.slice(8, 14).map((value) => value * 100),
  };

  const nodeType = toNodeType(obs.slice(19, 22));

  const neighbors: NeighborObservation[] = [];
  for (let slot = 0; slot < N_NEIGHBORS; slot += 1) {
    const base = 24 + 7 * slot;
    const present = obs[base] > 0.5;
    if (!present) {
      continue;
    }

    neighbors.push({
      slot,
      nodeType: toNodeType([obs[base + 1], obs[base + 2], obs[base + 3]]),
      travelTimeNorm: obs[base + 4],
      travelFuelNorm: obs[base + 5],
      threat: obs[base + 6],
    });
  }

  const asteroids: AsteroidObservation[] = [];
  for (let index = 0; index < N_ASTEROIDS; index += 1) {
    const base = 68 + 11 * index;
    const valid = obs[base] > 0.5;
    asteroids.push({
      index,
      valid,
      composition: obs.slice(base + 1, base + 7),
      stability: obs[base + 7],
      depletion: obs[base + 8],
      scanConf: obs[base + 9],
      selected: obs[base + 10] > 0.5,
    });
  }

  const market: MarketObservation = {
    priceNorm: obs.slice(244, 250),
    deltaPriceNorm: obs.slice(250, 256),
    stationInventoryNorm: obs.slice(256, 260),
  };

  return {
    ship,
    atStation: obs[17] > 0.5,
    selectedAsteroidValid: obs[18] > 0.5,
    nodeType,
    currentNodeNorm: obs[22],
    stepsToStationNorm: obs[23],
    neighbors,
    asteroids,
    market,
  };
}
