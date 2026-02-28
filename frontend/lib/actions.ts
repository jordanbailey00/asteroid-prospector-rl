const BASE_ACTION_LABELS: string[] = [
  "TRAVEL_NEIGHBOR_0",
  "TRAVEL_NEIGHBOR_1",
  "TRAVEL_NEIGHBOR_2",
  "TRAVEL_NEIGHBOR_3",
  "TRAVEL_NEIGHBOR_4",
  "TRAVEL_NEIGHBOR_5",
  "HOLD_DRIFT",
  "EMERGENCY_BURN",
  "WIDE_SCAN",
  "FOCUSED_SCAN_SELECTED",
  "DEEP_SCAN_SELECTED",
  "PASSIVE_THREAT_LISTEN",
  ...Array.from({ length: 16 }, (_, idx) => `SELECT_ASTEROID_${idx}`),
  "MINE_CONSERVATIVE_SELECTED",
  "MINE_STANDARD_SELECTED",
  "MINE_AGGRESSIVE_SELECTED",
  "STABILIZE_SELECTED",
  "REFINE_ONBOARD",
  "ACTIVE_COOLDOWN",
  "TOOL_MAINTENANCE",
  "HULL_PATCH",
  ...Array.from({ length: 6 }, (_, idx) => `JETTISON_COMMODITY_${idx}`),
  "DOCK",
  ...Array.from({ length: 18 }, (_, idx) => {
    const commodity = Math.floor(idx / 3);
    const bucket = idx % 3;
    const bucketLabel = bucket === 0 ? "25" : bucket === 1 ? "50" : "100";
    return `SELL_COMMODITY_${commodity}_${bucketLabel}PCT`;
  }),
  "BUY_FUEL_SMALL",
  "BUY_FUEL_MED",
  "BUY_FUEL_LARGE",
  "BUY_REPAIR_KIT",
  "BUY_STABILIZER",
  "BUY_DECOY",
  "FULL_REPAIR_OVERHAUL",
  "CASH_OUT_END_EPISODE",
];

if (BASE_ACTION_LABELS.length !== 69) {
  throw new Error(`Expected 69 actions, got ${BASE_ACTION_LABELS.length}`);
}

export function actionLabel(action: number): string {
  if (!Number.isInteger(action) || action < 0 || action >= BASE_ACTION_LABELS.length) {
    return `UNKNOWN_ACTION_${action}`;
  }
  return BASE_ACTION_LABELS[action];
}

export function allActionLabels(): string[] {
  return BASE_ACTION_LABELS.slice();
}
