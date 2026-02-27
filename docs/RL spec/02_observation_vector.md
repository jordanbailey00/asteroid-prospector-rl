## 1) Observation vector (exact fields, fixed layout)

### 1.1 High-level structure

Observation is `np.ndarray(shape=(OBS_DIM,), dtype=np.float32)`.

- **Ship / global state:** 0–23
- **Node / route context:** 24–67
- **Asteroid slots (current node):** 68–243
- **Market features:** 244–259

### 1.2 Exact field mapping (indices)

#### A) Ship + episode scalars (0–23)

| Idx | Name | Type | Range | Meaning |
|---:|---|---|---|---|
| 0 | fuel_frac | float | [0,1] | `fuel / FUEL_MAX` fileciteturn2file3L7-L15 |
| 1 | hull_frac | float | [0,1] | `hull / HULL_MAX` fileciteturn2file3L7-L15 |
| 2 | heat_frac | float | [0,1] | `heat / HEAT_MAX` fileciteturn2file3L7-L15 |
| 3 | tool_frac | float | [0,1] | `tool_condition / TOOL_MAX` fileciteturn2file3L7-L15 |
| 4 | cargo_load_frac | float | [0,1] | `cargo_total / CARGO_MAX` fileciteturn2file3L7-L15 |
| 5 | alert_frac | float | [0,1] | `alert / ALERT_MAX` fileciteturn2file3L14-L15 |
| 6 | time_frac | float | [0,1] | `time_remaining / TIME_MAX` fileciteturn2file3L13-L15 |
| 7 | credits_log_norm | float | [0,1] | `log1p(credits)/log1p(credits_cap)` (cap e.g. 1e7) |
| 8..13 | cargo_by_commodity_frac[c] | float | [0,1] | `cargo[c] / CARGO_MAX` (6 dims) |
| 14 | repair_kits_norm | float | [0,1] | `repair_kits / repair_kits_cap` |
| 15 | stabilizers_norm | float | [0,1] | `stabilizers / stabilizers_cap` |
| 16 | decoys_norm | float | [0,1] | `decoys / decoys_cap` |
| 17 | at_station_flag | float | {0,1} | 1 if docked/station node |
| 18 | mining_active_flag | float | {0,1} | 1 if selected asteroid is “locked” for mining |
| 19..21 | current_node_type_onehot | float | {0,1} | station/cluster/hazard (3 dims) |
| 22 | current_node_index_norm | float | [0,1] | `node_idx / (MAX_NODES-1)` |
| 23 | steps_to_station_norm | float | [0,1] | shortest-path steps / (MAX_NODES-1) |

#### B) Neighbor slots (24–67) — `MAX_NEIGHBORS=6`, each slot 7 dims (total 42)

For neighbor slot `k in [0..5]`, base = `24 + 7*k`:

| Offset | Name | Range | Meaning |
|---:|---|---|---|
| +0 | neigh_valid | {0,1} | 1 if neighbor exists in slot |
| +1 | neigh_type_station | {0,1} | onehot type |
| +2 | neigh_type_cluster | {0,1} | onehot type |
| +3 | neigh_type_hazard | {0,1} | onehot type |
| +4 | travel_time_norm | [0,1] | `travel_time / travel_time_max` |
| +5 | travel_fuel_cost_norm | [0,1] | `fuel_cost / fuel_cost_max` |
| +6 | neigh_threat_norm | [0,1] | pirate+hazard combined risk estimate from passive model |

> This directly supports graph travel (“Travel to Node i (neighbor)”) fileciteturn1file3L15-L20

#### C) Asteroid slots in current node (68–243) — `MAX_ASTEROIDS=16`, each slot 11 dims (total 176)

For asteroid slot `a in [0..15]`, base = `68 + 11*a`:

| Offset | Name | Range | Meaning |
|---:|---|---|---|
| +0 | ast_valid | {0,1} | asteroid exists |
| +1..+6 | comp_est[c] | [0,1] | estimated composition distribution (6 commodities; sums ~1) |
| +7 | stability_est | [0,1] | estimated stability (higher = safer) fileciteturn2file3L20-L23 |
| +8 | depletion_frac | [0,1] | 0 fresh → 1 depleted fileciteturn2file3L22-L23 |
| +9 | scan_conf | [0,1] | confidence in estimates (higher after deep scan) |
| +10 | selected_flag | {0,1} | 1 if this asteroid is currently selected/targeted |

#### D) Market features (244–259) — 16 dims

| Idx | Name | Range | Meaning |
|---:|---|---|---|
| 244..249 | price_norm[c] | [0,1] | normalized commodity prices (per-commodity) fileciteturn2file3L25-L31 |
| 250..255 | price_delta_norm[c] | [-1,1] | `(price - prev_price) / price_scale` |
| 256..259 | top4_station_inventory_norm | [0,1] | normalized inventory for 4 key commodities (pick: iron, water_ice, pge, rare_isotopes) |

> Note: you can expand inventory dims to 6 later, but that changes OBS_DIM.

---

