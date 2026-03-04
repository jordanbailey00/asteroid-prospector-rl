from __future__ import annotations

from collections.abc import Callable

import numpy as np
from asteroid_prospector.constants import MAX_ASTEROIDS, MAX_NEIGHBORS, N_ACTIONS, N_COMMODITIES

BaselinePolicy = Callable[[np.ndarray], int]

HOLD_ACTION = 6
WIDE_SCAN_ACTION = 8
DEEP_SCAN_ACTION = 10
STABILIZE_SELECTED_ACTION = 31
COOLDOWN_ACTION = 33
BUY_FUEL_MED_ACTION = 62
BUY_FUEL_LARGE_ACTION = 63
BUY_REPAIR_KIT_ACTION = 64

S_FUEL = 0
S_HULL = 1
S_HEAT = 2
S_TOOL = 3
S_CARGO_LOAD = 4
S_TIME = 6
S_CREDITS = 7
S_CARGO0 = 8
S_REPAIR_KITS = 14
S_STABILIZERS = 15
S_AT_STATION = 17

NEIGH_BASE = 24
NEIGH_STRIDE = 7

AST_BASE = 68
AST_STRIDE = 11

MKT_PRICE_BASE = 244
MKT_DPRICE_BASE = 250

DEFAULT_CREDITS_CAP = 1.0e7
BUY_FUEL_MED_COST = 120.0
BUY_FUEL_LARGE_COST = 210.0
BUY_REPAIR_KIT_COST = 150.0

BASELINE_BOT_NAMES = ("greedy_miner", "cautious_scanner", "market_timer")


def _sanitize_action(action: int) -> int:
    action_i = int(action)
    if 0 <= action_i < N_ACTIONS:
        return action_i
    return HOLD_ACTION


def _estimate_credits(obs: np.ndarray) -> float:
    credits_norm = float(np.clip(obs[S_CREDITS], 0.0, 1.0))
    return float(np.expm1(credits_norm * np.log1p(DEFAULT_CREDITS_CAP)))


def _iter_valid_asteroids(obs: np.ndarray):
    for asteroid_idx in range(MAX_ASTEROIDS):
        base = AST_BASE + asteroid_idx * AST_STRIDE
        if float(obs[base]) > 0.5:
            yield asteroid_idx, base


def _best_asteroid(obs: np.ndarray, *, target_commodity: int | None = None) -> int | None:
    prices = obs[MKT_PRICE_BASE : MKT_PRICE_BASE + N_COMMODITIES]
    best_idx: int | None = None
    best_score = -1.0e12

    for asteroid_idx, base in _iter_valid_asteroids(obs):
        comp = obs[base + 1 : base + 1 + N_COMMODITIES]
        stability = float(obs[base + 7])
        depletion = float(obs[base + 8])
        conf = float(obs[base + 9])

        if target_commodity is None:
            score = float(np.dot(prices, comp)) * stability * (1.0 - depletion)
        else:
            score = (
                float(comp[target_commodity]) * (0.5 + 0.5 * conf) * stability * (1.0 - depletion)
            )

        if score > best_score:
            best_score = score
            best_idx = asteroid_idx

    return best_idx


def _asteroid_selected(obs: np.ndarray, asteroid_idx: int) -> bool:
    base = AST_BASE + asteroid_idx * AST_STRIDE
    return float(obs[base + 10]) > 0.5


def _station_neighbor_action(obs: np.ndarray) -> int | None:
    for slot in range(MAX_NEIGHBORS):
        base = NEIGH_BASE + slot * NEIGH_STRIDE
        if float(obs[base]) > 0.5 and float(obs[base + 1]) > 0.5:
            return slot
    return None


def _first_valid_neighbor_action(obs: np.ndarray) -> int | None:
    for slot in range(MAX_NEIGHBORS):
        base = NEIGH_BASE + slot * NEIGH_STRIDE
        if float(obs[base]) > 0.5:
            return slot
    return None


def _field_neighbor_action(obs: np.ndarray) -> int | None:
    cluster_candidate: int | None = None
    non_station_candidate: int | None = None

    for slot in range(MAX_NEIGHBORS):
        base = NEIGH_BASE + slot * NEIGH_STRIDE
        if float(obs[base]) <= 0.5:
            continue

        is_station = float(obs[base + 1]) > 0.5
        is_cluster = float(obs[base + 2]) > 0.5

        if is_cluster and cluster_candidate is None:
            cluster_candidate = slot
        if not is_station and non_station_candidate is None:
            non_station_candidate = slot

    if cluster_candidate is not None:
        return cluster_candidate
    if non_station_candidate is not None:
        return non_station_candidate
    return _first_valid_neighbor_action(obs)


def greedy_miner_policy(obs: np.ndarray) -> int:
    at_station = float(obs[S_AT_STATION]) > 0.5
    cargo_load = float(obs[S_CARGO_LOAD])
    heat = float(obs[S_HEAT])
    hull = float(obs[S_HULL])
    tool = float(obs[S_TOOL])
    fuel = float(obs[S_FUEL])
    credits = _estimate_credits(obs)

    if at_station:
        for commodity_idx in range(N_COMMODITIES):
            if float(obs[S_CARGO0 + commodity_idx]) > 0.01:
                return _sanitize_action(43 + commodity_idx * 3 + 2)
        if fuel < 0.40 and credits >= BUY_FUEL_LARGE_COST:
            return BUY_FUEL_LARGE_ACTION
        action = _field_neighbor_action(obs)
        return _sanitize_action(HOLD_ACTION if action is None else action)

    if cargo_load > 0.85 or heat > 0.80 or hull < 0.40 or fuel < 0.15:
        action = _station_neighbor_action(obs)
        if action is None:
            action = _first_valid_neighbor_action(obs)
        return _sanitize_action(HOLD_ACTION if action is None else action)

    asteroid_idx = _best_asteroid(obs)
    if asteroid_idx is None:
        action = _field_neighbor_action(obs)
        return _sanitize_action(HOLD_ACTION if action is None else action)

    if not _asteroid_selected(obs, asteroid_idx):
        return _sanitize_action(12 + asteroid_idx)

    if tool > 0.60 and heat < 0.60:
        return 30
    return 29


def cautious_scanner_policy(obs: np.ndarray) -> int:
    at_station = float(obs[S_AT_STATION]) > 0.5
    cargo_load = float(obs[S_CARGO_LOAD])
    heat = float(obs[S_HEAT])
    hull = float(obs[S_HULL])
    fuel = float(obs[S_FUEL])
    credits = _estimate_credits(obs)

    if at_station:
        for commodity_idx in range(N_COMMODITIES):
            if float(obs[S_CARGO0 + commodity_idx]) > 0.01:
                return _sanitize_action(43 + commodity_idx * 3 + 1)
        if float(obs[S_REPAIR_KITS]) < 0.30 and credits >= BUY_REPAIR_KIT_COST:
            return BUY_REPAIR_KIT_ACTION
        if fuel < 0.50 and credits >= BUY_FUEL_MED_COST:
            return BUY_FUEL_MED_ACTION
        action = _field_neighbor_action(obs)
        return _sanitize_action(HOLD_ACTION if action is None else action)

    if cargo_load > 0.70 or heat > 0.75 or hull < 0.55 or fuel < 0.20:
        action = _station_neighbor_action(obs)
        if action is None:
            action = _first_valid_neighbor_action(obs)
        return _sanitize_action(HOLD_ACTION if action is None else action)

    conf_total = 0.0
    conf_count = 0
    for _asteroid_idx, base in _iter_valid_asteroids(obs):
        conf_total += float(obs[base + 9])
        conf_count += 1

    if conf_count > 0 and (conf_total / float(conf_count)) < 0.35:
        return WIDE_SCAN_ACTION

    asteroid_idx = _best_asteroid(obs)
    if asteroid_idx is None:
        action = _field_neighbor_action(obs)
        return _sanitize_action(HOLD_ACTION if action is None else action)

    if not _asteroid_selected(obs, asteroid_idx):
        return _sanitize_action(12 + asteroid_idx)

    base = AST_BASE + asteroid_idx * AST_STRIDE
    stability = float(obs[base + 7])
    conf = float(obs[base + 9])

    if conf < 0.60:
        return DEEP_SCAN_ACTION

    if stability < 0.40 and float(obs[S_STABILIZERS]) > 0.10:
        return STABILIZE_SELECTED_ACTION

    if heat > 0.70:
        return COOLDOWN_ACTION
    return 28


def market_timer_policy(obs: np.ndarray, *, target_commodity: int = 3) -> int:
    if target_commodity < 0 or target_commodity >= N_COMMODITIES:
        raise ValueError(
            f"target_commodity must be in [0, {N_COMMODITIES - 1}], got {target_commodity}"
        )

    at_station = float(obs[S_AT_STATION]) > 0.5
    heat = float(obs[S_HEAT])
    cargo_load = float(obs[S_CARGO_LOAD])
    time_remaining = float(obs[S_TIME])
    fuel = float(obs[S_FUEL])
    credits = _estimate_credits(obs)

    price = float(obs[MKT_PRICE_BASE + target_commodity])
    d_price = float(obs[MKT_DPRICE_BASE + target_commodity])

    if at_station:
        force_sell = cargo_load > 0.70 or time_remaining < 0.15
        target_cargo = float(obs[S_CARGO0 + target_commodity])

        if target_cargo > 0.01 and ((d_price > 0.01 and price > 0.50) or force_sell):
            return _sanitize_action(43 + target_commodity * 3)

        if force_sell:
            for commodity_idx in range(N_COMMODITIES):
                if commodity_idx == target_commodity:
                    continue
                if float(obs[S_CARGO0 + commodity_idx]) > 0.05:
                    return _sanitize_action(43 + commodity_idx * 3)

        if fuel < 0.50 and credits >= BUY_FUEL_MED_COST:
            return BUY_FUEL_MED_ACTION

        action = _field_neighbor_action(obs)
        return _sanitize_action(HOLD_ACTION if action is None else action)

    if cargo_load > 0.12:
        if heat > 0.65:
            return COOLDOWN_ACTION
        action = _station_neighbor_action(obs)
        if action is not None:
            return _sanitize_action(action)

    if d_price < -0.02 and cargo_load > 0.25:
        if heat > 0.60:
            return COOLDOWN_ACTION
        action = _station_neighbor_action(obs)
        if action is None:
            action = _first_valid_neighbor_action(obs)
        return _sanitize_action(HOLD_ACTION if action is None else action)

    if cargo_load > 0.85 or heat > 0.80:
        if heat > 0.70:
            return COOLDOWN_ACTION
        action = _station_neighbor_action(obs)
        if action is not None:
            return _sanitize_action(action)
        return HOLD_ACTION

    asteroid_idx = _best_asteroid(obs, target_commodity=target_commodity)
    if asteroid_idx is None:
        action = _field_neighbor_action(obs)
        return _sanitize_action(HOLD_ACTION if action is None else action)

    if not _asteroid_selected(obs, asteroid_idx):
        return _sanitize_action(12 + asteroid_idx)

    if heat < 0.60:
        return 29
    return 28


def make_market_timer_policy(*, target_commodity: int = 3) -> BaselinePolicy:
    if target_commodity < 0 or target_commodity >= N_COMMODITIES:
        raise ValueError(
            f"target_commodity must be in [0, {N_COMMODITIES - 1}], got {target_commodity}"
        )

    def _policy(obs: np.ndarray) -> int:
        return market_timer_policy(obs, target_commodity=target_commodity)

    return _policy


def list_baseline_bots() -> tuple[str, ...]:
    return BASELINE_BOT_NAMES


def get_baseline_bot(name: str, *, target_commodity: int = 3) -> BaselinePolicy:
    key = str(name).strip().lower()
    if key == "greedy_miner":
        return greedy_miner_policy
    if key == "cautious_scanner":
        return cautious_scanner_policy
    if key == "market_timer":
        return make_market_timer_policy(target_commodity=target_commodity)

    supported = ", ".join(BASELINE_BOT_NAMES)
    raise ValueError(f"Unsupported baseline bot: {name!r}. Supported: {supported}")


__all__ = [
    "BaselinePolicy",
    "BASELINE_BOT_NAMES",
    "cautious_scanner_policy",
    "get_baseline_bot",
    "greedy_miner_policy",
    "list_baseline_bots",
    "make_market_timer_policy",
    "market_timer_policy",
]
