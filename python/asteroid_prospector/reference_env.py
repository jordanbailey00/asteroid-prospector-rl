"""Pure-Python reference environment for Asteroid Belt Prospector (M1).

This module implements the frozen RL interface contract:
- OBS_DIM == 260
- N_ACTIONS == 69 (0..68)
- Gymnasium-style reset/step returns
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import (
    ALERT_MAX,
    CARGO_MAX,
    CREDIT_SCALE,
    FUEL_MAX,
    HEAT_MAX,
    HULL_MAX,
    MAX_ASTEROIDS,
    MAX_NEIGHBORS,
    MAX_NODES,
    N_ACTIONS,
    N_COMMODITIES,
    NODE_CLUSTER,
    NODE_HAZARD,
    NODE_STATION,
    OBS_DIM,
    TIME_MAX,
    TOOL_MAX,
)
from .pcg32_rng import Pcg32Rng

# Observation indices intentionally mirror the frozen specification.
MKT_PRICE_BASE = 244
MKT_DPRICE_BASE = 250
MKT_INV_BASE = 256


@dataclass(frozen=True)
class DiscreteActionSpace:
    n: int

    def contains(self, action: int) -> bool:
        return isinstance(action, int | np.integer) and 0 <= int(action) < self.n


@dataclass(frozen=True)
class BoxObservationSpace:
    low: float
    high: float
    shape: tuple[int, ...]
    dtype: Any


@dataclass(frozen=True)
class RewardCfg:
    credit_scale: float = CREDIT_SCALE
    alpha_extract: float = 0.02
    beta_fuel: float = 0.10
    gamma_time: float = 0.001
    delta_wear: float = 0.05
    epsilon_heat: float = 0.20
    zeta_damage: float = 1.00
    kappa_pirate: float = 1.00
    scan_cost: float = 0.005
    invalid_action_pen: float = 0.01
    heat_safe_frac: float = 0.70
    stranded_pen: float = 50.0
    destroyed_pen: float = 100.0
    terminal_bonus_b: float = 0.002


@dataclass(frozen=True)
class ReferenceEnvConfig:
    credits_cap: float = 1.0e7
    time_max: float = TIME_MAX

    repair_kits_cap: int = 12
    stabilizers_cap: int = 12
    decoys_cap: int = 12

    travel_time_max: float = 8.0
    travel_fuel_cost_max: float = 160.0

    price_base: tuple[float, ...] = (45.0, 55.0, 85.0, 145.0, 210.0, 120.0)
    price_min: tuple[float, ...] = (12.0, 15.0, 20.0, 50.0, 80.0, 30.0)
    price_max: tuple[float, ...] = (180.0, 200.0, 240.0, 320.0, 420.0, 300.0)
    price_scale: float = 100.0

    station_inventory_norm_cap: float = 500.0

    wide_scan_time: int = 3
    focused_scan_time: int = 2
    deep_scan_time: int = 4
    threat_listen_time: int = 2
    stabilize_time: int = 2
    refine_time: int = 2
    cooldown_time: int = 2
    maint_time: int = 2
    patch_time: int = 2
    dock_time: int = 1
    overhaul_time: int = 3

    wide_scan_fuel: float = 5.0
    focused_scan_fuel: float = 4.0
    deep_scan_fuel: float = 8.0
    refine_fuel: float = 4.0
    cooldown_fuel: float = 2.0
    emergency_burn_fuel: float = 18.0

    refine_heat: float = 6.0
    cooldown_amount: float = 20.0

    emergency_burn_alert: float = 10.0
    wide_scan_alert: float = 4.0
    focused_scan_alert: float = 3.0
    deep_scan_alert: float = 6.0
    refine_alert: float = 3.0
    cooldown_alert: float = 1.0
    alert_decay_hold: float = 3.0
    dock_alert_drop: float = 20.0
    jettison_alert_relief: float = 8.0

    heat_dissipation_per_tick: float = 2.5
    overheat_damage_per_unit: float = 1.25

    tool_repair_amount: float = 25.0
    hull_patch_amount: float = 20.0

    escape_buff_ticks: int = 4
    stabilize_buff_ticks: int = 6

    fracture_depletion_rate: float = 0.01

    hazard_damage_per_tick: float = 0.7
    hazard_heat_per_tick: float = 0.5
    hazard_alert_per_tick: float = 0.8

    pirate_bias: float = -4.0
    pirate_intensity_w: float = 3.0
    pirate_alert_w: float = 2.2
    pirate_cargo_w: float = 0.8
    pirate_escape_w: float = 2.8

    slippage_k: float = 0.25
    slippage_root: float = 0.2

    inventory_pressure_k: float = 0.04
    sales_pressure_k: float = 0.05
    market_noise_k: float = 0.03
    sales_decay_tau: float = 14.0

    buy_fuel_small_qty: float = 120.0
    buy_fuel_med_qty: float = 260.0
    buy_fuel_large_qty: float = 480.0

    buy_fuel_small_cost: float = 60.0
    buy_fuel_med_cost: float = 120.0
    buy_fuel_large_cost: float = 210.0
    buy_repair_kit_cost: float = 150.0
    buy_stabilizer_cost: float = 175.0
    buy_decoy_cost: float = 110.0

    overhaul_cost: float = 280.0

    reward_cfg: RewardCfg = RewardCfg()


@dataclass
class _StepSnapshot:
    credits_before: float
    fuel_before: float
    hull_before: float
    heat_before: float
    tool_before: float
    cargo_before: np.ndarray
    cargo_value_before: float
    value_lost_to_pirates_before: float


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(high, max(low, value)))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))


def _normalize_probs(values: np.ndarray) -> np.ndarray:
    arr = np.clip(values.astype(np.float64, copy=False), 1.0e-8, None)
    total = float(np.sum(arr))
    if total <= 0.0:
        return np.full_like(arr, 1.0 / float(arr.size), dtype=np.float64)
    return arr / total


def compute_reward(
    snapshot: _StepSnapshot,
    *,
    credits_after: float,
    fuel_after: float,
    hull_after: float,
    heat_after: float,
    tool_after: float,
    cargo_after: np.ndarray,
    cargo_value_after: float,
    value_lost_to_pirates_after: float,
    action: int,
    dt: int,
    invalid: bool,
    destroyed: bool,
    stranded: bool,
    done: bool,
    cfg: RewardCfg,
) -> float:
    del cargo_after  # Included for parity with the published reward signature.

    delta_credits = credits_after - snapshot.credits_before
    r_sell = delta_credits / cfg.credit_scale

    delta_cargo_value = max(0.0, cargo_value_after - snapshot.cargo_value_before)
    r_extract = cfg.alpha_extract * (delta_cargo_value / cfg.credit_scale)

    r_fuel = -cfg.beta_fuel * max(0.0, snapshot.fuel_before - fuel_after) / 100.0
    r_time = -cfg.gamma_time * float(dt)
    r_wear = -cfg.delta_wear * max(0.0, snapshot.tool_before - tool_after) / 10.0
    r_damage = -cfg.zeta_damage * max(0.0, snapshot.hull_before - hull_after) / 10.0

    heat_safe = cfg.heat_safe_frac * HEAT_MAX
    heat_excess = max(0.0, heat_after - heat_safe)
    r_heat = -cfg.epsilon_heat * (heat_excess / HEAT_MAX) ** 2

    r_scan = -cfg.scan_cost if action in (8, 9, 10) else 0.0
    r_invalid = -cfg.invalid_action_pen if invalid else 0.0

    delta_pirate_loss = max(
        0.0, value_lost_to_pirates_after - snapshot.value_lost_to_pirates_before
    )
    r_pirate = -cfg.kappa_pirate * (delta_pirate_loss / cfg.credit_scale)

    r_terminal = 0.0
    if stranded:
        r_terminal -= cfg.stranded_pen
    if destroyed:
        r_terminal -= cfg.destroyed_pen
    if done and not destroyed and not stranded:
        r_terminal += cfg.terminal_bonus_b * (credits_after / cfg.credit_scale)

    reward = (
        r_sell
        + r_extract
        + r_fuel
        + r_time
        + r_wear
        + r_heat
        + r_damage
        + r_scan
        + r_invalid
        + r_pirate
        + r_terminal
    )
    return float(reward)


class ProspectorReferenceEnv:
    """Reference Python environment used as the semantic baseline for parity."""

    def __init__(self, config: ReferenceEnvConfig | None = None, seed: int | None = None) -> None:
        self.config = config or ReferenceEnvConfig()

        self.action_space = DiscreteActionSpace(N_ACTIONS)
        self.observation_space = BoxObservationSpace(
            low=-1.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        self._rng = Pcg32Rng(0 if seed is None else seed)
        self._obs = np.zeros((OBS_DIM,), dtype=np.float32)
        self._needs_reset = False

        self._init_buffers()
        self.reset(seed=seed)

    def _init_buffers(self) -> None:
        self.node_count = 0
        self.current_node = 0

        self.node_type = np.full((MAX_NODES,), NODE_CLUSTER, dtype=np.int32)
        self.node_hazard = np.zeros((MAX_NODES,), dtype=np.float32)
        self.node_pirate = np.zeros((MAX_NODES,), dtype=np.float32)

        self.neighbors = np.full((MAX_NODES, MAX_NEIGHBORS), -1, dtype=np.int32)
        self.edge_travel_time = np.ones((MAX_NODES, MAX_NEIGHBORS), dtype=np.int32)
        self.edge_fuel_cost = np.zeros((MAX_NODES, MAX_NEIGHBORS), dtype=np.float32)
        self.edge_threat_true = np.zeros((MAX_NODES, MAX_NEIGHBORS), dtype=np.float32)
        self.edge_threat_est = np.full((MAX_NODES, MAX_NEIGHBORS), 0.5, dtype=np.float32)

        self.ast_valid = np.zeros((MAX_NODES, MAX_ASTEROIDS), dtype=np.int8)
        self.true_comp = np.zeros((MAX_NODES, MAX_ASTEROIDS, N_COMMODITIES), dtype=np.float32)
        self.richness = np.zeros((MAX_NODES, MAX_ASTEROIDS), dtype=np.float32)
        self.stability_true = np.zeros((MAX_NODES, MAX_ASTEROIDS), dtype=np.float32)
        self.noise_profile = np.zeros((MAX_NODES, MAX_ASTEROIDS), dtype=np.float32)

        self.comp_est = np.zeros((MAX_NODES, MAX_ASTEROIDS, N_COMMODITIES), dtype=np.float32)
        self.stability_est = np.zeros((MAX_NODES, MAX_ASTEROIDS), dtype=np.float32)
        self.scan_conf = np.zeros((MAX_NODES, MAX_ASTEROIDS), dtype=np.float32)
        self.depletion = np.zeros((MAX_NODES, MAX_ASTEROIDS), dtype=np.float32)

        self.price = np.array(self.config.price_base, dtype=np.float32)
        self.prev_price = np.array(self.config.price_base, dtype=np.float32)
        self.price_phase = np.zeros((N_COMMODITIES,), dtype=np.float32)
        self.price_period = np.full((N_COMMODITIES,), 256.0, dtype=np.float32)
        self.price_amp = np.zeros((N_COMMODITIES,), dtype=np.float32)

        self.station_inventory = np.zeros((N_COMMODITIES,), dtype=np.float32)
        self.recent_sales = np.zeros((N_COMMODITIES,), dtype=np.float32)

        self.selected_asteroid = -1
        self.escape_buff_ticks = 0
        self.stabilize_buff_ticks = np.zeros((MAX_ASTEROIDS,), dtype=np.int32)

        self.fuel = float(FUEL_MAX)
        self.hull = float(HULL_MAX)
        self.heat = 0.0
        self.tool_condition = float(TOOL_MAX)
        self.alert = 0.0
        self.time_remaining = float(self.config.time_max)
        self.credits = 0.0
        self.cargo = np.zeros((N_COMMODITIES,), dtype=np.float32)

        self.repair_kits = 0
        self.stabilizers = 0
        self.decoys = 0

        self.ticks_elapsed = 0
        self.total_spend = 0.0

        self.overheat_ticks = 0
        self.pirate_encounters = 0
        self.value_lost_to_pirates = 0.0
        self.scan_count = 0
        self.mining_ticks = 0

        self._fuel_start = float(FUEL_MAX)
        self._hull_start = float(HULL_MAX)
        self._tool_start = float(TOOL_MAX)

        self._cargo_util_sum = 0.0
        self._cargo_util_count = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if seed is not None:
            self._rng = Pcg32Rng(0 if seed is None else seed)

        self._init_buffers()
        self._generate_world()

        self.selected_asteroid = -1
        self.escape_buff_ticks = 0
        self.stabilize_buff_ticks.fill(0)

        self.fuel = float(FUEL_MAX)
        self.hull = float(HULL_MAX)
        self.heat = 0.0
        self.tool_condition = float(TOOL_MAX)
        self.alert = 0.0
        self.time_remaining = float(self.config.time_max)
        self.credits = 0.0
        self.cargo.fill(0.0)

        self.repair_kits = 3
        self.stabilizers = 2
        self.decoys = 1

        self.ticks_elapsed = 0
        self.total_spend = 0.0

        self.overheat_ticks = 0
        self.pirate_encounters = 0
        self.value_lost_to_pirates = 0.0
        self.scan_count = 0
        self.mining_ticks = 0

        self._fuel_start = self.fuel
        self._hull_start = self.hull
        self._tool_start = self.tool_condition
        self._cargo_util_sum = 0.0
        self._cargo_util_count = 0.0

        self._needs_reset = False

        obs = self._build_observation()
        info = self._build_info(
            action=-1,
            dt=0,
            invalid_action=False,
            destroyed=False,
            stranded=False,
            terminated=False,
            truncated=False,
        )
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise RuntimeError("Episode ended. Call reset() before step().")

        action_int = int(action)

        snapshot = _StepSnapshot(
            credits_before=self.credits,
            fuel_before=self.fuel,
            hull_before=self.hull,
            heat_before=self.heat,
            tool_before=self.tool_condition,
            cargo_before=self.cargo.copy(),
            cargo_value_before=self._est_cargo_value(),
            value_lost_to_pirates_before=self.value_lost_to_pirates,
        )

        terminated = False
        truncated = False
        invalid_action = False
        dt = 1

        if not self.action_space.contains(action_int):
            invalid_action = True
            action_int = 6

        if 0 <= action_int <= 5:
            dt, invalid_action = self._apply_travel(action_int)
        elif action_int == 6:
            self._apply_hold()
        elif action_int == 7:
            self._apply_emergency_burn()
        elif action_int == 8:
            dt = self.config.wide_scan_time
            self.fuel -= self.config.wide_scan_fuel
            self.alert += self.config.wide_scan_alert
            self._update_cluster_priors_with_noise()
            self.scan_count += 1
        elif action_int == 9:
            dt = self.config.focused_scan_time
            self.fuel -= self.config.focused_scan_fuel
            self.alert += self.config.focused_scan_alert
            if not self._selected_asteroid_valid():
                invalid_action = True
            else:
                self._update_asteroid_estimates(self.selected_asteroid, mode="focused")
                self.scan_count += 1
        elif action_int == 10:
            dt = self.config.deep_scan_time
            self.fuel -= self.config.deep_scan_fuel
            self.alert += self.config.deep_scan_alert
            if not self._selected_asteroid_valid():
                invalid_action = True
            else:
                self._update_asteroid_estimates(self.selected_asteroid, mode="deep")
                self.scan_count += 1
        elif action_int == 11:
            dt = self.config.threat_listen_time
            self._update_neighbor_threat_estimates()
        elif 12 <= action_int <= 27:
            invalid_action = not self._select_asteroid(action_int - 12)
        elif action_int in (28, 29, 30):
            if not self._selected_asteroid_valid():
                invalid_action = True
            else:
                self._mine_selected(action_int)
        elif action_int == 31:
            dt = self.config.stabilize_time
            if not self._selected_asteroid_valid() or self.stabilizers <= 0:
                invalid_action = True
            else:
                self.stabilizers -= 1
                self.stabilize_buff_ticks[self.selected_asteroid] = self.config.stabilize_buff_ticks
        elif action_int == 32:
            dt = self.config.refine_time
            self.fuel -= self.config.refine_fuel
            self.heat += self.config.refine_heat
            self.alert += self.config.refine_alert
            self._refine_some_cargo()
        elif action_int == 33:
            dt = self.config.cooldown_time
            self.fuel -= self.config.cooldown_fuel
            self.heat = max(0.0, self.heat - self.config.cooldown_amount)
            self.alert += self.config.cooldown_alert
        elif action_int == 34:
            dt = self.config.maint_time
            if self.repair_kits <= 0:
                invalid_action = True
            else:
                self.repair_kits -= 1
                self.tool_condition = min(
                    TOOL_MAX, self.tool_condition + self.config.tool_repair_amount
                )
        elif action_int == 35:
            dt = self.config.patch_time
            if self.repair_kits <= 0:
                invalid_action = True
            else:
                self.repair_kits -= 1
                self.hull = min(HULL_MAX, self.hull + self.config.hull_patch_amount)
        elif 36 <= action_int <= 41:
            c_idx = action_int - 36
            self.cargo[c_idx] = 0.0
            self.alert = max(0.0, self.alert - self.config.jettison_alert_relief)
        elif action_int == 42:
            dt = self.config.dock_time
            if not self._is_at_station():
                invalid_action = True
            else:
                self.alert = max(0.0, self.alert - self.config.dock_alert_drop)
        elif 43 <= action_int <= 60:
            if not self._is_at_station():
                invalid_action = True
            else:
                self._sell_action(action_int)
        elif 61 <= action_int <= 66:
            if not self._is_at_station() or not self._purchase_station_item(action_int):
                invalid_action = True
        elif action_int == 67:
            dt = self.config.overhaul_time
            if not self._is_at_station() or self.credits < self.config.overhaul_cost:
                invalid_action = True
            else:
                self.credits -= self.config.overhaul_cost
                self.total_spend += self.config.overhaul_cost
                self.hull = HULL_MAX
                self.tool_condition = TOOL_MAX
        elif action_int == 68:
            terminated = True
        else:
            invalid_action = True

        if invalid_action:
            dt = 1
            self._apply_hold()

        self._apply_global_dynamics(dt=dt)
        self.ticks_elapsed += int(dt)

        destroyed = self.hull <= 0.0
        stranded = self.fuel <= 0.0 and not self._is_at_station()

        if destroyed or stranded:
            terminated = True

        if self.time_remaining <= 0.0 and not terminated:
            truncated = True

        done = terminated or truncated

        cargo_value_after = self._est_cargo_value()
        reward = compute_reward(
            snapshot,
            credits_after=self.credits,
            fuel_after=self.fuel,
            hull_after=self.hull,
            heat_after=self.heat,
            tool_after=self.tool_condition,
            cargo_after=self.cargo,
            cargo_value_after=cargo_value_after,
            value_lost_to_pirates_after=self.value_lost_to_pirates,
            action=action_int,
            dt=dt,
            invalid=invalid_action,
            destroyed=destroyed,
            stranded=stranded,
            done=done,
            cfg=self.config.reward_cfg,
        )

        obs = self._build_observation()
        info = self._build_info(
            action=action_int,
            dt=dt,
            invalid_action=invalid_action,
            destroyed=destroyed,
            stranded=stranded,
            terminated=terminated,
            truncated=truncated,
        )

        self._needs_reset = done
        return obs, reward, terminated, truncated, info

    def _generate_world(self) -> None:
        self.node_count = int(self._rng.integers(8, MAX_NODES + 1))
        self.current_node = 0

        self.node_type.fill(NODE_CLUSTER)
        self.node_hazard.fill(0.0)
        self.node_pirate.fill(0.0)
        self.node_type[0] = NODE_STATION

        for node in range(1, self.node_count):
            is_hazard = float(self._rng.random()) < 0.25
            self.node_type[node] = NODE_HAZARD if is_hazard else NODE_CLUSTER
            self.node_hazard[node] = np.float32(self._rng.uniform(0.05, 0.35))
            self.node_pirate[node] = np.float32(self._rng.uniform(0.05, 0.30))
            if is_hazard:
                self.node_hazard[node] = np.float32(
                    _clamp(float(self.node_hazard[node]) + 0.25, 0.0, 1.0)
                )
                self.node_pirate[node] = np.float32(
                    _clamp(float(self.node_pirate[node]) + 0.12, 0.0, 1.0)
                )

        self.neighbors.fill(-1)
        self.edge_travel_time.fill(1)
        self.edge_fuel_cost.fill(0.0)
        self.edge_threat_true.fill(0.0)
        self.edge_threat_est.fill(0.5)

        for node in range(1, self.node_count):
            parent = int(self._rng.integers(0, node))
            self._add_edge(node, parent)

        extra_attempts = max(0, self.node_count)
        for _ in range(extra_attempts):
            u = int(self._rng.integers(0, self.node_count))
            v = int(self._rng.integers(0, self.node_count))
            if u == v:
                continue
            self._add_edge(u, v)

        self._generate_asteroids()
        self._generate_market()

    def _add_edge(self, u: int, v: int) -> None:
        if u >= self.node_count or v >= self.node_count:
            return
        if self._edge_exists(u, v):
            return

        u_slot = self._first_free_slot(u)
        v_slot = self._first_free_slot(v)
        if u_slot < 0 or v_slot < 0:
            return

        t_time = int(self._rng.integers(1, int(self.config.travel_time_max) + 1))
        fuel_cost = float(self._rng.uniform(20.0, self.config.travel_fuel_cost_max * 0.7))

        threat = float(
            _clamp(
                0.5 * (self.node_hazard[u] + self.node_hazard[v])
                + 0.5 * (self.node_pirate[u] + self.node_pirate[v])
                + float(self._rng.normal(0.0, 0.05)),
                0.0,
                1.0,
            )
        )

        self.neighbors[u, u_slot] = v
        self.neighbors[v, v_slot] = u

        self.edge_travel_time[u, u_slot] = t_time
        self.edge_travel_time[v, v_slot] = t_time

        self.edge_fuel_cost[u, u_slot] = fuel_cost
        self.edge_fuel_cost[v, v_slot] = fuel_cost

        self.edge_threat_true[u, u_slot] = threat
        self.edge_threat_true[v, v_slot] = threat
        self.edge_threat_est[u, u_slot] = 0.5
        self.edge_threat_est[v, v_slot] = 0.5

    def _edge_exists(self, u: int, v: int) -> bool:
        return bool(np.any(self.neighbors[u] == v))

    def _first_free_slot(self, node: int) -> int:
        slots = np.where(self.neighbors[node] < 0)[0]
        if slots.size == 0:
            return -1
        return int(slots[0])

    def _generate_asteroids(self) -> None:
        self.ast_valid.fill(0)
        self.true_comp.fill(0.0)
        self.richness.fill(0.0)
        self.stability_true.fill(0.0)
        self.noise_profile.fill(0.0)

        self.comp_est.fill(0.0)
        self.stability_est.fill(0.0)
        self.scan_conf.fill(0.0)
        self.depletion.fill(0.0)

        for node in range(self.node_count):
            if self.node_type[node] == NODE_STATION:
                continue

            n_ast = int(self._rng.integers(5, MAX_ASTEROIDS + 1))
            self.ast_valid[node, :n_ast] = 1

            for a_idx in range(n_ast):
                true_comp = self._rng.dirichlet(np.ones((N_COMMODITIES,), dtype=np.float64))
                self.true_comp[node, a_idx] = true_comp.astype(np.float32)

                self.richness[node, a_idx] = float(
                    _clamp(float(self._rng.lognormal(mean=-0.2, sigma=0.65)), 0.2, 4.0)
                )
                self.stability_true[node, a_idx] = float(self._rng.beta(3.0, 2.0))
                self.noise_profile[node, a_idx] = float(self._rng.uniform(0.04, 0.22))

                self.comp_est[node, a_idx] = (
                    self._rng.dirichlet(np.ones((N_COMMODITIES,), dtype=np.float64))
                ).astype(np.float32)
                self.stability_est[node, a_idx] = 0.5
                self.scan_conf[node, a_idx] = 0.10
                self.depletion[node, a_idx] = 0.0

    def _generate_market(self) -> None:
        self.recent_sales.fill(0.0)

        base = np.array(self.config.price_base, dtype=np.float64)

        for c_idx in range(N_COMMODITIES):
            self.station_inventory[c_idx] = np.float32(self._rng.uniform(20.0, 120.0))

            phase = float(self._rng.uniform(0.0, 2.0 * np.pi))
            period = float(self._rng.uniform(180.0, 380.0))
            amp_factor = float(self._rng.uniform(0.10, 0.30))
            self.price_phase[c_idx] = np.float32(phase)
            self.price_period[c_idx] = np.float32(period)
            self.price_amp[c_idx] = np.float32(base[c_idx] * amp_factor)

            cycle = float(self.price_amp[c_idx]) * float(np.sin(float(self.price_phase[c_idx])))
            self.price[c_idx] = np.float32(
                _clamp(
                    float(base[c_idx] + cycle),
                    float(self.config.price_min[c_idx]),
                    float(self.config.price_max[c_idx]),
                )
            )
            self.prev_price[c_idx] = self.price[c_idx]

    def _apply_hold(self) -> None:
        self.alert = max(0.0, self.alert - self.config.alert_decay_hold)
        self._passive_heat_dissipation(1)

    def _apply_emergency_burn(self) -> None:
        self.fuel -= self.config.emergency_burn_fuel
        self.alert += self.config.emergency_burn_alert
        self.escape_buff_ticks = max(self.escape_buff_ticks, self.config.escape_buff_ticks)

    def _apply_travel(self, slot: int) -> tuple[int, bool]:
        neighbor = int(self.neighbors[self.current_node, slot])
        if neighbor < 0:
            return 1, True

        dt = int(self.edge_travel_time[self.current_node, slot])
        mass_factor = 1.0 + 0.5 * (float(np.sum(self.cargo)) / CARGO_MAX)
        fuel_cost = float(self.edge_fuel_cost[self.current_node, slot]) * mass_factor
        threat = float(self.edge_threat_true[self.current_node, slot])

        self.fuel -= fuel_cost
        self.current_node = neighbor
        self.selected_asteroid = -1

        self._apply_edge_hazards_and_pirates(dt=dt, edge_threat=threat)
        return dt, False

    def _apply_edge_hazards_and_pirates(self, dt: int, edge_threat: float) -> None:
        if dt <= 0:
            return

        hazard_dmg = float(dt) * edge_threat * self.config.hazard_damage_per_tick
        hazard_dmg *= float(self._rng.uniform(0.85, 1.15))
        self.hull -= hazard_dmg

        self.heat += float(dt) * edge_threat * self.config.hazard_heat_per_tick
        self.alert += float(dt) * edge_threat * self.config.hazard_alert_per_tick

        self._maybe_pirate_encounter(dt=dt, intensity=edge_threat)

    def _update_cluster_priors_with_noise(self) -> None:
        node = self.current_node
        valid_indices = np.where(self.ast_valid[node] > 0)[0]
        for a_idx in valid_indices:
            self._update_asteroid_estimates(int(a_idx), mode="wide")

    def _update_asteroid_estimates(self, asteroid: int, mode: str) -> None:
        if self.ast_valid[self.current_node, asteroid] == 0:
            return

        if mode == "wide":
            blend = 0.22
            conf_gain = 0.10
            noise_mult = 1.35
        elif mode == "focused":
            blend = 0.42
            conf_gain = 0.20
            noise_mult = 1.00
        else:
            blend = 0.80
            conf_gain = 0.45
            noise_mult = 0.55

        node = self.current_node
        base_noise = float(self.noise_profile[node, asteroid])
        conf = float(self.scan_conf[node, asteroid])
        sigma = base_noise * (1.0 - conf + 0.1) * noise_mult

        noisy_truth = self.true_comp[node, asteroid].astype(np.float64) + self._rng.normal(
            0.0, sigma, N_COMMODITIES
        )
        noisy_truth = _normalize_probs(noisy_truth)

        prev_est = self.comp_est[node, asteroid].astype(np.float64)
        new_est = _normalize_probs((1.0 - blend) * prev_est + blend * noisy_truth)
        self.comp_est[node, asteroid] = new_est.astype(np.float32)

        stable_truth = float(self.stability_true[node, asteroid])
        stable_noisy = _clamp(stable_truth + float(self._rng.normal(0.0, sigma)), 0.0, 1.0)
        stable_est = (1.0 - blend) * float(
            self.stability_est[node, asteroid]
        ) + blend * stable_noisy
        self.stability_est[node, asteroid] = np.float32(_clamp(stable_est, 0.0, 1.0))

        conf_next = _clamp(conf + conf_gain, 0.0, 1.0)
        self.scan_conf[node, asteroid] = np.float32(conf_next)

    def _update_neighbor_threat_estimates(self) -> None:
        for slot in range(MAX_NEIGHBORS):
            neighbor = int(self.neighbors[self.current_node, slot])
            if neighbor < 0:
                continue
            truth = float(self.edge_threat_true[self.current_node, slot])
            est = float(self.edge_threat_est[self.current_node, slot])
            noisy = _clamp(truth + float(self._rng.normal(0.0, 0.08)), 0.0, 1.0)
            self.edge_threat_est[self.current_node, slot] = np.float32(0.25 * est + 0.75 * noisy)

    def _select_asteroid(self, asteroid: int) -> bool:
        if asteroid < 0 or asteroid >= MAX_ASTEROIDS:
            return False
        if self.ast_valid[self.current_node, asteroid] == 0:
            return False
        if float(self.depletion[self.current_node, asteroid]) >= 1.0:
            return False
        self.selected_asteroid = asteroid
        return True

    def _selected_asteroid_valid(self) -> bool:
        if self.selected_asteroid < 0 or self.selected_asteroid >= MAX_ASTEROIDS:
            return False
        if self.ast_valid[self.current_node, self.selected_asteroid] == 0:
            return False
        if float(self.depletion[self.current_node, self.selected_asteroid]) >= 1.0:
            return False
        return True

    def _mine_selected(self, action: int) -> None:
        mode = {28: "cons", 29: "std", 30: "agg"}[action]
        a_idx = self.selected_asteroid
        node = self.current_node

        mode_mult = {"cons": 0.80, "std": 1.15, "agg": 1.55}[mode]
        heat_gain = {"cons": 2.0, "std": 4.0, "agg": 7.0}[mode]
        wear_gain = {"cons": 0.8, "std": 1.6, "agg": 2.8}[mode]
        alert_gain = {"cons": 1.2, "std": 2.2, "agg": 4.0}[mode]
        sigma = {"cons": 0.05, "std": 0.10, "agg": 0.16}[mode]
        fracture_bias = {"cons": -0.7, "std": 0.0, "agg": 0.8}[mode]

        richness = float(self.richness[node, a_idx])
        depletion = float(self.depletion[node, a_idx])
        base = richness * max(0.0, 1.0 - depletion)

        tool_frac = _clamp(self.tool_condition / TOOL_MAX, 0.0, 1.0)
        heat_frac = _clamp(self.heat / HEAT_MAX, 0.0, 2.0)

        eff_tool = 0.4 + 0.6 * tool_frac
        eff_heat = 1.0 if heat_frac <= 0.7 else max(0.1, 1.0 - (heat_frac - 0.7) / 0.3)

        noise = float(np.exp(self._rng.normal(0.0, sigma)))
        extracted = (
            base
            * eff_tool
            * eff_heat
            * mode_mult
            * noise
            * self.true_comp[node, a_idx].astype(np.float64)
        )

        available_capacity = max(0.0, CARGO_MAX - float(np.sum(self.cargo)))
        total_extracted = float(np.sum(extracted))
        if total_extracted > available_capacity and total_extracted > 0.0:
            extracted *= available_capacity / total_extracted
            total_extracted = available_capacity

        self.cargo += extracted.astype(np.float32)
        self.heat += heat_gain
        self.tool_condition -= wear_gain
        self.alert += alert_gain

        self.depletion[node, a_idx] = np.float32(
            _clamp(
                float(self.depletion[node, a_idx])
                + self.config.fracture_depletion_rate * total_extracted,
                0.0,
                1.0,
            )
        )

        self.mining_ticks += 1

        stabilized = self.stabilize_buff_ticks[a_idx] > 0
        logit = (
            -3.1
            + fracture_bias
            + 2.5 * (1.0 - float(self.stability_true[node, a_idx]))
            + 2.2 * max(0.0, heat_frac - 0.7)
            + 1.5 * (1.0 - tool_frac)
            - (1.1 if stabilized else 0.0)
        )

        if self._rng.random() < _sigmoid(logit):
            severity = float(self._rng.uniform(0.5, 1.0))
            self.hull -= 12.0 * severity
            self.depletion[node, a_idx] = 1.0
            self.node_hazard[node] = np.float32(
                _clamp(float(self.node_hazard[node]) + 0.1, 0.0, 1.0)
            )

    def _refine_some_cargo(self) -> None:
        low_value = float(self.cargo[0] + self.cargo[1])
        if low_value <= 0.0:
            return

        refine_input = 0.15 * low_value
        total_low = float(self.cargo[0] + self.cargo[1])
        if total_low <= 0.0:
            return

        take_ratio = min(1.0, refine_input / total_low)
        self.cargo[0] *= np.float32(1.0 - take_ratio)
        self.cargo[1] *= np.float32(1.0 - take_ratio)

        output = 0.65 * refine_input
        self.cargo[4] += np.float32(output)

    def _sell_action(self, action: int) -> None:
        c_idx = (action - 43) // 3
        bucket = (action - 43) % 3
        frac = (0.25, 0.50, 1.00)[bucket]

        qty = float(self.cargo[c_idx]) * frac
        if qty <= 0.0:
            return

        slippage = self._slippage(qty, float(self.station_inventory[c_idx]))
        effective_price = float(self.price[c_idx]) * (1.0 - slippage)

        self.credits += qty * effective_price
        self.cargo[c_idx] = np.float32(max(0.0, float(self.cargo[c_idx]) - qty))
        self.station_inventory[c_idx] += np.float32(qty)
        self.recent_sales[c_idx] += np.float32(qty)

    def _purchase_station_item(self, action: int) -> bool:
        if action == 61:
            return self._buy_fuel(self.config.buy_fuel_small_qty, self.config.buy_fuel_small_cost)
        if action == 62:
            return self._buy_fuel(self.config.buy_fuel_med_qty, self.config.buy_fuel_med_cost)
        if action == 63:
            return self._buy_fuel(self.config.buy_fuel_large_qty, self.config.buy_fuel_large_cost)
        if action == 64:
            return self._buy_supply(kind="repair")
        if action == 65:
            return self._buy_supply(kind="stabilizer")
        if action == 66:
            return self._buy_supply(kind="decoy")
        return False

    def _buy_fuel(self, qty: float, cost: float) -> bool:
        if self.credits < cost:
            return False
        self.credits -= cost
        self.total_spend += cost
        self.fuel = min(FUEL_MAX, self.fuel + qty)
        return True

    def _buy_supply(self, kind: str) -> bool:
        if kind == "repair":
            cost = self.config.buy_repair_kit_cost
            if self.credits < cost or self.repair_kits >= self.config.repair_kits_cap:
                return False
            self.credits -= cost
            self.total_spend += cost
            self.repair_kits += 1
            return True

        if kind == "stabilizer":
            cost = self.config.buy_stabilizer_cost
            if self.credits < cost or self.stabilizers >= self.config.stabilizers_cap:
                return False
            self.credits -= cost
            self.total_spend += cost
            self.stabilizers += 1
            return True

        cost = self.config.buy_decoy_cost
        if self.credits < cost or self.decoys >= self.config.decoys_cap:
            return False
        self.credits -= cost
        self.total_spend += cost
        self.decoys += 1
        return True

    def _slippage(self, qty: float, inventory: float) -> float:
        if qty <= 0.0:
            return 0.0
        ratio = qty / max(1.0, inventory + qty)
        raw = self.config.slippage_k * ratio + self.config.slippage_root * np.sqrt(ratio)
        return float(_clamp(float(raw), 0.0, 0.70))

    def _apply_global_dynamics(self, dt: int) -> None:
        self.time_remaining -= float(dt)

        self._passive_heat_dissipation(dt)
        if self.escape_buff_ticks > 0:
            self.escape_buff_ticks = max(0, self.escape_buff_ticks - dt)

        for a_idx in range(MAX_ASTEROIDS):
            if self.stabilize_buff_ticks[a_idx] > 0:
                self.stabilize_buff_ticks[a_idx] = max(0, self.stabilize_buff_ticks[a_idx] - dt)

        if self.heat > HEAT_MAX:
            overflow = self.heat - HEAT_MAX
            self.hull -= self.config.overheat_damage_per_unit * overflow
            self.heat = HEAT_MAX
            self.overheat_ticks += dt

        if not self._is_at_station():
            self._apply_node_hazards(dt)
            self._maybe_pirate_encounter(dt=dt)

        self._update_market(dt)
        self._clamp_state()
        self._track_cargo_utilization(dt)

    def _passive_heat_dissipation(self, dt: int) -> None:
        self.heat = max(0.0, self.heat - self.config.heat_dissipation_per_tick * float(dt))

    def _apply_node_hazards(self, dt: int) -> None:
        hazard = float(self.node_hazard[self.current_node])
        if hazard <= 0.0:
            return

        hull_damage = (
            float(dt)
            * hazard
            * self.config.hazard_damage_per_tick
            * float(self._rng.uniform(0.8, 1.2))
        )
        heat_gain = float(dt) * hazard * self.config.hazard_heat_per_tick
        alert_gain = float(dt) * hazard * self.config.hazard_alert_per_tick

        self.hull -= hull_damage
        self.heat += heat_gain
        self.alert += alert_gain

    def _maybe_pirate_encounter(self, dt: int, intensity: float | None = None) -> None:
        if self._is_at_station():
            return

        pirate_intensity = (
            float(self.node_pirate[self.current_node]) if intensity is None else intensity
        )
        cargo_value_before = self._est_cargo_value()

        logit = (
            self.config.pirate_bias
            + self.config.pirate_intensity_w * pirate_intensity
            + self.config.pirate_alert_w * _clamp(self.alert / ALERT_MAX, 0.0, 1.0)
            + self.config.pirate_cargo_w * np.log1p(cargo_value_before / CREDIT_SCALE)
            - self.config.pirate_escape_w * float(self.escape_buff_ticks > 0)
        )

        base_prob = _sigmoid(float(logit))
        p_encounter = 1.0 - (1.0 - base_prob) ** float(max(1, dt))
        if self._rng.random() >= p_encounter:
            return

        self.pirate_encounters += 1
        loss_frac = float(self._rng.uniform(0.08, 0.20))

        if self.decoys > 0 and self._rng.random() < 0.6:
            self.decoys -= 1
            loss_frac *= 0.3

        self.cargo *= np.float32(1.0 - loss_frac)
        cargo_value_after = self._est_cargo_value()
        self.value_lost_to_pirates += max(0.0, cargo_value_before - cargo_value_after)

        self.hull -= float(self._rng.uniform(1.0, 4.0))
        self.alert += 8.0

    def _update_market(self, dt: int) -> None:
        self.prev_price = self.price.copy()

        base = np.array(self.config.price_base, dtype=np.float64)
        p_min = np.array(self.config.price_min, dtype=np.float64)
        p_max = np.array(self.config.price_max, dtype=np.float64)

        t = float(self.ticks_elapsed + dt)
        cycles = self.price_amp.astype(np.float64) * np.sin(
            2.0 * np.pi * (t / self.price_period.astype(np.float64))
            + self.price_phase.astype(np.float64)
        )
        inv_pressure = self.config.inventory_pressure_k * self.station_inventory.astype(np.float64)
        sale_pressure = self.config.sales_pressure_k * self.recent_sales.astype(np.float64)

        noise_std = self.config.market_noise_k * base * np.sqrt(float(max(dt, 1)))
        noise = self._rng.normal(0.0, noise_std)

        new_price = np.clip(base + cycles - inv_pressure - sale_pressure + noise, p_min, p_max)
        self.price = new_price.astype(np.float32)

        decay = float(np.exp(-float(dt) / self.config.sales_decay_tau))
        self.recent_sales *= np.float32(decay)
        self.station_inventory = np.maximum(self.station_inventory * np.float32(0.998), 0.0)

    def _clamp_state(self) -> None:
        self.fuel = _clamp(self.fuel, 0.0, FUEL_MAX)
        self.hull = _clamp(self.hull, 0.0, HULL_MAX)
        self.heat = _clamp(self.heat, 0.0, HEAT_MAX)
        self.tool_condition = _clamp(self.tool_condition, 0.0, TOOL_MAX)
        self.alert = _clamp(self.alert, 0.0, ALERT_MAX)
        self.time_remaining = _clamp(self.time_remaining, 0.0, self.config.time_max)

        self.cargo = np.clip(self.cargo, 0.0, CARGO_MAX).astype(np.float32)
        total_cargo = float(np.sum(self.cargo))
        if total_cargo > CARGO_MAX and total_cargo > 0.0:
            self.cargo *= np.float32(CARGO_MAX / total_cargo)

    def _track_cargo_utilization(self, dt: int) -> None:
        frac = _clamp(float(np.sum(self.cargo)) / CARGO_MAX, 0.0, 1.0)
        self._cargo_util_sum += frac * float(dt)
        self._cargo_util_count += float(dt)

    def _est_cargo_value(self) -> float:
        return float(np.dot(self.cargo.astype(np.float64), self.price.astype(np.float64)))

    def _is_at_station(self) -> bool:
        return int(self.node_type[self.current_node]) == NODE_STATION

    def _steps_to_station(self) -> int:
        if self.current_node == 0:
            return 0

        visited = np.zeros((MAX_NODES,), dtype=np.int8)
        queue: deque[tuple[int, int]] = deque()

        visited[self.current_node] = 1
        queue.append((self.current_node, 0))

        while queue:
            node, dist = queue.popleft()
            for slot in range(MAX_NEIGHBORS):
                nxt = int(self.neighbors[node, slot])
                if nxt < 0 or nxt >= self.node_count or visited[nxt]:
                    continue
                if nxt == 0:
                    return dist + 1
                visited[nxt] = 1
                queue.append((nxt, dist + 1))

        return MAX_NODES - 1

    def _build_observation(self) -> np.ndarray:
        obs = self._obs
        obs.fill(0.0)

        cargo_total = float(np.sum(self.cargo))

        obs[0] = np.float32(_clamp(self.fuel / FUEL_MAX, 0.0, 1.0))
        obs[1] = np.float32(_clamp(self.hull / HULL_MAX, 0.0, 1.0))
        obs[2] = np.float32(_clamp(self.heat / HEAT_MAX, 0.0, 1.0))
        obs[3] = np.float32(_clamp(self.tool_condition / TOOL_MAX, 0.0, 1.0))
        obs[4] = np.float32(_clamp(cargo_total / CARGO_MAX, 0.0, 1.0))
        obs[5] = np.float32(_clamp(self.alert / ALERT_MAX, 0.0, 1.0))
        obs[6] = np.float32(_clamp(self.time_remaining / self.config.time_max, 0.0, 1.0))

        credits_norm = np.log1p(max(0.0, self.credits)) / np.log1p(self.config.credits_cap)
        obs[7] = np.float32(_clamp(float(credits_norm), 0.0, 1.0))

        obs[8:14] = np.clip(self.cargo / CARGO_MAX, 0.0, 1.0).astype(np.float32)
        obs[14] = np.float32(
            _clamp(self.repair_kits / float(self.config.repair_kits_cap), 0.0, 1.0)
        )
        obs[15] = np.float32(
            _clamp(self.stabilizers / float(self.config.stabilizers_cap), 0.0, 1.0)
        )
        obs[16] = np.float32(_clamp(self.decoys / float(self.config.decoys_cap), 0.0, 1.0))

        obs[17] = np.float32(float(self._is_at_station()))
        obs[18] = np.float32(float(self._selected_asteroid_valid()))

        node_type = int(self.node_type[self.current_node])
        obs[19:22] = 0.0
        obs[19 + node_type] = 1.0

        obs[22] = np.float32(_clamp(self.current_node / float(MAX_NODES - 1), 0.0, 1.0))
        obs[23] = np.float32(_clamp(self._steps_to_station() / float(MAX_NODES - 1), 0.0, 1.0))

        for k in range(MAX_NEIGHBORS):
            base = 24 + 7 * k
            neighbor = int(self.neighbors[self.current_node, k])
            if neighbor < 0:
                continue

            obs[base] = 1.0
            neigh_type = int(self.node_type[neighbor])
            obs[base + 1 + neigh_type] = 1.0

            travel_time = float(self.edge_travel_time[self.current_node, k])
            travel_fuel = float(self.edge_fuel_cost[self.current_node, k])
            threat = float(self.edge_threat_est[self.current_node, k])

            obs[base + 4] = np.float32(_clamp(travel_time / self.config.travel_time_max, 0.0, 1.0))
            obs[base + 5] = np.float32(
                _clamp(travel_fuel / self.config.travel_fuel_cost_max, 0.0, 1.0)
            )
            obs[base + 6] = np.float32(_clamp(threat, 0.0, 1.0))

        for a_idx in range(MAX_ASTEROIDS):
            base = 68 + 11 * a_idx
            if self.ast_valid[self.current_node, a_idx] == 0:
                continue

            obs[base] = 1.0
            comp = _normalize_probs(self.comp_est[self.current_node, a_idx].astype(np.float64))
            obs[base + 1 : base + 7] = comp.astype(np.float32)
            obs[base + 7] = np.float32(
                _clamp(float(self.stability_est[self.current_node, a_idx]), 0.0, 1.0)
            )
            obs[base + 8] = np.float32(
                _clamp(float(self.depletion[self.current_node, a_idx]), 0.0, 1.0)
            )
            obs[base + 9] = np.float32(
                _clamp(float(self.scan_conf[self.current_node, a_idx]), 0.0, 1.0)
            )
            obs[base + 10] = np.float32(float(a_idx == self.selected_asteroid))

        price_base = np.array(self.config.price_base, dtype=np.float64)
        price_norm = np.divide(
            self.price.astype(np.float64),
            price_base,
            out=np.zeros_like(price_base),
            where=price_base > 0,
        )
        obs[MKT_PRICE_BASE : MKT_PRICE_BASE + N_COMMODITIES] = np.clip(price_norm, 0.0, 1.0).astype(
            np.float32
        )

        d_price = (
            self.price.astype(np.float64) - self.prev_price.astype(np.float64)
        ) / self.config.price_scale
        obs[MKT_DPRICE_BASE : MKT_DPRICE_BASE + N_COMMODITIES] = np.clip(d_price, -1.0, 1.0).astype(
            np.float32
        )

        inv_indices = (0, 2, 3, 4)
        for i, c_idx in enumerate(inv_indices):
            obs[MKT_INV_BASE + i] = np.float32(
                _clamp(
                    float(self.station_inventory[c_idx]) / self.config.station_inventory_norm_cap,
                    0.0,
                    1.0,
                )
            )

        return obs.copy()

    def _build_info(
        self,
        *,
        action: int,
        dt: int,
        invalid_action: bool,
        destroyed: bool,
        stranded: bool,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, Any]:
        net_profit = float(self.credits - self.total_spend)
        profit_per_tick = net_profit / float(max(1, self.ticks_elapsed))

        cargo_util_avg = 0.0
        if self._cargo_util_count > 0.0:
            cargo_util_avg = self._cargo_util_sum / self._cargo_util_count

        info = {
            "action": int(action),
            "dt": int(dt),
            "invalid_action": bool(invalid_action),
            "credits": float(self.credits),
            "net_profit": float(net_profit),
            "profit_per_tick": float(profit_per_tick),
            "survival": float(0.0 if destroyed or stranded else 1.0),
            "overheat_ticks": float(self.overheat_ticks),
            "pirate_encounters": float(self.pirate_encounters),
            "value_lost_to_pirates": float(self.value_lost_to_pirates),
            "fuel_used": float(max(0.0, self._fuel_start - self.fuel)),
            "hull_damage": float(max(0.0, self._hull_start - self.hull)),
            "tool_wear": float(max(0.0, self._tool_start - self.tool_condition)),
            "scan_count": float(self.scan_count),
            "mining_ticks": float(self.mining_ticks),
            "cargo_utilization_avg": float(_clamp(cargo_util_avg, 0.0, 1.0)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "node_context": "station" if self._is_at_station() else "field",
            "time_remaining": float(self.time_remaining),
        }
        return info


__all__ = [
    "ProspectorReferenceEnv",
    "ReferenceEnvConfig",
    "RewardCfg",
    "compute_reward",
]
