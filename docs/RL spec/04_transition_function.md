## 3) Transition function (Gym step) — pseudocode

This is written as macro-action simulation: one `step(action)` may consume variable “time units” (`dt`) (travel_time, deep_scan_time, etc.) but still returns a single transition.

```python id="guat47"
def step(action: int):
    # --- snapshot for reward deltas ---
    credits_before = credits
    fuel_before    = fuel
    hull_before    = hull
    heat_before    = heat
    tool_before    = tool_condition
    cargo_before   = cargo.copy()          # per commodity
    cargo_value_before = est_cargo_value() # using current market prices

    invalid = False
    dt = 1  # default time cost per action

    # --- decode + apply action ---
    if 0 <= action <= 5:  # TRAVEL_NEIGHBOR[k]
        k = action
        if neighbor_valid[k] == 0:
            invalid = True
        else:
            dt = travel_time[k]
            fuel -= travel_fuel_cost[k]
            node_idx = neighbor_node_idx[k]
            # travel exposure: hazards/pirates along edge (scaled by dt)
            apply_edge_hazards_and_pirates(dt)

    elif action == 6:  # HOLD_DRIFT
        dt = 1
        passive_heat_dissipation(dt)
        alert = max(0, alert - alert_decay_hold * dt)

    elif action == 7:  # EMERGENCY_BURN
        dt = 1
        fuel -= emergency_burn_fuel
        alert += emergency_burn_alert
        set_escape_buff(ticks=escape_buff_ticks)  # reduces pirate encounter chance

    elif action == 8:  # WIDE_SCAN
        dt = wide_scan_time
        fuel -= wide_scan_fuel
        alert += wide_scan_alert
        update_cluster_priors_with_noise()  # improves comp_est priors for asteroids in node

    elif action == 9:  # FOCUSED_SCAN_SELECTED
        dt = focused_scan_time
        fuel -= focused_scan_fuel
        alert += focused_scan_alert
        if not selected_asteroid_valid():
            invalid = True
        else:
            update_asteroid_estimates(selected, mode="focused")

    elif action == 10:  # DEEP_SCAN_SELECTED
        dt = deep_scan_time
        fuel -= deep_scan_fuel
        alert += deep_scan_alert
        if not selected_asteroid_valid():
            invalid = True
        else:
            update_asteroid_estimates(selected, mode="deep")

    elif action == 11:  # PASSIVE_THREAT_LISTEN
        dt = threat_listen_time
        update_neighbor_threat_estimates()  # observation-level improvement (neigh_threat_norm)

    elif 12 <= action <= 27:  # SELECT_ASTEROID[a]
        a = action - 12
        if asteroid_valid[a] == 0:
            invalid = True
        else:
            selected = a
            mining_active = 1

    elif action in (28, 29, 30):  # MINE_*_SELECTED
        if not selected_asteroid_valid():
            invalid = True
        else:
            mode = {28:"cons", 29:"std", 30:"agg"}[action]
            dt = 1
            # stochastic yield based on hidden true composition, richness, depletion, tool, heat
            extracted = mine_one_tick(selected, mode)  # returns per-commodity amounts
            cargo += extracted
            # heat & wear
            heat += heat_gain[mode]
            tool_condition -= wear_gain[mode]
            alert += mining_alert_gain[mode]
            # fracture check (nonlinear with heat, wear, stability)
            if fracture_occurs(selected, mode):
                apply_fracture_consequences(selected)  # asteroid depleted, hull dmg, local hazard

    elif action == 31:  # STABILIZE_SELECTED
        dt = stabilize_time
        if not selected_asteroid_valid() or stabilizers <= 0:
            invalid = True
        else:
            stabilizers -= 1
            apply_stabilize_buff(selected, ticks=stabilize_buff_ticks)

    elif action == 32:  # REFINE_ONBOARD
        dt = refine_time
        fuel -= refine_fuel
        heat += refine_heat
        alert += refine_alert
        refine_some_cargo()  # reduces cargo volume, increases effective value

    elif action == 33:  # ACTIVE_COOLDOWN
        dt = cooldown_time
        fuel -= cooldown_fuel
        heat = max(0, heat - cooldown_amount)
        alert += cooldown_alert

    elif action == 34:  # TOOL_MAINTENANCE
        dt = maint_time
        if repair_kits <= 0:
            invalid = True
        else:
            repair_kits -= 1
            tool_condition = min(TOOL_MAX, tool_condition + tool_repair_amount)

    elif action == 35:  # HULL_PATCH
        dt = patch_time
        if repair_kits <= 0:
            invalid = True
        else:
            repair_kits -= 1
            hull = min(HULL_MAX, hull + hull_patch_amount)

    elif 36 <= action <= 41:  # JETTISON_COMMODITY[c]
        c = action - 36
        jettison_frac = 1.0  # jettison all of that commodity
        cargo[c] *= (1.0 - jettison_frac)
        # optional: reduces pirate attraction immediately
        alert = max(0, alert - jettison_alert_relief)

    elif action == 42:  # DOCK
        dt = dock_time
        if node_type[node_idx] != NODE_STATION:
            invalid = True
        else:
            docked = True
            alert = max(0, alert - dock_alert_drop)

    elif 43 <= action <= 60:  # SELL[c,b]
        if node_type[node_idx] != NODE_STATION:
            invalid = True
        else:
            c = (action - 43) // 3
            b = (action - 43) % 3
            frac = (0.25, 0.50, 1.00)[b]
            qty = cargo[c] * frac
            credits += qty * price[c] * (1.0 - slippage(qty, station_inventory[c]))
            cargo[c] -= qty
            station_inventory[c] += qty
            register_recent_sale(c, qty)  # affects future price impact

    elif 61 <= action <= 66:  # BUY_*
        if node_type[node_idx] != NODE_STATION:
            invalid = True
        else:
            purchase_station_item(action)

    elif action == 67:  # FULL_REPAIR_OVERHAUL
        dt = overhaul_time
        if node_type[node_idx] != NODE_STATION:
            invalid = True
        else:
            credits -= overhaul_cost
            hull = HULL_MAX
            tool_condition = TOOL_MAX

    elif action == 68:  # CASH_OUT_END_EPISODE
        done = True

    else:
        invalid = True

    if invalid:
        # treat as hold
        dt = 1
        passive_heat_dissipation(dt)
        credits -= 0.0  # no change; handled by reward penalty below

    # --- global dynamics each step ---
    time_remaining -= dt
    # passive heat dissipation always applies
    passive_heat_dissipation(dt)

    # overheat effects
    if heat > HEAT_MAX:
        overflow = heat - HEAT_MAX
        hull -= overheat_damage_per_unit * overflow
        heat = HEAT_MAX

    # pirates/hazards (node exposure) unless at station
    if node_type[node_idx] != NODE_STATION:
        apply_node_hazards(dt)
        maybe_pirate_encounter(dt)  # may reduce cargo/credits, damage hull

    # market evolution every step
    update_market(dt)

    # terminal conditions
    destroyed = (hull <= 0)
    stranded  = (fuel <= 0 and node_type[node_idx] != NODE_STATION)
    timeout   = (time_remaining <= 0)

    done = done or destroyed or stranded or timeout

    # --- reward ---
    reward = compute_reward(
        credits_before, fuel_before, hull_before, heat_before, tool_before,
        cargo_before, cargo_value_before,
        invalid, destroyed, stranded, done
    )

    obs = build_observation_vector()
    info = build_info_metrics()

    return obs, reward, done, False, info
```

---

