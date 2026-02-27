## 6) Baseline bots (benchmark policies)

All bots below assume they only see `(obs)` and maintain a tiny internal memory (optional). They’re intentionally imperfect but useful for sanity checks and PPO baselines.

### 6.1 Shared helpers (slice constants)

```python id="w8dqt6"
# Observation slices (must match the table above)
S_FUEL = 0
S_HULL = 1
S_HEAT = 2
S_TOOL = 3
S_CARGO_LOAD = 4
S_ALERT = 5
S_TIME = 6
S_CRED = 7
S_CARGO0 = 8  # 8..13
S_AT_STATION = 17
S_NODETYPE0 = 19  # 19..21

NEIGH_BASE = 24
NEIGH_STRIDE = 7

AST_BASE = 68
AST_STRIDE = 11

MKT_PRICE_BASE = 244  # 244..249
MKT_DPRICE_BASE = 250 # 250..255
```

---

### 6.2 Greedy Miner (fast profit, risky, minimal scans)

**Idea:** Mine immediately based on current comp estimates × price, return when cargo/heat are high, sell everything at station in one go.

Policy rules:
1. If at station: sell 100% of all commodities; buy fuel if low.
2. Else if cargo_load > 0.85 or heat > 0.80 or hull < 0.40: travel toward station (pick neighbor with smallest `steps_to_station_norm` if you expose it; otherwise neighbor with `neigh_type_station=1` if available).
3. Else: select best asteroid by `argmax(expected_value_per_tick)` and mine **aggressive** if tool>0.6 and heat<0.6 else standard.

```python id="bqp7bm"
def greedy_miner_policy(obs: np.ndarray) -> int:
    at_station = obs[S_AT_STATION] > 0.5
    cargo_load = obs[S_CARGO_LOAD]
    heat = obs[S_HEAT]
    hull = obs[S_HULL]
    tool = obs[S_TOOL]

    # station behavior
    if at_station:
        # sell all in order, pick the first commodity with any cargo
        for c in range(N_COMMODITIES):
            if obs[S_CARGO0 + c] > 0.01:
                return 43 + (c * 3) + 2  # SELL[c,100%]
        if obs[S_FUEL] < 0.4:
            return 63  # BUY_FUEL_LARGE
        return 6  # HOLD

    # retreat conditions
    if cargo_load > 0.85 or heat > 0.80 or hull < 0.40:
        # travel to station neighbor if present, else travel neighbor 0
        for k in range(MAX_NEIGHBORS):
            base = NEIGH_BASE + k * NEIGH_STRIDE
            if obs[base] > 0.5 and obs[base + 1] > 0.5:  # valid and station type
                return k  # TRAVEL_NEIGHBOR[k]
        return 0  # fallback travel

    # choose asteroid maximizing expected value (price dot comp_est)
    prices = obs[MKT_PRICE_BASE:MKT_PRICE_BASE+N_COMMODITIES]
    best_a, best_score = 0, -1e9
    for a in range(MAX_ASTEROIDS):
        base = AST_BASE + a * AST_STRIDE
        if obs[base] < 0.5:
            continue
        comp = obs[base+1:base+1+N_COMMODITIES]
        stability = obs[base+7]
        depletion = obs[base+8]
        score = float(np.dot(prices, comp)) * stability * (1.0 - depletion)
        if score > best_score:
            best_score, best_a = score, a

    # select asteroid if not already selected
    selected = (obs[AST_BASE + best_a*AST_STRIDE + 10] > 0.5)
    if not selected:
        return 12 + best_a  # SELECT_ASTEROID[a]

    # mine mode
    if tool > 0.60 and heat < 0.60:
        return 30  # MINE_AGGRESSIVE_SELECTED
    return 29      # MINE_STANDARD_SELECTED
```

---

### 6.3 Cautious Scanner (safer, scan-first, stabilizes)

**Idea:** Doesn’t commit without scanning; uses deep scan on top candidate; mines conservatively; stabilizes good finds; returns earlier to avoid pirate loss spirals.

Policy rules:
1. At station: sell in 50% buckets (reduce slippage exposure), top up repair kits.
2. In field: if current node has not been wide-scanned recently (you can infer via low `scan_conf` across asteroids), do `WIDE_SCAN`.
3. Select top asteroid by expected value; if `scan_conf < 0.6`, `DEEP_SCAN_SELECTED`.
4. If stability < 0.4 and have stabilizers, `STABILIZE_SELECTED`.
5. Mine conservative unless heat low and stability high.

```python id="slyfdo"
def cautious_scanner_policy(obs: np.ndarray) -> int:
    at_station = obs[S_AT_STATION] > 0.5
    cargo_load = obs[S_CARGO_LOAD]
    heat = obs[S_HEAT]
    hull = obs[S_HULL]

    if at_station:
        # sell 50% buckets first (more granular)
        for c in range(N_COMMODITIES):
            if obs[S_CARGO0 + c] > 0.01:
                return 43 + (c * 3) + 1  # SELL[c,50%]
        # buy some safety supplies if low
        if obs[14] < 0.3:
            return 64  # BUY_REPAIR_KIT
        if obs[S_FUEL] < 0.5:
            return 62  # BUY_FUEL_MED
        return 6

    # retreat earlier than greedy
    if cargo_load > 0.70 or heat > 0.75 or hull < 0.55:
        for k in range(MAX_NEIGHBORS):
            base = NEIGH_BASE + k * NEIGH_STRIDE
            if obs[base] > 0.5 and obs[base + 1] > 0.5:
                return k
        return 0

    # If overall scan confidence is low, wide scan first
    avg_conf = 0.0
    n = 0
    for a in range(MAX_ASTEROIDS):
        base = AST_BASE + a * AST_STRIDE
        if obs[base] < 0.5:
            continue
        avg_conf += obs[base+9]
        n += 1
    if n > 0 and (avg_conf / n) < 0.35:
        return 8  # WIDE_SCAN

    # pick best asteroid
    prices = obs[MKT_PRICE_BASE:MKT_PRICE_BASE+N_COMMODITIES]
    best_a, best_score = 0, -1e9
    for a in range(MAX_ASTEROIDS):
        base = AST_BASE + a * AST_STRIDE
        if obs[base] < 0.5:
            continue
        comp = obs[base+1:base+1+N_COMMODITIES]
        stability = obs[base+7]
        depletion = obs[base+8]
        conf = obs[base+9]
        score = float(np.dot(prices, comp)) * (0.5 + 0.5*conf) * stability * (1.0 - depletion)
        if score > best_score:
            best_score, best_a = score, a

    if not (obs[AST_BASE + best_a*AST_STRIDE + 10] > 0.5):
        return 12 + best_a

    # deep scan if needed
    stability = obs[AST_BASE + best_a*AST_STRIDE + 7]
    conf = obs[AST_BASE + best_a*AST_STRIDE + 9]
    if conf < 0.6:
        return 10  # DEEP_SCAN_SELECTED

    # stabilize risky-but-good asteroids if supplies
    if stability < 0.4 and obs[15] > 0.1:
        return 31  # STABILIZE_SELECTED

    # mine conservative to reduce fracture risk
    return 28
```

---

### 6.4 Market Timer (timing + staggered sells, holds near station)

**Idea:** Mines primarily when target commodity price trend is favorable; sells in 25% buckets when price is rising; otherwise holds cargo (risk-aware) and drifts/cools or returns closer to station.

Policy rules:
1. Define a **target commodity** (e.g., PGE index `3`).
2. If at station and price_delta for target is positive and price_norm high: sell 25% buckets repeatedly.
3. If in field and price trend is negative: stop mining, head toward station or hold/cool to reduce risk.
4. If trend positive: mine best asteroid weighted toward target commodity.

```python id="lqoz8f"
def market_timer_policy(obs: np.ndarray, target_c: int = 3) -> int:
    at_station = obs[S_AT_STATION] > 0.5
    heat = obs[S_HEAT]
    cargo_load = obs[S_CARGO_LOAD]
    price = obs[MKT_PRICE_BASE + target_c]
    dprice = obs[MKT_DPRICE_BASE + target_c]

    if at_station:
        # only sell when trend favorable and price high
        if obs[S_CARGO0 + target_c] > 0.01 and dprice > 0.02 and price > 0.6:
            return 43 + (target_c * 3) + 0  # SELL[target,25%]
        # otherwise, sell other junk slowly or just wait
        if obs[S_FUEL] < 0.5:
            return 62
        return 6

    # if trend is bad, reduce exposure: head toward station or cool
    if dprice < -0.02:
        if heat > 0.6:
            return 33  # ACTIVE_COOLDOWN
        # retreat
        for k in range(MAX_NEIGHBORS):
            base = NEIGH_BASE + k * NEIGH_STRIDE
            if obs[base] > 0.5 and obs[base + 1] > 0.5:
                return k
        return 6

    # trend OK: mine targeted
    if cargo_load > 0.85 or heat > 0.80:
        return 33 if heat > 0.7 else 6

    # pick asteroid maximizing (target commodity weight)
    best_a, best_score = 0, -1e9
    for a in range(MAX_ASTEROIDS):
        base = AST_BASE + a * AST_STRIDE
        if obs[base] < 0.5:
            continue
        comp = obs[base+1:base+1+N_COMMODITIES]
        stability = obs[base+7]
        depletion = obs[base+8]
        conf = obs[base+9]
        score = float(comp[target_c]) * (0.5 + 0.5*conf) * stability * (1.0 - depletion)
        if score > best_score:
            best_score, best_a = score, a

    if not (obs[AST_BASE + best_a*AST_STRIDE + 10] > 0.5):
        return 12 + best_a
    return 29 if heat < 0.6 else 28
```

---

