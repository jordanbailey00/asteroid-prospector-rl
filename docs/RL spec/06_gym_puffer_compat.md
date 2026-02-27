## 5) Gymnasium + PufferLib compatibility requirements

### 5.1 Spaces

```python id="8667jw"
import gymnasium as gym
import numpy as np

observation_space = gym.spaces.Box(
    low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
)

action_space = gym.spaces.Discrete(N_ACTIONS)
```

Why `[-1,1]`? Because you’re normalizing everything (fractions, deltas). If you prefer raw values, widen bounds.

### 5.2 `info` metrics for benchmarking (recommended)

Return these every step (or at least at episode end):

```python id="iss6nz"
info = dict(
    credits=credits,
    net_profit=credits - total_spend,         # track spends internally
    profit_per_tick=(credits - total_spend)/max(1, ticks_elapsed),
    survival=1.0 if not (destroyed or stranded) else 0.0,
    overheat_ticks=overheat_ticks,
    pirate_encounters=pirate_encounters,
    value_lost_to_pirates=value_lost_to_pirates,
    fuel_used=(fuel_start - fuel),
    hull_damage=(hull_start - hull),
    tool_wear=(tool_start - tool_condition),
    scan_count=scan_count,
    mining_ticks=mining_ticks,
    cargo_utilization_avg=cargo_utilization_avg,
)
```

These match the GDD’s success/monitoring metrics. fileciteturn2file0L3-L20

---

