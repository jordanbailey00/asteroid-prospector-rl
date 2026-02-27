## 4) Reward function (code-ready definition)

This is a direct “shaped training reward” implementation consistent with the GDD reward components (`r_sell`, `r_extract`, `r_fuel`, `r_time`, `r_wear`, `r_heat`, `r_damage`, terminal penalties/bonus). fileciteturn2file0L39-L68 fileciteturn2file2L1-L33

```python id="s0s3kl"
from dataclasses import dataclass
import numpy as np

@dataclass
class RewardCfg:
    # scaling
    credit_scale: float = 1000.0

    # shaped components
    alpha_extract: float = 0.02   # weight on estimated value created by extraction
    beta_fuel: float = 0.10       # fuel penalty weight
    gamma_time: float = 0.001     # per dt penalty
    delta_wear: float = 0.05      # tool wear penalty
    epsilon_heat: float = 0.20    # heat penalty weight
    zeta_damage: float = 1.00     # hull damage penalty
    kappa_pirate: float = 1.00    # (optional) value lost to pirates penalty

    # extras
    scan_cost: float = 0.005
    invalid_action_pen: float = 0.01

    # thresholds
    heat_safe_frac: float = 0.70  # heat safe threshold (fraction of HEAT_MAX)

    # terminal
    stranded_pen: float = 50.0
    destroyed_pen: float = 100.0
    terminal_bonus_B: float = 0.002  # multiply by normalized net_profit

def compute_reward(
    credits_before: float,
    fuel_before: float,
    hull_before: float,
    heat_before: float,
    tool_before: float,
    cargo_before: np.ndarray,
    cargo_value_before: float,
    invalid: bool,
    destroyed: bool,
    stranded: bool,
    done: bool,
    *,
    credits_after: float,
    fuel_after: float,
    hull_after: float,
    heat_after: float,
    tool_after: float,
    cargo_after: np.ndarray,
    cargo_value_after: float,
    action: int,
    dt: int,
    cfg: RewardCfg,
    HEAT_MAX: float = 100.0,
):
    # A) Profit / selling
    delta_credits = credits_after - credits_before
    r_sell = delta_credits / cfg.credit_scale

    # A2) Value created by extraction (estimated via market value of cargo delta)
    delta_cargo_value = max(0.0, cargo_value_after - cargo_value_before)
    r_extract = cfg.alpha_extract * (delta_cargo_value / cfg.credit_scale)

    # B) Operating costs
    r_fuel = -cfg.beta_fuel * max(0.0, fuel_before - fuel_after) / 100.0  # scale fuel
    r_time = -cfg.gamma_time * float(dt)
    r_wear = -cfg.delta_wear * max(0.0, tool_before - tool_after) / 10.0  # scale wear
    r_damage = -cfg.zeta_damage * max(0.0, hull_before - hull_after) / 10.0

    # Heat penalty (nonlinear beyond safe threshold)
    heat_safe = cfg.heat_safe_frac * HEAT_MAX
    heat_excess = max(0.0, heat_after - heat_safe)
    r_heat = -cfg.epsilon_heat * (heat_excess / HEAT_MAX) ** 2

    # D) Scan discipline (optional)
    # actions: 8=wide_scan, 9=focused_scan, 10=deep_scan
    r_scan = -cfg.scan_cost if action in (8, 9, 10) else 0.0

    # Invalid action penalty
    r_invalid = -cfg.invalid_action_pen if invalid else 0.0

    # Terminal penalties
    r_terminal = 0.0
    if stranded:
        r_terminal -= cfg.stranded_pen
    if destroyed:
        r_terminal -= cfg.destroyed_pen

    # Terminal bonus (end-to-end alignment)
    # net_profit can be approximated by credits_after (or credits_after - start_credits - spend)
    if done and (not destroyed) and (not stranded):
        net_profit_norm = credits_after / cfg.credit_scale
        r_terminal += cfg.terminal_bonus_B * net_profit_norm

    return (r_sell + r_extract + r_fuel + r_time + r_wear + r_heat + r_damage +
            r_scan + r_invalid + r_terminal)
```

**Notes on scaling**
- Keep rewards roughly in `[-1, +1]` per step for stable PPO-style learning.
- Use `credit_scale` and the small `terminal_bonus_B` to align long-horizon behavior.

---

