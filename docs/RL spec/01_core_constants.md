## 0) Core constants (frozen interface)

```python id="mrjtdx"
# Environment interface constants (do not change once training begins)
N_COMMODITIES      = 6   # [iron, nickel, water_ice, pge, rare_isotopes, volatiles]
COMMODITY_NAMES    = ("iron","nickel","water_ice","pge","rare_isotopes","volatiles")

MAX_NODES          = 32  # padded
MAX_NEIGHBORS      = 6   # padded neighbor slots per node
MAX_ASTEROIDS      = 16  # padded asteroids per node

NODE_TYPES         = 3   # [station, cluster, hazard]
NODE_STATION       = 0
NODE_CLUSTER       = 1
NODE_HAZARD        = 2

OBS_DIM            = 260
N_ACTIONS          = 69

# Normalization scales
CREDIT_SCALE       = 1000.0
FUEL_MAX           = 1000.0
HULL_MAX           = 100.0
HEAT_MAX           = 100.0
TOOL_MAX           = 100.0
CARGO_MAX          = 200.0
ALERT_MAX          = 100.0
TIME_MAX           = 20000.0  # example; can be configured per env instance
```

---

