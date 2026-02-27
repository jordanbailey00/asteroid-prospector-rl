## 2) Action space (Discrete, exact indexing 0..68)

We use **selection actions** to avoid “mine(asteroid_i)” blowing up action count. The selected asteroid is stored in env state and exposed via `selected_flag`.

```python id="kvqdrv"
action_space = gym.spaces.Discrete(N_ACTIONS)  # N_ACTIONS=69
```

### 2.1 Index map

#### Travel / movement
- **0..5**: `TRAVEL_NEIGHBOR[k]` (neighbor slot k)
- **6**: `HOLD_DRIFT`
- **7**: `EMERGENCY_BURN`

#### Sensing
- **8**: `WIDE_SCAN` (cluster-level)
- **9**: `FOCUSED_SCAN_SELECTED`
- **10**: `DEEP_SCAN_SELECTED`
- **11**: `PASSIVE_THREAT_LISTEN`

#### Target selection
- **12..27**: `SELECT_ASTEROID[a]` where `a = action - 12`

#### Mining / extraction
- **28**: `MINE_CONSERVATIVE_SELECTED` fileciteturn2file4L10-L13  
- **29**: `MINE_STANDARD_SELECTED` fileciteturn2file4L15-L16  
- **30**: `MINE_AGGRESSIVE_SELECTED` fileciteturn2file4L18-L21  
- **31**: `STABILIZE_SELECTED` fileciteturn2file4L23-L26  
- **32**: `REFINE_ONBOARD` fileciteturn2file4L28-L31  

#### Thermal / maintenance / safety
- **33**: `ACTIVE_COOLDOWN` fileciteturn2file4L34-L36  
- **34**: `TOOL_MAINTENANCE` fileciteturn2file4L38-L41  
- **35**: `HULL_PATCH` fileciteturn2file1L1-L4  
- **36..41**: `JETTISON_COMMODITY[c]` where `c = action - 36` fileciteturn2file1L6-L9  

#### Station-only
- **42**: `DOCK` fileciteturn2file1L12-L15  
- **43..60**: `SELL[c,b]` where `c in [0..5]`, `b in [0..2]`
  - `c = (action-43)//3`, `b = (action-43)%3`
  - bucket `b`: 0→25%, 1→50%, 2→100% fileciteturn2file1L16-L19  
- **61**: `BUY_FUEL_SMALL`
- **62**: `BUY_FUEL_MED`
- **63**: `BUY_FUEL_LARGE`
- **64**: `BUY_REPAIR_KIT`
- **65**: `BUY_STABILIZER`
- **66**: `BUY_DECOY`
- **67**: `FULL_REPAIR_OVERHAUL` fileciteturn2file1L30-L33  
- **68**: `CASH_OUT_END_EPISODE` fileciteturn2file1L39-L41  

### 2.2 Invalid action handling (important for stable RL)
If an action is invalid (e.g., sell while not at station; mine with invalid asteroid; travel to empty neighbor slot):
- treat as `HOLD_DRIFT`
- apply `invalid_action_penalty` (small, e.g. `-0.01`)

This prevents agents from exploiting invalids and keeps learning stable.

---

