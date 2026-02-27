# Asteroid Belt Prospector — Acceptance Tests & Parity Harness Spec

## 1) Purpose

This document defines the **acceptance test suite** and the **Python↔C parity harness** required before:
- long-running PufferLib training,
- replay generation (Option A eval-run recording),
- and any “learning progress” claims.

The environment is intentionally stochastic and non-linear, but it must remain:
- **API-correct** (Gymnasium-style step/reset),
- **deterministic under fixed seeds**,
- **stable in its frozen interface** (observation layout + action indexing + reward definition),
- and **numerically consistent** across implementations (Python reference vs C core).

---

## 2) Definitions and invariants (frozen contract)

### Frozen interface invariants
These values must never change once training begins:
- `OBS_DIM = 260`
- `N_ACTIONS = 69` (actions indexed `0..68`)
- Observation field layout and normalization rules
- Action decoding / validity behavior
- Reward function definition and scaling

### Episode end semantics (Gymnasium)
`step()` must return `(obs, reward, terminated, truncated, info)`.
- `terminated=True` means the episode ended due to a terminal state in the environment (e.g., destroyed/stranded/cash-out).
- `truncated=True` means the episode ended due to an external time/step limit (e.g., time budget exhausted), even if the environment did not “fail”.

When either `terminated` or `truncated` is true, the caller must `reset()` before stepping again.

---

## 3) Test suite structure (tiers)

The test suite is organized into tiers so CI can run fast checks always and heavier checks on demand.

### Tier 0 — Static contract checks (always run)
- Shapes, dtypes, value ranges, and deterministic seeding behavior
- Action indexing bounds and invalid action handling
- Reward outputs are finite (no NaN/Inf) and within expected magnitude

### Tier 1 — Engine unit tests (always run)
- Mining, scan update, market tick, pirate/hazard models: local correctness properties
- Resource clamps (fuel/hull/heat/tool/cargo bounds)
- Station-only behaviors (sell/buy/repair) enforced

### Tier 2 — Integration tests (always run)
- Full env rollouts (short episodes) do not crash
- `info` metrics keys exist and are consistent
- Windowing logic (`window_env_steps`) increments correctly

### Tier 3 — Parity harness (run on main branch, nightly, or manually)
- Python reference env vs C core match under identical seeds and action sequences

### Tier 4 — Performance and soak tests (manual / nightly)
- steps/sec benchmarks
- multi-hour stability (no leaks, no drift)

---

## 4) Required test tooling

### Python test framework
- Use `pytest` for unit + integration tests, including parametrization.  
- Use `hypothesis` for property-based tests (randomized edge-case generation).

### Native core test harness
- A small CLI executable in the C core:
  - `core_test_runner --seed <seed> --actions <actions.bin> --out <trace.bin>`
- A Python harness that runs both Python and C, then compares traces.

---

## 5) Determinism contract (required for parity and reproducible replays)

Because NumPy’s RNG and a custom C RNG will not match bit-for-bit by default, define one authoritative RNG:

### 5.1 Authoritative RNG requirement
- The C core must implement a deterministic RNG algorithm (e.g., PCG32 / xoshiro / splitmix) with:
  - explicit seed initialization
  - explicit “next_u32/next_f32” functions
- The Python reference environment MUST use the *same RNG algorithm* (either by:
  - calling into the C RNG via bindings, or
  - implementing the same RNG in Python with exact integer arithmetic)

### 5.2 RNG usage rules (to avoid divergence)
- All stochastic draws must happen through the authoritative RNG:
  - mining yield noise
  - fracture checks
  - pirate encounter checks
  - market shock checks
  - procedural generation on reset
- Never use:
  - `rand()` in C,
  - `np.random` directly in Python,
  - time-based seeds,
  - nondeterministic hash iteration.

### 5.3 Floating point consistency rules
- Observation vectors are `float32`.
- Internal computations may use `float32` or `float64`, but parity harness comparisons must define:
  - which outputs require exact match vs tolerance
  - acceptable tolerance thresholds (below)

---

## 6) Parity harness (Python reference vs C core)

### 6.1 Why a Python reference env exists
Before optimizing in C, implement a “golden” Python environment that:
- follows the frozen interface exactly,
- is readable/debuggable,
- is used for parity comparisons only (not for production training once C core is ready).

### 6.2 What parity means (outputs compared)
Given:
- identical env configuration,
- identical seed,
- identical action sequence length `T`,
the two envs must match on each step `t`:

**Required comparisons**
1) `terminated` and `truncated` must match exactly.
2) `done = terminated or truncated` must match exactly.
3) `reward` must match within tolerance.
4) `obs` must match within tolerance (elementwise).
5) `info` must contain required keys; selected keys must match within tolerance.

**Info keys to compare (minimum)**
- `credits`
- `net_profit`
- `profit_per_tick`
- `survival`
- `overheat_ticks`
- `pirate_encounters`
- `value_lost_to_pirates`
- `fuel_used`
- `hull_damage`
- `tool_wear`
- `scan_count`
- `mining_ticks`
- `cargo_utilization_avg`

### 6.3 Trace format (for comparing runs)
Define a trace record per step:

- `t: uint32`
- `action: uint8`
- `dt: uint16` (macro action time)
- `reward: float32`
- `terminated: uint8`
- `truncated: uint8`
- `obs[260]: float32`
- `info_selected[K]: float32` (fixed order list of K metrics)

Store traces as:
- `trace.npy` (easy) or
- a simple binary format (fast, C-friendly)

### 6.4 Tolerance rules
Because float math may differ slightly across compilers/platforms:

- For flags and discrete indices: **exact match**
- For float32 outputs:
  - `abs_diff <= 1e-6` OR `rtol <= 1e-5` (choose one standard)
- For cumulative quantities (credits, profit) allow slightly larger tolerance:
  - `abs_diff <= 1e-4 * max(1, |value|)` (or equivalent)

If tolerances fail:
- dump a “mismatch bundle”:
  - seed
  - config hash
  - action sequence
  - both traces
  - the first mismatch step index and per-field diffs

### 6.5 Action sequence generation for parity
Use three test action suites:

**Suite A: Random legal/illegal mix**
- Generate actions uniformly from `0..68` for `T=500..5000` steps.

**Suite B: Adversarial sequences**
- repeated invalid station sells in field
- mining without selection
- travel to invalid neighbor slots
- deep-scan without selection
- repeated emergency burns and cooldown loops

**Suite C: Scenario scripts (semi-deterministic)**
- “scan → select → mine → cool → return → dock → sell”
- “risk run”: aggressive mine until heat threshold, test overheat clamping

### 6.6 Parity test matrix (minimum)
Run parity across:
- 10 distinct seeds
- 3 action suites
- 2 episode configs (short and medium TIME_MAX)

Minimum parity workload:
- 10 seeds × 3 suites × T=2000 steps = 60k step comparisons

---

## 7) Acceptance tests for the frozen interface

### 7.1 Observation contract tests
- `obs.shape == (260,)`
- `obs.dtype == np.float32`
- Expected bounds:
  - most fields in `[0,1]`
  - `price_delta_norm` in `[-1,1]`
- Onehot groups sum to 1 or 0 as appropriate:
  - `current_node_type_onehot`
  - neighbor type onehot fields
- Exactly one selected asteroid flag is 1 if `mining_active_flag==1` else 0 or 1 depending on design

### 7.2 Action contract tests
- `action_space.n == 69`
- All actions `0..68` accepted; others rejected/treated as invalid
- Invalid action handling:
  - treated as `HOLD_DRIFT`
  - `invalid_action_penalty` applied
  - no state corruption

### 7.3 Reward contract tests
- reward is finite (no NaN/Inf)
- reward magnitude is stable (roughly within a bounded range per step)
- spot-check:
  - selling increases reward via `r_sell`
  - fuel use decreases reward via `r_fuel`
  - overheat induces nonlinear penalty

---

## 8) Engine dynamics acceptance tests (core game behavior)

### 8.1 Resource invariants
Always enforce:
- `0 <= fuel <= FUEL_MAX`
- `0 <= hull <= HULL_MAX`
- `0 <= heat <= HEAT_MAX` (after clamping)
- `0 <= tool_condition <= TOOL_MAX`
- `0 <= cargo_total <= CARGO_MAX` (or clamp/penalize if overflow allowed)

### 8.2 Station gatekeeping
- Selling/buying/overhaul only works at station node.
- Dock only works at station node.
- At station, pirate/hazard dynamics should not apply.

### 8.3 Macro-time (`dt`) consistency
- Actions that consume variable time must:
  - reduce time_remaining by exactly `dt`
  - apply passive dynamics scaled by `dt`
  - update market by `dt`
- Ensure `dt` is included in trace for parity.

### 8.4 Partial observability sanity checks
- After scans:
  - `scan_conf` increases appropriately
  - composition estimates remain normalized (sum ~1)
- Deep scan should increase confidence more than wide scan.

---

## 9) Replay acceptance tests

Replays are produced by eval-run recording. Acceptance tests must verify:

### 9.1 Replay frame schema
Each frame contains:
- `t`, `action`, `reward`, `dt`
- `render_state` with required sub-fields
- `events` list (possibly empty)
- `info` (optional, but recommended)

### 9.2 Replay fidelity test (key)
For an eval episode recorded from a checkpoint:
- replay frames must match the env’s step outputs for:
  - action sequence
  - reward sequence
  - terminated/truncated location
- This is easiest when replays are frame-recorded (not event-sourced).

---

## 10) Windowing (`window_env_steps`) acceptance tests

### 10.1 Window counter logic
Given `window_env_steps = W`:
- window_id increments each time cumulative env steps crosses a multiple of `W`.
- Metrics aggregation includes exactly the steps/episodes in that window.

### 10.2 Replay generation trigger
- Exactly once per window:
  - a checkpoint is saved
  - an eval-run replay is generated and tagged `every_window` (or `every_n`)
  - a metrics row is emitted for that window

---

## 11) W&B acceptance tests (metrics + artifacts)

### 11.1 Metrics cadence
- Log window metrics once per window to W&B (not per-step).
- Required metric keys must exist (return_mean, profit_mean, survival_rate, etc.).

### 11.2 Artifact logging
- Checkpoints must be logged as W&B artifacts with deterministic naming.
- Replays must be logged as W&B artifacts (or linked deterministically).

### 11.3 Offline test mode
Tests should run without network by default:
- use W&B offline mode or mocked W&B client
- verify local artifact directory contents and naming conventions

---

## 12) API acceptance tests (backend)

Use FastAPI test client to validate:
- `GET /api/runs` returns run metadata and latest_window
- `GET /api/runs/{run_id}/metrics/windows` returns window rows
- `GET /api/runs/{run_id}/replays` returns replay catalog
- `POST /api/play/session` and `/step` work end-to-end (basic smoke)

---

## 13) CI gating rules

### Must-pass on every PR
- Tier 0 + Tier 1 + Tier 2

### Must-pass before long training runs
- Tier 3 (parity harness) on at least:
  - 10 seeds × Suite A and Suite B

### Performance checks (non-gating unless explicitly enabled)
- Run `bench_steps_per_sec` and report in CI logs

---

## 14) Commands (developer ergonomics)

- `pytest -q` (fast checks)
- `pytest -q -m parity` (parity suite)
- `pytest -q -m perf` (performance suite)
- `python tools/run_parity.py --seeds 10 --suite A --steps 2000`

---

## 15) Required files to implement

- `tests/test_contract_obs.py`
- `tests/test_contract_actions.py`
- `tests/test_reward_sanity.py`
- `tests/test_station_rules.py`
- `tests/test_dt_and_time.py`
- `tests/test_replay_schema.py`
- `tests/test_windowing.py`
- `tests/test_wandb_offline.py`
- `tests/test_api_smoke.py`
- `tools/run_parity.py`
- `engine_core/core_test_runner.c` (or equivalent)

---

## 16) “Stop-the-line” failures (must fix immediately)

If any of these occur, do not proceed with training:
- parity harness mismatch beyond tolerance
- nondeterminism for fixed seed/action sequence
- NaN/Inf in obs/reward
- violated resource bounds (negative fuel/hull, etc.)
- window metrics misaligned with window_env_steps
- replay schema incompatible with frontend playback
