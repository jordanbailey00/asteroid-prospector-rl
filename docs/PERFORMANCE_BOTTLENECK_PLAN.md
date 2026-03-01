# Game Bottleneck Performance Plan

Last updated: 2026-03-01
Owner: Training/runtime workstream
Status: Active execution plan

## Objective

Maximize practical training throughput by treating simulation runtime as the primary bottleneck.

- Aspirational target: `100,000` env steps/sec.
- Practical target: highest stable throughput achievable on target hardware with reproducible evidence.
- Constraint: frozen RL interface remains unchanged (`OBS_DIM=260`, `N_ACTIONS=69`, reward/obs/action contracts frozen).

## Current bottleneck diagnosis

Observed in current code paths:

1. PPO hot loop uses `ProspectorReferenceEnv` (Python) via Gym wrapper in `training/puffer_backend.py`.
2. Training loop processes `on_step` callbacks per env, per step in Python (`for i in range(cfg.num_envs)`), increasing interpreter overhead.
3. Native core exists, but trainer path is not yet using it as default stepping authority for PPO.
4. Native bridge is single-step oriented (`abp_core_step`), with no `step_many` API to amortize Python<->C call overhead.

## Performance-first principles

1. Move simulation stepping into C before tuning PPO hyperparameters.
2. Reduce Python work inside step loops to aggregate/batched operations.
3. Keep eval/replay overhead off hot training path where possible.
4. Benchmark after each major change; do not optimize blindly.

## Workstream P0: Measurement discipline

### P0.1 Baseline artifacts
- Use `tools/profile_training_throughput.py` to capture:
  - `env_only`
  - `trainer`
  - `trainer_eval`
- Store JSON baseline artifacts under `artifacts/throughput/` with run IDs tied to commit/date.

### P0.2 Target enforcement mode
- Use `--target-steps-per-sec 100000 --enforce-target` for hard checks once stable.
- Until then, record measured floor and trendline by commit.

## Workstream P1: Native env path for PPO (highest ROI)

### P1.1 Add selectable PPO env implementation
- Extend PPO runtime config with `ppo_env_impl`:
  - `reference`
  - `native`
  - `auto` (prefer native, fallback reference)
- Default to `auto` for local benchmarking.

### P1.2 Implement native Gym wrapper
- Add a native-backed Gym-compatible env wrapper around `NativeProspectorCore`.
- Keep output semantics identical to existing wrapper (`obs, reward, terminated, truncated, info`).
- Ensure deterministic reset/seed behavior stays consistent with existing contracts.

### P1.3 Validation
- Add tests for:
  - native wrapper shape/dtype/action bounds,
  - reset/step contract parity (keys + types),
  - smoke PPO run with `ppo_env_impl=native`.

## Workstream P2: Reduce Python callback overhead

### P2.1 Replace per-env callback hot loop
Current pattern:
- per step, iterate `for i in range(num_envs)` and call `on_step(...)` per env.

Target pattern:
- process vectorized arrays once per step batch,
- only materialize full `info` payloads for envs with terminal transitions or sampled telemetry,
- aggregate metrics via vector operations where possible.

### P2.2 Aggregator interface update
- Introduce batch-aware callback contract (or adapter) for window metrics.
- Preserve existing trainer summary semantics.

### P2.3 Validation
- Ensure metrics totals and window boundaries match previous implementation for fixed seeds.

## Workstream P3: C-level batch stepping

### P3.1 Extend C API for batch operations
Add APIs (design target):
- `abp_core_reset_many(...)`
- `abp_core_step_many(...)`

Requirements:
- no per-step heap allocation,
- contiguous buffers for obs/reward/done/info-selected metrics,
- deterministic results given same seeds/actions.

### P3.2 Python bridge for batch APIs
- Extend `native_core.py` with batched reset/step methods.
- Avoid Python object churn in hot loop (favor NumPy views / preallocated buffers).

### P3.3 Validation
- Add parity/contract tests for batch vs single-step outputs.

## Workstream P4: Native-core micro-optimizations

### P4.1 Build flags and release profile
- Ensure release benchmarking uses optimized compiler flags.
- Keep debug/sanitizer configs separate from perf builds.

### P4.2 Hot function optimization candidates
- Observation packing (`abp_pack_obs`)
- Market/hazard update loops
- Any repeated transcendental-heavy code on critical path

Rules:
- Preserve determinism and frozen behavior.
- Guard with tests before/after optimization.

## Workstream P5: Training path hygiene

### P5.1 Decouple eval overhead
- Keep eval replay generation out of throughput measurements unless explicitly profiling `trainer_eval`.
- For raw throughput runs, set eval generation off.

### P5.2 Metadata/logging overhead control
- Reduce write frequency in hot path where safe.
- Keep window checkpoints/metrics as authoritative artifacts.

## Workstream P6: Throughput tuning matrix (after P1/P2)

- Run controlled matrix over:
  - `num_envs`
  - `num_workers`
  - `rollout_steps`
  - `num_minibatches`
- Select config by highest stable mean throughput, then verify policy training still converges.

## Acceptance criteria

1. Throughput report demonstrates improved trend from baseline after each major workstream.
2. Native PPO path is available and validated.
3. Python per-step callback overhead is measurably reduced.
4. If 100k is still unattainable:
   - calibrated stable floor is documented,
   - enforcement gates use floor while tracking delta-to-target.

## Ordered implementation sequence

1. P1 native PPO env path.
2. P2 batch/aggregated callback processing.
3. P3 C-level batched stepping APIs and Python bridge.
4. P4 micro-optimizations in C hot functions.
5. P6 tuning matrix and updated threshold policy.

## Risks

- C batch API complexity can introduce parity drift.
  - Mitigation: parity/contract tests for batch vs single-step before enabling by default.
- Native path may expose platform/library friction.
  - Mitigation: `auto` fallback and explicit runtime checks.
- Throughput gains may shift bottleneck to PPO compute on some hardware.
  - Mitigation: separate env-only vs trainer measurements in profiler artifacts.
