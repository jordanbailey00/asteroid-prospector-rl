# M7 Benchmark Protocol Execution (2026-03-04)

## Chunk objective

Chunk 2 objective from the MVP backlog:

- Deliver M7.2 benchmark protocol automation that:
  - trains a seeded policy candidate per seed in a reproducible matrix,
  - evaluates the trained policy and all baseline bots on the same seeded episode sets,
  - emits a local JSON report with required KPI comparisons.

## Planned scope

- Add a benchmark protocol runner under `tools/`.
- Reuse frozen training/eval contracts (`TrainConfig`, `run_training`, frozen obs/action spaces).
- Compare trained policy vs baseline bots on:
  - `net_profit_mean`
  - `survival_rate`
  - `profit_per_tick_mean`
  - `overheat_ticks_mean`
  - `pirate_encounters_mean`
- Add deterministic regression tests and a small local smoke artifact.
- Update tracking/docs for M7.2 completion and M7.3 remaining work.

## Implemented deliverables

- `tools/run_m7_benchmark_protocol.py`
  - Added `BenchmarkProtocolConfig` and `run_m7_benchmark_protocol(...)`.
  - Added CLI for seed-matrix execution and PPO/random training backend selection.
  - Added per-seed training orchestration using `training/train_puffer.py` contracts.
  - Added seeded contender evaluation for:
    - trained policy (`ppo` when `trainer_backend=puffer_ppo`, otherwise configured backend),
    - `greedy_miner`,
    - `cautious_scanner`,
    - `market_timer`.
  - Added aggregate KPI summaries and per-bot comparison deltas.
  - Added optional protocol expectations section and strict enforcement flag (`--enforce-protocol-expectations`).

- `tests/test_run_m7_benchmark_protocol.py`
  - Added parser seed-matrix validation coverage.
  - Added report schema/comparison coverage for random-backend benchmark smoke.
  - Added guard coverage for pre-existing training run directory collision behavior.

- Smoke evidence artifact:
  - `artifacts/benchmarks/m7-protocol-smoke.json`

## Corrections and hardening during implementation

- Added explicit run-directory collision checks per seed (`{run_id}-{policy}-seed{seed}`) to prevent silent reuse/mixing of stale benchmark runs.
- Added structured checkpoint resolution guards so protocol runs fail clearly when training metadata is incomplete.
- Kept cross-platform regression path by allowing `trainer_backend=random` in tests while preserving PPO-first protocol defaults in CLI.

## Validation evidence

- `python -m ruff check tools/run_m7_benchmark_protocol.py tests/test_run_m7_benchmark_protocol.py`
- `python -m black --check tools/run_m7_benchmark_protocol.py tests/test_run_m7_benchmark_protocol.py`
- `python -m pytest -q tests/test_run_m7_benchmark_protocol.py`
  - Result: `3 passed`.
- `python -m pytest -q`
  - Result: `121 passed, 2 skipped`.
- `python tools/run_m7_benchmark_protocol.py --trainer-backend random --seed-matrix 7,9 --episodes-per-seed 2 --trainer-total-env-steps 80 --trainer-window-env-steps 40 --env-time-max 4000 --max-steps-per-episode 12000 --run-id m7-protocol-smoke --output-path artifacts/benchmarks/m7-protocol-smoke.json`
  - Result: report emitted successfully.

## Remaining M7 work after Chunk 2

- M7.3 W&B benchmark logging and artifact lineage (`job_type=eval`) for protocol reports.
