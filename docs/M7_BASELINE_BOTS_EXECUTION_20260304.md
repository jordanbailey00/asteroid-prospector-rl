# M7.1 Baseline Bots Execution - 2026-03-04

## Scope (planned)

Chunk 1 objective from the MVP backlog:

- Implement baseline bots:
  - `greedy_miner`
  - `cautious_scanner`
  - `market_timer`
- Add reproducible CLI runs for `N` episodes across deterministic seed schedules.
- Add tests for:
  - action validity (`0..68`)
  - deterministic policy behavior for fixed observations
  - runner output/report behavior and deterministic replayability

## Implemented

Code deliverables:

- `training/baseline_bots.py`
  - Added three observation-only baseline policies using frozen obs/action contracts.
  - Added baseline bot registry helpers:
    - `list_baseline_bots()`
    - `get_baseline_bot(...)`
    - `make_market_timer_policy(...)`
- `tools/run_baseline_bots.py`
  - Added deterministic baseline runner with configurable:
    - bot selection (`--bot` repeatable, `all` supported)
    - `--episodes`
    - `--base-seed`
    - `--env-time-max`
    - `--max-steps-per-episode`
    - `--market-timer-target-commodity`
    - `--run-id`
    - `--output-path`
  - Emits structured JSON report with per-episode rows and per-bot summary metrics.

Test deliverables:

- `tests/test_baseline_bots.py`
  - policy action range checks
  - deterministic decision checks
  - target-commodity validation checks
- `tests/test_run_baseline_bots.py`
  - report + artifact schema checks
  - deterministic run reproducibility checks
  - invalid bot-name rejection checks

## Corrections made during implementation

The following issues were found and corrected before finalizing Chunk 1:

1. Station deadlock behavior in initial bot implementations
- Initial behavior kept some bots idling at station with no outbound action.
- Correction: added explicit station-to-field travel preference (`_field_neighbor_action`) so policies leave station when not selling/buying.

2. Invalid station purchase attempts in cautious scanner
- Initial behavior attempted `BUY_REPAIR_KIT` even with insufficient credits, creating high invalid-action rates.
- Correction: added credit estimation from normalized credits obs and affordability checks before station purchases.

3. CLI `--bot` selection bug
- Initial parser default (`["all"]`) made `--bot market_timer` still execute all bots.
- Correction: changed parser default to empty list and resolved default-to-all only when no explicit bot values are supplied.

4. Market timer never realizing profit in smoke runs
- Initial behavior over-constrained sell/retreat timing and could terminate before liquidation.
- Correction: added forced-liquidation path and cargo-based retreat behavior so market timer exits risk and sells opportunistically.

## Validation evidence

Local checks executed:

- `python -m ruff check training/baseline_bots.py tools/run_baseline_bots.py tests/test_baseline_bots.py tests/test_run_baseline_bots.py`
- `python -m black --check training/baseline_bots.py tools/run_baseline_bots.py tests/test_baseline_bots.py tests/test_run_baseline_bots.py`
- `python -m pytest -q tests/test_baseline_bots.py tests/test_run_baseline_bots.py`

Sample runner artifact:

- `artifacts/benchmarks/m7-baseline-smoke.json`

## Remaining M7 work after Chunk 1

- `M7.2` Benchmark protocol automation (PPO vs baselines across seed matrix and KPI report).
- `M7.3` W&B benchmark logging path (`job_type=eval`) with reproducible artifact lineage.
