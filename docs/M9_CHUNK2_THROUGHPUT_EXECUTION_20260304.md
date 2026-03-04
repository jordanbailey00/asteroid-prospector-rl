# M9 Chunk 2 Throughput Execution (2026-03-04)

## Objective

Execute Chunk 2 throughput hardening:

- rerun throughput profiler/matrix/floor artifacts,
- close an evidence-quality gap where trainer floors could be calibrated from non-target backends,
- preserve deterministic floor-calibration behavior while adding explicit backend coverage gates.

## Implemented changes

- `tools/run_throughput_matrix.py`
  - Added coverage controls:
    - `required_trainer_backends` (CSV/tuple)
    - `fail_on_coverage_gap` (optional hard fail)
  - Added coverage telemetry to matrix reports:
    - `summary.coverage_pass`
    - `summary.coverage_gaps`
    - `mode_summaries.<trainer_mode>.successful_trainer_backends`
  - Added CLI flags:
    - `--required-trainer-backends`
    - `--fail-on-coverage-gap`

- `tests/test_run_throughput_matrix.py`
  - Added coverage behavior tests for:
    - warning-only coverage gaps (default pass),
    - enforced coverage gap failure when `--fail-on-coverage-gap` is enabled.

## Validation

- `python -m ruff check tools/run_throughput_matrix.py tests/test_run_throughput_matrix.py`
- `python -m black --check tools/run_throughput_matrix.py tests/test_run_throughput_matrix.py`
- `python -m pytest -q tests/test_run_throughput_matrix.py`

## Throughput evidence artifacts

- Profile:
  - `artifacts/throughput/m9-chunk2-profile-20260304-r2.json`

- Matrix (coverage warning mode):
  - `artifacts/throughput/m9-chunk2-matrix-20260304-r2.json`
  - Result: `pass=true`, `coverage_pass=false`, gap reports missing `puffer_ppo` in trainer mode.

- Matrix (coverage enforced):
  - `artifacts/throughput/m9-chunk2-matrix-20260304-r2-enforced.json`
  - CLI exit: `2`
  - Result: `pass=false` due enforced missing backend coverage (`puffer_ppo`).

- Floor gate replay:
  - `artifacts/throughput/m9-chunk2-floor-gate-20260304-r2.json`
  - Result: `pass=true` for calibrated floors from the non-enforced matrix run.

## Outcome

- Chunk 2 throughput hardening is complete.
- Throughput evidence now distinguishes runtime performance regressions from evidence-coverage gaps, allowing strict enforcement when target trainer backend coverage is required.
