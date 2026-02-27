# Decision Log

Use this file for non-trivial project decisions.

## Entry format

- ID: `ADR-XXXX`
- Date: `YYYY-MM-DD`
- Status: `Accepted`, `Superseded`, or `Proposed`
- Context
- Decision
- Consequences
- Related commits/docs

## Entries

### ADR-0001 - Freeze RL interface v1

- Date: 2026-02-27
- Status: Accepted
- Context: Training and replay compatibility require stable contracts across milestones.
- Decision: Keep `OBS_DIM=260`, `N_ACTIONS=69`, action indexing `0..68`, observation field layout, and reward definition frozen unless a versioned `v2` is explicitly created with docs/tests updates.
- Consequences: Feature work must fit inside the existing interface; incompatible changes require explicit versioning.
- Related commits/docs: `docs/RL spec/01_core_constants.md` through `docs/RL spec/06_gym_puffer_compat.md`

### ADR-0002 - Build order follows reference-first parity flow

- Date: 2026-02-27
- Status: Accepted
- Context: Native implementation without a correctness oracle increases risk of hard-to-debug divergence.
- Decision: Implement milestones in this order: M0 scaffold, M1 Python reference environment, M2 native C core, then M2.5 parity harness gating further training work.
- Consequences: Native progress is blocked from advancing to training integration until parity requirements pass.
- Related commits/docs: `001a004`, `b009f53`, `287a79f`, `docs/BUILD_CHECKLIST.md`, `docs/ACCEPTANCE_TESTS_PARITY_HARNESS.md`

### ADR-0003 - Use handle-based C API with Python ctypes wrapper for initial bridge

- Date: 2026-02-27
- Status: Accepted
- Context: Need a minimal, testable path from Python to native core early in M2.
- Decision: Expose `abp_core_create/destroy/init/reset/step` in C and bind with `ctypes` in Python first; defer heavier binding stacks until needed.
- Consequences: Fast bootstrap for parity work with low dependency cost; future binding migrations must preserve API semantics.
- Related commits/docs: `287a79f`, `87c2fc0`, `python/asteroid_prospector/native_core.py`

### ADR-0004 - Enforce commit-level project tracking updates

- Date: 2026-02-27
- Status: Accepted
- Context: Existing status information was split across docs and not guaranteed to stay synchronized with commits.
- Decision: Require each commit to update at least one tracking artifact (`docs/PROJECT_STATUS.md`, `docs/DECISION_LOG.md`, or `CHANGELOG.md`) and record significant decisions in this log.
- Consequences: Higher documentation discipline and clearer current-state visibility; minor overhead added to each commit.
- Related commits/docs: `AGENTS.md`, `.pre-commit-config.yaml`, `tools/check_project_tracking.py`
