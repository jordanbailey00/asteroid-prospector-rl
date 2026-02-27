# Asteroid Prospector RL

Asteroid Prospector RL is a reinforcement learning project focused on a strategic asteroid-belt simulation where agents balance scanning, mining, travel risk, fuel, heat, and market timing over long horizons.

This repository is building both:
- a high-throughput training environment for RL experimentation, and
- a replay and web visualization stack so training progress is understandable to humans.

## Why this project exists

Most RL demos are hard to interpret from raw metrics alone. This project combines environment correctness, deterministic evaluation, and replay tooling so each training window can be inspected with concrete gameplay traces.

Design goals:
- Keep a frozen RL interface contract for stable training.
- Use a Python reference implementation first, then a C core for performance.
- Produce replay artifacts and analytics that map directly to training progress.
- Provide a human-play mode against the same environment rules.

## Current status

Milestones completed so far:
- M0: Repository scaffold, quality gates, and hello-environment contract stub.
- M1: Pure Python reference environment and contract/determinism tests.
- M2 (in progress): Native C core scaffold and Python wrapper smoke path.

Interface contract currently held constant:
- Observation dimension: `260`
- Action count: `69` (indexed `0..68`)

## Tech stack

- Simulation core: C (`engine_core/`) with deterministic RNG and handle-based API.
- RL/env layer: Python (`python/`) with Gym-style reset/step interfaces.
- Testing: `pytest` (+ property tests where needed).
- Code quality: Black, Ruff, clang-format, pre-commit.
- CI: GitHub Actions (`.github/workflows/ci.yml`) running format/lint/tests.
- Planned platform components: API server + frontend replay viewer/human-play mode.

## Repository structure

- `engine_core/` - native C environment core.
- `python/` - Python package, wrappers, and reference env.
- `server/` - API server scaffold.
- `frontend/` - frontend scaffold.
- `training/` - trainer/evaluator scaffolding.
- `replay/` - replay pipeline scaffolding.
- `tests/` - contract, parity, determinism, and wrapper tests.
- `docs/` - specs, milestone brief, and authoritative design docs.
- `tools/` - local check/build scripts.

For module-level details, see:
- `python/README.md`
- `engine_core/README.md`

## Quick start

### 1) Install Python tooling

```powershell
python -m pip install -U pip
python -m pip install numpy pytest hypothesis black ruff pre-commit clang-format
pre-commit install
```

### 2) Run quality checks

```powershell
.\tools\run_checks.ps1
```

### 3) Run tests only

```powershell
pytest -q
```

### 4) Build native C scaffold (optional, current M2 path)

```powershell
.\tools\build_native_core.ps1
```

## Documentation order

Project docs are intentionally ordered. Start with:
- `docs/DOCS_INDEX.md`
- `docs/AGENT_HANDOFF_BRIEF.md`

These point to the frozen RL contract docs and milestone plan.

## License

This project is licensed under the MIT License. See `LICENSE`.
