# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- M0 repository scaffold (`engine_core`, `python`, `server`, `frontend`, `training`, `replay`, `tests`, `tools`).
- Pre-commit hooks for Black, Ruff, clang-format, and basic hygiene checks.
- CI workflow and local `tools/run_checks.ps1` gate for format/lint/tests.
- `HelloProspectorEnv` contract-only stub with frozen interface shape/action checks.
- `ProspectorReferenceEnv` M1 pure-Python baseline with deterministic reset, full action decode, obs packing, reward computation, and required info metrics.
- Tier-1/Tier-2 pytest coverage for resource bounds, station rules, dt/time behavior, scan normalization, market stability, rollout stability, and info key presence.

### Docs
- Added authoritative root `AGENTS.md` instructions requiring commit/push after each completed change.
- Updated README files to reflect M1 status and run commands.
