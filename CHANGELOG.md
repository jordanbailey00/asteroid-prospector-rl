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
- Additional M1 hardening tests for determinism, reward sanity, and frozen observation layout contract checks.
- M2 native scaffold for `engine_core` with deterministic PCG32 RNG and C API skeleton (`abp_core_init/reset/step`).
- Handle-based native API (`abp_core_create/destroy`) for Python bindings.
- `engine_core/core_test_runner.c` smoke CLI for action-trace execution.
- `tools/build_native_core.ps1` for building `abp_core.dll` and `core_test_runner.exe`.
- Python ctypes native wrapper `NativeProspectorCore` and wrapper tests.

### Environment
- Installed missing development dependencies and toolchains:
  - `pre-commit` (Python package)
  - `hypothesis` (Python package)
  - `Kitware.CMake`
  - `LLVM.LLVM`
  - `BrechtSanders.WinLibs.MCF.UCRT` (GCC/MinGW toolchain)
- Configured user PATH to include CMake, LLVM, and WinLibs binaries.

### Docs
- Added authoritative root `AGENTS.md` instructions requiring commit/push after each completed change.
- Updated README files to reflect M2 scaffold status and native build commands.
