# Asteroid Belt Prospector

Current milestone: **M2 scaffold + M1 reference env**.

## Repo layout

- `engine_core/`: C core scaffold (authoritative engine target in later milestones)
- `python/`: Python env package (`HelloProspectorEnv`, `ProspectorReferenceEnv`, native core wrapper)
- `server/`: API server scaffold
- `frontend/`: Web UI scaffold
- `training/`: trainer/evaluator scaffold (placeholder)
- `replay/`: replay pipeline scaffold (placeholder)
- `tests/`: contract + Tier-1/2 + determinism/reward checks
- `tools/`: local quality/check and native build scripts

## Toolchain prerequisites

The project currently needs:
- Python 3.14+
- Node.js/npm (for future frontend milestones)
- C toolchain with `gcc` available in PATH for native core builds

Installed and configured in this workspace:
- `pre-commit`
- `cmake`
- LLVM tools
- WinLibs GCC toolchain

## Quick start (PowerShell)

```powershell
python -m pip install -U pip
python -m pip install numpy pytest hypothesis black ruff pre-commit clang-format
pre-commit install
```

## Quality gates

Run local format/lint/test checks:

```powershell
.\tools\run_checks.ps1
```

Run tests only:

```powershell
pytest -q
```

## Native core build (M2 scaffold)

```powershell
.\tools\build_native_core.ps1
```

This produces:
- `engine_core/build/abp_core.dll`
- `engine_core/build/core_test_runner.exe`

Run trace smoke test (requires an actions file):

```powershell
.\engine_core\build\core_test_runner.exe --seed 42 --actions .\actions.bin --out .\trace.bin
```

## Environment stubs

M0 contract-only env:

```powershell
$env:PYTHONPATH = "python"
python -c "from asteroid_prospector import HelloProspectorEnv; e=HelloProspectorEnv(); o,_=e.reset(seed=1); print(o.shape, o.dtype, e.action_space.n)"
```

M1 reference env:

```powershell
$env:PYTHONPATH = "python"
python -c "from asteroid_prospector import ProspectorReferenceEnv; e=ProspectorReferenceEnv(seed=1); o,_=e.reset(seed=1); print(o.shape, o.dtype, e.action_space.n); print(e.step(6)[1:])"
```

M2 native core wrapper (after build):

```powershell
$env:PYTHONPATH = "python"
python -c "from asteroid_prospector import NativeProspectorCore; c=NativeProspectorCore(seed=1); o=c.reset(1); print(o.shape); print(c.step(6)[1:]); c.close()"
```

Frozen interface values remain unchanged:
- observation shape `(260,)`
- action space size `69` (`0..68`)
