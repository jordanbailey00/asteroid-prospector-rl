# engine_core

Native C scaffolding for the authoritative simulation core.

Current status (M2 scaffold):
- deterministic RNG module (`abp_rng`, PCG32)
- core state/config structs (`abp_core`)
- handle-based API (`abp_core_create/destroy`) plus `reset`/`step` and batched `reset_many`/`step_many` entry points
- smoke CLI runner `core_test_runner` for trace generation

Build with:

```powershell
.\tools\build_native_core.ps1
```

Artifacts:
- `engine_core/build/abp_core.dll`
- `engine_core/build/core_test_runner.exe`

This is still scaffold-only and does not yet implement full game dynamics parity with the Python reference env.
