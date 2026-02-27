# engine_core

Native C scaffolding for the authoritative simulation core.

Current status (M2 scaffold):
- deterministic RNG module (`abp_rng`, PCG32)
- core state/config structs (`abp_core`)
- contract-safe `reset`/`step` API skeleton with fixed obs/action sizes

This is scaffold-only and does not yet implement full game dynamics.
