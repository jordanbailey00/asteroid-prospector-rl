# python

Python-facing environment package for Asteroid Belt Prospector.

Current components:
- `HelloProspectorEnv`: M0 contract-only stub.
- `ProspectorReferenceEnv`: M1 pure-Python reference implementation used as the correctness baseline for parity.
- `NativeProspectorCore`: ctypes wrapper for the M2 C core scaffold (`engine_core/build/abp_core.dll`).

Both env implementations preserve the frozen interface contract:
- observation shape `(260,)`
- action space size `69` (`0..68`)
- Gymnasium-style `(obs, reward, terminated, truncated, info)` step return
