# python

Python-facing environment package for Asteroid Belt Prospector.

Current components:
- `HelloProspectorEnv`: M0 contract-only stub.
- `ProspectorReferenceEnv`: M1 pure-Python reference implementation used as the correctness baseline for future C parity.

Both preserve the frozen interface contract:
- observation shape `(260,)`
- action space size `69` (`0..68`)
- Gymnasium-style `(obs, reward, terminated, truncated, info)` step return
