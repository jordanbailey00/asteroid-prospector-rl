# Asteroid Belt Prospector - Coding Agent Hygiene and Guardrails

This document defines repo-level guardrails for coding agents.
If any instruction conflicts with frozen RL interface docs, the frozen interface docs take precedence.

## 1) Non-negotiables

1. Do not change the frozen RL interface (`OBS_DIM`, `N_ACTIONS`, obs layout, action indexing, reward semantics) unless explicitly versioning and approved.
2. Determinism must hold for fixed seed + fixed action sequence.
3. Keep diffs focused. No broad mixed-scope rewrites.
4. Do not add dependencies without clear justification and docs updates.

## 2) Change scope discipline

- One change set should have one clear theme.
- Avoid drive-by edits in unrelated files.
- Keep refactors separate from feature work when possible.

## 3) Required project tracking hygiene

Every commit must update at least one of:
- `docs/PROJECT_STATUS.md`
- `docs/DECISION_LOG.md`
- `CHANGELOG.md`

Any non-trivial technical decision must add a same-commit ADR entry in `docs/DECISION_LOG.md`.

## 4) Testing and quality gates

- Add or update tests in the same change as behavior changes.
- Do not merge changes that break default test gates.
- Run parity checks when touching simulation logic.
- Run benchmark/profile evidence when claiming performance improvement.

## 5) Architecture boundaries

- `engine_core/`: authoritative simulation dynamics.
- `python/`: wrappers and reference implementation for parity/debug.
- `training/`: trainer/eval/replay orchestration.
- `server/`: API contracts, replay transport, play session orchestration.
- `frontend/`: presentation and UX only.

No business-rule authority should move into frontend code.

## 6) Performance guardrails

- Prefer native-path and batch-path optimization before micro-tuning.
- Avoid per-step heap allocation in C hot loops.
- Keep Python callback overhead out of hot stepping paths where practical.
- For performance claims, include reproducible before/after command + artifact.

## 7) Observability guardrails

- Window-level metrics are the canonical cadence (not per-step spam).
- Keep run metadata and replay index contracts stable for API/frontend consumers.
- Do not expose W&B secrets in frontend/runtime client code.

## 8) Documentation guardrails

- Keep `docs/DOCS_INDEX.md` aligned with actual docs.
- Keep milestone naming consistent across checklist/status/readme docs.
- Remove or rewrite stale docs rather than letting parallel conflicting guidance accumulate.

## 9) When uncertain

1. Check the frozen RL interface docs first.
2. Apply the smallest change that satisfies current requirements.
3. Add tests before optimization changes.
4. Record decisions and consequences in ADR form.
