# Asteroid Belt Prospector — Coding Agent Hygiene & Guardrails

This document is a **hard guardrail set** for any coding agent working on this repo. It is designed to prevent “going off the rails,” keep the codebase clean, and preserve performance and determinism.

If any instruction below conflicts with a frozen-interface doc, **the frozen-interface doc wins**.

---

## 1) Non‑negotiables (stop-the-line rules)

1. **Do not change the frozen RL interface** (obs layout, action indexing, reward definition) without explicit versioning and approval.
2. **Determinism must hold under fixed seed + action sequence** (required by the parity harness).
3. **No “clever” rewrites**: preserve readability and minimize diff size.
4. **No large refactors mixed with feature work**.
5. **No new dependencies** without a written rationale + approval.

These are aligned with “code health improves over time” code review standards. citeturn0search0turn2search7

---

## 2) Change management (how to keep the repo clean)

### 2.1 Small changes only
- Prefer **small, focused PRs** (one purpose).
- Avoid “fully formed projects” in one PR; reviewers can reject large changes as too big to digest. citeturn2search7turn0search12

### 2.2 One PR = one theme
Examples of acceptable “themes”:
- Implement `engine_core/reset + step` in C
- Add parity test harness
- Add replay recorder schema validation
- Add a frontend replay player

Unacceptable:
- “Implement engine core + rewrite docs + redesign frontend + add new logging stack”

### 2.3 No drive-by formatting
If you run formatters, keep it limited to touched files.
Use auto-format in pre-commit/CI so formatting stays consistent. citeturn1search0turn1search2

---

## 3) Required repo hygiene (files the agent must maintain)

### 3.1 Keep the following always accurate
- `DOCS_INDEX.md`
- `AGENT_HANDOFF_BRIEF.md`
- `ACCEPTANCE_TESTS_PARITY_HARNESS.md`
- Replay frame schema (if changed, must be versioned)

### 3.2 Changelog discipline
- Maintain `CHANGELOG.md` in a human-readable format (not a git log dump). citeturn0search3
- Use Semantic Versioning when you publish versioned releases/APIs. citeturn0search11

### 3.3 Commit message discipline
Use **Conventional Commits**:
- `feat: ...`, `fix: ...`, `refactor: ...`, `test: ...`, `docs: ...`
This helps automated changelog/version tooling and keeps history readable. citeturn0search2

---

## 4) Automated code quality (must be enforced)

### 4.1 Pre-commit hooks (required)
Use the `pre-commit` framework to run checks before every commit. citeturn1search0

Required hooks:
- Python formatting: **Black** citeturn1search2
- Python linting/format checks: **Ruff** citeturn1search1
- C/C++ formatting: **clang-format** citeturn1search7turn1search3
- Basic hygiene: trailing whitespace, EOF newline, large file checks

### 4.2 Python style rules
- Follow **PEP 8** conventions. citeturn0search1
- Use Black formatting to minimize diff noise and make reviews faster. citeturn1search2

### 4.3 C/C++ style rules
- Adopt one clang-format style and stick to it (LLVM or Google are fine).
- Use LLVM coding standards as a baseline for large-scale maintainable native code. citeturn2search2

---

## 5) Testing & CI gates (no broken main)

### 5.1 Rule: tests ship with the change
Tests should be added in the same change as the behavior they validate. citeturn0search0turn0search20

### 5.2 Must-pass checks on every PR
- Tier 0/1/2 tests from `ACCEPTANCE_TESTS_PARITY_HARNESS.md`
- Lint/format checks (Ruff/Black/clang-format)
- Replay schema validation (if replay code touched)

### 5.3 Parity harness is mandatory before “real training”
Before any long PufferLib run:
- parity harness must pass for the configured seed/action suites
- determinism under fixed seed must be verified

---

## 6) Architectural guardrails (how to avoid spaghetti)

### 6.1 One authoritative engine
- There is exactly **one** authoritative simulation implementation:
  - C core owns the hot loop state + dynamics
  - Python wrapper is thin, does not duplicate logic
- A Python reference env may exist for parity/debug only, but must not diverge.

### 6.2 “No business logic in the UI”
- Frontend only renders:
  - replay frames
  - human-play session outputs
  - historical metrics
- All authoritative rules are backend/engine-owned.

### 6.3 Filesystem boundaries
- `engine_core/` (C): deterministic step/reset, obs packing, reward
- `python/` wrapper: Gym API + PufferLib integration
- `server/`: HTTP/WS API, replay serving, play sessions
- `frontend/`: UI only
Do not blur these boundaries.

---

## 7) Performance guardrails (keep it fast)

### 7.1 C core performance principles
- No per-step heap allocations in the hot loop.
- Preallocate observation buffers; fill in-place.
- Avoid Python↔C call overhead per env if possible:
  - implement batch stepping (step_many) when needed.

### 7.2 Profiling rule
Any “performance improvement” PR must include:
- before/after steps/sec benchmark (same seed/config)
- method used to measure (script + command)

### 7.3 Safety rule (native code)
- Prefer well-defined behavior; avoid undefined behavior.
- Use sanitizer builds during development (ASan/UBSan) when feasible.

---

## 8) Telemetry & observability guardrails (W&B + Puffer dashboards)

### 8.1 W&B logging cadence
- Log **window-level metrics** (once per window_env_steps), not per-step.
- Use `wandb.init(config=...)` for run configuration and `Run.log()` for metrics. citeturn2search4
- Write end-of-run summary metrics to `run.summary`. citeturn2search0

### 8.2 W&B artifacts (checkpoints + replays)
Use W&B Artifacts to track and version:
- model checkpoints
- replay files (or a replay bundle)
Artifacts are designed for input/output versioning and lineage. citeturn2search1turn2search12

Naming rules:
- checkpoint artifact type includes `model` if you want model registry linkage. citeturn2search9
- replay artifacts use deterministic naming based on `run_id/window_id/replay_id`.

### 8.3 Puffer dashboards
- Keep PufferLib’s dashboard outputs enabled for immediate feedback.
- Treat “Constellation” as an observability surface; do not couple core system logic to it.

---

## 9) Dependency hygiene

### 9.1 Python dependencies
- Pin runtime dependencies in `pyproject.toml`.
- Avoid adding heavy libraries unless needed.
- Prefer Ruff over multiple separate lint tools where possible (it is designed to replace many linters/formatters). citeturn1search1

### 9.2 JS/TS dependencies
- No new UI libraries without explicit approval.
- Keep bundles small; avoid heavy visualization frameworks if Recharts suffices.

---

## 10) Documentation hygiene

### 10.1 “Why” comments only
- Comments should explain **why**, not what the code already states.
- Document invariants and constraints (seed handling, tolerance choices).

### 10.2 ADRs for big decisions
Any of these requires an ADR:
- changing replay format
- changing telemetry approach
- changing C/Python boundary
- changing any frozen interface element

---

## 11) PR checklist (agent must include in PR description)

- [ ] Scope is one theme; diff is small and reviewable. citeturn2search7turn0search12
- [ ] No frozen interface changes, or version bump/approval included.
- [ ] Tests added/updated and passing. citeturn0search20
- [ ] Pre-commit hooks pass (Ruff/Black/clang-format). citeturn1search0turn1search2turn1search7
- [ ] Parity harness updated if C core touched.
- [ ] W&B logging keys/artifacts updated if training/eval changed. citeturn2search1turn2search4
- [ ] Docs updated if behavior changed (DOCS_INDEX + relevant spec).
- [ ] Benchmarks included for performance changes.

---

## 12) What to do when uncertain

If the agent is unsure, it must:
1) Point to the relevant spec file(s).
2) Propose the minimal change to satisfy the spec.
3) Add/extend tests before implementing “optimizations”.
4) Avoid guessing on frozen contracts.

The priority is long-term code health and maintainability. citeturn0search0turn2search7
