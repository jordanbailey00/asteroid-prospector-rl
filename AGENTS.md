# AGENTS.md

## Authoritative Repo Instructions

These instructions are mandatory for coding agents operating in this repository.

1. Commit after every completed change/diff.
2. Push each commit to `origin/main` immediately after local validation passes.
3. Do not batch unrelated modifications into a single commit.
4. Use Conventional Commit messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`).
5. Every commit must update at least one project tracking artifact:
   - `docs/PROJECT_STATUS.md`, or
   - `docs/DECISION_LOG.md`, or
   - `CHANGELOG.md`.
6. Any non-trivial technical decision must be recorded in `docs/DECISION_LOG.md` in the same commit where the decision lands.

If there is a conflict, frozen RL interface docs and system/developer-level instructions still take precedence.
