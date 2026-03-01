# Asteroid Belt Prospector - Docs Index (Read Order)

This index is the table of contents for project specs and operating docs.
Read top to bottom.

## 1) Product and RL intent

1. `game design/asteroid_belt_prospector_rl_gdd.md`
   - High-level game design and strategic goals.

## 2) Frozen RL interface (highest precedence)

2. `RL spec/01_core_constants.md`
   - `OBS_DIM`, `N_ACTIONS`, normalization constants.
3. `RL spec/02_observation_vector.md`
   - Exact observation layout and indices.
4. `RL spec/03_action_space.md`
   - Exact action indexing (`0..68`).
5. `RL spec/04_transition_function.md`
   - Step ordering and transition logic.
6. `RL spec/05_reward_function.md`
   - Reward definition.
7. `RL spec/06_gym_puffer_compat.md`
   - Gym/Puffer compatibility and required info keys.

## 3) Benchmarks

8. `RL spec/07_baseline_bots.md`
   - Baseline bot definitions.
9. `RL spec/08_benchmarking_protocol.md`
   - Benchmark comparison protocol.

## 4) Backend implementation

10. `backend/asteroid_belt_prospector_backend_engine_c_wandb.md`
    - Authoritative backend architecture and observability requirements.

## 5) Frontend implementation

11. `frontend/asteroid_belt_prospector_frontend_spec_wandb.md`
    - Replay/player/analytics frontend requirements.

## 6) Graphics and audio

12. `graphics/asteroid_belt_prospector_graphics_spec.md`
    - Asset mapping and rendering/audio manifests.

## 7) Agent operating docs

13. `AGENT_HANDOFF_BRIEF.md`
    - Purpose, milestones, acceptance criteria.
14. `AGENT_HYGIENE_GUARDRAILS.md`
    - Non-negotiables and repo hygiene rules.
15. `ACCEPTANCE_TESTS_PARITY_HARNESS.md`
    - Determinism and parity gates.
16. `BUILD_CHECKLIST.md`
    - Ordered implementation checklist.
17. `PROJECT_STATUS.md`
    - Current project state, milestone board, and ordered next work.
18. `DECISION_LOG.md`
    - Architecture/process decisions with consequences.
19. `M65_MANUAL_VERIFICATION.md`
    - Final M6.5 replay/play checklist evidence and sampled replay trace.

## Notes

- If any docs conflict, frozen interface docs (Section 2) win.
- Do not change observation/action/reward contracts unless intentionally versioning to `v2` and updating tests/docs.
