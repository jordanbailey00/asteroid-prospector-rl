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
12. `PUBLIC_UX_REALIGNMENT_PLAN_20260303.md`
    - Gap analysis and phased redesign plan for Replay/Play/Analytics plus operator-tooling boundary.

## 6) Graphics and audio

13. `graphics/asteroid_belt_prospector_graphics_spec.md`
    - Asset mapping and rendering/audio manifests.

## 7) Agent operating docs

14. `AGENT_HANDOFF_BRIEF.md`
    - Purpose, milestones, acceptance criteria.
15. `AGENT_HYGIENE_GUARDRAILS.md`
    - Non-negotiables and repo hygiene rules.
16. `ACCEPTANCE_TESTS_PARITY_HARNESS.md`
    - Determinism and parity gates.
17. `BUILD_CHECKLIST.md`
    - Ordered implementation checklist.
18. `PROJECT_STATUS.md`
    - Current project state, milestone board, and ordered next work.
19. `DECISION_LOG.md`
    - Architecture/process decisions with consequences.
20. `M65_MANUAL_VERIFICATION.md`
    - Final M6.5 replay/play checklist evidence and sampled replay trace.
21. `M9_DEPLOYMENT_RUNBOOK.md`
    - Production deployment wiring and smoke-check procedure for M9.
22. `M9_DEPLOYMENT_EVIDENCE_20260303.md`
    - Captured live endpoint values and smoke artifacts for the 2026-03-03 M9 deployment dry run.
23. `M7_BASELINE_BOTS_EXECUTION_20260304.md`
    - Chunk 1 execution record for baseline bot implementation, corrections, validation, and remaining M7 scope.
24. `M7_BENCHMARK_PROTOCOL_EXECUTION_20260304.md`
    - Chunk 2 execution record for benchmark protocol automation, validation evidence, and remaining M7 scope.
25. `M7_WANDB_BENCHMARK_EXECUTION_20260304.md`
    - Chunk 3 execution record for W&B benchmark logging + artifact lineage implementation and validation evidence.
26. `M9_CHUNK2_THROUGHPUT_EXECUTION_20260304.md`
    - Chunk 2 execution record for throughput reruns, backend-coverage gating, and calibrated-floor evidence updates.
27. `M9_CHUNK3_DRIFT_GUARDRAILS_EXECUTION_20260304.md`
    - Chunk 3 execution record for W&B/websocket/CORS drift guardrails and strict smoke evidence updates.

28. `M9_CHUNK4_MVP_CLOSEOUT_EXECUTION_20260304.md`
    - Final MVP closeout execution record, including full validation sweep and strict production smoke artifact.
29. `MVP_EXTENSIVE_TEST_PLAN_20260305.md`
    - End-to-end extensive validation plan for post-MVP verification campaign.
## Notes

- If any docs conflict, frozen interface docs (Section 2) win.
- Do not change observation/action/reward contracts unless intentionally versioning to `v2` and updating tests/docs.
