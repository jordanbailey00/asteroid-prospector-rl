# Asteroid Belt Prospector — Docs Index (Read Order)

This index is the “table of contents” for the project specs. Start at the top and read in order.

---

## 1) Product and RL intent

1. `asteroid_belt_prospector_rl_gdd.md`
   - High-level game design, strategic depth, and what success means.

---

## 2) Frozen RL interface (do not change once training begins)

These define the stable interface contract for training and replays.

2. `01_core_constants.md`
   - OBS_DIM, N_ACTIONS, padding limits, normalization constants.
3. `02_observation_vector.md`
   - Exact observation vector field mapping (indices).
4. `03_action_space.md`
   - Exact action indexing (0..68).
5. `04_transition_function.md`
   - Step ordering + transition pseudocode.
6. `05_reward_function.md`
   - Code-ready reward definition.
7. `06_gym_puffer_compat.md`
   - Gym/Puffer compatibility requirements + required info metrics.

---

## 3) Benchmarks

8. `07_baseline_bots.md`
   - Greedy miner, cautious scanner, market timer.
9. `08_benchmarking_protocol.md`
   - How to compare learning vs baselines.

---

## 4) Backend implementation

10. `asteroid_belt_prospector_backend_engine_c_wandb.md`
   - Authoritative engine architecture (includes C/native core boundary + W&B logging expectations).

---

## 5) Frontend implementation

11. `asteroid_belt_prospector_frontend_spec_wandb.md`
   - Vercel-hosted UI requirements: replay, play mode, analytics (with W&B/Constellation integration links).

---

## 6) Graphics and audio

12. `asteroid_belt_prospector_graphics_spec.md`
   - How to map Kenney pack assets + sound cues to entities/actions/events + manifest-driven rendering.

---

## 7) Project context (agent-facing)

13. `AGENT_HANDOFF_BRIEF.md`
   - Why this exists, system overview, build order, acceptance criteria, naming conventions.

---

## Notes

- If anything conflicts: the “Frozen RL interface” docs win.
- Avoid changes to the obs/action/reward contract unless you intentionally version the environment.
