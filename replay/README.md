# replay

Replay schema and index utilities used by the M4 eval runner.

## Files

- `replay/schema.py`
  - `REPLAY_SCHEMA_VERSION = 1`
  - `frame_from_step(...)` for canonical frame creation
  - `validate_replay_frame(...)` for schema checks
- `replay/index.py`
  - `REPLAY_INDEX_SCHEMA_VERSION = 1`
  - `load_replay_index(...)` and `append_replay_entry(...)`

## Frame format (`jsonl.gz`)

Each line is one frame with required keys:
- `schema_version`
- `frame_index`
- `t`, `dt`
- `action`
- `reward`
- `terminated`, `truncated`
- `render_state`
- `events`

Optional key:
- `info` (controlled by trainer flag `--eval-include-info` / `--no-eval-include-info`)

## Replay index format (`replay_index.json`)

Top-level keys:
- `schema_version`
- `run_id`
- `updated_at`
- `entries`

Each entry includes:
- `run_id`, `window_id`, `replay_id`
- `replay_path`, `checkpoint_path`
- `tags` (`every_window`, optional `best_so_far`)
- `return_total`, `profit`, `survival`, `steps`
- `terminated`, `truncated`
- `checkpoint_env_steps_total`
- `created_at`
