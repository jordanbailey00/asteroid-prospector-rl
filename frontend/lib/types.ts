export interface RunSummary {
  run_id: string;
  status: string | null;
  trainer_backend: string | null;
  env_steps_total: number | null;
  windows_emitted: number | null;
  checkpoints_written: number | null;
  latest_checkpoint: Record<string, unknown> | null;
  latest_replay: ReplayEntry | null;
  replay_count: number;
  updated_at: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface RunsResponse {
  runs: RunSummary[];
  count: number;
  total: number;
}

export interface ReplayEntry {
  run_id: string;
  window_id: number;
  replay_id: string;
  replay_path: string;
  checkpoint_path: string;
  tags: string[];
  return_total: number;
  profit: number;
  survival: number;
  steps: number;
  created_at?: string;
  [key: string]: unknown;
}

export interface ReplayListResponse {
  run_id: string;
  index_path: string;
  count: number;
  replays: ReplayEntry[];
}

export interface ReplayFramesResponse {
  run_id: string;
  replay_id: string;
  offset: number;
  next_offset: number;
  count: number;
  has_more: boolean;
  frames: ReplayFrame[];
}

export interface ReplayFrame {
  frame_index: number;
  t?: number;
  dt?: number;
  action: number;
  reward: number;
  terminated?: boolean;
  truncated?: boolean;
  events?: string[];
  render_state?: Record<string, unknown>;
  info?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface RunDetailResponse {
  run_id: string;
  replay_count: number;
  metadata: Record<string, unknown>;
}

export interface MetricsWindowsResponse {
  run_id: string;
  metrics_path: string;
  count: number;
  total: number;
  windows: WindowMetric[];
}

export interface WindowMetric {
  window_id: number;
  env_steps_total?: number;
  reward_mean?: number;
  return_mean?: number;
  profit_mean?: number;
  survival_rate?: number;
  invalid_action_rate?: number;
  [key: string]: unknown;
}

export interface PlaySessionCreateRequest {
  seed?: number;
  env_time_max?: number;
}

export interface PlaySessionResetRequest {
  seed?: number;
}

export interface PlaySessionStepRequest {
  action: number;
}

export interface PlaySessionState {
  session_id: string;
  seed?: number;
  env_time_max?: number;
  obs: number[];
  reward: number;
  terminated: boolean;
  truncated: boolean;
  info: Record<string, unknown>;
  steps: number;
  created_at?: string;
  updated_at?: string;
  action?: number;
}

export interface PlaySessionDeleteResponse {
  session_id: string;
  deleted: boolean;
}
