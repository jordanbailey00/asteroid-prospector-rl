from __future__ import annotations

from dataclasses import dataclass

INFO_METRIC_KEYS = (
    "credits",
    "net_profit",
    "profit_per_tick",
    "survival",
    "overheat_ticks",
    "pirate_encounters",
    "value_lost_to_pirates",
    "fuel_used",
    "hull_damage",
    "tool_wear",
    "scan_count",
    "mining_ticks",
    "cargo_utilization_avg",
)


@dataclass(frozen=True)
class WindowRecord:
    run_id: str
    window_id: int
    window_complete: bool
    env_steps_start: int
    env_steps_end: int
    env_steps_in_window: int
    env_steps_total: int
    episodes_completed: int
    terminated_episodes: int
    truncated_episodes: int
    reward_sum: float
    reward_mean: float
    return_mean: float
    invalid_action_rate: float
    metric_means: dict[str, float]

    def to_dict(self) -> dict[str, float | int | str | bool]:
        payload: dict[str, float | int | str | bool] = {
            "run_id": self.run_id,
            "window_id": self.window_id,
            "window_complete": self.window_complete,
            "env_steps_start": self.env_steps_start,
            "env_steps_end": self.env_steps_end,
            "env_steps_in_window": self.env_steps_in_window,
            "env_steps_total": self.env_steps_total,
            "episodes_completed": self.episodes_completed,
            "terminated_episodes": self.terminated_episodes,
            "truncated_episodes": self.truncated_episodes,
            "reward_sum": self.reward_sum,
            "reward_mean": self.reward_mean,
            "return_mean": self.return_mean,
            "invalid_action_rate": self.invalid_action_rate,
            # Friendly aliases expected by downstream analytics specs.
            "profit_mean": self.metric_means.get("net_profit", 0.0),
            "survival_rate": self.metric_means.get("survival", 0.0),
        }
        for key, value in self.metric_means.items():
            payload[f"{key}_mean"] = value
        return payload


class WindowMetricsAggregator:
    """Aggregates step metrics into fixed-size window records."""

    def __init__(
        self,
        *,
        run_id: str,
        window_env_steps: int,
        info_metric_keys: tuple[str, ...] = INFO_METRIC_KEYS,
    ) -> None:
        if window_env_steps <= 0:
            raise ValueError("window_env_steps must be positive")

        self.run_id = run_id
        self.window_env_steps = int(window_env_steps)
        self.info_metric_keys = tuple(info_metric_keys)

        self.env_steps_total = 0
        self.episodes_total = 0
        self.current_window_id = 0

        self._window_start_step = 0
        self._window_steps = 0
        self._window_reward_sum = 0.0
        self._window_invalid_steps = 0
        self._window_metric_weighted_sums = {key: 0.0 for key in self.info_metric_keys}

        self._window_episodes_completed = 0
        self._window_terminated_episodes = 0
        self._window_truncated_episodes = 0
        self._window_episode_return_sum = 0.0

        self._current_episode_return = 0.0

    def record_step(
        self,
        *,
        reward: float,
        info: dict[str, object],
        terminated: bool,
        truncated: bool,
    ) -> list[WindowRecord]:
        emitted: list[WindowRecord] = []

        dt = int(info.get("dt", 1))
        if dt <= 0:
            dt = 1

        invalid_action = bool(info.get("invalid_action", False))
        reward_f = float(reward)
        self._current_episode_return += reward_f

        remaining = dt
        while remaining > 0:
            room = self.window_env_steps - self._window_steps
            take = min(room, remaining)
            frac = float(take) / float(dt)
            is_final_segment = take == remaining

            self._window_steps += take
            self.env_steps_total += take
            self._window_reward_sum += reward_f * frac

            if invalid_action:
                self._window_invalid_steps += take

            for key in self.info_metric_keys:
                self._window_metric_weighted_sums[key] += float(info.get(key, 0.0)) * float(take)

            if is_final_segment and (terminated or truncated):
                self.episodes_total += 1
                self._window_episodes_completed += 1
                self._window_terminated_episodes += int(terminated)
                self._window_truncated_episodes += int(truncated)
                self._window_episode_return_sum += self._current_episode_return
                self._current_episode_return = 0.0

            remaining -= take

            if self._window_steps == self.window_env_steps:
                emitted.append(self._emit_window(window_complete=True))

        return emitted

    def flush_partial(self) -> WindowRecord | None:
        if self._window_steps == 0:
            return None
        return self._emit_window(window_complete=False)

    def _emit_window(self, *, window_complete: bool) -> WindowRecord:
        steps = max(1, self._window_steps)
        metric_means = {
            key: self._window_metric_weighted_sums[key] / float(steps)
            for key in self.info_metric_keys
        }

        record = WindowRecord(
            run_id=self.run_id,
            window_id=self.current_window_id,
            window_complete=window_complete,
            env_steps_start=self._window_start_step,
            env_steps_end=self._window_start_step + self._window_steps,
            env_steps_in_window=self._window_steps,
            env_steps_total=self.env_steps_total,
            episodes_completed=self._window_episodes_completed,
            terminated_episodes=self._window_terminated_episodes,
            truncated_episodes=self._window_truncated_episodes,
            reward_sum=self._window_reward_sum,
            reward_mean=self._window_reward_sum / float(steps),
            return_mean=(
                self._window_episode_return_sum / float(self._window_episodes_completed)
                if self._window_episodes_completed > 0
                else 0.0
            ),
            invalid_action_rate=float(self._window_invalid_steps) / float(steps),
            metric_means=metric_means,
        )

        self.current_window_id += 1
        self._window_start_step = self.env_steps_total
        self._window_steps = 0
        self._window_reward_sum = 0.0
        self._window_invalid_steps = 0
        self._window_metric_weighted_sums = {key: 0.0 for key in self.info_metric_keys}
        self._window_episodes_completed = 0
        self._window_terminated_episodes = 0
        self._window_truncated_episodes = 0
        self._window_episode_return_sum = 0.0

        return record
