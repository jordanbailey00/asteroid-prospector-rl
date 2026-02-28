import pytest

from replay.schema import REPLAY_SCHEMA_VERSION, frame_from_step, validate_replay_frame


def test_frame_from_step_round_trip_validation() -> None:
    frame = frame_from_step(
        frame_index=0,
        t=7,
        dt=2,
        action=3,
        reward=1.5,
        terminated=False,
        truncated=False,
        render_state={"x": 1.0},
        events=["scan"],
        info={"credits": 42.0},
        include_info=True,
    )

    assert frame["schema_version"] == REPLAY_SCHEMA_VERSION
    assert frame["t"] == 7
    assert frame["dt"] == 2
    assert frame["action"] == 3
    assert frame["reward"] == 1.5
    assert frame["info"] == {"credits": 42.0}

    validate_replay_frame(frame)


def test_frame_from_step_can_omit_info() -> None:
    frame = frame_from_step(
        frame_index=1,
        t=9,
        dt=1,
        action=4,
        reward=0.0,
        terminated=False,
        truncated=False,
        render_state={"x": 0.0},
        events=[],
        info={"ignored": True},
        include_info=False,
    )

    assert "info" not in frame
    validate_replay_frame(frame)


def test_validate_replay_frame_rejects_missing_required_keys() -> None:
    with pytest.raises(ValueError, match="missing keys"):
        validate_replay_frame({"schema_version": REPLAY_SCHEMA_VERSION})


def test_validate_replay_frame_rejects_invalid_dt() -> None:
    frame = frame_from_step(
        frame_index=0,
        t=0,
        dt=1,
        action=0,
        reward=0.0,
        terminated=False,
        truncated=False,
        render_state={},
        events=[],
        info=None,
        include_info=False,
    )
    frame["dt"] = 0

    with pytest.raises(ValueError, match="dt must be positive"):
        validate_replay_frame(frame)
