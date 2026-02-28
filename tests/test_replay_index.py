from replay.index import filter_replay_entries, get_replay_entry_by_id


def _entry(window_id: int, replay_id: str, tags: list[str]) -> dict:
    return {
        "window_id": window_id,
        "replay_id": replay_id,
        "created_at": f"2026-02-28T00:00:0{window_id}+00:00",
        "tags": tags,
    }


def test_filter_replay_entries_by_single_tag_and_limit() -> None:
    entries = [
        _entry(0, "r0", ["every_window"]),
        _entry(1, "r1", ["every_window", "best_so_far"]),
        _entry(2, "r2", ["milestone:profit:100"]),
    ]

    filtered = filter_replay_entries(entries, tag="every_window", limit=1)

    assert len(filtered) == 1
    assert filtered[0]["replay_id"] == "r1"


def test_filter_replay_entries_by_tags_any_all_and_window() -> None:
    entries = [
        _entry(0, "r0", ["every_window", "milestone:survival:1"]),
        _entry(1, "r1", ["every_window", "best_so_far"]),
        _entry(1, "r2", ["every_window", "milestone:profit:500"]),
    ]

    filtered = filter_replay_entries(
        entries,
        tags_any=["best_so_far", "milestone:profit:500"],
        tags_all=["every_window"],
        window_id=1,
    )

    assert [item["replay_id"] for item in filtered] == ["r2", "r1"]


def test_get_replay_entry_by_id_returns_match_or_none() -> None:
    index_payload = {
        "entries": [
            _entry(0, "r0", ["every_window"]),
            _entry(1, "r1", ["every_window", "best_so_far"]),
        ]
    }

    assert get_replay_entry_by_id(index_payload, "r1") is not None
    assert get_replay_entry_by_id(index_payload, "missing") is None
