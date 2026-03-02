import pytest

from tools import smoke_m9_deployment as smoke


def _cfg(**overrides) -> smoke.SmokeConfig:
    payload = {
        "backend_http_base": "https://api.example.com",
        "backend_ws_base": "wss://api.example.com",
        "frontend_base": "https://app.example.com",
        "timeout_seconds": 5.0,
        "run_id": None,
        "replay_id": None,
        "allow_empty_runs": True,
        "skip_wandb": False,
        "require_clean_wandb_status": False,
        "wandb_entity": "team-astro",
        "wandb_project": "asteroid-prospector",
        "output_path": None,
    }
    payload.update(overrides)
    return smoke.SmokeConfig(**payload)


def test_derive_ws_base_from_http_and_https() -> None:
    assert smoke.derive_ws_base("http://example.com") == "ws://example.com"
    assert smoke.derive_ws_base("https://example.com") == "wss://example.com"
    assert smoke.derive_ws_base("ws://example.com") == "ws://example.com"


def test_normalize_base_trims_and_removes_trailing_slash() -> None:
    assert smoke.normalize_base(" https://api.example.com/ ") == "https://api.example.com"
    assert smoke.normalize_base("http://localhost:8000") == "http://localhost:8000"


def test_build_url_without_query() -> None:
    assert (
        smoke._build_url("https://api.example.com", "/health") == "https://api.example.com/health"
    )


def test_build_url_with_query() -> None:
    built = smoke._build_url("https://api.example.com", "/api/runs", {"limit": 10, "order": "desc"})
    assert built in {
        "https://api.example.com/api/runs?limit=10&order=desc",
        "https://api.example.com/api/runs?order=desc&limit=10",
    }


def test_parse_wandb_status_payload_allows_available_non_strict() -> None:
    available, reason, notes = smoke._parse_wandb_status_payload(
        {
            "available": True,
            "reason": None,
            "notes": [" cache ttl is low ", ""],
        },
        require_clean_wandb_status=False,
    )
    assert available is True
    assert reason is None
    assert notes == ["cache ttl is low"]


def test_parse_wandb_status_payload_rejects_unavailable() -> None:
    with pytest.raises(RuntimeError, match="reported unavailable"):
        smoke._parse_wandb_status_payload(
            {
                "available": False,
                "reason": "missing api key",
                "notes": [],
            },
            require_clean_wandb_status=False,
        )


def test_parse_wandb_status_payload_rejects_notes_in_strict_mode() -> None:
    with pytest.raises(RuntimeError, match="strict mode"):
        smoke._parse_wandb_status_payload(
            {
                "available": True,
                "reason": None,
                "notes": ["missing defaults"],
            },
            require_clean_wandb_status=True,
        )


def test_check_wandb_latest_returns_first_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_http_json(*, session, url: str, timeout_seconds: float):
        del session
        del timeout_seconds
        assert "/api/wandb/runs/latest" in url
        return 200, {
            "runs": [
                {"run_id": " wb-iter-009 "},
                {"run_id": "wb-iter-008"},
            ]
        }

    monkeypatch.setattr(smoke, "_http_json", fake_http_json)

    detail, run_id = smoke._check_wandb_latest(session=object(), cfg=_cfg())
    assert run_id == "wb-iter-009"
    assert "returned 2 rows" in detail


def test_wandb_run_detail_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_http_json(*, session, url: str, timeout_seconds: float):
        del session
        del timeout_seconds
        if "/summary" in url:
            return 200, {"run": {"run_id": "wb-iter-009"}}
        if "/history" in url and "/iteration-view" not in url:
            return 200, {"count": 1, "rows": [{"_step": 1, "window_id": 1}]}
        if "/iteration-view" in url:
            return 200, {
                "history": {"count": 1, "rows": [{"_step": 1, "window_id": 1}]},
                "kpis": {"window_id": 1, "return_mean": 2.0},
            }
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(smoke, "_http_json", fake_http_json)

    cfg = _cfg()
    summary_detail = smoke._check_wandb_run_summary(session=object(), cfg=cfg, run_id="wb-iter-009")
    history_detail = smoke._check_wandb_run_history(session=object(), cfg=cfg, run_id="wb-iter-009")
    view_detail = smoke._check_wandb_iteration_view(session=object(), cfg=cfg, run_id="wb-iter-009")

    assert "run_id=wb-iter-009" in summary_detail
    assert "returned 1 rows" in history_detail
    assert "history_rows=1" in view_detail


def test_run_smoke_executes_extended_wandb_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smoke, "_check_backend_health", lambda **kwargs: "ok")
    monkeypatch.setattr(smoke, "_check_backend_runs", lambda **kwargs: "ok")
    monkeypatch.setattr(smoke, "_discover_run_and_replay", lambda **kwargs: (None, None))
    monkeypatch.setattr(smoke, "_check_wandb_status", lambda **kwargs: "status-ok")
    monkeypatch.setattr(smoke, "_check_wandb_latest", lambda **kwargs: ("latest-ok", "wb-iter-009"))
    monkeypatch.setattr(smoke, "_check_wandb_run_summary", lambda **kwargs: "summary-ok")
    monkeypatch.setattr(smoke, "_check_wandb_run_history", lambda **kwargs: "history-ok")
    monkeypatch.setattr(smoke, "_check_wandb_iteration_view", lambda **kwargs: "view-ok")

    report = smoke.run_smoke(_cfg(frontend_base=None, allow_empty_runs=True))
    names = [row["name"] for row in report["checks"]]

    assert "wandb-run-summary" in names
    assert "wandb-run-history" in names
    assert "wandb-iteration-view" in names
    assert "wandb-status-post" in names
    assert names.index("wandb-status-post") > names.index("wandb-iteration-view")
    assert report["summary"]["pass"] is True


def test_run_smoke_skips_wandb_run_detail_checks_when_latest_has_no_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(smoke, "_check_backend_health", lambda **kwargs: "ok")
    monkeypatch.setattr(smoke, "_check_backend_runs", lambda **kwargs: "ok")
    monkeypatch.setattr(smoke, "_discover_run_and_replay", lambda **kwargs: (None, None))
    monkeypatch.setattr(smoke, "_check_wandb_status", lambda **kwargs: "status-ok")
    monkeypatch.setattr(smoke, "_check_wandb_latest", lambda **kwargs: ("latest-empty", None))

    report = smoke.run_smoke(_cfg(frontend_base=None, allow_empty_runs=True))
    rows = {row["name"]: row for row in report["checks"]}

    assert rows["wandb-run-summary"]["ok"] is True
    assert "returned no run_id" in rows["wandb-run-summary"]["detail"]
    assert rows["wandb-run-history"]["ok"] is True
    assert rows["wandb-iteration-view"]["ok"] is True
