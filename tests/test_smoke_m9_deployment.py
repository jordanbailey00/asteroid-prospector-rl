import pytest

from tools.smoke_m9_deployment import (
    _build_url,
    _parse_wandb_status_payload,
    derive_ws_base,
    normalize_base,
)


def test_derive_ws_base_from_http_and_https() -> None:
    assert derive_ws_base("http://example.com") == "ws://example.com"
    assert derive_ws_base("https://example.com") == "wss://example.com"
    assert derive_ws_base("ws://example.com") == "ws://example.com"


def test_normalize_base_trims_and_removes_trailing_slash() -> None:
    assert normalize_base(" https://api.example.com/ ") == "https://api.example.com"
    assert normalize_base("http://localhost:8000") == "http://localhost:8000"


def test_build_url_without_query() -> None:
    assert _build_url("https://api.example.com", "/health") == "https://api.example.com/health"


def test_build_url_with_query() -> None:
    built = _build_url("https://api.example.com", "/api/runs", {"limit": 10, "order": "desc"})
    assert built in {
        "https://api.example.com/api/runs?limit=10&order=desc",
        "https://api.example.com/api/runs?order=desc&limit=10",
    }


def test_parse_wandb_status_payload_allows_available_non_strict() -> None:
    available, reason, notes = _parse_wandb_status_payload(
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
        _parse_wandb_status_payload(
            {
                "available": False,
                "reason": "missing api key",
                "notes": [],
            },
            require_clean_wandb_status=False,
        )


def test_parse_wandb_status_payload_rejects_notes_in_strict_mode() -> None:
    with pytest.raises(RuntimeError, match="strict mode"):
        _parse_wandb_status_payload(
            {
                "available": True,
                "reason": None,
                "notes": ["missing defaults"],
            },
            require_clean_wandb_status=True,
        )
