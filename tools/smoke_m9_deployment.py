from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import ssl
import struct
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlparse

import requests

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


@dataclass(frozen=True)
class SmokeConfig:
    backend_http_base: str
    backend_ws_base: str
    frontend_base: str | None
    cors_origin: str | None
    timeout_seconds: float
    ws_check_attempts: int
    run_id: str | None
    replay_id: str | None
    allow_empty_runs: bool
    skip_wandb: bool
    require_clean_wandb_status: bool
    wandb_entity: str | None
    wandb_project: str | None
    output_path: Path | None


@dataclass(frozen=True)
class SmokeCheckResult:
    name: str
    ok: bool
    elapsed_ms: float
    detail: str


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def normalize_base(base: str) -> str:
    text = base.strip()
    if text.endswith("/"):
        return text[:-1]
    return text


def derive_ws_base(http_base: str) -> str:
    if http_base.startswith("https://"):
        return f"wss://{http_base[len('https://') :]}"
    if http_base.startswith("http://"):
        return f"ws://{http_base[len('http://') :]}"
    return http_base


def normalize_origin(origin: str) -> str:
    parsed = urlparse(origin.strip())
    if parsed.scheme not in {"http", "https"} or parsed.netloc.strip() == "":
        raise ValueError(f"Invalid CORS origin: {origin!r}")
    return f"{parsed.scheme}://{parsed.netloc}"


def _origin_from_base(base: str | None) -> str | None:
    if base is None:
        return None
    parsed = urlparse(base)
    if parsed.scheme in {"http", "https"} and parsed.netloc.strip() != "":
        return f"{parsed.scheme}://{parsed.netloc}"
    return None


def _resolve_cors_origin(cfg: SmokeConfig) -> str | None:
    if cfg.cors_origin is not None and cfg.cors_origin.strip() != "":
        return normalize_origin(cfg.cors_origin)
    return _origin_from_base(cfg.frontend_base)


def _build_url(
    base: str,
    path: str,
    query: dict[str, str | int] | None = None,
) -> str:
    if query is None or len(query) == 0:
        return f"{base}{path}"
    return f"{base}{path}?{urlencode(query)}"


def _recv_exact(
    sock: socket.socket,
    count: int,
    prefetched: bytearray | None = None,
) -> bytes:
    chunks: list[bytes] = []
    remaining = int(count)

    if prefetched is not None and len(prefetched) > 0:
        take = min(remaining, len(prefetched))
        chunks.append(bytes(prefetched[:take]))
        del prefetched[:take]
        remaining -= take

    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise RuntimeError("Unexpected websocket EOF")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_until(sock: socket.socket, marker: bytes, max_bytes: int = 65536) -> bytes:
    data = bytearray()
    while marker not in data:
        if len(data) >= max_bytes:
            raise RuntimeError("Websocket handshake response exceeded max_bytes")
        chunk = sock.recv(1024)
        if not chunk:
            raise RuntimeError("Unexpected websocket EOF during handshake")
        data.extend(chunk)
    return bytes(data)


def _ws_send_frame(sock: socket.socket, opcode: int, payload: bytes) -> None:
    first = 0x80 | (opcode & 0x0F)
    payload_len = len(payload)

    mask = os.urandom(4)
    if payload_len < 126:
        header = bytes([first, 0x80 | payload_len])
    elif payload_len < 65536:
        header = bytes([first, 0x80 | 126]) + struct.pack("!H", payload_len)
    else:
        header = bytes([first, 0x80 | 127]) + struct.pack("!Q", payload_len)

    masked = bytes(payload[idx] ^ mask[idx % 4] for idx in range(payload_len))
    sock.sendall(header + mask + masked)


def _ws_open(url: str, timeout_seconds: float) -> tuple[socket.socket, bytes]:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in {"ws", "wss"}:
        raise ValueError(f"Unsupported websocket scheme: {parsed.scheme!r}")

    host = parsed.hostname
    if not host:
        raise ValueError("Websocket URL missing hostname")

    port = parsed.port
    if port is None:
        port = 443 if scheme == "wss" else 80

    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    sock = socket.create_connection((host, int(port)), timeout=timeout_seconds)
    sock.settimeout(timeout_seconds)
    if scheme == "wss":
        context = ssl.create_default_context()
        sock = context.wrap_socket(sock, server_hostname=host)
        sock.settimeout(timeout_seconds)

    key = base64.b64encode(os.urandom(16)).decode("ascii")
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    ).encode("ascii")
    sock.sendall(request)

    response = _recv_until(sock, b"\r\n\r\n")
    header_end = response.find(b"\r\n\r\n")
    if header_end < 0:
        raise RuntimeError("Websocket handshake response missing header terminator")
    header_bytes = response[: header_end + 4]
    prefetched = response[header_end + 4 :]
    header_text = header_bytes.decode("iso-8859-1")
    lines = header_text.split("\r\n")
    status_line = lines[0] if lines else ""
    if " 101 " not in status_line:
        raise RuntimeError(f"Websocket handshake failed: {status_line}")

    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        key_part, value_part = line.split(":", 1)
        headers[key_part.strip().lower()] = value_part.strip()

    expected_accept = base64.b64encode(
        hashlib.sha1((key + WS_GUID).encode("ascii")).digest()
    ).decode("ascii")
    actual_accept = headers.get("sec-websocket-accept")
    if actual_accept != expected_accept:
        raise RuntimeError("Websocket handshake returned invalid Sec-WebSocket-Accept")

    return sock, prefetched


def _ws_recv_text(
    sock: socket.socket,
    timeout_seconds: float,
    prefetched: bytes = b"",
) -> str:
    sock.settimeout(timeout_seconds)
    read_buffer = bytearray(prefetched)
    while True:
        header = _recv_exact(sock, 2, prefetched=read_buffer)
        first = header[0]
        second = header[1]

        opcode = first & 0x0F
        masked = (second & 0x80) != 0
        payload_len = second & 0x7F

        if payload_len == 126:
            payload_len = struct.unpack("!H", _recv_exact(sock, 2, prefetched=read_buffer))[0]
        elif payload_len == 127:
            payload_len = struct.unpack("!Q", _recv_exact(sock, 8, prefetched=read_buffer))[0]

        mask = _recv_exact(sock, 4, prefetched=read_buffer) if masked else b""
        payload = _recv_exact(sock, payload_len, prefetched=read_buffer)

        if masked:
            payload = bytes(value ^ mask[idx % 4] for idx, value in enumerate(payload))

        if opcode == 0x1:
            return payload.decode("utf-8")
        if opcode == 0x8:
            raise RuntimeError("Websocket closed before text payload was received")
        if opcode == 0x9:
            _ws_send_frame(sock, opcode=0xA, payload=payload)


def _http_json(
    *,
    session: requests.Session,
    url: str,
    timeout_seconds: float,
) -> tuple[int, Any]:
    response = session.get(url, timeout=timeout_seconds)
    try:
        payload = response.json()
    except Exception:
        payload = response.text
    return int(response.status_code), payload


def _record_check(
    *,
    name: str,
    fn,
    results: list[SmokeCheckResult],
) -> bool:
    start = time.perf_counter()
    try:
        detail = fn()
        elapsed = (time.perf_counter() - start) * 1000.0
        results.append(SmokeCheckResult(name=name, ok=True, elapsed_ms=elapsed, detail=str(detail)))
        return True
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000.0
        results.append(
            SmokeCheckResult(
                name=name, ok=False, elapsed_ms=elapsed, detail=f"{type(exc).__name__}: {exc}"
            )
        )
        return False


def _discover_run_and_replay(
    *,
    session: requests.Session,
    cfg: SmokeConfig,
) -> tuple[str | None, str | None]:
    if cfg.run_id and cfg.replay_id:
        return cfg.run_id, cfg.replay_id

    runs_url = _build_url(cfg.backend_http_base, "/api/runs", {"limit": 10})
    status_code, payload = _http_json(
        session=session, url=runs_url, timeout_seconds=cfg.timeout_seconds
    )
    if status_code != 200:
        raise RuntimeError(f"Failed to list runs: status={status_code}, payload={payload!r}")
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected /api/runs payload type")

    run_id = cfg.run_id
    if run_id is None:
        runs = payload.get("runs", [])
        if isinstance(runs, list) and len(runs) > 0 and isinstance(runs[0], dict):
            run_id_value = runs[0].get("run_id")
            if isinstance(run_id_value, str) and run_id_value.strip() != "":
                run_id = run_id_value.strip()

    if run_id is None:
        return None, None

    replay_id = cfg.replay_id
    if replay_id is None:
        replay_url = _build_url(
            cfg.backend_http_base,
            f"/api/runs/{quote(run_id, safe='')}/replays",
            {"limit": 1},
        )
        replay_status, replay_payload = _http_json(
            session=session,
            url=replay_url,
            timeout_seconds=cfg.timeout_seconds,
        )
        if replay_status != 200:
            raise RuntimeError(
                "Failed to list replays for "
                f"run={run_id}: status={replay_status}, payload={replay_payload!r}"
            )
        if not isinstance(replay_payload, dict):
            raise RuntimeError("Unexpected replay list payload type")

        rows = replay_payload.get("replays", [])
        if isinstance(rows, list) and len(rows) > 0 and isinstance(rows[0], dict):
            replay_value = rows[0].get("replay_id")
            if isinstance(replay_value, str) and replay_value.strip() != "":
                replay_id = replay_value.strip()

    return run_id, replay_id


def run_smoke(cfg: SmokeConfig) -> dict[str, Any]:
    session = requests.Session()
    results: list[SmokeCheckResult] = []

    _record_check(
        name="backend-health",
        results=results,
        fn=lambda: _check_backend_health(session=session, cfg=cfg),
    )
    _record_check(
        name="backend-runs-catalog",
        results=results,
        fn=lambda: _check_backend_runs(session=session, cfg=cfg),
    )

    cors_origin = _resolve_cors_origin(cfg)
    if cors_origin is None:
        results.append(
            SmokeCheckResult(
                name="backend-cors-simple",
                ok=True,
                elapsed_ms=0.0,
                detail="Skipped CORS simple check (no frontend-base/cors-origin provided).",
            )
        )
        results.append(
            SmokeCheckResult(
                name="backend-cors-preflight",
                ok=True,
                elapsed_ms=0.0,
                detail="Skipped CORS preflight check (no frontend-base/cors-origin provided).",
            )
        )
    else:
        _record_check(
            name="backend-cors-simple",
            results=results,
            fn=lambda: _check_backend_cors_simple(
                session=session,
                cfg=cfg,
                origin=cors_origin,
            ),
        )
        _record_check(
            name="backend-cors-preflight",
            results=results,
            fn=lambda: _check_backend_cors_preflight(
                session=session,
                cfg=cfg,
                origin=cors_origin,
            ),
        )

    run_id, replay_id = _discover_run_and_replay(session=session, cfg=cfg)
    if run_id is None or replay_id is None:
        if cfg.allow_empty_runs:
            results.append(
                SmokeCheckResult(
                    name="backend-replay-discovery",
                    ok=True,
                    elapsed_ms=0.0,
                    detail="Skipped replay checks because no run/replay was discoverable.",
                )
            )
        else:
            results.append(
                SmokeCheckResult(
                    name="backend-replay-discovery",
                    ok=False,
                    elapsed_ms=0.0,
                    detail=(
                        "No run/replay discovered. Provide --run-id/--replay-id "
                        "or deploy with populated runs."
                    ),
                )
            )
    else:
        _record_check(
            name="backend-replay-frames-http",
            results=results,
            fn=lambda: _check_backend_replay_frames_http(
                session=session,
                cfg=cfg,
                run_id=run_id,
                replay_id=replay_id,
            ),
        )
        _record_check(
            name="backend-replay-frames-ws",
            results=results,
            fn=lambda: _check_backend_replay_frames_ws(cfg=cfg, run_id=run_id, replay_id=replay_id),
        )

    if cfg.frontend_base is not None:
        for path in ("/", "/play", "/analytics"):
            _record_check(
                name=f"frontend-route-{path}",
                results=results,
                fn=lambda path=path: _check_frontend_route(session=session, cfg=cfg, path=path),
            )
    else:
        results.append(
            SmokeCheckResult(
                name="frontend-routes",
                ok=True,
                elapsed_ms=0.0,
                detail="Skipped frontend checks (no --frontend-base provided).",
            )
        )

    if cfg.skip_wandb:
        results.append(
            SmokeCheckResult(
                name="wandb-status",
                ok=True,
                elapsed_ms=0.0,
                detail="Skipped W&B status check (--skip-wandb).",
            )
        )
        results.append(
            SmokeCheckResult(
                name="wandb-latest-runs",
                ok=True,
                elapsed_ms=0.0,
                detail="Skipped W&B proxy check (--skip-wandb).",
            )
        )
        results.append(
            SmokeCheckResult(
                name="wandb-run-summary",
                ok=True,
                elapsed_ms=0.0,
                detail="Skipped W&B run summary check (--skip-wandb).",
            )
        )
        results.append(
            SmokeCheckResult(
                name="wandb-run-history",
                ok=True,
                elapsed_ms=0.0,
                detail="Skipped W&B run history check (--skip-wandb).",
            )
        )
        results.append(
            SmokeCheckResult(
                name="wandb-iteration-view",
                ok=True,
                elapsed_ms=0.0,
                detail="Skipped W&B iteration view check (--skip-wandb).",
            )
        )
        results.append(
            SmokeCheckResult(
                name="wandb-status-post",
                ok=True,
                elapsed_ms=0.0,
                detail="Skipped post-operation W&B status check (--skip-wandb).",
            )
        )
    else:
        _record_check(
            name="wandb-status",
            results=results,
            fn=lambda: _check_wandb_status(session=session, cfg=cfg),
        )
        wandb_run_id: str | None = None
        latest_start = time.perf_counter()
        try:
            latest_detail, wandb_run_id = _check_wandb_latest(session=session, cfg=cfg)
            results.append(
                SmokeCheckResult(
                    name="wandb-latest-runs",
                    ok=True,
                    elapsed_ms=(time.perf_counter() - latest_start) * 1000.0,
                    detail=latest_detail,
                )
            )
        except Exception as exc:
            results.append(
                SmokeCheckResult(
                    name="wandb-latest-runs",
                    ok=False,
                    elapsed_ms=(time.perf_counter() - latest_start) * 1000.0,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )

        if wandb_run_id is None:
            results.append(
                SmokeCheckResult(
                    name="wandb-run-summary",
                    ok=True,
                    elapsed_ms=0.0,
                    detail=(
                        "Skipped W&B run summary check because "
                        "`/api/wandb/runs/latest` returned no run_id."
                    ),
                )
            )
            results.append(
                SmokeCheckResult(
                    name="wandb-run-history",
                    ok=True,
                    elapsed_ms=0.0,
                    detail=(
                        "Skipped W&B run history check because "
                        "`/api/wandb/runs/latest` returned no run_id."
                    ),
                )
            )
            results.append(
                SmokeCheckResult(
                    name="wandb-iteration-view",
                    ok=True,
                    elapsed_ms=0.0,
                    detail=(
                        "Skipped W&B iteration view check because "
                        "`/api/wandb/runs/latest` returned no run_id."
                    ),
                )
            )
        else:
            _record_check(
                name="wandb-run-summary",
                results=results,
                fn=lambda: _check_wandb_run_summary(session=session, cfg=cfg, run_id=wandb_run_id),
            )
            _record_check(
                name="wandb-run-history",
                results=results,
                fn=lambda: _check_wandb_run_history(session=session, cfg=cfg, run_id=wandb_run_id),
            )
            _record_check(
                name="wandb-iteration-view",
                results=results,
                fn=lambda: _check_wandb_iteration_view(
                    session=session, cfg=cfg, run_id=wandb_run_id
                ),
            )

        _record_check(
            name="wandb-status-post",
            results=results,
            fn=lambda: _check_wandb_status(session=session, cfg=cfg),
        )
    pass_count = sum(1 for row in results if row.ok)
    fail_count = len(results) - pass_count
    config_payload = asdict(cfg)
    if cfg.output_path is not None:
        config_payload["output_path"] = str(cfg.output_path)

    report = {
        "generated_at": now_iso(),
        "config": config_payload,
        "summary": {
            "pass": fail_count == 0,
            "checks": len(results),
            "pass_count": pass_count,
            "fail_count": fail_count,
        },
        "checks": [asdict(row) for row in results],
    }

    if cfg.output_path is not None:
        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def _check_backend_health(*, session: requests.Session, cfg: SmokeConfig) -> str:
    url = _build_url(cfg.backend_http_base, "/health")
    status_code, payload = _http_json(session=session, url=url, timeout_seconds=cfg.timeout_seconds)
    if status_code != 200:
        raise RuntimeError(f"Expected 200, got {status_code}: {payload!r}")
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise RuntimeError(f"Unexpected /health payload: {payload!r}")
    return "Backend health is ok"


def _check_backend_runs(*, session: requests.Session, cfg: SmokeConfig) -> str:
    url = _build_url(cfg.backend_http_base, "/api/runs", {"limit": 10})
    status_code, payload = _http_json(session=session, url=url, timeout_seconds=cfg.timeout_seconds)
    if status_code != 200:
        raise RuntimeError(f"Expected 200, got {status_code}: {payload!r}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected /api/runs payload: {payload!r}")
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        raise RuntimeError(f"Unexpected /api/runs.runs payload: {runs!r}")
    return f"Runs endpoint returned {len(runs)} rows"


def _check_backend_cors_simple(
    *,
    session: requests.Session,
    cfg: SmokeConfig,
    origin: str,
) -> str:
    url = _build_url(cfg.backend_http_base, "/health")
    response = session.get(url, timeout=cfg.timeout_seconds, headers={"Origin": origin})
    if int(response.status_code) != 200:
        raise RuntimeError(f"Expected 200, got {response.status_code}")

    allow_origin = str(response.headers.get("Access-Control-Allow-Origin", "")).strip()
    if allow_origin not in {"*", origin}:
        raise RuntimeError(
            "CORS simple request does not allow frontend origin; "
            f"origin={origin!r}, allow_origin={allow_origin!r}"
        )
    return f"CORS simple request allowed origin={allow_origin}"


def _check_backend_cors_preflight(
    *,
    session: requests.Session,
    cfg: SmokeConfig,
    origin: str,
) -> str:
    url = _build_url(cfg.backend_http_base, "/api/runs")
    response = session.options(
        url,
        timeout=cfg.timeout_seconds,
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    if int(response.status_code) not in {200, 204}:
        raise RuntimeError(f"Expected preflight status 200/204, got {response.status_code}")

    allow_origin = str(response.headers.get("Access-Control-Allow-Origin", "")).strip()
    if allow_origin not in {"*", origin}:
        raise RuntimeError(
            "CORS preflight does not allow frontend origin; "
            f"origin={origin!r}, allow_origin={allow_origin!r}"
        )

    allow_methods_raw = str(response.headers.get("Access-Control-Allow-Methods", "")).strip()
    allow_methods_upper = allow_methods_raw.upper()
    if allow_methods_raw == "" or (
        "*" not in allow_methods_upper and "GET" not in allow_methods_upper
    ):
        raise RuntimeError(
            "CORS preflight missing GET in Access-Control-Allow-Methods: " f"{allow_methods_raw!r}"
        )

    allow_headers_raw = str(response.headers.get("Access-Control-Allow-Headers", "")).strip()
    allow_headers_upper = allow_headers_raw.upper()
    if allow_headers_raw != "" and (
        "*" not in allow_headers_upper and "CONTENT-TYPE" not in allow_headers_upper
    ):
        raise RuntimeError(
            "CORS preflight missing content-type in Access-Control-Allow-Headers: "
            f"{allow_headers_raw!r}"
        )

    return f"CORS preflight allowed origin={allow_origin} methods={allow_methods_raw}"


def _check_backend_replay_frames_http(
    *,
    session: requests.Session,
    cfg: SmokeConfig,
    run_id: str,
    replay_id: str,
) -> str:
    path = f"/api/runs/{quote(run_id, safe='')}/replays/{quote(replay_id, safe='')}/frames"
    url = _build_url(cfg.backend_http_base, path, {"offset": 0, "limit": 1})
    status_code, payload = _http_json(session=session, url=url, timeout_seconds=cfg.timeout_seconds)
    if status_code != 200:
        raise RuntimeError(f"Expected 200, got {status_code}: {payload!r}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected replay frames payload: {payload!r}")

    count = payload.get("count")
    if not isinstance(count, int):
        raise RuntimeError(f"Replay frames payload missing integer count: {payload!r}")
    return f"HTTP replay frames returned count={count}"


def _is_retryable_ws_exception(exc: Exception) -> bool:
    if isinstance(exc, socket.timeout | TimeoutError | ConnectionError | OSError):
        return True

    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "unexpected websocket eof",
            "websocket closed before text payload was received",
            "no close frame received",
            "connection reset",
            "timed out",
        )
    )


def _check_backend_replay_frames_ws_once(*, cfg: SmokeConfig, run_id: str, replay_id: str) -> str:
    ws_path = f"/ws/runs/{quote(run_id, safe='')}/replays/{quote(replay_id, safe='')}/frames"
    ws_url = _build_url(
        cfg.backend_ws_base,
        ws_path,
        {"offset": 0, "limit": 1, "batch_size": 1},
    )

    sock, prefetched = _ws_open(url=ws_url, timeout_seconds=cfg.timeout_seconds)
    try:
        next_prefetched = prefetched
        max_messages = 12
        for _ in range(max_messages):
            message_text = _ws_recv_text(
                sock=sock,
                timeout_seconds=cfg.timeout_seconds,
                prefetched=next_prefetched,
            )
            next_prefetched = b""

            try:
                payload = json.loads(message_text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid websocket JSON payload: {message_text!r}") from exc

            if not isinstance(payload, dict):
                raise RuntimeError(f"Unexpected websocket payload type: {payload!r}")

            message_type = str(payload.get("type", ""))
            if message_type == "error":
                raise RuntimeError(f"Websocket route returned error payload: {payload!r}")
            if message_type == "frames":
                frames = payload.get("frames")
                count = payload.get("count")
                if isinstance(frames, list) and len(frames) > 0:
                    return "WS replay frames returned message type=frames"
                if isinstance(count, int) and count > 0:
                    return "WS replay frames returned message type=frames"
                continue
            if message_type == "complete":
                count = payload.get("count")
                if isinstance(count, int) and count > 0:
                    return "WS replay frames returned message type=complete"
                raise RuntimeError(
                    "Websocket stream completed before any frame payload was received"
                )
            raise RuntimeError(f"Unexpected websocket message type: {message_type!r}")

        raise RuntimeError("Websocket stream did not produce frame payload within message budget")
    finally:
        try:
            _ws_send_frame(sock, opcode=0x8, payload=struct.pack("!H", 1000))
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass


def _check_backend_replay_frames_ws(*, cfg: SmokeConfig, run_id: str, replay_id: str) -> str:
    attempts_total = max(1, int(cfg.ws_check_attempts))
    last_error: Exception | None = None

    for attempt in range(1, attempts_total + 1):
        try:
            detail = _check_backend_replay_frames_ws_once(
                cfg=cfg, run_id=run_id, replay_id=replay_id
            )
            if attempt == 1:
                return detail
            return f"{detail} (attempt={attempt}/{attempts_total})"
        except Exception as exc:
            last_error = exc
            if attempt >= attempts_total or not _is_retryable_ws_exception(exc):
                raise
            time.sleep(min(0.75, 0.15 * float(attempt)))

    if last_error is not None:
        raise last_error
    raise RuntimeError("Websocket replay check failed without recorded error")


def _check_frontend_route(*, session: requests.Session, cfg: SmokeConfig, path: str) -> str:
    if cfg.frontend_base is None:
        raise ValueError("frontend_base is required for frontend route checks")

    url = _build_url(cfg.frontend_base, path)
    response = session.get(url, timeout=cfg.timeout_seconds)
    if int(response.status_code) != 200:
        raise RuntimeError(f"Expected 200, got {response.status_code}")

    body = response.text.lower()
    if "<html" not in body:
        raise RuntimeError("Expected HTML response body")
    return f"Frontend route {path} returned status=200"


def _wandb_scope_query(
    cfg: SmokeConfig,
    *,
    limit: int | None = None,
) -> dict[str, str | int]:
    query: dict[str, str | int] = {}
    if limit is not None:
        query["limit"] = int(limit)
    if cfg.wandb_entity is not None:
        query["entity"] = cfg.wandb_entity
    if cfg.wandb_project is not None:
        query["project"] = cfg.wandb_project
    return query


def _check_wandb_latest(*, session: requests.Session, cfg: SmokeConfig) -> tuple[str, str | None]:
    query = _wandb_scope_query(cfg, limit=1)

    url = _build_url(cfg.backend_http_base, "/api/wandb/runs/latest", query)
    status_code, payload = _http_json(session=session, url=url, timeout_seconds=cfg.timeout_seconds)
    if status_code != 200:
        raise RuntimeError(f"Expected 200, got {status_code}: {payload!r}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected W&B latest payload: {payload!r}")

    rows = payload.get("runs", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"Unexpected W&B latest runs payload: {rows!r}")

    selected_run_id: str | None = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        run_id_raw = row.get("run_id")
        if isinstance(run_id_raw, str) and run_id_raw.strip() != "":
            selected_run_id = run_id_raw.strip()
            break

    detail = f"W&B latest endpoint returned {len(rows)} rows"
    if selected_run_id is not None:
        detail += f" (first run_id={selected_run_id})"
    return detail, selected_run_id


def _check_wandb_run_summary(
    *,
    session: requests.Session,
    cfg: SmokeConfig,
    run_id: str,
) -> str:
    query = _wandb_scope_query(cfg)
    url = _build_url(
        cfg.backend_http_base,
        f"/api/wandb/runs/{quote(run_id, safe='')}/summary",
        query if len(query) > 0 else None,
    )
    status_code, payload = _http_json(session=session, url=url, timeout_seconds=cfg.timeout_seconds)
    if status_code != 200:
        raise RuntimeError(f"Expected 200, got {status_code}: {payload!r}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected W&B summary payload: {payload!r}")

    run_payload = payload.get("run")
    if not isinstance(run_payload, dict):
        raise RuntimeError(f"Unexpected W&B summary.run payload: {run_payload!r}")

    resolved_run_id = run_payload.get("run_id")
    if not isinstance(resolved_run_id, str) or resolved_run_id.strip() == "":
        raise RuntimeError(f"W&B summary payload missing run_id: {run_payload!r}")
    return f"W&B summary endpoint returned run_id={resolved_run_id.strip()}"


def _check_wandb_run_history(
    *,
    session: requests.Session,
    cfg: SmokeConfig,
    run_id: str,
) -> str:
    query = _wandb_scope_query(cfg)
    query["keys"] = "_step,window_id,env_steps_total,return_mean,profit_mean,survival_rate"
    query["max_points"] = 64
    url = _build_url(
        cfg.backend_http_base,
        f"/api/wandb/runs/{quote(run_id, safe='')}/history",
        query,
    )
    status_code, payload = _http_json(session=session, url=url, timeout_seconds=cfg.timeout_seconds)
    if status_code != 200:
        raise RuntimeError(f"Expected 200, got {status_code}: {payload!r}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected W&B history payload: {payload!r}")

    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"Unexpected W&B history rows payload: {rows!r}")
    return f"W&B history endpoint returned {len(rows)} rows"


def _check_wandb_iteration_view(
    *,
    session: requests.Session,
    cfg: SmokeConfig,
    run_id: str,
) -> str:
    query = _wandb_scope_query(cfg)
    query["keys"] = "_step,window_id,env_steps_total,return_mean,profit_mean,survival_rate"
    query["max_points"] = 64
    url = _build_url(
        cfg.backend_http_base,
        f"/api/wandb/runs/{quote(run_id, safe='')}/iteration-view",
        query,
    )
    status_code, payload = _http_json(session=session, url=url, timeout_seconds=cfg.timeout_seconds)
    if status_code != 200:
        raise RuntimeError(f"Expected 200, got {status_code}: {payload!r}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected W&B iteration-view payload: {payload!r}")

    history_payload = payload.get("history")
    if not isinstance(history_payload, dict):
        raise RuntimeError(f"Unexpected W&B iteration-view history payload: {history_payload!r}")
    rows = history_payload.get("rows", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"Unexpected W&B iteration-view history rows payload: {rows!r}")

    kpis = payload.get("kpis", {})
    if not isinstance(kpis, dict):
        raise RuntimeError(f"Unexpected W&B iteration-view kpis payload: {kpis!r}")
    return f"W&B iteration-view endpoint returned history_rows={len(rows)}"


def _parse_wandb_status_payload(
    payload: Any,
    *,
    require_clean_wandb_status: bool,
) -> tuple[bool, str | None, list[str]]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected W&B status payload: {payload!r}")

    available_raw = payload.get("available")
    if not isinstance(available_raw, bool):
        raise RuntimeError(f"W&B status missing boolean 'available': {payload!r}")

    reason_raw = payload.get("reason")
    reason: str | None = None
    if isinstance(reason_raw, str) and reason_raw.strip() != "":
        reason = reason_raw.strip()

    notes_raw = payload.get("notes", [])
    if notes_raw is None:
        notes_raw = []
    if not isinstance(notes_raw, list):
        raise RuntimeError(f"W&B status notes must be a list: {payload!r}")

    notes: list[str] = []
    for item in notes_raw:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text != "":
            notes.append(text)

    if not available_raw:
        reason_suffix = f": {reason}" if reason is not None else ""
        raise RuntimeError(f"W&B status reported unavailable{reason_suffix}")

    if require_clean_wandb_status and len(notes) > 0:
        raise RuntimeError(
            "W&B status reported actionable notes while strict mode is enabled: " + "; ".join(notes)
        )

    return available_raw, reason, notes


def _check_wandb_status(*, session: requests.Session, cfg: SmokeConfig) -> str:
    url = _build_url(cfg.backend_http_base, "/api/wandb/status")
    status_code, payload = _http_json(session=session, url=url, timeout_seconds=cfg.timeout_seconds)
    if status_code != 200:
        raise RuntimeError(f"Expected 200, got {status_code}: {payload!r}")

    _, _, notes = _parse_wandb_status_payload(
        payload,
        require_clean_wandb_status=cfg.require_clean_wandb_status,
    )

    mode = "strict" if cfg.require_clean_wandb_status else "non-strict"
    if len(notes) == 0:
        return f"W&B status endpoint reported available=true (notes=0, mode={mode})"
    return (
        "W&B status endpoint reported available=true "
        f"(notes={len(notes)}, mode={mode}): " + "; ".join(notes)
    )


def _parse_args() -> SmokeConfig:
    parser = argparse.ArgumentParser(description="Deployment smoke checks for M9 split hosting")
    parser.add_argument("--backend-http-base", type=str, required=True)
    parser.add_argument("--backend-ws-base", type=str, default=None)
    parser.add_argument("--frontend-base", type=str, default=None)
    parser.add_argument("--cors-origin", type=str, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=12.0)
    parser.add_argument("--ws-check-attempts", type=int, default=3)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--replay-id", type=str, default=None)
    parser.add_argument("--allow-empty-runs", action="store_true")
    parser.add_argument("--skip-wandb", action="store_true")
    parser.add_argument("--require-clean-wandb-status", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--output-path", type=Path, default=None)

    args = parser.parse_args()

    backend_http_base = normalize_base(args.backend_http_base)
    backend_ws_base_raw = args.backend_ws_base
    if backend_ws_base_raw is None or backend_ws_base_raw.strip() == "":
        backend_ws_base = normalize_base(derive_ws_base(backend_http_base))
    else:
        backend_ws_base = normalize_base(backend_ws_base_raw)

    frontend_base = None
    if isinstance(args.frontend_base, str) and args.frontend_base.strip() != "":
        frontend_base = normalize_base(args.frontend_base)

    cors_origin = None
    if isinstance(args.cors_origin, str) and args.cors_origin.strip() != "":
        cors_origin = normalize_origin(args.cors_origin)

    return SmokeConfig(
        backend_http_base=backend_http_base,
        backend_ws_base=backend_ws_base,
        frontend_base=frontend_base,
        cors_origin=cors_origin,
        timeout_seconds=float(args.timeout_seconds),
        ws_check_attempts=max(1, int(args.ws_check_attempts)),
        run_id=args.run_id,
        replay_id=args.replay_id,
        allow_empty_runs=bool(args.allow_empty_runs),
        skip_wandb=bool(args.skip_wandb),
        require_clean_wandb_status=bool(args.require_clean_wandb_status),
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        output_path=args.output_path,
    )


def main() -> int:
    cfg = _parse_args()
    report = run_smoke(cfg)

    for row in report["checks"]:
        status = "PASS" if bool(row["ok"]) else "FAIL"
        print(f"[{status}] {row['name']} ({row['elapsed_ms']:.1f} ms) - {row['detail']}")

    print(json.dumps(report["summary"], indent=2))
    return 0 if bool(report["summary"]["pass"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
