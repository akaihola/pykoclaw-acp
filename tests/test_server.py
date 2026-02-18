"""Tests for the ACP server JSON-RPC dispatch logic."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pykoclaw_acp.protocol import JsonRpcError
from pykoclaw_acp.server import AcpServer


@pytest.fixture()
def tmp_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(
        "CREATE TABLE IF NOT EXISTS conversations ("
        "    name TEXT PRIMARY KEY,"
        "    session_id TEXT,"
        "    cwd TEXT,"
        "    created_at TEXT NOT NULL"
        ");"
    )
    return db


@pytest.fixture()
def server(tmp_db: sqlite3.Connection, tmp_path: Path) -> AcpServer:
    return AcpServer(db=tmp_db, data_dir=tmp_path)


def _collect_writes(server: AcpServer) -> list[dict[str, Any]]:
    """Capture all JSON-RPC messages written by the server."""
    written: list[dict[str, Any]] = []
    original_write = server._write

    def capture(message: str) -> None:
        written.append(json.loads(message))
        original_write(message)

    server._write = capture  # type: ignore[assignment]
    return written


@pytest.mark.asyncio
async def test_initialize(server: AcpServer) -> None:
    written = _collect_writes(server)
    await server._dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    )
    assert len(written) == 1
    result = written[0]["result"]
    assert result["protocolVersion"] == 1
    assert result["agentInfo"]["name"] == "pykoclaw"


@pytest.mark.asyncio
async def test_session_new(server: AcpServer) -> None:
    written = _collect_writes(server)
    await server._dispatch(
        {"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {"cwd": "/tmp"}}
    )
    assert len(written) == 1
    session_id = written[0]["result"]["sessionId"]
    assert isinstance(session_id, str)
    assert len(session_id) == 36


@pytest.mark.asyncio
async def test_method_not_found(server: AcpServer) -> None:
    written = _collect_writes(server)
    await server._dispatch(
        {"jsonrpc": "2.0", "id": 3, "method": "nonexistent", "params": {}}
    )
    assert len(written) == 1
    assert written[0]["error"]["code"] == JsonRpcError.METHOD_NOT_FOUND


@pytest.mark.asyncio
async def test_session_prompt_invalid_session(server: AcpServer) -> None:
    written = _collect_writes(server)
    await server._dispatch(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "session/prompt",
            "params": {
                "sessionId": "nonexistent",
                "prompt": [{"type": "text", "text": "hi"}],
            },
        }
    )
    assert len(written) == 1
    assert written[0]["error"]["code"] == JsonRpcError.INVALID_SESSION


@pytest.mark.asyncio
async def test_session_prompt_empty_prompt(server: AcpServer) -> None:
    written = _collect_writes(server)

    await server._dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {}}
    )
    session_id = written[0]["result"]["sessionId"]

    await server._dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/prompt",
            "params": {"sessionId": session_id, "prompt": []},
        }
    )
    assert written[-1]["error"]["code"] == JsonRpcError.INVALID_PARAMS


@pytest.mark.asyncio
async def test_session_prompt_calls_pool_send(server: AcpServer) -> None:
    written = _collect_writes(server)

    await server._dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {}}
    )
    session_id = written[0]["result"]["sessionId"]

    server._pool.send = AsyncMock(return_value=None)

    await server._dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": "hello"}],
            },
        }
    )

    server._pool.send.assert_awaited_once()
    call_args = server._pool.send.call_args
    assert call_args[0][0] == session_id  # first positional: session_id
    assert call_args[0][1] == "hello"  # second positional: content
    assert call_args[1]["on_text"] is not None  # keyword: on_text callback

    # Final message is the stop response
    stop = written[-1]
    assert stop["id"] == 2
    assert stop["result"]["stopReason"] == "end_turn"


@pytest.mark.asyncio
async def test_response_without_method_is_ignored(server: AcpServer) -> None:
    written = _collect_writes(server)
    await server._dispatch({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})
    assert len(written) == 0


@pytest.mark.asyncio
async def test_session_prompt_pool_error_sends_notification(
    server: AcpServer,
) -> None:
    written = _collect_writes(server)

    await server._dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {}}
    )
    session_id = written[0]["result"]["sessionId"]

    server._pool.send = AsyncMock(side_effect=RuntimeError("API timeout"))

    await server._dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": "hello"}],
            },
        }
    )

    # Error notification should be sent
    error_notif = written[1]
    assert error_notif["method"] == "session/update"
    assert error_notif["params"]["sessionId"] == session_id
    assert error_notif["params"]["update"]["sessionUpdate"] == "error"
    assert "error" in error_notif["params"]["update"]

    # Stop response still sent even after error
    stop = written[-1]
    assert stop["id"] == 2
    assert stop["result"]["stopReason"] == "end_turn"


@pytest.mark.asyncio
async def test_session_prompt_pool_error_server_survives(
    server: AcpServer,
) -> None:
    written = _collect_writes(server)

    await server._dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {}}
    )
    session_id = written[0]["result"]["sessionId"]

    server._pool.send = AsyncMock(side_effect=RuntimeError("boom"))

    await server._dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/prompt",
            "params": {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": "hello"}],
            },
        }
    )

    # Server should still work after the error
    await server._dispatch(
        {"jsonrpc": "2.0", "id": 3, "method": "initialize", "params": {}}
    )

    init_response = written[-1]
    assert init_response["id"] == 3
    assert "result" in init_response
    assert init_response["result"]["protocolVersion"] == 1


@pytest.mark.asyncio
async def test_main_loop_continues_after_dispatch_error(
    server: AcpServer,
) -> None:
    written = _collect_writes(server)

    original_dispatch = server._dispatch
    call_count = 0

    async def mock_dispatch_with_error(msg: dict[str, Any]) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient error")
        await original_dispatch(msg)

    server._dispatch = mock_dispatch_with_error  # type: ignore[assignment]

    try:
        await server._dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    except RuntimeError:
        pass

    await server._dispatch(
        {"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {}}
    )

    init_response = written[-1]
    assert init_response["id"] == 2
    assert "result" in init_response


@pytest.mark.asyncio
async def test_pool_cancelled_error_not_swallowed(
    server: AcpServer,
) -> None:
    """CancelledError from pool.send must propagate — not be caught by the
    generic Exception handler — so that asyncio shutdown can cancel tasks."""
    written = _collect_writes(server)

    await server._dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {}}
    )
    session_id = written[0]["result"]["sessionId"]

    server._pool.send = AsyncMock(side_effect=asyncio.CancelledError())

    # The bare `except Exception` in _handle_session_prompt does NOT catch
    # CancelledError (which inherits from BaseException), so it should
    # propagate out.
    with pytest.raises(asyncio.CancelledError):
        await server._dispatch(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "hello"}],
                },
            }
        )
