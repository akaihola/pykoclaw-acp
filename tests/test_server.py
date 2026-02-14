"""Tests for the ACP server JSON-RPC dispatch logic."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

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
async def test_session_prompt_streams_via_dispatch(server: AcpServer) -> None:
    written = _collect_writes(server)

    await server._dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {}}
    )
    session_id = written[0]["result"]["sessionId"]

    mock_dispatch = AsyncMock()
    mock_dispatch.return_value = None

    with patch("pykoclaw_acp.server.dispatch_to_agent", mock_dispatch):
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

    # ACP requires an empty acknowledgment before streaming begins
    ack = written[1]
    assert ack["id"] == 2
    assert ack["result"] == {}

    mock_dispatch.assert_awaited_once()
    call_kwargs = mock_dispatch.call_args.kwargs
    assert call_kwargs["prompt"] == "hello"
    assert call_kwargs["channel_prefix"] == "acp"
    assert call_kwargs["channel_id"] == session_id[:8]
    assert call_kwargs["on_text"] is not None


@pytest.mark.asyncio
async def test_response_without_method_is_ignored(server: AcpServer) -> None:
    written = _collect_writes(server)
    await server._dispatch({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})
    assert len(written) == 0
