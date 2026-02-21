"""Tests for session/load (resume across restarts) feature."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from pykoclaw_acp.protocol import JsonRpcError
from pykoclaw_acp.server import AcpServer
from pykoclaw_acp.worker_protocol import WorkerConfig, decode_config, encode


# -- Fixtures ----------------------------------------------------------------


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


# -- Initialize: loadSession capability -------------------------------------


@pytest.mark.asyncio
async def test_initialize_advertises_load_session(server: AcpServer) -> None:
    written = _collect_writes(server)
    await server._dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    )
    assert len(written) == 1
    caps = written[0]["result"]["agentCapabilities"]
    assert caps["loadSession"] is True


# -- session/load: success (known conversation) ------------------------------


@pytest.mark.asyncio
async def test_session_load_known_conversation(
    server: AcpServer, tmp_db: sqlite3.Connection
) -> None:
    # Pre-populate DB with a prior conversation
    acp_session_id = "abcd1234-5678-9abc-def0-111111111111"
    conversation_name = f"acp-{acp_session_id[:8]}"
    claude_session_id = "claude-sess-xyz-999"
    tmp_db.execute(
        "INSERT INTO conversations (name, session_id, cwd, created_at) "
        "VALUES (?, ?, ?, ?)",
        (conversation_name, claude_session_id, "/some/dir", "2025-01-01T00:00:00Z"),
    )
    tmp_db.commit()

    written = _collect_writes(server)
    await server._dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/load",
            "params": {"sessionId": acp_session_id, "cwd": "/tmp/project"},
        }
    )
    assert len(written) == 1
    assert "result" in written[0]
    assert written[0]["id"] == 1

    # Session should be registered
    assert acp_session_id in server._sessions
    session = server._sessions[acp_session_id]
    assert session["resume_session_id"] == claude_session_id
    assert session["cwd"] == "/tmp/project"


# -- session/load: miss (unknown conversation) --------------------------------


@pytest.mark.asyncio
async def test_session_load_unknown_conversation(server: AcpServer) -> None:
    written = _collect_writes(server)
    await server._dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/load",
            "params": {"sessionId": "unknown-id-does-not-exist-1234"},
        }
    )
    assert len(written) == 1
    assert written[0]["error"]["code"] == JsonRpcError.SESSION_ERROR
    assert "No prior session found" in written[0]["error"]["message"]


# -- session/load: missing sessionId param ------------------------------------


@pytest.mark.asyncio
async def test_session_load_missing_session_id(server: AcpServer) -> None:
    written = _collect_writes(server)
    await server._dispatch(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/load",
            "params": {},
        }
    )
    assert len(written) == 1
    assert written[0]["error"]["code"] == JsonRpcError.INVALID_PARAMS
    assert "sessionId is required" in written[0]["error"]["message"]


# -- session/load: conversation exists but has no session_id ------------------


@pytest.mark.asyncio
async def test_session_load_conversation_without_session_id(
    server: AcpServer, tmp_db: sqlite3.Connection
) -> None:
    acp_session_id = "beef0000-1111-2222-3333-444444444444"
    conversation_name = f"acp-{acp_session_id[:8]}"
    # Insert conversation with empty session_id
    tmp_db.execute(
        "INSERT INTO conversations (name, session_id, cwd, created_at) "
        "VALUES (?, ?, ?, ?)",
        (conversation_name, "", "/some/dir", "2025-01-01T00:00:00Z"),
    )
    tmp_db.commit()

    written = _collect_writes(server)
    await server._dispatch(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "session/load",
            "params": {"sessionId": acp_session_id},
        }
    )
    assert len(written) == 1
    assert written[0]["error"]["code"] == JsonRpcError.SESSION_ERROR


# -- session/load → session/prompt: resume_session_id passed to pool ----------


@pytest.mark.asyncio
async def test_session_load_then_prompt_passes_resume_id(
    server: AcpServer, tmp_db: sqlite3.Connection
) -> None:
    acp_session_id = "face1234-aaaa-bbbb-cccc-dddddddddddd"
    conversation_name = f"acp-{acp_session_id[:8]}"
    claude_session_id = "claude-resume-target-42"
    tmp_db.execute(
        "INSERT INTO conversations (name, session_id, cwd, created_at) "
        "VALUES (?, ?, ?, ?)",
        (conversation_name, claude_session_id, "/dir", "2025-01-01T00:00:00Z"),
    )
    tmp_db.commit()

    written = _collect_writes(server)

    # Load session
    await server._dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/load",
            "params": {"sessionId": acp_session_id, "cwd": "/tmp"},
        }
    )
    assert "result" in written[0]

    # Mock pool.send to capture the resume_session_id
    server._pool.send = AsyncMock(return_value=None)

    # Send prompt
    await server._dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/prompt",
            "params": {
                "sessionId": acp_session_id,
                "prompt": [{"type": "text", "text": "hello"}],
            },
        }
    )

    server._pool.send.assert_awaited_once()
    call_kwargs = server._pool.send.call_args[1]
    assert call_kwargs["resume_session_id"] == claude_session_id


# -- resume_session_id consumed after first prompt ----------------------------


@pytest.mark.asyncio
async def test_resume_session_id_consumed_after_first_prompt(
    server: AcpServer, tmp_db: sqlite3.Connection
) -> None:
    acp_session_id = "aaaa1111-2222-3333-4444-555555555555"
    conversation_name = f"acp-{acp_session_id[:8]}"
    claude_session_id = "claude-one-shot"
    tmp_db.execute(
        "INSERT INTO conversations (name, session_id, cwd, created_at) "
        "VALUES (?, ?, ?, ?)",
        (conversation_name, claude_session_id, "/dir", "2025-01-01T00:00:00Z"),
    )
    tmp_db.commit()

    # Load session
    await server._dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/load",
            "params": {"sessionId": acp_session_id},
        }
    )

    server._pool.send = AsyncMock(return_value=None)

    # First prompt — should have resume_session_id
    await server._dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/prompt",
            "params": {
                "sessionId": acp_session_id,
                "prompt": [{"type": "text", "text": "first"}],
            },
        }
    )
    first_kwargs = server._pool.send.call_args[1]
    assert first_kwargs["resume_session_id"] == claude_session_id

    server._pool.send.reset_mock()

    # Second prompt — resume_session_id should be None (consumed)
    await server._dispatch(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {
                "sessionId": acp_session_id,
                "prompt": [{"type": "text", "text": "second"}],
            },
        }
    )
    second_kwargs = server._pool.send.call_args[1]
    assert second_kwargs["resume_session_id"] is None


# -- session/new does NOT set resume_session_id --------------------------------


@pytest.mark.asyncio
async def test_session_new_has_no_resume_id(server: AcpServer) -> None:
    written = _collect_writes(server)

    await server._dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"cwd": "/tmp"}}
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

    call_kwargs = server._pool.send.call_args[1]
    assert call_kwargs["resume_session_id"] is None


# -- WorkerConfig: resume_session_id round-trip --------------------------------


class TestWorkerConfigResume:
    def test_config_with_resume_session_id(self) -> None:
        config = WorkerConfig(
            cwd="/tmp/work",
            model="claude-sonnet-4-20250514",
            conversation_name="acp-abc123",
            db_path="/data/pykoclaw.db",
            resume_session_id="sess-to-resume",
        )
        decoded = decode_config(encode(config))
        assert decoded == config
        assert decoded.resume_session_id == "sess-to-resume"

    def test_config_without_resume_session_id(self) -> None:
        config = WorkerConfig(
            cwd="/tmp/work",
            model="claude-sonnet-4-20250514",
            conversation_name="acp-abc123",
            db_path="/data/pykoclaw.db",
        )
        decoded = decode_config(encode(config))
        assert decoded == config
        assert decoded.resume_session_id is None

    def test_config_resume_none_round_trip(self) -> None:
        config = WorkerConfig(resume_session_id=None)
        decoded = decode_config(encode(config))
        assert decoded.resume_session_id is None
