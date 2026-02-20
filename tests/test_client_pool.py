"""Tests for ClientPool._query() — result text fallback for empty streams."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

import pytest

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock


# Minimal stub that replaces a real ClaudeSDKClient inside _Entry.
@dataclass
class FakeClient:
    """Stub ClaudeSDKClient whose receive_response yields canned messages."""

    messages: list[Any] = field(default_factory=list)
    _queried: str | None = None

    async def query(self, prompt: str) -> None:
        self._queried = prompt

    async def receive_response(self):  # noqa: ANN201
        for msg in self.messages:
            yield msg


def _make_entry(client: FakeClient, conversation_name: str = "acp-test1234") -> Any:
    """Build a _Entry without importing the private dataclass."""
    from pykoclaw_acp.client_pool import _Entry

    return _Entry(client=client, conversation_name=conversation_name)


def _make_pool(tmp_path, tmp_db) -> Any:  # noqa: ANN001
    """Build a ClientPool backed by tmp fixtures."""
    from pykoclaw_acp.client_pool import ClientPool

    return ClientPool(db=tmp_db, data_dir=tmp_path)


@pytest.fixture()
def tmp_db():  # noqa: ANN201
    import sqlite3

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


# ---------- the actual tests ----------


@pytest.mark.asyncio
async def test_query_streams_text_blocks(tmp_path, tmp_db) -> None:  # noqa: ANN001
    """Baseline: TextBlock content is forwarded via on_text."""
    client = FakeClient(
        messages=[
            AssistantMessage(
                content=[TextBlock(text="Hello from stream")],
                model="test",
            ),
            ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=80,
                is_error=False,
                num_turns=1,
                session_id="sess-ok",
                result="Hello from stream",
            ),
        ]
    )
    entry = _make_entry(client)
    pool = _make_pool(tmp_path, tmp_db)

    chunks: list[str] = []

    async def on_text(text: str) -> None:
        chunks.append(text)

    sid = await pool._query(entry, "hi", on_text)

    assert chunks == ["Hello from stream"]
    assert sid == "sess-ok"


@pytest.mark.asyncio
async def test_query_result_text_fallback_when_no_text_blocks(tmp_path, tmp_db) -> None:  # noqa: ANN001
    """When no TextBlock messages are streamed, ResultMessage.result must
    be forwarded via on_text so the Mitto user sees the reply."""
    client = FakeClient(
        messages=[
            ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=80,
                is_error=False,
                num_turns=1,
                session_id="sess-fb",
                result="Fallback reply text",
            ),
        ]
    )
    entry = _make_entry(client)
    pool = _make_pool(tmp_path, tmp_db)

    chunks: list[str] = []

    async def on_text(text: str) -> None:
        chunks.append(text)

    sid = await pool._query(entry, "hi", on_text)

    assert sid == "sess-fb"
    assert chunks == ["Fallback reply text"]


@pytest.mark.asyncio
async def test_query_no_duplication_when_streamed_and_result(tmp_path, tmp_db) -> None:  # noqa: ANN001
    """When TextBlock IS streamed, ResultMessage.result must NOT cause
    duplicate delivery."""
    client = FakeClient(
        messages=[
            AssistantMessage(
                content=[TextBlock(text="Streamed text")],
                model="test",
            ),
            ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=80,
                is_error=False,
                num_turns=1,
                session_id="sess-dup",
                result="Streamed text",
            ),
        ]
    )
    entry = _make_entry(client)
    pool = _make_pool(tmp_path, tmp_db)

    chunks: list[str] = []

    async def on_text(text: str) -> None:
        chunks.append(text)

    sid = await pool._query(entry, "hi", on_text)

    assert sid == "sess-dup"
    assert chunks == ["Streamed text"]  # exactly once, no duplication


@pytest.mark.asyncio
async def test_disconnect_shields_cancelled_error(tmp_path, tmp_db) -> None:  # noqa: ANN001
    """_disconnect must not leak CancelledError into the caller.

    Regression test: ClaudeSDKClient.disconnect() calls anyio's
    cancel_scope.cancel() internally, which can propagate a CancelledError
    into the asyncio task that called it.  _disconnect() must shield this
    so that callers (especially _sweep_loop and the main server loop) are
    not disrupted.
    """
    from pykoclaw_acp.client_pool import _Entry

    class CancellingClient:
        """Simulates the real SDK: disconnect raises CancelledError."""

        async def disconnect(self) -> None:
            raise asyncio.CancelledError("cancel scope leak")

    entry = _Entry(
        client=CancellingClient(),  # type: ignore[arg-type]
        conversation_name="acp-cancel1",
    )
    pool = _make_pool(tmp_path, tmp_db)
    pool._entries["cancel-session"] = entry

    # Must NOT raise CancelledError
    await pool._disconnect("cancel-session")
    assert "cancel-session" not in pool._entries


@pytest.mark.asyncio
async def test_query_no_on_text_callback(tmp_path, tmp_db) -> None:  # noqa: ANN001
    """When on_text is None, _query still completes without error."""
    client = FakeClient(
        messages=[
            ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=80,
                is_error=False,
                num_turns=1,
                session_id="sess-none",
                result="Some text",
            ),
        ]
    )
    entry = _make_entry(client)
    pool = _make_pool(tmp_path, tmp_db)

    sid = await pool._query(entry, "hi", None)
    assert sid == "sess-none"
