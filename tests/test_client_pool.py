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
async def test_disconnect_removes_entry(tmp_path, tmp_db) -> None:  # noqa: ANN001
    """_disconnect must remove the entry and clean up without errors."""
    from pykoclaw_acp.client_pool import _Entry

    class FakeSDKClient:
        """Minimal stub with the attributes _kill_client accesses."""

        _query = None
        _transport = None

    entry = _Entry(
        client=FakeSDKClient(),  # type: ignore[arg-type]
        conversation_name="acp-remove1",
    )
    pool = _make_pool(tmp_path, tmp_db)
    pool._entries["remove-session"] = entry

    await pool._disconnect("remove-session")
    assert "remove-session" not in pool._entries


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


@pytest.mark.asyncio
async def test_sweep_eviction_does_not_cancel_other_tasks(tmp_path, tmp_db) -> None:  # noqa: ANN001
    """Regression: _sweep_loop evicting idle clients must not leak
    CancelledError into other asyncio tasks running in the same loop.

    In production, ClaudeSDKClient.disconnect() uses anyio cancel scopes
    that inject CancelledError into the host task.  _kill_client() avoids
    this by terminating the subprocess directly without calling disconnect().
    """
    from pykoclaw_acp.client_pool import IDLE_TIMEOUT_S, _Entry

    class FakeSDKClient:
        _query = None
        _transport = None

    pool = _make_pool(tmp_path, tmp_db)

    entry = _Entry(
        client=FakeSDKClient(),  # type: ignore[arg-type]
        conversation_name="acp-stale123",
    )
    import time
    entry.last_used = time.monotonic() - IDLE_TIMEOUT_S - 100
    pool._entries["stale-session"] = entry

    survivor_cancelled = False

    async def survivor_task() -> None:
        nonlocal survivor_cancelled
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            survivor_cancelled = True
            raise

    survivor = asyncio.create_task(survivor_task())

    import pykoclaw_acp.client_pool as pool_mod
    orig_sweep = pool_mod.SWEEP_INTERVAL_S
    pool_mod.SWEEP_INTERVAL_S = 0.05
    try:
        await pool.start()

        for _ in range(50):
            await asyncio.sleep(0.02)
            if "stale-session" not in pool._entries:
                break

        assert "stale-session" not in pool._entries, "sweep didn't evict stale entry"
        assert not survivor_cancelled, (
            "CancelledError from disconnect leaked to unrelated task"
        )
    finally:
        pool_mod.SWEEP_INTERVAL_S = orig_sweep
        survivor.cancel()
        try:
            await survivor
        except asyncio.CancelledError:
            pass
        await pool.close()


@pytest.mark.asyncio
async def test_sweep_evicts_multiple_stale_entries(tmp_path, tmp_db) -> None:  # noqa: ANN001
    """Multiple stale entries should all be evicted in a single sweep pass."""
    from pykoclaw_acp.client_pool import IDLE_TIMEOUT_S, _Entry

    class FakeSDKClient:
        _query = None
        _transport = None

    pool = _make_pool(tmp_path, tmp_db)

    import time
    for i in range(3):
        entry = _Entry(
            client=FakeSDKClient(),  # type: ignore[arg-type]
            conversation_name=f"acp-stale{i}",
        )
        entry.last_used = time.monotonic() - IDLE_TIMEOUT_S - 100
        pool._entries[f"stale-{i}"] = entry

    import pykoclaw_acp.client_pool as pool_mod
    orig_sweep = pool_mod.SWEEP_INTERVAL_S
    pool_mod.SWEEP_INTERVAL_S = 0.05
    try:
        await pool.start()

        for _ in range(50):
            await asyncio.sleep(0.02)
            if not pool._entries:
                break

        assert len(pool._entries) == 0, (
            f"Expected all stale entries evicted, but {len(pool._entries)} remain"
        )
    finally:
        pool_mod.SWEEP_INTERVAL_S = orig_sweep
        await pool.close()


@pytest.mark.asyncio
async def test_kill_client_terminates_subprocess(tmp_path) -> None:  # noqa: ANN001
    """_kill_client must terminate the subprocess and null out references."""
    from unittest.mock import AsyncMock, MagicMock, PropertyMock

    from pykoclaw_acp.client_pool import _kill_client

    # Build a fake client with the structure _kill_client expects
    proc = AsyncMock()
    proc.returncode = None  # process is "running"
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = AsyncMock()

    transport = MagicMock()
    transport._process = proc

    query = MagicMock()
    query.transport = transport
    query._closed = False

    client = MagicMock()
    client._query = query
    client._transport = transport

    await _kill_client(client, "test-label")

    # Process should have been terminated
    proc.terminate.assert_called_once()
    proc.wait.assert_awaited_once()

    # References should be nulled
    assert client._query is None
    assert client._transport is None

    # Query should be marked closed
    assert query._closed is True
