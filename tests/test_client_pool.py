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


@pytest.mark.asyncio
async def test_sweep_eviction_does_not_cancel_other_tasks(tmp_path, tmp_db) -> None:  # noqa: ANN001
    """Regression: _sweep_loop evicting idle clients must not leak
    CancelledError into other asyncio tasks running in the same loop.

    In production, ClaudeSDKClient.disconnect() uses anyio cancel scopes
    that can propagate CancelledError into the asyncio event loop.  When
    _sweep_loop calls _disconnect() and that raises CancelledError, it
    must be fully contained — not reach the main server loop or any other
    concurrent task.
    """
    from pykoclaw_acp.client_pool import IDLE_TIMEOUT_S, _Entry

    class CancelLeakingClient:
        """Simulates anyio cancel scope leak: disconnect raises CancelledError."""

        async def disconnect(self) -> None:
            raise asyncio.CancelledError("anyio cancel scope leak")

    pool = _make_pool(tmp_path, tmp_db)

    # Plant a stale entry that the sweep will evict
    entry = _Entry(
        client=CancelLeakingClient(),  # type: ignore[arg-type]
        conversation_name="acp-stale123",
    )
    # Make it look very stale (older than IDLE_TIMEOUT_S)
    import time
    entry.last_used = time.monotonic() - IDLE_TIMEOUT_S - 100

    pool._entries["stale-session"] = entry

    # A concurrent task that should survive the eviction
    survivor_cancelled = False

    async def survivor_task() -> None:
        nonlocal survivor_cancelled
        try:
            # Wait long enough for the sweep to run and evict the stale client
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            survivor_cancelled = True
            raise

    survivor = asyncio.create_task(survivor_task())

    # Temporarily lower the sweep interval so it fires quickly
    import pykoclaw_acp.client_pool as pool_mod
    orig_sweep = pool_mod.SWEEP_INTERVAL_S
    pool_mod.SWEEP_INTERVAL_S = 0.05  # 50ms
    try:
        await pool.start()

        # Give the sweep loop time to detect and evict the stale entry
        for _ in range(50):
            await asyncio.sleep(0.02)
            if "stale-session" not in pool._entries:
                break

        # Stale entry should have been evicted
        assert "stale-session" not in pool._entries, "sweep didn't evict stale entry"

        # The survivor task must NOT have been cancelled
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
async def test_sweep_loop_survives_disconnect_errors(tmp_path, tmp_db) -> None:  # noqa: ANN001
    """_sweep_loop must keep running even when disconnect raises exceptions.

    Regression: if _disconnect leaks an exception (CancelledError or other),
    the sweep loop must catch it and continue — not crash and leave all
    future idle clients unreaped.
    """
    from pykoclaw_acp.client_pool import IDLE_TIMEOUT_S, _Entry

    disconnect_call_count = 0

    class FailingClient:
        """disconnect() always raises."""

        async def disconnect(self) -> None:
            nonlocal disconnect_call_count
            disconnect_call_count += 1
            raise RuntimeError("subprocess died")

    pool = _make_pool(tmp_path, tmp_db)

    import time

    # Plant two stale entries — both should be evicted even if the first
    # disconnect raises an error.
    for i in range(2):
        entry = _Entry(
            client=FailingClient(),  # type: ignore[arg-type]
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
        assert disconnect_call_count == 2
    finally:
        pool_mod.SWEEP_INTERVAL_S = orig_sweep
        await pool.close()


@pytest.mark.asyncio
async def test_disconnect_isolates_anyio_cancel_scope(tmp_path, tmp_db) -> None:  # noqa: ANN001
    """_disconnect runs client.disconnect() in a separate Task so that anyio
    cancel scope effects are fully isolated from the calling task.

    This tests the specific fix: even if disconnect() cancels the *current*
    asyncio task (which is what anyio cancel scopes can do), _disconnect's
    caller must not be affected.
    """
    from pykoclaw_acp.client_pool import _Entry

    class AnyioCancelScopeClient:
        """Simulates the real SDK: disconnect cancels the current task via
        anyio's cancel scope, not just by raising CancelledError."""

        async def disconnect(self) -> None:
            # This simulates what happens when anyio's
            # TaskGroup.cancel_scope.cancel() runs — it cancels the
            # *current* asyncio task.
            current_task = asyncio.current_task()
            if current_task:
                current_task.cancel()
            # Yield control so the cancellation takes effect
            await asyncio.sleep(0)

    entry = _Entry(
        client=AnyioCancelScopeClient(),  # type: ignore[arg-type]
        conversation_name="acp-cancel-scope",
    )
    pool = _make_pool(tmp_path, tmp_db)
    pool._entries["cancel-scope-session"] = entry

    # _disconnect must NOT propagate the task cancellation to its caller
    await pool._disconnect("cancel-scope-session")
    assert "cancel-scope-session" not in pool._entries
