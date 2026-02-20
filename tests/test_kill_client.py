"""Tests for _kill_client — the safe disconnect that avoids anyio cancel scope leaks.

These tests use the REAL ClaudeSDKClient to verify that _kill_client does not
inject CancelledError into the calling task or any concurrent tasks.  This is
the exact bug that ClaudeSDKClient.disconnect() triggers due to anyio's cancel
scope tracking the host task.

Requires: ``claude`` CLI on PATH (skipped automatically if unavailable).
"""

from __future__ import annotations

import asyncio
import shutil

import pytest

_has_claude = shutil.which("claude") is not None
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not _has_claude, reason="claude CLI not on PATH"),
]


def _make_options():  # noqa: ANN202
    from claude_agent_sdk import ClaudeAgentOptions

    return ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        model="claude-sonnet-4-20250514",
        setting_sources=["project"],
        env={"CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1"},
    )


async def test_kill_client_no_cancelled_error_leak() -> None:
    """_kill_client must not inject CancelledError into the calling task."""
    from claude_agent_sdk import ClaudeSDKClient

    from pykoclaw_acp.client_pool import _kill_client

    client = ClaudeSDKClient(_make_options())
    await client.connect()

    # Kill it — must not raise CancelledError
    await _kill_client(client, "test")

    # Verify the caller can still await things
    await asyncio.sleep(0.05)


async def test_kill_client_does_not_cancel_concurrent_tasks() -> None:
    """Concurrent asyncio tasks must not be affected by _kill_client."""
    from claude_agent_sdk import ClaudeSDKClient

    from pykoclaw_acp.client_pool import _kill_client

    client = ClaudeSDKClient(_make_options())
    await client.connect()

    survivor_cancelled = False

    async def survivor() -> None:
        nonlocal survivor_cancelled
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            survivor_cancelled = True
            raise

    task = asyncio.create_task(survivor())

    await _kill_client(client, "test")
    await asyncio.sleep(0.1)

    assert not survivor_cancelled, "CancelledError leaked to concurrent task"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_sdk_disconnect_DOES_leak_cancelled_error() -> None:
    """Prove that the real SDK's disconnect() leaks CancelledError.

    This is the bug we're working around.  The leak happens when
    disconnect() is called via asyncio.shield() (which is how the old
    _disconnect code called it).  The anyio cancel scope targets the
    *host task* that called connect(), and shield creates a new task,
    so the CancelledError hits the original task.

    If this test starts passing (disconnect stops leaking), we can
    switch back to client.disconnect() and remove _kill_client.
    """
    from claude_agent_sdk import ClaudeSDKClient

    client = ClaudeSDKClient(_make_options())
    await client.connect()

    # When shielded (as in the old _disconnect), the cancel scope leak
    # causes CancelledError in the calling task.
    leaked = False
    try:
        await asyncio.shield(client.disconnect())
    except asyncio.CancelledError:
        leaked = True

    # Also check if a subsequent await gets CancelledError (async leak)
    if not leaked:
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            leaked = True

    assert leaked, (
        "SDK disconnect no longer leaks CancelledError — "
        "_kill_client workaround may no longer be needed"
    )
