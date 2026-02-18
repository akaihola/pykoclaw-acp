"""Tests for the bounded shutdown logic in _run_server / _cancel_remaining_tasks."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from pykoclaw_acp import _cancel_remaining_tasks


def test_cancel_remaining_tasks_no_tasks() -> None:
    loop = asyncio.new_event_loop()
    try:
        _cancel_remaining_tasks(loop)
    finally:
        loop.close()


def test_cancel_remaining_tasks_all_cancel_cleanly() -> None:
    cancelled = False

    async def cooperative_task() -> None:
        nonlocal cancelled
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled = True

    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(cooperative_task())
        loop.run_until_complete(asyncio.sleep(0))

        _cancel_remaining_tasks(loop)

        assert cancelled
        assert task.done()
    finally:
        loop.close()


def test_cancel_remaining_tasks_force_exits_on_stuck_task() -> None:
    stuck_forever = asyncio.Event()

    async def uncancellable_task() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await stuck_forever.wait()

    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(uncancellable_task())
        loop.run_until_complete(asyncio.sleep(0))

        with (
            patch("pykoclaw_acp._SHUTDOWN_TIMEOUT_S", 0.1),
            patch("pykoclaw_acp.os._exit") as mock_exit,
        ):
            _cancel_remaining_tasks(loop)
            mock_exit.assert_called_once_with(0)

        task.cancel()
        stuck_forever.set()
        loop.run_until_complete(asyncio.sleep(0.01))
    finally:
        loop.close()


def test_cancel_remaining_tasks_skips_done_tasks() -> None:
    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(asyncio.sleep(0))
        loop.run_until_complete(task)

        with patch("pykoclaw_acp.os._exit") as mock_exit:
            _cancel_remaining_tasks(loop)
            mock_exit.assert_not_called()
    finally:
        loop.close()
