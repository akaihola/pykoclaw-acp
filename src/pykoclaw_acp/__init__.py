"""ACP (Agent Client Protocol) plugin for pykoclaw."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click

from pykoclaw.plugins import PykoClawPluginBase

# Seconds to wait for tasks to cancel during shutdown before force-killing.
_SHUTDOWN_TIMEOUT_S = 5


async def _run_with_graceful_shutdown(server: object) -> None:
    """Run the ACP server and enforce a bounded shutdown.

    ``asyncio.run()`` calls ``_cancel_all_tasks()`` which waits *indefinitely*
    for every task to finish cancellation.  If any task is stuck on a
    non-cancellable I/O or lock, the whole process hangs (the exact bug the
    watchdog was catching).

    This helper does the same work as ``asyncio.run()`` but with a timeout on
    the cancellation phase so the process always exits.
    """
    try:
        await server.run()  # type: ignore[attr-defined]
    finally:
        await server.stop()  # type: ignore[attr-defined]

        # Cancel remaining tasks with a bounded timeout.
        loop = asyncio.get_event_loop()
        tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
        if tasks:
            for t in tasks:
                t.cancel()
            results = await asyncio.wait(tasks, timeout=_SHUTDOWN_TIMEOUT_S)
            # Log tasks that refused to cancel.
            still_pending = results[1] if isinstance(results, tuple) else set()
            if still_pending:
                logging.getLogger(__name__).warning(
                    "SHUTDOWN: %d task(s) did not cancel within %ds — "
                    "abandoning them: %s",
                    len(still_pending),
                    _SHUTDOWN_TIMEOUT_S,
                    [t.get_name() for t in still_pending],
                )


class AcpPlugin(PykoClawPluginBase):
    def register_commands(self, group: click.Group) -> None:
        @group.command()
        def acp() -> None:
            """Start ACP server (JSON-RPC over stdio)."""
            import faulthandler
            import os
            import signal

            # Dump Python tracebacks on SIGUSR1 for live debugging of stuck processes.
            # Write to a file (not stderr) because mitto pipes stderr and may swallow it.
            fault_dir = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "pykoclaw"
            fault_dir.mkdir(parents=True, exist_ok=True)
            fault_file = (fault_dir / f"faulthandler-{os.getpid()}.txt").open("w")
            faulthandler.enable(file=fault_file, all_threads=True)
            faulthandler.register(signal.SIGUSR1, file=fault_file, all_threads=True)

            logging.basicConfig(
                stream=sys.stderr,
                level=logging.ERROR,
                format="%(levelname)s %(name)s: %(message)s",
            )

            from pykoclaw.config import settings
            from pykoclaw.db import init_db

            from .server import AcpServer
            from .watchdog import Watchdog

            # Watchdog: daemon thread that auto-captures diagnostics and kills
            # the process if the asyncio event loop stops responding.
            watchdog = Watchdog(fault_file=fault_file)

            db = init_db(settings.db_path)
            server = AcpServer(db=db, data_dir=settings.data, watchdog=watchdog)

            asyncio.run(_run_with_graceful_shutdown(server))
