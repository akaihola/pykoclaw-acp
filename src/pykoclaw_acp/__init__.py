"""ACP (Agent Client Protocol) plugin for pykoclaw."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import click

from pykoclaw.plugins import PykoClawPluginBase

log = logging.getLogger(__name__)

# Seconds to wait for tasks to cancel during shutdown before force-killing.
_SHUTDOWN_TIMEOUT_S = 5


def _run_server(server: object) -> None:
    """Run the ACP server with a bounded shutdown.

    We manage the event loop manually instead of using ``asyncio.run()``
    because its cleanup phase calls ``_cancel_all_tasks()`` which waits
    *indefinitely* for every task to finish cancellation.  If any task is
    stuck on non-cancellable I/O (e.g. a Claude SDK subprocess), the process
    hangs forever — the exact bug the watchdog was catching and SIGKILL-ing,
    leaving zombies that Mitto can't reap.

    By owning the loop ourselves we can enforce a hard timeout and call
    ``os._exit()`` when tasks refuse to cancel, guaranteeing the process
    never lingers.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(server.run())  # type: ignore[attr-defined]
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(server.stop())  # type: ignore[attr-defined]
        _cancel_remaining_tasks(loop)
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


def _cancel_remaining_tasks(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel all remaining tasks with a hard timeout.

    If tasks don't cancel within ``_SHUTDOWN_TIMEOUT_S`` seconds, force-exit
    the process with ``os._exit(0)`` to prevent ``asyncio.run()``-style
    infinite hangs in ``_cancel_all_tasks()``.
    """
    tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
    if not tasks:
        return

    for t in tasks:
        t.cancel()

    results = loop.run_until_complete(asyncio.wait(tasks, timeout=_SHUTDOWN_TIMEOUT_S))
    still_pending = results[1] if isinstance(results, tuple) else set()
    if still_pending:
        log.warning(
            "SHUTDOWN: %d task(s) did not cancel within %ds — force-exiting: %s",
            len(still_pending),
            _SHUTDOWN_TIMEOUT_S,
            [t.get_name() for t in still_pending],
        )
        # Hard exit: the only way to prevent the process from becoming a
        # zombie when non-cancellable tasks (Claude SDK subprocesses) refuse
        # to die.  This is safe because server.stop() already ran — all
        # graceful cleanup that *can* happen has happened.
        os._exit(0)


class AcpPlugin(PykoClawPluginBase):
    def register_commands(self, group: click.Group) -> None:
        @group.command()
        def acp() -> None:
            """Start ACP server (JSON-RPC over stdio)."""
            import faulthandler
            import signal

            # Dump Python tracebacks on SIGUSR1 for live debugging of stuck processes.
            # Write to a file (not stderr) because mitto pipes stderr and may swallow it.
            fault_dir = (
                Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
                / "pykoclaw"
            )
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

            _run_server(server)
