"""ACP (Agent Client Protocol) plugin for pykoclaw."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click

from pykoclaw.plugins import PykoClawPluginBase


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

            asyncio.run(server.run())
