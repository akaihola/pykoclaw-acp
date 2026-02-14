"""ACP (Agent Client Protocol) plugin for pykoclaw."""

from __future__ import annotations

import asyncio
import logging
import sys

import click

from pykoclaw.plugins import PykoClawPluginBase


class AcpPlugin(PykoClawPluginBase):
    def register_commands(self, group: click.Group) -> None:
        @group.command()
        def acp() -> None:
            """Start ACP server (JSON-RPC over stdio)."""
            logging.basicConfig(
                stream=sys.stderr,
                level=logging.ERROR,
                format="%(levelname)s %(name)s: %(message)s",
            )

            from pykoclaw.config import settings
            from pykoclaw.db import init_db

            from .server import AcpServer

            db = init_db(settings.db_path)
            server = AcpServer(db=db, data_dir=settings.data)
            asyncio.run(server.run())
