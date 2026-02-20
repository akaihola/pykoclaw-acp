"""ACP (Agent Client Protocol) plugin for pykoclaw."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path

import click

from pykoclaw.plugins import PykoClawPluginBase

log = logging.getLogger(__name__)

# Seconds to wait for tasks to cancel during shutdown before force-killing.
_SHUTDOWN_TIMEOUT_S = 5


def _run_server(server: object) -> None:
    """Run the ACP server with a bounded shutdown.

    We manage the event loop manually for defense-in-depth.  The server is
    now pure asyncio (no anyio/SDK code in-process — that runs in worker
    subprocesses), so ``asyncio.run()`` would likely work fine.  We keep
    the manual loop + bounded cancellation as a safety net in case a
    worker pipe read ever gets stuck.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    log.info("ACP server starting (PID %d)", os.getpid())
    try:
        loop.run_until_complete(server.run())  # type: ignore[attr-defined]
    except KeyboardInterrupt:
        log.info("SHUTDOWN: KeyboardInterrupt received")
    finally:
        log.info("SHUTDOWN: stopping server (graceful cleanup)")
        loop.run_until_complete(server.stop())  # type: ignore[attr-defined]
        log.info("SHUTDOWN: server stopped, cancelling remaining tasks")
        _cancel_remaining_tasks(loop)
        log.info("SHUTDOWN: shutting down async generators")
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        log.info("SHUTDOWN: event loop closed — clean exit")


def _cancel_remaining_tasks(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel all remaining tasks with a hard timeout.

    With process-isolated workers, all tasks in this event loop are pure
    asyncio (no anyio cancel scope leaks).  The ``os._exit()`` fallback
    is kept as defense-in-depth but should never trigger.
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
        os._exit(0)


_HEALTHCHECK_TIMEOUT_S = 10


def _run_healthcheck(cli_path: str) -> None:
    """Spawn ``pykoclaw acp`` and verify it responds to ``initialize``.

    Exits with code 0 on success, 1 on failure.  Designed to be run after
    a deploy to catch import errors, config problems, and protocol regressions
    without needing a full Mitto session.
    """
    request = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {},
    }) + "\n"

    click.echo(f"Starting ACP server ({cli_path} acp) ...", err=True)
    start = time.monotonic()

    try:
        proc = subprocess.Popen(
            [cli_path, "acp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        click.echo(f"FAIL: command not found: {cli_path}", err=True)
        sys.exit(1)

    try:
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(request)
        proc.stdin.flush()
        proc.stdin.close()  # signal EOF so the server exits its read loop

        # Read response with timeout
        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ)

        response_line = ""
        deadline = start + _HEALTHCHECK_TIMEOUT_S
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            events = sel.select(timeout=max(0, remaining))
            if events:
                chunk = proc.stdout.readline()
                if chunk:
                    response_line = chunk.strip()
                    break
                break  # EOF

        sel.close()
        elapsed = time.monotonic() - start

        if not response_line:
            stderr_out = proc.stderr.read() if proc.stderr else ""
            click.echo(
                f"FAIL: no response within {_HEALTHCHECK_TIMEOUT_S}s", err=True,
            )
            if stderr_out.strip():
                click.echo(f"stderr: {stderr_out.strip()}", err=True)
            sys.exit(1)

        msg = json.loads(response_line)
        result = msg.get("result", {})
        agent_info = result.get("agentInfo", {})

        if (
            msg.get("jsonrpc") == "2.0"
            and msg.get("id") == 1
            and "protocolVersion" in result
        ):
            click.echo(
                f"OK: {agent_info.get('name', '?')} v{agent_info.get('version', '?')}"
                f" (protocol v{result['protocolVersion']})"
                f" responded in {elapsed:.2f}s",
                err=True,
            )
            sys.exit(0)
        else:
            click.echo(f"FAIL: unexpected response: {response_line}", err=True)
            sys.exit(1)

    except json.JSONDecodeError as exc:
        click.echo(f"FAIL: invalid JSON response: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"FAIL: {exc}", err=True)
        sys.exit(1)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


class AcpPlugin(PykoClawPluginBase):
    def register_commands(self, group: click.Group) -> None:
        @group.command()
        @click.option(
            "--healthcheck",
            is_flag=True,
            help="Send initialize request, verify response, and exit.",
        )
        @click.option(
            "--log-level",
            type=click.Choice(
                ["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False,
            ),
            default="INFO",
            envvar="PYKOCLAW_ACP_LOG_LEVEL",
            help="Logging level (default: INFO, env: PYKOCLAW_ACP_LOG_LEVEL).",
        )
        def acp(healthcheck: bool, log_level: str) -> None:
            """Start ACP server (JSON-RPC over stdio)."""
            if healthcheck:
                from pykoclaw.config import settings

                cli_path = getattr(settings, "cli_path", None) or "pykoclaw"
                _run_healthcheck(cli_path)
                return

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

            effective_level = getattr(logging, log_level.upper())

            # Log to stderr (visible when run directly or via mitto cli;
            # captured in 8KB ring buffer by mitto web — only at DEBUG level).
            stderr_handler = logging.StreamHandler(sys.stderr)
            stderr_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S",
                )
            )

            # Also log to a file — survives regardless of how Mitto handles
            # stderr.  Rotated by PID so concurrent processes don't clobber.
            log_file = fault_dir / f"acp-{os.getpid()}.log"
            file_handler = logging.FileHandler(log_file, mode="w")
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )

            logging.basicConfig(
                level=effective_level,
                handlers=[stderr_handler, file_handler],
            )
            log.info(
                "Logging to stderr + %s (level=%s, PID=%d)",
                log_file, log_level, os.getpid(),
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
