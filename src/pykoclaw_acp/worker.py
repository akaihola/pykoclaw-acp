"""SDK worker subprocess — runs ClaudeSDKClient in process isolation."""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
)

from pykoclaw.db import DbConnection, init_db, upsert_conversation
from pykoclaw.sdk_consume import consume_sdk_response
from pykoclaw.tools import make_mcp_server

from .worker_protocol import (
    ErrorMessage,
    HeartbeatMessage,
    QueryMessage,
    ReadyMessage,
    ShutdownMessage,
    TextChunkMessage,
    WorkerConfig,
    WorkerMessage,
    WorkerResultMessage,
    decode_config,
    decode_server_message,
    encode,
)

log = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S = 5

_ALLOWED_TOOLS = [
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "WebSearch",
    "WebFetch",
    "mcp__pykoclaw__*",
]


def _write_msg(msg: WorkerMessage) -> None:
    """Write a protocol message to stdout (the pipe to the server)."""
    sys.stdout.write(encode(msg))
    sys.stdout.flush()


async def _handle_query(
    client: ClaudeSDKClient,
    msg: QueryMessage,
    *,
    db: DbConnection,
    conversation_name: str,
    cwd: str,
    write_msg: Callable[[WorkerMessage], None],
) -> None:
    """Handle a single query message."""
    try:
        await client.query(msg.prompt)

        async def on_text(text: str) -> None:
            write_msg(TextChunkMessage(id=msg.id, text=text))

        async def on_result(result: ResultMessage) -> None:
            upsert_conversation(db, conversation_name, result.session_id or "", cwd)
            write_msg(
                WorkerResultMessage(id=msg.id, session_id=result.session_id or "")
            )

        final = await consume_sdk_response(client, on_text=on_text, on_result=on_result)

        # If consume_sdk_response returned None (no ResultMessage), send an
        # empty result so the server always gets a completion signal.
        if final is None:
            write_msg(WorkerResultMessage(id=msg.id, session_id=""))

    except Exception as exc:
        log.exception("Query %s failed", msg.id)
        write_msg(ErrorMessage(id=msg.id, error=str(exc)))


async def _heartbeat_loop(write_msg: Callable[[WorkerMessage], None]) -> None:
    """Periodically send heartbeat messages."""
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)
        write_msg(HeartbeatMessage())


async def _run_worker() -> None:
    """Main worker coroutine."""
    loop = asyncio.get_event_loop()

    # Read config from stdin (first line)
    config_line = await loop.run_in_executor(None, sys.stdin.readline)
    if not config_line:
        log.error("No config received on stdin")
        return

    config = decode_config(config_line)
    log.info("Worker starting for conversation %s", config.conversation_name)

    # Set up DB and MCP server
    db = init_db(Path(config.db_path))
    mcp_server = make_mcp_server(db, config.conversation_name)

    # Create and connect the SDK client
    options = ClaudeAgentOptions(
        cwd=config.cwd,
        permission_mode="bypassPermissions",
        mcp_servers={"pykoclaw": mcp_server},
        model=config.model,
        cli_path=config.cli_path,
        allowed_tools=(
            list(config.allowed_tools) if config.allowed_tools else list(_ALLOWED_TOOLS)
        ),
        system_prompt="If a tool call fails, retry it before concluding the tool is unavailable.",
        setting_sources=["project"],
        env={"SHELL": "/bin/bash"},
        resume=config.resume_session_id,
    )

    client = ClaudeSDKClient(options)
    await client.connect()

    _write_msg(ReadyMessage())
    log.info("Worker ready")

    # Start heartbeat
    heartbeat_task = asyncio.create_task(_heartbeat_loop(_write_msg))

    try:
        # Main message loop
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                log.info("Stdin closed, shutting down")
                break

            line = line.strip()
            if not line:
                continue

            try:
                server_msg = decode_server_message(line)
            except (ValueError, Exception) as exc:
                log.warning("Failed to decode message: %s", exc)
                continue

            if isinstance(server_msg, ShutdownMessage):
                log.info("Shutdown requested")
                break

            if isinstance(server_msg, QueryMessage):
                await _handle_query(
                    client,
                    server_msg,
                    db=db,
                    conversation_name=config.conversation_name,
                    cwd=config.cwd,
                    write_msg=_write_msg,
                )
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

        # Disconnect is safe here — we own the entire async runtime
        try:
            await client.disconnect()
        except Exception:
            log.warning("Error during client disconnect", exc_info=True)

        log.info("Worker shutdown complete")


def main() -> None:
    """Entry point for the worker subprocess."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,  # stdout is the protocol channel
    )
    asyncio.run(_run_worker())


if __name__ == "__main__":
    main()
