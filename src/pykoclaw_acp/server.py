"""ACP server: async stdio loop speaking JSON-RPC 2.0."""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from pykoclaw.db import DbConnection
from pykoclaw_messaging import dispatch_to_agent

from .protocol import AcpProtocolHandler, JsonRpcError

log = logging.getLogger(__name__)

PROTOCOL_VERSION = 1


class AcpServer:
    def __init__(self, *, db: DbConnection, data_dir: Path) -> None:
        self._db = db
        self._data_dir = data_dir
        self._protocol = AcpProtocolHandler()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._running = False

    async def run(self) -> None:
        self._running = True
        log.info("Starting ACP channel (JSON-RPC over stdio)")

        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while self._running:
            try:
                line = await reader.readline()
                if not line:
                    log.info("Stdin closed, stopping ACP channel")
                    break

                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                msg = self._protocol.parse_message(line_str)
                if msg is None:
                    self._write(
                        self._protocol.format_error(
                            None, JsonRpcError.PARSE_ERROR, "Parse error"
                        )
                    )
                    continue

                await self._dispatch(msg)

            except Exception:
                log.exception("Error reading stdin")
                continue

        self._running = False

    async def stop(self) -> None:
        self._running = False
        self._sessions.clear()

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        method = msg.get("method")
        params = msg.get("params", {})
        msg_id = msg.get("id")

        if not method:
            return

        handler = {
            "initialize": self._handle_initialize,
            "session/new": self._handle_session_new,
            "session/prompt": self._handle_session_prompt,
        }.get(method)

        if handler is None:
            self._write(
                self._protocol.format_error(
                    msg_id,
                    JsonRpcError.METHOD_NOT_FOUND,
                    f"Method not found: {method}",
                )
            )
            return

        await handler(msg_id, params)

    async def _handle_initialize(self, msg_id: Any, params: dict[str, Any]) -> None:
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "agentCapabilities": {},
            "agentInfo": {"name": "pykoclaw", "version": "0.1.0"},
        }
        self._write(self._protocol.format_response(msg_id, result))

    async def _handle_session_new(self, msg_id: Any, params: dict[str, Any]) -> None:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "cwd": params.get("cwd", ""),
        }
        self._write(self._protocol.format_response(msg_id, {"sessionId": session_id}))

    async def _handle_session_prompt(self, msg_id: Any, params: dict[str, Any]) -> None:
        session_id = params.get("sessionId")
        prompt_items = params.get("prompt", [])

        if not session_id or session_id not in self._sessions:
            self._write(
                self._protocol.format_error(
                    msg_id,
                    JsonRpcError.INVALID_SESSION,
                    f"Invalid or unknown session ID: {session_id}",
                )
            )
            return

        text_parts = [
            item.get("text", "")
            for item in prompt_items
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        content = "\n".join(text_parts)

        if not content:
            self._write(
                self._protocol.format_error(
                    msg_id,
                    JsonRpcError.INVALID_PARAMS,
                    "Prompt must contain at least one text content item",
                )
            )
            return

        # Acknowledge immediately.
        self._write(self._protocol.format_response(msg_id, {}))

        async def _send_chunk(text: str) -> None:
            self._write(
                self._protocol.format_notification(
                    "session/update",
                    {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": text},
                        },
                    },
                )
            )

        try:
            await dispatch_to_agent(
                prompt=content,
                channel_prefix="acp",
                channel_id=session_id[:8],
                db=self._db,
                data_dir=self._data_dir,
                on_text=_send_chunk,
            )
        except Exception:
            log.exception("Agent dispatch failed for session %s", session_id)
            self._write(
                self._protocol.format_notification(
                    "session/update",
                    {
                        "sessionId": session_id,
                        "update": {
                            "sessionUpdate": "error",
                            "error": "Agent processing failed. Please try again.",
                        },
                    },
                )
            )

    def _write(self, message: str) -> None:
        try:
            sys.stdout.write(message)
            sys.stdout.flush()
        except Exception:
            log.exception("Error writing to stdout")
