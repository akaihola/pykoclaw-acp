"""ACP server: async stdio loop speaking JSON-RPC 2.0."""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

from pykoclaw.db import (
    DbConnection,
    get_pending_deliveries,
    mark_delivered,
    mark_delivery_failed,
)

from .client_pool import ClientPool
from .protocol import AcpProtocolHandler, JsonRpcError
from .watchdog import Watchdog

log = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
DELIVERY_POLL_INTERVAL_S = 10
HEARTBEAT_INTERVAL_S = 5


class AcpServer:
    def __init__(
        self,
        *,
        db: DbConnection,
        data_dir: Path,
        watchdog: Watchdog | None = None,
    ) -> None:
        self._db = db
        self._data_dir = data_dir
        self._protocol = AcpProtocolHandler()
        self._sessions: dict[str, dict[str, Any]] = {}
        self._pool = ClientPool(db=db, data_dir=data_dir)
        self._running = False
        self._watchdog = watchdog

    async def run(self) -> None:
        self._running = True
        await self._pool.start()
        self._delivery_task = asyncio.create_task(self._delivery_poll_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        if self._watchdog:
            self._watchdog.start()
        log.info("Starting ACP channel (JSON-RPC over stdio)")

        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        consecutive_errors = 0
        max_consecutive_errors = 10

        while self._running:
            try:
                line = await reader.readline()
                if not line:
                    log.info("Stdin closed, stopping ACP channel")
                    break

                line_str = line.decode("utf-8").strip()
                if not line_str:
                    await asyncio.sleep(0.001)  # prevent busy-loop on empty lines
                    continue

                consecutive_errors = 0  # successful read resets counter

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
                consecutive_errors += 1
                log.exception(
                    "Error reading stdin (consecutive: %d/%d)",
                    consecutive_errors,
                    max_consecutive_errors,
                )
                if consecutive_errors >= max_consecutive_errors:
                    log.error("Too many consecutive errors, stopping ACP server")
                    break
                await asyncio.sleep(0.1)  # back off on errors
                continue

        self._running = False

    async def stop(self) -> None:
        self._running = False
        if self._watchdog:
            self._watchdog.stop()
        if hasattr(self, "_delivery_task"):
            self._delivery_task.cancel()
        if hasattr(self, "_heartbeat_task"):
            self._heartbeat_task.cancel()
        await self._pool.close()
        self._sessions.clear()

    async def _heartbeat_loop(self) -> None:
        """Periodically ping the watchdog to prove the event loop is alive."""
        try:
            while self._running:
                if self._watchdog:
                    self._watchdog.heartbeat()
                await asyncio.sleep(HEARTBEAT_INTERVAL_S)
        except asyncio.CancelledError:
            pass

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
            await self._pool.send(session_id, content, on_text=_send_chunk)
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

        # Send prompt response AFTER streaming completes, per ACP protocol.
        # The stopReason tells the client the turn is finished.
        self._write(self._protocol.format_response(msg_id, {"stopReason": "end_turn"}))

    async def _delivery_poll_loop(self) -> None:
        log.info("ACP delivery polling started")
        try:
            while self._running:
                await asyncio.sleep(DELIVERY_POLL_INTERVAL_S)
                try:
                    self._process_pending_deliveries()
                except Exception:
                    log.exception("Error processing ACP delivery queue")
        except asyncio.CancelledError:
            log.info("ACP delivery polling stopped")

    def _process_pending_deliveries(self) -> None:
        pending = get_pending_deliveries(self._db, "acp")
        if not pending:
            return

        active_conversations = {
            entry.conversation_name for entry in self._pool._entries.values()
        }

        for delivery in pending:
            if delivery.conversation not in active_conversations:
                continue

            session_id = self._find_session_for_conversation(delivery.conversation)
            if not session_id:
                continue

            try:
                self._write(
                    self._protocol.format_notification(
                        "session/update",
                        {
                            "sessionId": session_id,
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "content": {
                                    "type": "text",
                                    "text": delivery.message,
                                },
                            },
                        },
                    )
                )
                mark_delivered(self._db, delivery.id)
                log.info("Delivered task result to ACP session %s", session_id)
            except Exception:
                mark_delivery_failed(self._db, delivery.id, "write failed")
                log.exception("Failed to deliver to ACP session %s", session_id)

    def _find_session_for_conversation(self, conversation: str) -> str | None:
        for session_id, entry in self._pool._entries.items():
            if entry.conversation_name == conversation:
                return session_id
        return None

    def _write(self, message: str) -> None:
        try:
            sys.stdout.write(message)
            sys.stdout.flush()
        except Exception:
            log.exception("Error writing to stdout")
