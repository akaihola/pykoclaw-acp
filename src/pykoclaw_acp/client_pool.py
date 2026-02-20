"""Long-lived ClaudeSDKClient pool — one subprocess per ACP conversation."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

from pykoclaw.config import settings
from pykoclaw.db import DbConnection, upsert_conversation
from pykoclaw.tools import make_mcp_server

log = logging.getLogger(__name__)

IDLE_TIMEOUT_S = 600
SWEEP_INTERVAL_S = 60
QUERY_TIMEOUT_S = 600  # 10 min max per query to prevent infinite hangs

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


@dataclass
class _Entry:
    client: ClaudeSDKClient
    conversation_name: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = field(default_factory=time.monotonic)


class ClientPool:
    """Keyed pool of long-lived Claude clients for ACP sessions.

    API: ``start()``, ``close()``, ``send()``.
    """

    def __init__(self, *, db: DbConnection, data_dir: Path) -> None:
        self._db = db
        self._data_dir = data_dir
        self._entries: dict[str, _Entry] = {}
        self._create_lock = asyncio.Lock()
        self._sweep_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._sweep_task = asyncio.create_task(self._sweep_loop())

    async def close(self) -> None:
        if self._sweep_task:
            self._sweep_task.cancel()
            self._sweep_task = None
        sids = list(self._entries)
        log.info("ClientPool.close() — disconnecting %d client(s)", len(sids))
        results = await asyncio.gather(
            *(self._disconnect(sid) for sid in sids),
            return_exceptions=True,
        )
        errors = [(sid, r) for sid, r in zip(sids, results) if isinstance(r, Exception)]
        if errors:
            for sid, exc in errors:
                log.warning("ClientPool: error disconnecting %s: %s", sid, exc)
        log.info("ClientPool.close() complete")

    async def send(
        self,
        session_id: str,
        prompt: str,
        *,
        on_text: Callable[[str], Awaitable[None]] | None = None,
    ) -> str | None:
        """Send *prompt* on the client for *session_id*, streaming via *on_text*.

        Creates the client on first call.  Returns the Claude session ID.
        """
        entry = await self._get_or_create(session_id)
        async with entry.lock:
            entry.last_used = time.monotonic()
            try:
                return await self._query(entry, prompt, on_text)
            except Exception:
                log.warning("Client %s crashed, recreating", session_id, exc_info=True)
                await self._disconnect(session_id)
                entry = await self._get_or_create(session_id)
                async with entry.lock:
                    return await self._query(entry, prompt, on_text)

    async def _query(
        self,
        entry: _Entry,
        prompt: str,
        on_text: Callable[[str], Awaitable[None]] | None,
    ) -> str | None:
        async with asyncio.timeout(QUERY_TIMEOUT_S):
            await entry.client.query(prompt)
            session_id: str | None = None
            had_text_blocks = False
            async for message in entry.client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            had_text_blocks = True
                            if on_text:
                                await on_text(block.text)
                elif isinstance(message, ResultMessage):
                    session_id = message.session_id
                    # Fallback: if no TextBlock was streamed but
                    # ResultMessage carries the response text, forward
                    # it so the Mitto user still sees the reply.
                    if (
                        not had_text_blocks
                        and message.result
                        and on_text
                    ):
                        await on_text(message.result)
                    conv_dir = (
                        self._data_dir / "conversations" / entry.conversation_name
                    )
                    upsert_conversation(
                        self._db, entry.conversation_name, session_id, str(conv_dir)
                    )
            return session_id

    async def _get_or_create(self, session_id: str) -> _Entry:
        if session_id in self._entries:
            return self._entries[session_id]

        async with self._create_lock:
            if session_id in self._entries:
                return self._entries[session_id]

            conversation_name = f"acp-{session_id[:8]}"
            conv_dir = self._data_dir / "conversations" / conversation_name
            conv_dir.mkdir(parents=True, exist_ok=True)

            options = ClaudeAgentOptions(
                cwd=str(conv_dir),
                permission_mode="bypassPermissions",
                mcp_servers={"pykoclaw": make_mcp_server(self._db, conversation_name)},
                model=settings.model,
                cli_path=settings.cli_path,
                allowed_tools=list(_ALLOWED_TOOLS),
                setting_sources=["project"],
                env={"SHELL": "/bin/bash"},
            )

            client = ClaudeSDKClient(options)
            await client.connect()

            entry = _Entry(client=client, conversation_name=conversation_name)
            self._entries[session_id] = entry
            log.info("Created pooled client for session %s", session_id)
            return entry

    async def _disconnect(self, session_id: str) -> None:
        entry = self._entries.pop(session_id, None)
        if entry:
            try:
                # Shield the disconnect so that anyio's cancel-scope
                # cancellation inside ClaudeSDKClient.disconnect() cannot
                # propagate a CancelledError into the caller's task (the
                # _sweep_loop or the main server loop).  Without this, the
                # cancel scope set by Query.close() leaks into the asyncio
                # event loop and spin-cancels the server's readline().
                await asyncio.shield(entry.client.disconnect())
            except (asyncio.CancelledError, Exception):
                log.debug("Disconnect error for %s", session_id, exc_info=True)

    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(SWEEP_INTERVAL_S)
            now = time.monotonic()
            stale = [
                sid
                for sid, e in self._entries.items()
                if now - e.last_used > IDLE_TIMEOUT_S and not e.lock.locked()
            ]
            for sid in stale:
                log.info("Evicting idle client %s", sid)
                await self._disconnect(sid)
