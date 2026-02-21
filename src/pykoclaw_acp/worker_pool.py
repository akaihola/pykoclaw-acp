"""WorkerPool — manages process-isolated SDK worker subprocesses."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from pykoclaw.config import settings
from pykoclaw.db import DbConnection

from .worker_protocol import (
    ErrorMessage,
    HeartbeatMessage,
    QueryMessage,
    ReadyMessage,
    ShutdownMessage,
    TextChunkMessage,
    WorkerConfig,
    WorkerResultMessage,
    decode_worker_message,
    encode,
)

log = logging.getLogger(__name__)

IDLE_TIMEOUT_S = 600
SWEEP_INTERVAL_S = 60
QUERY_TIMEOUT_S = 600
WORKER_READY_TIMEOUT_S = 30
WORKER_SHUTDOWN_TIMEOUT_S = 5

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
class _WorkerHandle:
    process: asyncio.subprocess.Process
    conversation_name: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = field(default_factory=time.monotonic)
    heartbeat_last: float = field(default_factory=time.monotonic)
    stderr_task: asyncio.Task[None] | None = None


class WorkerPool:
    """Keyed pool of process-isolated SDK workers for ACP sessions.

    API-compatible replacement for ``ClientPool``.
    """

    def __init__(
        self,
        *,
        db: DbConnection,
        data_dir: Path,
        worker_cmd: list[str] | None = None,
    ) -> None:
        self._db = db
        self._data_dir = data_dir
        self._worker_cmd = worker_cmd or [sys.executable, "-m", "pykoclaw_acp.worker"]
        self._entries: dict[str, _WorkerHandle] = {}
        self._create_lock = asyncio.Lock()
        self._sweep_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._sweep_task = asyncio.create_task(self._sweep_loop())

    async def close(self) -> None:
        if self._sweep_task:
            self._sweep_task.cancel()
            self._sweep_task = None
        sids = list(self._entries)
        log.info("WorkerPool.close() — shutting down %d worker(s)", len(sids))
        results = await asyncio.gather(
            *(self._shutdown_worker(sid) for sid in sids),
            return_exceptions=True,
        )
        errors = [
            (sid, r) for sid, r in zip(sids, results) if isinstance(r, Exception)
        ]
        if errors:
            for sid, exc in errors:
                log.warning("WorkerPool: error shutting down %s: %s", sid, exc)
        log.info("WorkerPool.close() complete")

    async def send(
        self,
        session_id: str,
        prompt: str,
        *,
        on_text: Callable[[str], Awaitable[None]] | None = None,
        resume_session_id: str | None = None,
    ) -> str | None:
        """Send *prompt* to the worker for *session_id*, streaming via *on_text*.

        Creates the worker on first call.  Returns the session ID from the
        worker's result message.
        """
        handle = await self._get_or_create(
            session_id, resume_session_id=resume_session_id
        )
        async with handle.lock:
            handle.last_used = time.monotonic()
            try:
                return await self._query(handle, prompt, on_text)
            except Exception:
                log.warning(
                    "Worker %s crashed, recreating", session_id, exc_info=True
                )
                await self._shutdown_worker(session_id)
                handle = await self._get_or_create(session_id)
                async with handle.lock:
                    return await self._query(handle, prompt, on_text)

    # -- internal helpers -----------------------------------------------------

    async def _query(
        self,
        handle: _WorkerHandle,
        prompt: str,
        on_text: Callable[[str], Awaitable[None]] | None,
    ) -> str | None:
        msg_id = uuid.uuid4().hex[:8]
        query_msg = QueryMessage(id=msg_id, prompt=prompt)
        await self._write_to_worker(handle, encode(query_msg))

        async with asyncio.timeout(QUERY_TIMEOUT_S):
            while True:
                assert handle.process.stdout is not None
                raw = await handle.process.stdout.readline()
                if not raw:
                    raise RuntimeError("Worker process exited unexpectedly")

                line = raw.decode().strip()
                if not line:
                    continue

                msg = decode_worker_message(line)

                if isinstance(msg, TextChunkMessage) and msg.id == msg_id:
                    if on_text:
                        await on_text(msg.text)
                elif isinstance(msg, WorkerResultMessage) and msg.id == msg_id:
                    return msg.session_id or None
                elif isinstance(msg, ErrorMessage) and msg.id == msg_id:
                    raise RuntimeError(msg.error)
                elif isinstance(msg, HeartbeatMessage):
                    handle.heartbeat_last = time.monotonic()
                # Ignore messages with non-matching ids (shouldn't happen
                # since we hold the lock, but be defensive).

    async def _get_or_create(
        self, session_id: str, *, resume_session_id: str | None = None
    ) -> _WorkerHandle:
        if session_id in self._entries:
            return self._entries[session_id]

        async with self._create_lock:
            if session_id in self._entries:
                return self._entries[session_id]

            handle = await self._spawn_worker(
                session_id, resume_session_id=resume_session_id
            )
            self._entries[session_id] = handle
            log.info("Spawned worker for session %s", session_id)
            return handle

    async def _spawn_worker(
        self, session_id: str, *, resume_session_id: str | None = None
    ) -> _WorkerHandle:
        conversation_name = f"acp-{session_id[:8]}"
        conv_dir = self._data_dir / "conversations" / conversation_name
        conv_dir.mkdir(parents=True, exist_ok=True)

        config = WorkerConfig(
            cwd=str(conv_dir),
            model=settings.model,
            conversation_name=conversation_name,
            db_path=str(settings.db_path),
            cli_path=str(settings.cli_path) if settings.cli_path else None,
            allowed_tools=list(_ALLOWED_TOOLS),
            resume_session_id=resume_session_id,
        )

        process = await asyncio.create_subprocess_exec(
            *self._worker_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Send config as the first line
        assert process.stdin is not None
        await self._write_to_process(process, encode(config))

        # Wait for ready message
        assert process.stdout is not None
        try:
            async with asyncio.timeout(WORKER_READY_TIMEOUT_S):
                while True:
                    raw = await process.stdout.readline()
                    if not raw:
                        raise RuntimeError(
                            "Worker process exited before sending ready"
                        )
                    line = raw.decode().strip()
                    if not line:
                        continue
                    msg = decode_worker_message(line)
                    if isinstance(msg, ReadyMessage):
                        break
                    if isinstance(msg, HeartbeatMessage):
                        continue
                    raise RuntimeError(
                        f"Expected ready message, got: {msg}"
                    )
        except (TimeoutError, RuntimeError):
            process.kill()
            await process.wait()
            raise

        handle = _WorkerHandle(
            process=process,
            conversation_name=conversation_name,
        )

        # Start stderr forwarding
        assert process.stderr is not None
        handle.stderr_task = asyncio.create_task(
            self._forward_stderr(session_id, process.stderr)
        )

        return handle

    async def _shutdown_worker(self, session_id: str) -> None:
        handle = self._entries.pop(session_id, None)
        if not handle:
            return

        # Cancel stderr forwarder first
        if handle.stderr_task:
            handle.stderr_task.cancel()
            try:
                await handle.stderr_task
            except asyncio.CancelledError:
                pass

        # Send shutdown message if process is still alive
        if handle.process.returncode is None:
            try:
                await self._write_to_worker(handle, encode(ShutdownMessage()))
            except (BrokenPipeError, OSError, ConnectionResetError):
                pass

            try:
                async with asyncio.timeout(WORKER_SHUTDOWN_TIMEOUT_S):
                    await handle.process.wait()
            except TimeoutError:
                log.warning("Worker %s didn't exit gracefully, killing", session_id)
                handle.process.kill()
                await handle.process.wait()

        log.debug("Worker %s shut down (rc=%s)", session_id, handle.process.returncode)

    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(SWEEP_INTERVAL_S)
            now = time.monotonic()
            stale = [
                sid
                for sid, h in self._entries.items()
                if now - h.last_used > IDLE_TIMEOUT_S and not h.lock.locked()
            ]
            for sid in stale:
                log.info("Evicting idle worker %s", sid)
                try:
                    await self._shutdown_worker(sid)
                except BaseException:
                    log.warning(
                        "Eviction of worker %s raised; ignoring",
                        sid,
                        exc_info=True,
                    )

    async def _forward_stderr(
        self, session_id: str, stderr: asyncio.StreamReader
    ) -> None:
        """Read worker stderr line-by-line and log with session context."""
        try:
            while True:
                raw = await stderr.readline()
                if not raw:
                    break
                line = raw.decode().rstrip()
                if line:
                    log.debug("[worker %s] %s", session_id, line)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.debug(
                "stderr forwarder for %s stopped", session_id, exc_info=True
            )

    @staticmethod
    async def _write_to_worker(handle: _WorkerHandle, data: str) -> None:
        assert handle.process.stdin is not None
        handle.process.stdin.write(data.encode())
        await handle.process.stdin.drain()

    @staticmethod
    async def _write_to_process(
        process: asyncio.subprocess.Process, data: str
    ) -> None:
        assert process.stdin is not None
        process.stdin.write(data.encode())
        await process.stdin.drain()
