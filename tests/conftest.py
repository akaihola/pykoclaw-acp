from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import pytest

from pykoclaw_acp.server import AcpServer


@dataclass
class PromptResult:
    chunks: list[str]
    stop_response: dict[str, Any] | None
    error_updates: list[dict[str, Any]]
    messages: list[dict[str, Any]]


class MockClientPool:
    def __init__(self) -> None:
        self._started = False
        self._closed = False
        self._send_count = 0
        self.fail_prompts: set[str] = set()
        self.session_history: dict[str, list[str]] = {}
        self.prompt_response_map: dict[str, str] = {}
        self._entries: dict[str, Any] = {}

    async def start(self) -> None:
        self._started = True

    async def close(self) -> None:
        self._closed = True

    async def send(
        self,
        session_id: str,
        prompt: str,
        *,
        on_text: Callable[[str], Awaitable[None]] | None = None,
    ) -> str | None:
        self.session_history.setdefault(session_id, []).append(prompt)
        self._send_count += 1

        if prompt in self.fail_prompts:
            raise RuntimeError(f"mock failure for prompt: {prompt}")

        turn_index = len(self.session_history[session_id])
        prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
        base_response = f"Response[{prompt_hash}] to: {prompt}"
        response = f"{base_response} (turn={turn_index}, call={self._send_count})"
        self.prompt_response_map[prompt] = base_response

        if on_text is not None:
            for chunk in self._split_response(response, parts=3):
                await on_text(chunk)
                await asyncio.sleep(0.001)

        return f"mock-session-{session_id[:8]}"

    @staticmethod
    def _split_response(text: str, *, parts: int) -> list[str]:
        if not text:
            return [""]

        target_parts = max(2, parts)
        chunk_size = max(1, len(text) // target_parts)
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        if len(chunks) > 3:
            merged_tail = "".join(chunks[2:])
            chunks = [chunks[0], chunks[1], merged_tail]

        return chunks


class AcpTestClient:
    def __init__(self, server: AcpServer) -> None:
        self._server = server
        self._messages: list[dict[str, Any]] = []
        self._next_msg_id = 1
        self._msg_id_lock = asyncio.Lock()

        def capture(message: str) -> None:
            self._messages.append(json.loads(message))

        self._server._write = capture  # type: ignore[assignment]

    async def initialize(self) -> dict[str, Any]:
        messages = await self._request("initialize", {})
        return self._require_response(messages)["result"]

    async def new_session(self, cwd: str | None = None) -> str:
        params: dict[str, Any] = {}
        if cwd is not None:
            params["cwd"] = cwd
        messages = await self._request("session/new", params)
        return self._require_response(messages)["result"]["sessionId"]

    async def prompt(self, session_id: str, text: str) -> PromptResult:
        msg_id, start = await self._send_request(
            "session/prompt",
            {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": text}],
            },
        )
        messages = [
            msg
            for msg in self._messages[start:]
            if msg.get("id") == msg_id
            or (
                msg.get("method") == "session/update"
                and msg.get("params", {}).get("sessionId") == session_id
            )
        ]

        chunks: list[str] = []
        error_updates: list[dict[str, Any]] = []
        stop_response: dict[str, Any] | None = None

        for msg in messages:
            if msg.get("method") != "session/update":
                if "result" in msg:
                    stop_response = msg
                continue

            update = msg.get("params", {}).get("update", {})
            session_update = update.get("sessionUpdate")
            if session_update == "agent_message_chunk":
                chunk = update.get("content", {}).get("text", "")
                chunks.append(chunk)
            elif session_update == "error":
                error_updates.append(msg)

        return PromptResult(
            chunks=chunks,
            stop_response=stop_response,
            error_updates=error_updates,
            messages=messages,
        )

    def get_updates(self, session_id: str) -> list[dict[str, Any]]:
        return [
            msg
            for msg in self._messages
            if msg.get("method") == "session/update"
            and msg.get("params", {}).get("sessionId") == session_id
        ]

    async def request(
        self, method: str, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return await self._request(method, params)

    async def _request(
        self, method: str, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        _, start = await self._send_request(method, params)
        return self._messages[start:]

    async def _send_request(
        self, method: str, params: dict[str, Any]
    ) -> tuple[int, int]:
        async with self._msg_id_lock:
            msg_id = self._next_msg_id
            self._next_msg_id += 1
        start = len(self._messages)
        await self._server._dispatch(
            {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        )
        return msg_id, start

    @staticmethod
    def _require_response(messages: list[dict[str, Any]]) -> dict[str, Any]:
        for msg in messages:
            if "result" in msg and "id" in msg:
                return msg
        raise AssertionError("No JSON-RPC response message found")


@pytest.fixture()
def tmp_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(
        "CREATE TABLE IF NOT EXISTS conversations ("
        "    name TEXT PRIMARY KEY,"
        "    session_id TEXT,"
        "    cwd TEXT,"
        "    created_at TEXT NOT NULL"
        ");"
    )
    return db


@pytest.fixture()
def mock_pool() -> MockClientPool:
    return MockClientPool()


@pytest.fixture()
def acp_server(
    tmp_db: sqlite3.Connection,
    tmp_path: Path,
    mock_pool: MockClientPool,
) -> AcpServer:
    server = AcpServer(db=tmp_db, data_dir=tmp_path)
    server._pool = mock_pool
    return server


@pytest.fixture()
def acp_client(acp_server: AcpServer) -> AcpTestClient:
    return AcpTestClient(acp_server)
