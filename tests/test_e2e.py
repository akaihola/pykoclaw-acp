"""End-to-end tests for the ACP server over subprocess stdio.

These tests launch ``pykoclaw acp`` as a real subprocess with isolated
PYKOCLAW_DATA, send JSON-RPC messages over stdin, and verify responses
on stdout.  They require a working Claude API key in the environment.

Run with::

    uv run pytest pykoclaw-acp/tests/test_e2e.py -m e2e
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

pytestmark = pytest.mark.e2e

E2E_TIMEOUT_S = 60


class AcpSubprocessClient:
    """Talks JSON-RPC to a ``pykoclaw acp`` subprocess over stdin/stdout."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc
        self._next_id = 1

    async def send(self, method: str, params: dict[str, Any]) -> int:
        msg_id = self._next_id
        self._next_id += 1
        line = json.dumps(
            {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        )
        assert self._proc.stdin is not None
        self._proc.stdin.write((line + "\n").encode())
        await self._proc.stdin.drain()
        return msg_id

    async def read_line(self, timeout: float = 10.0) -> dict[str, Any] | None:
        assert self._proc.stdout is not None
        try:
            raw = await asyncio.wait_for(self._proc.stdout.readline(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        if not raw:
            return None
        return json.loads(raw.decode())

    async def read_until_response(
        self, msg_id: int, timeout: float = E2E_TIMEOUT_S
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Read lines until a response with *msg_id* appears.

        Returns ``(response, notifications)`` where *notifications* are
        any ``session/update`` messages received before the response.
        """
        notifications: list[dict[str, Any]] = []
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            remaining = deadline - asyncio.get_event_loop().time()
            msg = await self.read_line(timeout=max(0.1, remaining))
            if msg is None:
                continue
            if msg.get("id") == msg_id and ("result" in msg or "error" in msg):
                return msg, notifications
            notifications.append(msg)
        return None, notifications

    async def close(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            self._proc.kill()


@pytest_asyncio.fixture()
async def acp_proc(tmp_path: Path) -> AsyncGenerator[AcpSubprocessClient, None]:
    env = os.environ.copy()
    env["PYKOCLAW_DATA"] = str(tmp_path / "data")
    Path(env["PYKOCLAW_DATA"]).mkdir()

    uv = shutil.which("uv")
    assert uv is not None, "uv not found on PATH"

    proc = await asyncio.create_subprocess_exec(
        uv,
        "run",
        "pykoclaw",
        "acp",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    await asyncio.sleep(1)
    assert proc.returncode is None, "pykoclaw acp exited immediately"

    client = AcpSubprocessClient(proc)
    yield client
    await client.close()


@pytest.mark.asyncio
async def test_initialize(acp_proc: AcpSubprocessClient) -> None:
    msg_id = await acp_proc.send("initialize", {})
    resp, _ = await acp_proc.read_until_response(msg_id)

    assert resp is not None, "No response to initialize"
    result = resp["result"]
    assert result["agentInfo"]["name"] == "pykoclaw"
    assert "protocolVersion" in result


@pytest.mark.asyncio
async def test_session_new(acp_proc: AcpSubprocessClient) -> None:
    await acp_proc.send("initialize", {})
    await acp_proc.read_line()

    msg_id = await acp_proc.send("session/new", {})
    resp, _ = await acp_proc.read_until_response(msg_id)

    assert resp is not None
    session_id = resp["result"]["sessionId"]
    assert len(session_id) == 36


@pytest.mark.asyncio
async def test_prompt_streams_and_completes(
    acp_proc: AcpSubprocessClient,
) -> None:
    await acp_proc.send("initialize", {})
    await acp_proc.read_line()

    new_id = await acp_proc.send("session/new", {})
    resp, _ = await acp_proc.read_until_response(new_id)
    session_id = resp["result"]["sessionId"]

    prompt_id = await acp_proc.send(
        "session/prompt",
        {
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": "Reply with exactly: E2E_OK"}],
        },
    )
    resp, notifications = await acp_proc.read_until_response(prompt_id)

    assert resp is not None, "No stop response from session/prompt"
    assert resp["result"]["stopReason"] == "end_turn"

    chunks = [
        n["params"]["update"]["content"]["text"]
        for n in notifications
        if n.get("method") == "session/update"
        and n["params"]["update"].get("sessionUpdate") == "agent_message_chunk"
    ]
    full_text = "".join(chunks)
    assert len(full_text) > 0, "No streaming chunks received"


@pytest.mark.asyncio
async def test_invalid_session(acp_proc: AcpSubprocessClient) -> None:
    await acp_proc.send("initialize", {})
    await acp_proc.read_line()

    msg_id = await acp_proc.send(
        "session/prompt",
        {
            "sessionId": "nonexistent-session-id",
            "prompt": [{"type": "text", "text": "hello"}],
        },
    )
    resp, _ = await acp_proc.read_until_response(msg_id)

    assert resp is not None
    assert "error" in resp
    assert resp["error"]["code"] == -32000


@pytest.mark.asyncio
async def test_multi_turn(acp_proc: AcpSubprocessClient) -> None:
    await acp_proc.send("initialize", {})
    await acp_proc.read_line()

    new_id = await acp_proc.send("session/new", {})
    resp, _ = await acp_proc.read_until_response(new_id)
    session_id = resp["result"]["sessionId"]

    for i in range(2):
        prompt_id = await acp_proc.send(
            "session/prompt",
            {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": f"Reply with exactly: TURN_{i}"}],
            },
        )
        resp, notifications = await acp_proc.read_until_response(prompt_id)
        assert resp is not None, f"No response for turn {i}"
        assert resp["result"]["stopReason"] == "end_turn"

        chunks = [
            n["params"]["update"]["content"]["text"]
            for n in notifications
            if n.get("method") == "session/update"
            and n["params"]["update"].get("sessionUpdate") == "agent_message_chunk"
        ]
        assert len("".join(chunks)) > 0, f"No chunks for turn {i}"
