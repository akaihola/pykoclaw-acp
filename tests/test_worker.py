"""Tests for the SDK worker subprocess."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from pykoclaw_acp.worker import _handle_query
from pykoclaw_acp.worker_protocol import (
    ErrorMessage,
    QueryMessage,
    ReadyMessage,
    ShutdownMessage,
    TextChunkMessage,
    WorkerConfig,
    WorkerResultMessage,
    decode_worker_message,
    encode,
)


# ---------------------------------------------------------------------------
# Fake client stub (same pattern as test_client_pool.py / test_sdk_consume.py)
# ---------------------------------------------------------------------------


@dataclass
class FakeClient:
    """Stub ClaudeSDKClient whose receive_response yields canned messages."""

    messages: list[Any] = field(default_factory=list)
    _queried: str | None = None

    async def query(self, prompt: str) -> None:
        self._queried = prompt

    async def receive_response(self):  # noqa: ANN201
        for msg in self.messages:
            yield msg


class ErrorClient:
    """Stub that raises on query()."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def query(self, prompt: str) -> None:
        raise self._exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_msg(
    session_id: str = "sess-1",
    result: str = "",
) -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=100,
        duration_api_ms=80,
        is_error=False,
        num_turns=1,
        session_id=session_id,
        result=result,
    )


def _assistant_msg(*texts: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=t) for t in texts],
        model="test",
    )


def _fake_db() -> MagicMock:
    """Create a mock DB connection."""
    db = MagicMock()
    return db


# ---------------------------------------------------------------------------
# Tests for _handle_query
# ---------------------------------------------------------------------------


class TestHandleQuery:
    """Tests for the extracted _handle_query function."""

    @pytest.mark.asyncio
    async def test_normal_text_flow(self) -> None:
        """TextBlock messages produce TextChunkMessages, ResultMessage produces
        WorkerResultMessage."""
        client = FakeClient(
            messages=[
                _assistant_msg("Hello", " World"),
                _result_msg(session_id="sess-ok", result="Hello World"),
            ]
        )
        db = _fake_db()
        written: list[Any] = []

        await _handle_query(
            client,  # type: ignore[arg-type]
            QueryMessage(id="q-1", prompt="hi"),
            db=db,
            conversation_name="acp-test",
            cwd="/tmp/work",
            write_msg=written.append,
        )

        # Should have two text chunks + one result
        text_msgs = [m for m in written if isinstance(m, TextChunkMessage)]
        result_msgs = [m for m in written if isinstance(m, WorkerResultMessage)]

        assert len(text_msgs) == 2
        assert text_msgs[0].text == "Hello"
        assert text_msgs[0].id == "q-1"
        assert text_msgs[1].text == " World"
        assert len(result_msgs) == 1
        assert result_msgs[0].session_id == "sess-ok"
        assert result_msgs[0].id == "q-1"

    @pytest.mark.asyncio
    async def test_fallback_when_no_text_blocks(self) -> None:
        """When no TextBlocks are streamed, ResultMessage.result is forwarded
        as a TextChunkMessage (via consume_sdk_response fallback)."""
        client = FakeClient(
            messages=[_result_msg(session_id="sess-fb", result="Fallback reply")]
        )
        db = _fake_db()
        written: list[Any] = []

        await _handle_query(
            client,  # type: ignore[arg-type]
            QueryMessage(id="q-2", prompt="hello"),
            db=db,
            conversation_name="acp-test",
            cwd="/tmp/work",
            write_msg=written.append,
        )

        text_msgs = [m for m in written if isinstance(m, TextChunkMessage)]
        result_msgs = [m for m in written if isinstance(m, WorkerResultMessage)]

        assert len(text_msgs) == 1
        assert text_msgs[0].text == "Fallback reply"
        assert len(result_msgs) == 1
        assert result_msgs[0].session_id == "sess-fb"

    @pytest.mark.asyncio
    async def test_error_produces_error_message(self) -> None:
        """When the client raises an exception, an ErrorMessage is written."""
        client = ErrorClient(RuntimeError("SDK exploded"))
        db = _fake_db()
        written: list[Any] = []

        await _handle_query(
            client,  # type: ignore[arg-type]
            QueryMessage(id="q-err", prompt="boom"),
            db=db,
            conversation_name="acp-test",
            cwd="/tmp/work",
            write_msg=written.append,
        )

        assert len(written) == 1
        assert isinstance(written[0], ErrorMessage)
        assert written[0].id == "q-err"
        assert "SDK exploded" in written[0].error

    @pytest.mark.asyncio
    async def test_upsert_conversation_called_on_result(self) -> None:
        """upsert_conversation must be called when a ResultMessage arrives."""
        client = FakeClient(
            messages=[_result_msg(session_id="sess-upsert", result="done")]
        )
        db = _fake_db()
        written: list[Any] = []

        with patch("pykoclaw_acp.worker.upsert_conversation") as mock_upsert:
            await _handle_query(
                client,  # type: ignore[arg-type]
                QueryMessage(id="q-up", prompt="test"),
                db=db,
                conversation_name="acp-conv",
                cwd="/tmp/cwd",
                write_msg=written.append,
            )

            mock_upsert.assert_called_once_with(
                db, "acp-conv", "sess-upsert", "/tmp/cwd"
            )

    @pytest.mark.asyncio
    async def test_empty_stream_sends_empty_result(self) -> None:
        """When the stream has no messages at all, an empty
        WorkerResultMessage is sent."""
        client = FakeClient(messages=[])
        db = _fake_db()
        written: list[Any] = []

        await _handle_query(
            client,  # type: ignore[arg-type]
            QueryMessage(id="q-empty", prompt="nothing"),
            db=db,
            conversation_name="acp-test",
            cwd="/tmp/work",
            write_msg=written.append,
        )

        assert len(written) == 1
        assert isinstance(written[0], WorkerResultMessage)
        assert written[0].id == "q-empty"
        assert written[0].session_id == ""

    @pytest.mark.asyncio
    async def test_no_duplication_with_text_and_result(self) -> None:
        """When TextBlocks are streamed, ResultMessage.result must NOT
        produce a duplicate TextChunkMessage."""
        client = FakeClient(
            messages=[
                _assistant_msg("Streamed"),
                _result_msg(session_id="sess-dup", result="Streamed"),
            ]
        )
        db = _fake_db()
        written: list[Any] = []

        await _handle_query(
            client,  # type: ignore[arg-type]
            QueryMessage(id="q-dup", prompt="hi"),
            db=db,
            conversation_name="acp-test",
            cwd="/tmp/work",
            write_msg=written.append,
        )

        text_msgs = [m for m in written if isinstance(m, TextChunkMessage)]
        assert len(text_msgs) == 1
        assert text_msgs[0].text == "Streamed"


# ---------------------------------------------------------------------------
# End-to-end subprocess protocol test via mock worker script
# ---------------------------------------------------------------------------


def _create_mock_worker_script(tmp_path: Path) -> Path:
    """Write a tiny Python script that speaks the worker protocol without
    needing the real Claude SDK.  It reads config, sends ready, processes
    queries with canned responses, and handles shutdown."""
    script = tmp_path / "mock_worker.py"
    script.write_text(
        dedent(
            """\
            import json
            import sys

            def write_msg(data):
                sys.stdout.write(json.dumps(data) + "\\n")
                sys.stdout.flush()

            def main():
                # Read config line
                config_line = sys.stdin.readline()
                if not config_line:
                    return
                config = json.loads(config_line.strip())

                # Send ready
                write_msg({"type": "ready"})

                # Message loop
                while True:
                    line = sys.stdin.readline()
                    if not line:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    msg = json.loads(line)

                    if msg.get("type") == "shutdown":
                        break

                    if msg.get("type") == "query":
                        query_id = msg.get("id", "")
                        prompt = msg.get("prompt", "")
                        # Send a text chunk echoing the prompt
                        write_msg({
                            "type": "text",
                            "id": query_id,
                            "text": f"Echo: {prompt}",
                        })
                        # Send result
                        write_msg({
                            "type": "result",
                            "id": query_id,
                            "session_id": "mock-session-001",
                        })

            if __name__ == "__main__":
                main()
            """
        )
    )
    return script


class TestSubprocessProtocol:
    """End-to-end tests: spawn a mock worker as a subprocess and verify
    the protocol flow over real pipes."""

    @pytest.mark.asyncio
    async def test_ready_and_query_response(self, tmp_path: Path) -> None:
        """Worker sends ready, then responds to a query with text + result."""
        script = _create_mock_worker_script(tmp_path)

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None

        # Send config
        config = WorkerConfig(
            cwd="/tmp",
            model="test-model",
            conversation_name="acp-test",
            db_path="/tmp/test.db",
        )
        proc.stdin.write(encode(config).encode())
        await proc.stdin.drain()

        # Read ready message
        ready_line = await asyncio.wait_for(proc.stdout.readline(), timeout=5)
        ready_msg = decode_worker_message(ready_line.decode())
        assert isinstance(ready_msg, ReadyMessage)

        # Send a query
        query = QueryMessage(id="q-e2e-1", prompt="Hello world")
        proc.stdin.write(encode(query).encode())
        await proc.stdin.drain()

        # Read text chunk
        text_line = await asyncio.wait_for(proc.stdout.readline(), timeout=5)
        text_msg = decode_worker_message(text_line.decode())
        assert isinstance(text_msg, TextChunkMessage)
        assert text_msg.id == "q-e2e-1"
        assert text_msg.text == "Echo: Hello world"

        # Read result
        result_line = await asyncio.wait_for(proc.stdout.readline(), timeout=5)
        result_msg = decode_worker_message(result_line.decode())
        assert isinstance(result_msg, WorkerResultMessage)
        assert result_msg.id == "q-e2e-1"
        assert result_msg.session_id == "mock-session-001"

        # Send shutdown
        shutdown = ShutdownMessage()
        proc.stdin.write(encode(shutdown).encode())
        await proc.stdin.drain()

        # Wait for clean exit
        await asyncio.wait_for(proc.wait(), timeout=5)
        assert proc.returncode == 0

    @pytest.mark.asyncio
    async def test_multiple_queries(self, tmp_path: Path) -> None:
        """Worker handles multiple sequential queries correctly."""
        script = _create_mock_worker_script(tmp_path)

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None

        # Send config
        config = WorkerConfig(
            cwd="/tmp",
            model="test-model",
            conversation_name="acp-multi",
            db_path="/tmp/test.db",
        )
        proc.stdin.write(encode(config).encode())
        await proc.stdin.drain()

        # Read ready
        ready_line = await asyncio.wait_for(proc.stdout.readline(), timeout=5)
        assert isinstance(decode_worker_message(ready_line.decode()), ReadyMessage)

        # Send two queries back-to-back
        for i in range(2):
            query = QueryMessage(id=f"q-multi-{i}", prompt=f"Query {i}")
            proc.stdin.write(encode(query).encode())
            await proc.stdin.drain()

            # Read text + result for each query
            text_line = await asyncio.wait_for(proc.stdout.readline(), timeout=5)
            text_msg = decode_worker_message(text_line.decode())
            assert isinstance(text_msg, TextChunkMessage)
            assert text_msg.id == f"q-multi-{i}"

            result_line = await asyncio.wait_for(proc.stdout.readline(), timeout=5)
            result_msg = decode_worker_message(result_line.decode())
            assert isinstance(result_msg, WorkerResultMessage)
            assert result_msg.id == f"q-multi-{i}"

        # Clean shutdown
        proc.stdin.write(encode(ShutdownMessage()).encode())
        await proc.stdin.drain()
        await asyncio.wait_for(proc.wait(), timeout=5)
        assert proc.returncode == 0

    @pytest.mark.asyncio
    async def test_stdin_eof_causes_exit(self, tmp_path: Path) -> None:
        """Closing stdin should cause the worker to exit cleanly."""
        script = _create_mock_worker_script(tmp_path)

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None

        # Send config
        config = WorkerConfig(
            cwd="/tmp",
            model="test-model",
            conversation_name="acp-eof",
            db_path="/tmp/test.db",
        )
        proc.stdin.write(encode(config).encode())
        await proc.stdin.drain()

        # Read ready
        ready_line = await asyncio.wait_for(proc.stdout.readline(), timeout=5)
        assert isinstance(decode_worker_message(ready_line.decode()), ReadyMessage)

        # Close stdin — worker should exit
        proc.stdin.close()
        await asyncio.wait_for(proc.wait(), timeout=5)
        assert proc.returncode == 0
