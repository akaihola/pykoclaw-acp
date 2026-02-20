"""Tests for the worker subprocess protocol."""

from __future__ import annotations

import json

import pytest

from pykoclaw_acp.worker_protocol import (
    ErrorMessage,
    HeartbeatMessage,
    QueryMessage,
    ReadyMessage,
    ShutdownMessage,
    TextChunkMessage,
    WorkerConfig,
    WorkerResultMessage,
    decode_config,
    decode_server_message,
    decode_worker_message,
    encode,
)


# -- Round-trip: server messages ----------------------------------------------


class TestServerMessageRoundTrip:
    def test_query_message(self) -> None:
        original = QueryMessage(id="q-1", prompt="hello")
        decoded = decode_server_message(encode(original))
        assert decoded == original

    def test_shutdown_message(self) -> None:
        original = ShutdownMessage()
        decoded = decode_server_message(encode(original))
        assert decoded == original


# -- Round-trip: worker messages ----------------------------------------------


class TestWorkerMessageRoundTrip:
    def test_ready_message(self) -> None:
        original = ReadyMessage()
        decoded = decode_worker_message(encode(original))
        assert decoded == original

    def test_text_chunk_message(self) -> None:
        original = TextChunkMessage(id="q-1", text="some streamed text")
        decoded = decode_worker_message(encode(original))
        assert decoded == original

    def test_worker_result_message(self) -> None:
        original = WorkerResultMessage(id="q-1", session_id="sess-abc")
        decoded = decode_worker_message(encode(original))
        assert decoded == original

    def test_error_message(self) -> None:
        original = ErrorMessage(id="q-1", error="something broke")
        decoded = decode_worker_message(encode(original))
        assert decoded == original

    def test_heartbeat_message(self) -> None:
        original = HeartbeatMessage()
        decoded = decode_worker_message(encode(original))
        assert decoded == original


# -- Round-trip: WorkerConfig -------------------------------------------------


class TestWorkerConfigRoundTrip:
    def test_defaults(self) -> None:
        original = WorkerConfig()
        decoded = decode_config(encode(original))
        assert decoded == original
        assert decoded.cli_path is None
        assert decoded.allowed_tools == []

    def test_all_fields_populated(self) -> None:
        original = WorkerConfig(
            cwd="/tmp/work",
            model="claude-sonnet-4-20250514",
            conversation_name="acp-abc123",
            db_path="/data/pykoclaw.db",
            cli_path="/usr/local/bin/claude",
            allowed_tools=["Bash", "Read", "Write"],
        )
        decoded = decode_config(encode(original))
        assert decoded == original


# -- Field correctness --------------------------------------------------------


class TestFieldCorrectness:
    def test_query_fields(self) -> None:
        msg = decode_server_message('{"type":"query","id":"x","prompt":"hi"}')
        assert isinstance(msg, QueryMessage)
        assert msg.id == "x"
        assert msg.prompt == "hi"

    def test_text_chunk_fields(self) -> None:
        msg = decode_worker_message('{"type":"text","id":"t1","text":"chunk"}')
        assert isinstance(msg, TextChunkMessage)
        assert msg.id == "t1"
        assert msg.text == "chunk"

    def test_result_fields(self) -> None:
        msg = decode_worker_message('{"type":"result","id":"r1","session_id":"s1"}')
        assert isinstance(msg, WorkerResultMessage)
        assert msg.id == "r1"
        assert msg.session_id == "s1"

    def test_error_fields(self) -> None:
        msg = decode_worker_message('{"type":"error","id":"e1","error":"oops"}')
        assert isinstance(msg, ErrorMessage)
        assert msg.id == "e1"
        assert msg.error == "oops"


# -- Encoding format ----------------------------------------------------------


class TestEncodeFormat:
    def test_trailing_newline(self) -> None:
        for msg in [
            ReadyMessage(),
            QueryMessage(id="1", prompt="p"),
            ShutdownMessage(),
            TextChunkMessage(id="1", text="t"),
            WorkerResultMessage(id="1", session_id="s"),
            ErrorMessage(id="1", error="e"),
            HeartbeatMessage(),
            WorkerConfig(),
        ]:
            encoded = encode(msg)
            assert encoded.endswith("\n"), f"{type(msg).__name__} missing trailing newline"
            # Only one newline (it's a single JSON line).
            assert encoded.count("\n") == 1

    def test_valid_json(self) -> None:
        encoded = encode(QueryMessage(id="q", prompt="hello"))
        data = json.loads(encoded)
        assert data["type"] == "query"
        assert data["id"] == "q"
        assert data["prompt"] == "hello"


# -- Error handling -----------------------------------------------------------


class TestDecodeErrors:
    def test_unknown_server_message_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown server message type"):
            decode_server_message('{"type":"bogus"}')

    def test_unknown_worker_message_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown worker message type"):
            decode_worker_message('{"type":"bogus"}')

    def test_missing_type_server(self) -> None:
        with pytest.raises(ValueError, match="Unknown server message type"):
            decode_server_message('{"id":"1"}')

    def test_missing_type_worker(self) -> None:
        with pytest.raises(ValueError, match="Unknown worker message type"):
            decode_worker_message('{"id":"1"}')

    def test_invalid_json_server(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            decode_server_message("not json at all")

    def test_invalid_json_worker(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            decode_worker_message("not json at all")

    def test_invalid_json_config(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            decode_config("{broken")
