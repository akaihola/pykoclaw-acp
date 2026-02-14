"""Tests for the JSON-RPC 2.0 protocol handler."""

from __future__ import annotations

import json

from pykoclaw_acp.protocol import AcpProtocolHandler, JsonRpcError


class TestParseMessage:
    def test_valid_request(self) -> None:
        line = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )
        msg = AcpProtocolHandler.parse_message(line)
        assert msg is not None
        assert msg["method"] == "initialize"
        assert msg["id"] == 1

    def test_valid_notification(self) -> None:
        line = json.dumps(
            {"jsonrpc": "2.0", "method": "session/update", "params": {"x": 1}}
        )
        msg = AcpProtocolHandler.parse_message(line)
        assert msg is not None
        assert "id" not in msg

    def test_valid_response(self) -> None:
        line = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}})
        msg = AcpProtocolHandler.parse_message(line)
        assert msg is not None
        assert msg["result"] == {"ok": True}

    def test_invalid_json(self) -> None:
        assert AcpProtocolHandler.parse_message("not json") is None

    def test_wrong_version(self) -> None:
        line = json.dumps({"jsonrpc": "1.0", "method": "x", "params": {}})
        assert AcpProtocolHandler.parse_message(line) is None

    def test_missing_method_and_result(self) -> None:
        line = json.dumps({"jsonrpc": "2.0", "id": 1})
        assert AcpProtocolHandler.parse_message(line) is None

    def test_not_a_dict(self) -> None:
        assert AcpProtocolHandler.parse_message("[1,2,3]") is None

    def test_strips_whitespace(self) -> None:
        line = '  {"jsonrpc":"2.0","method":"x","params":{}}  \n'
        assert AcpProtocolHandler.parse_message(line) is not None


class TestFormatResponse:
    def test_structure(self) -> None:
        raw = AcpProtocolHandler.format_response(42, {"ok": True})
        assert raw.endswith("\n")
        msg = json.loads(raw)
        assert msg == {"jsonrpc": "2.0", "id": 42, "result": {"ok": True}}


class TestFormatNotification:
    def test_no_id_field(self) -> None:
        raw = AcpProtocolHandler.format_notification("session/update", {"s": "1"})
        msg = json.loads(raw)
        assert "id" not in msg
        assert msg["method"] == "session/update"
        assert msg["params"] == {"s": "1"}


class TestFormatError:
    def test_standard_error(self) -> None:
        raw = AcpProtocolHandler.format_error(1, JsonRpcError.METHOD_NOT_FOUND, "nope")
        msg = json.loads(raw)
        assert msg["error"]["code"] == -32601
        assert msg["error"]["message"] == "nope"
        assert "data" not in msg["error"]

    def test_error_with_data(self) -> None:
        raw = AcpProtocolHandler.format_error(
            None, JsonRpcError.INTERNAL_ERROR, "boom", data={"detail": "x"}
        )
        msg = json.loads(raw)
        assert msg["id"] is None
        assert msg["error"]["data"] == {"detail": "x"}

    def test_all_error_codes_are_negative(self) -> None:
        for attr in dir(JsonRpcError):
            if attr.startswith("_"):
                continue
            val = getattr(JsonRpcError, attr)
            assert isinstance(val, int)
            assert val < 0, f"{attr} should be negative"
