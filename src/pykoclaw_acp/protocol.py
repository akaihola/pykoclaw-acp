"""JSON-RPC 2.0 protocol handler for ACP."""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)


class JsonRpcError:
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    INVALID_SESSION = -32000
    SESSION_ERROR = -32001


class AcpProtocolHandler:
    """Newline-delimited JSON-RPC 2.0 message parsing and formatting."""

    @staticmethod
    def parse_message(line: str) -> dict[str, Any] | None:
        try:
            msg = json.loads(line.strip())
        except json.JSONDecodeError as exc:
            log.error("JSON parse error: %s", exc)
            return None
        except Exception as exc:
            log.error("Unexpected error parsing message: %s", exc)
            return None

        if not isinstance(msg, dict):
            return None
        if msg.get("jsonrpc") != "2.0":
            log.warning("Invalid JSON-RPC version")
            return None
        if "method" not in msg and "result" not in msg and "error" not in msg:
            return None
        return msg

    @staticmethod
    def format_response(msg_id: Any, result: Any) -> str:
        return json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result}) + "\n"

    @staticmethod
    def format_notification(method: str, params: dict[str, Any]) -> str:
        return json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n"

    @staticmethod
    def format_error(msg_id: Any, code: int, message: str, data: Any = None) -> str:
        error_obj: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error_obj["data"] = data
        return json.dumps({"jsonrpc": "2.0", "id": msg_id, "error": error_obj}) + "\n"
