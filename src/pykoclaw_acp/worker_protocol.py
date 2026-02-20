"""Worker subprocess protocol — JSON-newline over stdin/stdout pipes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal


# -- Server → Worker messages ------------------------------------------------


@dataclass
class QueryMessage:
    id: str = ""
    prompt: str = ""
    type: Literal["query"] = "query"


@dataclass
class ShutdownMessage:
    type: Literal["shutdown"] = "shutdown"


# -- Worker → Server messages ------------------------------------------------


@dataclass
class ReadyMessage:
    type: Literal["ready"] = "ready"


@dataclass
class TextChunkMessage:
    id: str = ""
    text: str = ""
    type: Literal["text"] = "text"


@dataclass
class WorkerResultMessage:
    """Named WorkerResultMessage to avoid clash with claude_agent_sdk.ResultMessage."""

    id: str = ""
    session_id: str = ""
    type: Literal["result"] = "result"


@dataclass
class ErrorMessage:
    id: str = ""
    error: str = ""
    type: Literal["error"] = "error"


@dataclass
class HeartbeatMessage:
    type: Literal["heartbeat"] = "heartbeat"


# -- Initial configuration ---------------------------------------------------


@dataclass
class WorkerConfig:
    cwd: str = ""
    model: str = ""
    conversation_name: str = ""
    db_path: str = ""
    cli_path: str | None = None
    allowed_tools: list[str] = field(default_factory=list)


# -- Type aliases -------------------------------------------------------------

ServerMessage = QueryMessage | ShutdownMessage
WorkerMessage = (
    ReadyMessage | TextChunkMessage | WorkerResultMessage | ErrorMessage | HeartbeatMessage
)


# -- Encode / decode ----------------------------------------------------------


def encode(msg: WorkerConfig | ServerMessage | WorkerMessage) -> str:
    """Serialize a protocol message to a JSON line (with trailing newline)."""
    return json.dumps(asdict(msg)) + "\n"


def decode_server_message(line: str) -> ServerMessage:
    """Decode a JSON line into a server→worker message."""
    data = json.loads(line.strip())
    match data.get("type"):
        case "query":
            return QueryMessage(id=data.get("id", ""), prompt=data.get("prompt", ""))
        case "shutdown":
            return ShutdownMessage()
        case _:
            raise ValueError(f"Unknown server message type: {data.get('type')}")


def decode_worker_message(line: str) -> WorkerMessage:
    """Decode a JSON line into a worker→server message."""
    data = json.loads(line.strip())
    match data.get("type"):
        case "ready":
            return ReadyMessage()
        case "text":
            return TextChunkMessage(id=data.get("id", ""), text=data.get("text", ""))
        case "result":
            return WorkerResultMessage(
                id=data.get("id", ""), session_id=data.get("session_id", "")
            )
        case "error":
            return ErrorMessage(id=data.get("id", ""), error=data.get("error", ""))
        case "heartbeat":
            return HeartbeatMessage()
        case _:
            raise ValueError(f"Unknown worker message type: {data.get('type')}")


def decode_config(line: str) -> WorkerConfig:
    """Decode the initial configuration line."""
    data = json.loads(line.strip())
    return WorkerConfig(
        cwd=data.get("cwd", ""),
        model=data.get("model", ""),
        conversation_name=data.get("conversation_name", ""),
        db_path=data.get("db_path", ""),
        cli_path=data.get("cli_path"),
        allowed_tools=data.get("allowed_tools", []),
    )
