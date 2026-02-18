"""Tests for the ACP healthcheck command."""

from __future__ import annotations

import json
import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pykoclaw_acp import _run_healthcheck


def _mock_popen_communicate(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> MagicMock:
    """Return a Popen mock whose stdout pipe delivers *stdout* via os.read."""
    proc = MagicMock(spec=subprocess.Popen)
    proc.returncode = returncode
    proc.terminate = MagicMock()
    proc.kill = MagicMock()
    proc.wait = MagicMock()

    # stdin: needs write/flush/close
    proc.stdin = MagicMock()

    # stdout: use a real pipe so selectors work
    read_fd, write_fd = os.pipe()
    os.write(write_fd, (stdout + "\n").encode() if stdout else b"")
    os.close(write_fd)
    proc.stdout = os.fdopen(read_fd, "r")

    # stderr
    proc.stderr = MagicMock()
    proc.stderr.read.return_value = stderr

    return proc


def test_healthcheck_success() -> None:
    response = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": 1,
            "agentCapabilities": {},
            "agentInfo": {"name": "pykoclaw", "version": "0.1.0"},
        },
    })

    proc = _mock_popen_communicate(stdout=response)
    with (
        patch("pykoclaw_acp.subprocess.Popen", return_value=proc),
        pytest.raises(SystemExit) as exc_info,
    ):
        _run_healthcheck("pykoclaw")

    assert exc_info.value.code == 0


def test_healthcheck_bad_response() -> None:
    response = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}})

    proc = _mock_popen_communicate(stdout=response)
    with (
        patch("pykoclaw_acp.subprocess.Popen", return_value=proc),
        pytest.raises(SystemExit) as exc_info,
    ):
        _run_healthcheck("pykoclaw")

    assert exc_info.value.code == 1


def test_healthcheck_no_response() -> None:
    proc = _mock_popen_communicate(stdout="")
    with (
        patch("pykoclaw_acp._HEALTHCHECK_TIMEOUT_S", 0.1),
        patch("pykoclaw_acp.subprocess.Popen", return_value=proc),
        pytest.raises(SystemExit) as exc_info,
    ):
        _run_healthcheck("pykoclaw")

    assert exc_info.value.code == 1


def test_healthcheck_command_not_found() -> None:
    with (
        patch(
            "pykoclaw_acp.subprocess.Popen",
            side_effect=FileNotFoundError("not found"),
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        _run_healthcheck("nonexistent-binary")

    assert exc_info.value.code == 1


def test_healthcheck_invalid_json() -> None:
    proc = _mock_popen_communicate(stdout="not json at all")
    with (
        patch("pykoclaw_acp.subprocess.Popen", return_value=proc),
        pytest.raises(SystemExit) as exc_info,
    ):
        _run_healthcheck("pykoclaw")

    assert exc_info.value.code == 1
