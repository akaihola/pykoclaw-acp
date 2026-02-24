"""Tests for WorkerPool using mock worker subprocesses."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import stat
import sys
import time
from pathlib import Path
from textwrap import dedent

import pytest

from pykoclaw_acp.worker_pool import (
    IDLE_TIMEOUT_S,
    WorkerPool,
    _WorkerHandle,
)


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


def _mock_worker_script(tmp_path: Path) -> Path:
    """Write a mock worker script that speaks the worker protocol."""
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
                # Read config (first line), ignore it
                config_line = sys.stdin.readline()
                if not config_line:
                    return

                # Send ready
                write_msg({"type": "ready"})

                # Main loop
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
                        msg_id = msg.get("id", "")
                        prompt = msg.get("prompt", "")

                        # Send heartbeat
                        write_msg({"type": "heartbeat"})

                        # Send text chunk
                        write_msg({
                            "type": "text",
                            "id": msg_id,
                            "text": f"Mock reply to: {prompt}",
                        })

                        # Send result
                        write_msg({
                            "type": "result",
                            "id": msg_id,
                            "session_id": "mock-sess-123",
                        })

            if __name__ == "__main__":
                main()
            """
        )
    )
    return script


def _crash_then_work_script(tmp_path: Path) -> Path:
    """Write a mock worker that crashes on the first query but works on the second spawn."""
    script = tmp_path / "crash_worker.py"
    state_file = tmp_path / "crash_state"
    script.write_text(
        dedent(
            f"""\
            import json
            import sys
            from pathlib import Path

            STATE_FILE = Path("{state_file}")

            def write_msg(data):
                sys.stdout.write(json.dumps(data) + "\\n")
                sys.stdout.flush()

            def main():
                config_line = sys.stdin.readline()
                if not config_line:
                    return

                write_msg({{"type": "ready"}})

                # Check if this is the first or second spawn
                first_run = not STATE_FILE.exists()
                if first_run:
                    STATE_FILE.write_text("crashed")

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
                        if first_run:
                            # Crash by exiting abruptly
                            sys.exit(1)
                        else:
                            msg_id = msg.get("id", "")
                            prompt = msg.get("prompt", "")
                            write_msg({{
                                "type": "text",
                                "id": msg_id,
                                "text": f"Recovered: {{prompt}}",
                            }})
                            write_msg({{
                                "type": "result",
                                "id": msg_id,
                                "session_id": "recovered-sess",
                            }})

            if __name__ == "__main__":
                main()
            """
        )
    )
    return script


def _make_pool(
    tmp_path: Path,
    tmp_db: sqlite3.Connection,
    script: Path,
) -> WorkerPool:
    return WorkerPool(
        db=tmp_db,
        data_dir=tmp_path,
        worker_cmd=[sys.executable, str(script)],
    )


@pytest.mark.asyncio
async def test_send_basic(tmp_path: Path, tmp_db: sqlite3.Connection) -> None:
    """Basic send: worker returns text via on_text and a session ID."""
    script = _mock_worker_script(tmp_path)
    pool = _make_pool(tmp_path, tmp_db, script)
    await pool.start()

    chunks: list[str] = []

    async def on_text(text: str) -> None:
        chunks.append(text)

    try:
        result = await pool.send("session-1", "Hello", on_text=on_text)

        assert result == "mock-sess-123"
        assert chunks == ["Mock reply to: Hello"]
        assert "session-1" in pool._entries
        assert pool._entries["session-1"].conversation_name == "acp-session-"
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_send_multiple_prompts(
    tmp_path: Path, tmp_db: sqlite3.Connection
) -> None:
    """Multiple sends on the same session reuse the same worker."""
    script = _mock_worker_script(tmp_path)
    pool = _make_pool(tmp_path, tmp_db, script)
    await pool.start()

    try:
        chunks_1: list[str] = []
        chunks_2: list[str] = []

        r1 = await pool.send(
            "sess-multi", "First", on_text=lambda t: _append(chunks_1, t)
        )
        r2 = await pool.send(
            "sess-multi", "Second", on_text=lambda t: _append(chunks_2, t)
        )

        assert r1 == "mock-sess-123"
        assert r2 == "mock-sess-123"
        assert chunks_1 == ["Mock reply to: First"]
        assert chunks_2 == ["Mock reply to: Second"]

        # Only one worker process should exist
        assert len(pool._entries) == 1
    finally:
        await pool.close()


async def _append(lst: list[str], text: str) -> None:
    lst.append(text)


@pytest.mark.asyncio
async def test_idle_eviction(tmp_path: Path, tmp_db: sqlite3.Connection) -> None:
    """Workers idle beyond timeout are evicted by the sweep loop."""
    import pykoclaw_acp.worker_pool as wp_mod

    script = _mock_worker_script(tmp_path)
    pool = _make_pool(tmp_path, tmp_db, script)

    # Send one prompt to create a worker
    await pool.send("evict-me", "hi")

    # Backdate last_used so it looks idle
    pool._entries["evict-me"].last_used = time.monotonic() - IDLE_TIMEOUT_S - 100

    orig_sweep = wp_mod.SWEEP_INTERVAL_S
    wp_mod.SWEEP_INTERVAL_S = 0.05
    try:
        await pool.start()

        for _ in range(50):
            await asyncio.sleep(0.02)
            if "evict-me" not in pool._entries:
                break

        assert "evict-me" not in pool._entries, "sweep didn't evict stale worker"
    finally:
        wp_mod.SWEEP_INTERVAL_S = orig_sweep
        await pool.close()


@pytest.mark.asyncio
async def test_worker_crash_retry(tmp_path: Path, tmp_db: sqlite3.Connection) -> None:
    """When a worker crashes mid-query, send() retries with a new worker."""
    script = _crash_then_work_script(tmp_path)
    pool = _make_pool(tmp_path, tmp_db, script)
    await pool.start()

    chunks: list[str] = []

    async def on_text(text: str) -> None:
        chunks.append(text)

    try:
        result = await pool.send("crash-sess", "test prompt", on_text=on_text)

        assert result == "recovered-sess"
        assert chunks == ["Recovered: test prompt"]
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_close_shuts_down_workers(
    tmp_path: Path, tmp_db: sqlite3.Connection
) -> None:
    """close() shuts down all worker processes."""
    script = _mock_worker_script(tmp_path)
    pool = _make_pool(tmp_path, tmp_db, script)
    await pool.start()

    # Create several workers
    await pool.send("w1", "hello")
    await pool.send("w2", "world")
    await pool.send("w3", "test")

    assert len(pool._entries) == 3

    # Grab process references before close clears entries
    processes = [h.process for h in pool._entries.values()]

    await pool.close()

    assert len(pool._entries) == 0

    # All processes should have exited
    for proc in processes:
        assert proc.returncode is not None


@pytest.mark.asyncio
async def test_resume_after_eviction(
    tmp_path: Path, tmp_db: sqlite3.Connection
) -> None:
    """After idle eviction, re-spawned worker is given the stored claude session ID.

    Regression test for: worker evicted → next prompt spawns fresh worker with
    resume_session_id=None → context lost.  Fix: _get_or_create looks up the
    stored session ID from the DB when resume_session_id is not explicitly given.
    """
    # Mock worker that echoes back the resume_session_id from its config
    script = tmp_path / "resume_echo_worker.py"
    script.write_text(
        dedent(
            """\
            import json
            import sys

            def write_msg(data):
                sys.stdout.write(json.dumps(data) + "\\n")
                sys.stdout.flush()

            def main():
                config_line = sys.stdin.readline()
                if not config_line:
                    return
                config = json.loads(config_line)
                resume_id = config.get("resume_session_id") or ""

                write_msg({"type": "ready"})

                while True:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    msg = json.loads(line.strip())
                    if msg.get("type") == "shutdown":
                        break
                    if msg.get("type") == "query":
                        msg_id = msg["id"]
                        # Echo the resume_session_id back as text
                        write_msg({"type": "text", "id": msg_id, "text": resume_id})
                        write_msg({"type": "result", "id": msg_id, "session_id": "new-sess-xyz"})

            if __name__ == "__main__":
                main()
            """
        )
    )

    pool = _make_pool(tmp_path, tmp_db, script)

    # Prime the DB: simulate that the conversation already has a stored session ID
    # (as if the worker had run before and written it via upsert_conversation).
    conv_name = "acp-evict-re"  # first 8 chars of "evict-resume-test"
    tmp_db.execute(
        "INSERT INTO conversations (name, session_id, cwd, created_at) VALUES (?, ?, ?, ?)",
        (conv_name, "stored-claude-sess-abc", "/tmp", "2026-01-01T00:00:00"),
    )
    tmp_db.commit()

    session_id = "evict-resume-test"

    # First send: worker is spawned with resume_session_id passed explicitly
    chunks_1: list[str] = []

    async def collect_1(text: str) -> None:
        chunks_1.append(text)

    await pool.send(
        session_id,
        "first",
        on_text=collect_1,
        resume_session_id="stored-claude-sess-abc",
    )
    # Worker echoes back the resume ID it received
    assert chunks_1 == ["stored-claude-sess-abc"]

    # Simulate eviction: remove entry from pool (kill the process cleanly)
    handle = pool._entries.pop(session_id)
    handle.process.kill()
    await handle.process.wait()

    # Second send: no explicit resume_session_id — should be looked up from DB
    chunks_2: list[str] = []

    async def collect_2(text: str) -> None:
        chunks_2.append(text)

    await pool.send(
        session_id,
        "after eviction",
        on_text=collect_2,
    )

    # The re-spawned worker must have received the stored session ID, not empty
    assert chunks_2 == ["stored-claude-sess-abc"], (
        f"Expected resume with 'stored-claude-sess-abc', got: {chunks_2!r}. "
        "Worker was spawned without resume_session_id after eviction."
    )

    await pool.close()


@pytest.mark.asyncio
async def test_concurrent_sessions(tmp_path: Path, tmp_db: sqlite3.Connection) -> None:
    """Multiple sessions each get their own worker."""
    script = _mock_worker_script(tmp_path)
    pool = _make_pool(tmp_path, tmp_db, script)
    await pool.start()

    try:
        results = await asyncio.gather(
            pool.send("s1", "msg1"),
            pool.send("s2", "msg2"),
            pool.send("s3", "msg3"),
        )

        assert all(r == "mock-sess-123" for r in results)
        assert len(pool._entries) == 3

        # Each should have a unique conversation name
        names = {h.conversation_name for h in pool._entries.values()}
        assert len(names) == 3
    finally:
        await pool.close()
