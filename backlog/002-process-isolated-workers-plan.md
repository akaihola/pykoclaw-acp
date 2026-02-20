# Implementation Plan: Process-Isolated SDK Workers

**Backlog ref:** [001-acp-architecture-fragility.md]
**Status:** Done
**Created:** 2026-02-21
**Feature branch:** `feature/process-isolated-workers`

## Goal

Move the Claude SDK (anyio) out of the ACP server process (asyncio)
entirely. Each SDK session runs in a dedicated **worker subprocess**,
communicating with the ACP server over a Unix socket or pipe using a
lightweight JSON protocol. The ACP server becomes a pure asyncio
router/proxy — boring, reliable, and immune to anyio cancel scope leaks.

## Target architecture

```
Mitto (Go) ─── JSON-RPC/stdio ───► AcpServer (pure asyncio)
                                       │
                                       ├── WorkerProcess 1 (anyio + SDK)
                                       │    └── ClaudeSDKClient → claude CLI
                                       ├── WorkerProcess 2 (anyio + SDK)
                                       │    └── ClaudeSDKClient → claude CLI
                                       └── ...
```

**Key properties:**
- ACP server: zero SDK imports, zero anyio dependency, pure asyncio
- Workers: own their entire anyio stack, die in isolation
- Communication: JSON-newline over pipes (stdin/stdout of worker process)
- One worker per ACP session (1:1 mapping, same as current ClientPool entries)

## What this eliminates

| Current problem                                 | After isolation                              |
| ------------------------------------------------ | -------------------------------------------- |
| anyio cancel scope leaks into asyncio event loop | Impossible — different process               |
| `_kill_client()` hack bypassing `disconnect()`   | Worker calls `disconnect()` safely in anyio  |
| `os._exit()` forced shutdown                     | Worker exits normally; server `waitpid()`s   |
| Watchdog SIGKILL → zombie chain                  | Worker crash ≠ server crash                  |
| SDK crash kills all sessions                     | Only one session affected                    |
| Two SDK message loops (agent_core + client_pool) | Shared `consume_sdk_response()` function     |

## Implementation phases

### Phase 0: Unify SDK message consumption (pykoclaw core)

**Repo:** `pykoclaw` (core)
**Files:** `pykoclaw/src/pykoclaw/sdk_consume.py` (new),
           `pykoclaw/src/pykoclaw/agent_core.py` (refactor)

Extract the "iterate SDK responses and collect text" logic into a shared
function that both `query_agent()` and the new worker script can call.
This removes the two-SDK-message-loops design smell flagged in the
architecture review and prevents the entire class of bugs where a fix
must be applied in three repos.

#### 0.1 — Create `sdk_consume.py`

```python
"""Single source of truth for consuming Claude SDK response streams."""

from collections.abc import Awaitable, Callable

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)


async def consume_sdk_response(
    client: ClaudeSDKClient,
    *,
    on_text: Callable[[str], Awaitable[None]] | None = None,
    on_result: Callable[[ResultMessage], Awaitable[None]] | None = None,
) -> ResultMessage | None:
    """Consume all messages from a ClaudeSDKClient response stream.

    Iterates ``client.receive_response()``, dispatches text blocks via
    *on_text*, and returns the final ``ResultMessage`` (if any).

    The ``ResultMessage.result`` field is forwarded via *on_text* as a
    fallback when no ``TextBlock`` was emitted — this prevents the
    "empty reply" bug (Issue #4) by construction.
    """
    had_text_blocks = False
    result_message: ResultMessage | None = None

    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock) and block.text:
                    had_text_blocks = True
                    if on_text:
                        await on_text(block.text)
        elif isinstance(message, ResultMessage):
            result_message = message
            if not had_text_blocks and message.result and on_text:
                await on_text(message.result)
            if on_result:
                await on_result(message)

    return result_message
```

#### 0.2 — Refactor `query_agent()` to use `consume_sdk_response()`

Replace the inline message iteration loop in `query_agent()` with a call
to `consume_sdk_response()`, yielding `AgentMessage` objects from the
callbacks. This ensures `query_agent()` (WhatsApp/scheduler path) and the
worker (ACP path) share identical consumption logic.

#### 0.3 — Tests

- Unit test `consume_sdk_response()` with a mock `ClaudeSDKClient` that
  yields various message sequences (text-only, result-only, mixed,
  empty).
- Verify existing `query_agent()` tests still pass after refactor.
- Regression test: `ResultMessage.result` fallback fires when no
  `TextBlock` is emitted.

---

### Phase 1: Worker subprocess protocol

**Repo:** `pykoclaw-acp`
**Files:** `pykoclaw_acp/worker_protocol.py` (new)

Define the JSON-newline protocol between the ACP server and worker
processes. Keep it minimal — just enough to proxy prompts and stream
responses.

#### Protocol messages (server → worker)

```json
{"type": "query", "id": "msg-001", "prompt": "Hello"}
{"type": "shutdown"}
```

#### Protocol messages (worker → server)

```json
{"type": "text", "id": "msg-001", "text": "Here is my response..."}
{"type": "result", "id": "msg-001", "session_id": "sess-abc123"}
{"type": "error", "id": "msg-001", "error": "Something went wrong"}
{"type": "ready"}
{"type": "heartbeat"}
```

#### Message semantics

| Message       | Direction        | Meaning                                               |
| ------------- | ---------------- | ----------------------------------------------------- |
| `ready`       | worker → server  | Worker has connected the SDK client and is ready       |
| `query`       | server → worker  | Send a prompt to the SDK client                       |
| `text`        | worker → server  | Streaming text chunk (from `on_text` callback)         |
| `result`      | worker → server  | Query complete, carries `session_id`                   |
| `error`       | worker → server  | Query failed                                           |
| `heartbeat`   | worker → server  | Periodic liveness signal                               |
| `shutdown`    | server → worker  | Graceful shutdown request                              |

#### Design decisions

- **JSON-newline over stdin/stdout pipes:** simplest possible IPC. No
  external dependencies. Pipes are already how we talk to Mitto.
- **One outstanding query at a time:** workers are single-session, and the
  current `_Entry.lock` already serializes queries per session. No need
  for concurrent query support.
- **`id` field on query/text/result/error:** correlates responses with
  requests. Future-proofs for potential pipelining without adding
  complexity now.
- **`heartbeat`:** replaces the watchdog's role for per-worker
  monitoring. Worker sends periodic heartbeats; if the server stops
  receiving them, the worker is assumed stuck and gets killed.

#### Implementation

- Pydantic models for each message type (or plain `TypedDict`s — stay
  light).
- `encode(msg) -> str` and `decode(line) -> WorkerMessage` functions.
- Thorough unit tests for serialization round-trips.

---

### Phase 2: Worker subprocess script

**Repo:** `pykoclaw-acp`
**Files:** `pykoclaw_acp/worker.py` (new)

A standalone async script that:

1. Receives configuration (JSON blob) on stdin as the first line
2. Creates a `ClaudeSDKClient` with that configuration
3. Calls `client.connect()`
4. Sends `{"type": "ready"}` on stdout
5. Enters a loop: read `query` messages from stdin, call
   `client.query()` + `consume_sdk_response()`, write `text`/`result`
   messages to stdout
6. On `shutdown` message or stdin EOF: call `client.disconnect()` (safe
   because we're in anyio's own process), exit cleanly
7. Periodic `heartbeat` messages on a background task

#### Key design points

- **Runs with `anyio`** — the worker can use `anyio.run()` or
  `asyncio.run()` with anyio tasks freely because there's no foreign
  async runtime to conflict with.
- **`client.disconnect()` is safe here** — the anyio cancel scope leak
  only matters when anyio and asyncio share a process. In the worker,
  anyio owns everything.
- **Crash = process exit** — if the SDK crashes, the worker dies. The
  server detects this via pipe EOF and handles retry.
- **No pykoclaw-acp imports** — the worker imports from `pykoclaw`
  (core) and `claude_agent_sdk` only. It must not import server code.
  The protocol module is the only shared code.
- **Stderr** forwarded to server's log (or to a per-worker log file).

#### Configuration blob (first stdin line)

```json
{
    "cwd": "/path/to/conversation/dir",
    "model": "claude-sonnet-4-20250514",
    "cli_path": "/path/to/claude",
    "conversation_name": "acp-abc12345",
    "db_path": "/path/to/pykoclaw.db",
    "allowed_tools": ["Bash", "Read", "Write", "Edit", ...],
    "mcp_server_config": { ... }
}
```

The worker creates its own `DbConnection` (separate process, separate
SQLite connection) and its own MCP server from the config. This is safe
because `ThreadSafeConnection` handles concurrent access and SQLite WAL
mode allows multiple readers.

#### Entry point

Register as a console script in `pyproject.toml`:

```toml
[project.scripts]
pykoclaw-acp-worker = "pykoclaw_acp.worker:main"
```

Or alternatively, run via `python -m pykoclaw_acp.worker`. The server
spawns it with `asyncio.create_subprocess_exec()`.

#### Tests

- Unit test the worker loop with a mock SDK client (inject via
  dependency parameter).
- Integration test: spawn a real worker subprocess, send a query,
  verify `text`/`result` messages come back.
- Test graceful shutdown: send `shutdown`, verify process exits cleanly
  with code 0.
- Test crash handling: make the mock SDK raise, verify `error` message
  is sent.

---

### Phase 3: WorkerPool (replaces ClientPool)

**Repo:** `pykoclaw-acp`
**Files:** `pykoclaw_acp/worker_pool.py` (new),
           `pykoclaw_acp/client_pool.py` (deleted at end)

Replace `ClientPool` with `WorkerPool` — same API surface
(`start()`, `close()`, `send()`), but manages worker subprocesses
instead of in-process `ClaudeSDKClient` instances.

#### 3.1 — `_WorkerHandle` (replaces `_Entry`)

```python
@dataclass
class _WorkerHandle:
    process: asyncio.subprocess.Process
    stdin_writer: asyncio.StreamWriter  # send queries
    stdout_reader: asyncio.StreamReader  # receive responses
    conversation_name: str
    lock: asyncio.Lock
    last_used: float
    _heartbeat_last: float  # last heartbeat received
```

#### 3.2 — `WorkerPool.send()`

1. Get or create a `_WorkerHandle` for the session ID
2. Acquire the session lock
3. Send `{"type": "query", "id": ..., "prompt": ...}` to worker stdin
4. Read worker stdout lines until `result` or `error` message with
   matching `id`
5. For each `text` message, call `on_text(text)`
6. On `result`, persist conversation via `upsert_conversation()`
7. On `error`, raise (or retry with a fresh worker)

#### 3.3 — Worker lifecycle management

- **Spawn:** `asyncio.create_subprocess_exec("pykoclaw-acp-worker", ...)`
  with `stdin=PIPE, stdout=PIPE, stderr=PIPE`
- **First line:** send JSON config blob
- **Wait for `ready`:** with a timeout (e.g. 30s)
- **Idle eviction:** same sweep loop as today, but `shutdown` message +
  `process.wait()` instead of `_kill_client()`
- **Crash detection:** pipe EOF or process exit → clean up handle,
  retry on next `send()`
- **Force kill:** if `shutdown` + wait times out, `process.kill()`

#### 3.4 — Heartbeat monitoring

Background task per worker (or a single sweep task) that checks
`_heartbeat_last` against a threshold. If a worker stops sending
heartbeats:

1. Log a warning
2. Send SIGTERM
3. Wait with timeout
4. SIGKILL if needed

This replaces the current process-level `Watchdog` for SDK-related
hangs. The server-level watchdog can remain for the ACP server's own
event loop health, but it should never trigger because the server
no longer runs any anyio code.

#### 3.5 — Stderr forwarding

Read worker stderr in a background task, line-by-line, and log each
line with the session ID as context:

```python
async def _forward_stderr(self, session_id: str, stderr: asyncio.StreamReader):
    while line := await stderr.readline():
        log.debug("[worker:%s] %s", session_id[:8], line.decode().rstrip())
```

#### 3.6 — Migration path

- `WorkerPool` implements the exact same public interface as
  `ClientPool`: `start()`, `close()`, `send(session_id, prompt, *,
  on_text)`.
- `AcpServer.__init__()` switches from `ClientPool(...)` to
  `WorkerPool(...)`.
- The `_entries` dict (accessed by `_find_session_for_conversation()`
  and `_process_pending_deliveries()`) is replaced with a comparable
  accessor on `WorkerPool`.
- After verification, `client_pool.py` and `_kill_client()` are
  deleted.

#### Tests

- Unit test `WorkerPool` with a mock worker (subprocess that speaks the
  protocol but doesn't use the real SDK).
- Test crash recovery: kill the worker mid-query, verify `send()`
  retries with a new worker.
- Test idle eviction: verify `shutdown` is sent and process exits.
- Test heartbeat timeout: stop heartbeats, verify worker gets killed.
- Test concurrent sessions: multiple workers running simultaneously.

---

### Phase 4: Server simplification

**Repo:** `pykoclaw-acp`
**Files:** `pykoclaw_acp/server.py`, `pykoclaw_acp/__init__.py`,
           `pykoclaw_acp/watchdog.py`

With the SDK out of process, simplify the server:

#### 4.1 — Remove anyio workarounds

- Delete the `CancelledError` safety net in the main loop — there are no
  more anyio cancel scopes in this process.
- Remove `_kill_client()` and all SDK-related imports from
  `client_pool.py` (which is now deleted).
- Simplify `_cancel_remaining_tasks()` — without SDK tasks, standard
  asyncio cancellation should work reliably. The `os._exit()` fallback
  can be kept as defense-in-depth but should never trigger.

#### 4.2 — Simplify watchdog

The watchdog remains as a server-level health check (event loop
liveness), but its role is reduced:

- No more SDK-related hangs to catch.
- Stale threshold can be tightened (30s instead of 60s) since the server
  is doing pure asyncio I/O.
- Consider whether the watchdog is even needed anymore — the server is
  now trivially simple. Keep it for safety initially, remove later if
  proven unnecessary.

#### 4.3 — Remove `asyncio.shield()` and related hacks

Grep for all `asyncio.shield()`, `_kill_client`, `BaseException` catches
that were added as anyio workarounds. Remove them.

#### 4.4 — Update `__init__.py`

- `_run_server()` can potentially return to `asyncio.run()` since there
  are no more anyio tasks. Evaluate whether the manual event loop
  management is still justified. If the only remaining tasks are pure
  asyncio, `asyncio.run()` with a `signal` handler is cleaner.
- Decision: keep manual loop management for now (it works, it's proven)
  but add a comment that `asyncio.run()` is now viable.

#### Tests

- Verify all existing server tests pass with `WorkerPool` replacing
  `MockClientPool` (the mock should implement the same interface).
- Verify no `CancelledError` leaks during idle eviction (the whole
  point of this project).
- Stress test: rapid create/evict cycles, concurrent prompts across
  sessions.

---

### Phase 5: Integration testing and deployment

#### 5.1 — Integration tests

- **Mock-worker integration test:** AcpServer + WorkerPool + mock
  worker subprocess. Full JSON-RPC flow from `initialize` through
  `session/prompt` with streaming.
- **Real-SDK integration test (e2e marker):** AcpServer + WorkerPool +
  real `pykoclaw-acp-worker` + real Claude SDK. Send a simple prompt,
  verify streamed response.

#### 5.2 — Staging verification

```bash
bin/staging.sh process-isolated-workers
```

- Verify Mitto → pykoclaw-acp → worker → Claude CLI flow works.
- Test multi-turn conversations.
- Test idle eviction (wait 10+ minutes, send another message).
- Test concurrent conversations (multiple Mitto sessions).
- Verify scheduled task delivery still works.

#### 5.3 — Deploy and monitor

```bash
bin/merge-feature.sh process-isolated-workers
./install-dev.sh
```

- Monitor `journalctl --user -u mitto-web` for 24h.
- Check for any `CancelledError` log lines (should be zero).
- Verify no zombie processes.
- Check ACP log files for clean worker lifecycle events.

---

## File inventory

### New files

| File                                         | Repo         | Purpose                          |
| -------------------------------------------- | ------------ | -------------------------------- |
| `pykoclaw/src/pykoclaw/sdk_consume.py`       | pykoclaw     | Shared SDK response consumer     |
| `pykoclaw/tests/test_sdk_consume.py`         | pykoclaw     | Tests for shared consumer        |
| `pykoclaw_acp/worker_protocol.py`            | pykoclaw-acp | Worker ↔ server protocol models  |
| `pykoclaw_acp/worker.py`                     | pykoclaw-acp | Worker subprocess entry point    |
| `pykoclaw_acp/worker_pool.py`                | pykoclaw-acp | WorkerPool (replaces ClientPool) |
| `pykoclaw-acp/tests/test_worker_protocol.py` | pykoclaw-acp | Protocol serialization tests     |
| `pykoclaw-acp/tests/test_worker.py`          | pykoclaw-acp | Worker subprocess tests          |
| `pykoclaw-acp/tests/test_worker_pool.py`     | pykoclaw-acp | WorkerPool lifecycle tests       |

### Modified files

| File                                      | Repo         | Change                                    |
| ----------------------------------------- | ------------ | ----------------------------------------- |
| `pykoclaw/src/pykoclaw/agent_core.py`     | pykoclaw     | Use `consume_sdk_response()`              |
| `pykoclaw_acp/server.py`                  | pykoclaw-acp | Use WorkerPool, remove anyio workarounds  |
| `pykoclaw_acp/__init__.py`                | pykoclaw-acp | Simplify `_run_server()`, update imports  |
| `pykoclaw_acp/watchdog.py`                | pykoclaw-acp | Reduce scope (optional)                   |
| `pykoclaw-acp/pyproject.toml`             | pykoclaw-acp | Add worker console script entry point     |
| `pykoclaw-acp/tests/conftest.py`          | pykoclaw-acp | Update MockClientPool → MockWorkerPool    |

### Deleted files

| File                          | Repo         | Reason                              |
| ----------------------------- | ------------ | ----------------------------------- |
| `pykoclaw_acp/client_pool.py` | pykoclaw-acp | Replaced by worker_pool.py + worker.py |

---

## Risk assessment

| Risk                                          | Likelihood | Impact | Mitigation                                           |
| --------------------------------------------- | ---------- | ------ | ---------------------------------------------------- |
| Worker startup latency (SDK connect)          | High       | Low    | Worker is kept alive across queries (same as today)   |
| Pipe buffering causes partial JSON lines      | Low        | Medium | Line-buffered stdout + explicit flush in worker       |
| Worker inherits unwanted env/file descriptors | Low        | Low    | Use `close_fds=True` (default for subprocess)         |
| MCP server can't run in worker process        | Medium     | High   | Worker creates its own MCP server from config; test early |
| Increased memory (N workers × Python process) | Medium     | Low    | Same as today — each ClaudeSDKClient already spawns a subprocess |
| SQLite contention from multiple workers       | Low        | Low    | WAL mode + ThreadSafeConnection; workers do minimal DB writes |
| Worker zombie accumulation                    | Low        | Medium | `process.wait()` in eviction + periodic reap task     |

## Sequencing and dependencies

```
Phase 0 (core: sdk_consume)
    ↓
Phase 1 (acp: worker_protocol)   ← can run in parallel with Phase 0
    ↓
Phase 2 (acp: worker.py)          ← depends on Phase 0 + Phase 1
    ↓
Phase 3 (acp: worker_pool.py)     ← depends on Phase 2
    ↓
Phase 4 (acp: server cleanup)     ← depends on Phase 3
    ↓
Phase 5 (integration + deploy)    ← depends on Phase 4
```

Phases 0 and 1 are independent and can be developed and tested in
parallel. Phase 2 needs both. Phases 3–5 are strictly sequential.

## Definition of done

- [ ] `consume_sdk_response()` exists in pykoclaw core and is used by
      both `query_agent()` and the worker
- [ ] Worker subprocess starts, connects SDK, handles queries, shuts
      down cleanly
- [ ] `WorkerPool` manages worker lifecycle with the same API as
      `ClientPool`
- [ ] ACP server has zero SDK/anyio imports
- [ ] `CancelledError` safety net in main loop is removed (no longer
      needed)
- [ ] `_kill_client()` is deleted
- [ ] `os._exit()` forced shutdown is either removed or demoted to
      a last-resort fallback that never triggers in practice
- [ ] All existing tests pass
- [ ] New tests cover worker protocol, worker lifecycle, WorkerPool
      crash recovery, and idle eviction
- [ ] Staging verification passes (multi-turn, idle eviction, concurrent
      sessions)
- [ ] 24h production monitoring shows zero `CancelledError`s and zero
      zombie processes

[001-acp-architecture-fragility.md]: 001-acp-architecture-fragility.md
