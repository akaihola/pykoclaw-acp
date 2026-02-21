# Implement `session/load` for conversation resume across restarts

**Priority:** High
**Status:** Backlog
**Created:** 2026-02-21
**Tags:** acp, session, resume, mitto

## Problem

Conversation context is lost every time the ACP process restarts
(deploy, crash, Mitto restart). Users see a blank conversation with
no history.

## Root cause

Pykoclaw returns `"agentCapabilities": {}` in the `initialize`
response. Mitto checks for `loadSession: true` — when absent, it
skips the resume path and calls `session/new` (creating a fresh
session) instead of `session/load` (resuming the old one).

Mitto already has the full resume mechanism built in
(`background_session.go:1199–1221`):

```go
if acpSessionID != "" && initResp.AgentCapabilities.LoadSession {
    loadResp, err := bs.acpConn.LoadSession(bs.ctx, acp.LoadSessionRequest{
        SessionId:  acp.SessionId(acpSessionID),
        Cwd:        cwd,
        McpServers: []acp.McpServer{},
    })
```

It stores the ACP session ID and passes it on restart. **No Mitto
changes are needed** — the fix is entirely on the pykoclaw side.

## What already works

- The DB stores `(conversation_name, claude_session_id)` via
  `upsert_conversation()`
- Claude CLI persists full conversations as JSONL in
  `~/.claude/projects/`
- `ClaudeAgentOptions` supports `resume=session_id`
- `query_agent()` (WhatsApp/scheduler path) already uses `resume=`
  successfully

## Implementation

### 1. Advertise the capability

In `_handle_initialize`, return `loadSession: true`:

```python
"agentCapabilities": {"loadSession": True},
```

### 2. Persist ACP session → conversation mapping

Currently the DB stores `(conversation_name, claude_session_id)`.
Add a new table or column to also store the ACP session ID → 
conversation name mapping, so `session/load` can look up by ACP
session ID:

```sql
CREATE TABLE IF NOT EXISTS acp_sessions (
    acp_session_id TEXT PRIMARY KEY,
    conversation_name TEXT NOT NULL
);
```

Populate it in `_handle_session_new` and `_handle_session_load`.

### 3. Implement `_handle_session_load`

```python
async def _handle_session_load(self, msg_id, params):
    acp_session_id = params.get("sessionId")
    
    # Look up conversation name from ACP session ID
    conversation_name = lookup_acp_session(self._db, acp_session_id)
    if not conversation_name:
        # Fall through — Mitto will call session/new
        self._write(self._protocol.format_error(
            msg_id, JsonRpcError.INVALID_SESSION,
            f"No conversation found for session {acp_session_id}",
        ))
        return
    
    # Look up Claude session ID from conversation
    conv = get_conversation(self._db, conversation_name)
    claude_session_id = conv.session_id if conv else None
    
    # Re-register in self._sessions
    self._sessions[acp_session_id] = {"cwd": params.get("cwd", "")}
    
    # Tell WorkerPool to use resume= when spawning the worker
    self._pool.set_resume(acp_session_id, claude_session_id)
    
    self._write(self._protocol.format_response(msg_id, {
        "sessionId": acp_session_id,
    }))
```

### 4. Add resume support to WorkerPool + worker

- Add `resume_session_id` field to `WorkerConfig`
- `WorkerPool._spawn_worker()`: accept optional `resume_session_id`,
  pass it in `WorkerConfig`
- `worker.py`: pass `resume=config.resume_session_id` to
  `ClaudeAgentOptions`

### 5. Stable conversation names

Currently `conversation_name = f"acp-{session_id[:8]}"` — derived
from the random ACP session UUID. With `session/load`, the same ACP
session ID comes back on restart, so the conversation name is stable
by construction. No change needed here.

### 6. Register in dispatch table

```python
handler = {
    "initialize": self._handle_initialize,
    "session/new": self._handle_session_new,
    "session/load": self._handle_session_load,
    "session/prompt": self._handle_session_prompt,
}.get(method)
```

## Tests

- Test `initialize` returns `loadSession: true`
- Test `session/load` with known ACP session ID → resumes
  conversation
- Test `session/load` with unknown ACP session ID → error response
  (Mitto falls through to `session/new`)
- Test that `session/new` persists the ACP session → conversation
  mapping
- Test WorkerPool spawns worker with `resume=` when set
- Test worker passes `resume=` to `ClaudeAgentOptions`
- Integration test: new session → prompt → "restart" → load session
  → prompt (verify context preserved)

## Verification

After deploy, check Mitto logs for:

```
"Resumed ACP session" acp_session_id=...
```

instead of:

```
"Created new ACP session" acp_session_id=...
```

## Scope

- **Pykoclaw only** — zero Mitto changes needed
- **Repos affected:** `pykoclaw-acp` (server, worker_pool, worker,
  worker_protocol), `pykoclaw` (db — new table/migration)
