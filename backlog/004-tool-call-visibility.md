# Surface tool calls and restore streaming in Mitto

**Priority:** Medium
**Status:** Backlog
**Created:** 2026-02-26
**Updated:** 2026-03-22
**Tags:** acp, mitto, tool-calls, streaming, ux

## What the user experiences today (problems)

### 1 — Responses appear all at once

When the agent is working on a long answer, Mitto shows nothing until the
entire response is ready. There is no progressive text — no sense that the
agent is thinking or writing. The UI appears frozen, then everything appears
at once.

This used to work differently: responses arrived word-by-word as the agent
generated them.

### 2 — Tool activity is invisible

While the agent reads files, runs commands, searches the codebase, or edits
code, the user sees a blank wait followed by the final answer. There is no
indication of what is happening, how long it will take, or whether anything
is working at all.

These two problems are independent and can be fixed separately.

## Why it matters

- **User trust**: Seeing "Reading file…" or "Running bash command…"
  makes long agent turns feel responsive rather than frozen.
- **Debugging**: When something goes wrong, knowing which tool failed
  is immediately useful.
- **Mitto already built the UI**: The frontend has full tool call
  rendering (spinning indicator, ✓/✗ status, clickable file paths) —
  it's waiting for data that never arrives.

## Current state

### What Mitto has (ready to go)

| Layer | Support | Key files |
|-------|---------|-----------|
| Frontend rendering | ✅ `ROLE_TOOL` messages with status icons | `web/static/components/Message.js` |
| WebSocket handler | ✅ `tool_call` / `tool_update` event types | `web/static/hooks/useWebSocket.js` |
| Go backend observer | ✅ `OnToolCall()` / `OnToolUpdate()` callbacks | `internal/acp/client.go` |
| ACP Go SDK | ✅ `SessionUpdateToolCall` / `SessionToolCallUpdate` structs | `acp-go-sdk` |
| Settings infrastructure | ✅ Per-session boolean flags + UI | `session_settings_api.go`, `SettingsDialog.js` |

### What pykoclaw sends today

Only two `session/update` types:
- `agent_message_chunk` — streaming text (with `---` separators at
  tool boundaries)
- `error` — processing errors

### What's missing (pykoclaw-acp side)

The pipeline from SDK → worker → server → client has no tool call
path at all:

```
SDK stream          Worker protocol        ACP notifications
────────────        ───────────────        ─────────────────
ToolUseBlock  ──✗── (no message type) ──✗── (not sent)
PreToolUse    ──✗── (hooks not used)  ──✗── (not sent)
PostToolUse   ──✗── (hooks not used)  ──✗── (not sent)
```

## Design options

### Option A: Stream-based detection (ToolUseBlock in sdk_consume)

Detect `ToolUseBlock` in `consume_sdk_response()`, call `on_tool_call`
callback.

**Pros:** Minimal change, no new SDK features needed.
**Cons:**
- `ToolUseBlock` appears *after* the model generates the tool call but
  *before* execution begins — timing is imprecise.
- No explicit completion signal. Must infer "completed" from the next
  stream event (next `AssistantMessage` or `ResultMessage`).
- Cannot detect tool *failure* vs *success* — both look the same
  (another message arrives).
- `ToolUseBlock.input` is the raw tool input dict — generating a
  human-readable title (e.g. "Read /home/agent/foo.py") requires
  per-tool formatting logic.

### Option B: SDK hooks (PreToolUse / PostToolUse) ← **Recommended**

Register `PreToolUse`, `PostToolUse`, and `PostToolUseFailure` hooks on
`ClaudeAgentOptions`. The SDK calls these at the exact moment tools
start and finish execution.

**Pros:**
- **Precise timing**: hook fires at tool start (`PreToolUse`) and
  tool completion (`PostToolUse`) / failure (`PostToolUseFailure`).
- **Rich data**: `tool_name`, `tool_input`, `tool_use_id` on all
  hooks; `tool_response` on PostToolUse; `error` on failure.
- **Correct status transitions**: in_progress → completed / failed.
- **Future-proof**: hooks also give us `SubagentStart`/`SubagentStop`
  for nested agent visibility.

**Cons:**
- Hooks run in the worker subprocess — must write protocol messages to
  stdout from within the hook callback. This is safe (worker owns
  stdout, hooks run in the same async loop).
- Hook return value must be `{"continue_": True}` — must not
  accidentally block tool execution.
- Adds complexity to worker.py setup.

### Option C: Hybrid (hooks + stream fallback)

Use hooks for precise timing, fall back to stream detection if hooks
are unavailable (e.g., older SDK version).

**Verdict:** Over-engineered for now. Go with **Option B** (hooks).

## Visibility toggle

The user should be able to choose between two modes:

### Mode 1: "Separator" (current default)

Tool calls are invisible. Text blocks are separated by `---` (`<hr>`)
at tool boundaries. This is what `sdk_consume.py` does today. Compact
and familiar.

### Mode 2: "Tool calls visible"

Tool calls appear as badge-like messages between text blocks (spinning
→ ✓/✗). Mitto already renders these — just needs data.

### Toggle mechanism: `/tools` slash command

**No Mitto changes needed.** The toggle is a slash command that
pykoclaw-acp intercepts before the prompt reaches the SDK:

```
/tools        — toggle tool call visibility
/tools on     — enable
/tools off    — disable
```

The user types `/tools` in the Mitto chat input. Pykoclaw's
`_handle_session_prompt` recognises it as a command, toggles the
per-session flag, sends a confirmation message back as an
`agent_message_chunk`, and does **not** forward anything to the worker.

**Per-session state:** Stored in `self._sessions[session_id]`:

```python
self._sessions[session_id] = {
    "cwd": "...",
    "show_tool_calls": False,  # new — default off
}
```

### How the two modes interact

The worker always produces both:
1. `---` text separators (from `sdk_consume.py`'s existing logic)
2. `ToolCallMessage` / `ToolCallUpdateMessage` (from SDK hooks — new)

The **server** decides what to forward to the ACP client based on the
per-session `show_tool_calls` flag:

| Flag | ACP tool notifications | `---` text separators |
|------|------------------------|-----------------------|
| `false` (default) | Suppressed | Forwarded |
| `true` | Forwarded | Suppressed |

**Suppressing `---` separators:** The server's `_send_chunk` callback
checks whether the text chunk is a tool boundary separator
(`"\n\n---\n\n"`) and drops it when `show_tool_calls` is true. This
marker is distinctive enough that false positives are negligible; the
worst case (an intentional `---` gets dropped) is cosmetic.

### Why server-side, not client-side?

- **Zero Mitto changes** — no new flags, no settings UI, no rendering
  logic changes. Mitto already renders tool calls when it receives
  them; it already renders `<hr>` from text. The server just controls
  which events reach the client.
- **Slash command is natural** — users already type prompts in the
  chat input. A `/tools` command feels native.
- **Per-session state is already there** — `self._sessions[session_id]`
  is the obvious place to store the flag.

## Implementation plan

### Phase 1: Worker protocol — new message types

**Repo:** `pykoclaw-acp`
**File:** `pykoclaw_acp/worker_protocol.py`

Add two new worker → server message types:

```python
@dataclass
class ToolCallMessage:
    """Emitted when a tool starts executing (PreToolUse hook)."""
    id: str = ""           # correlates with the query id
    tool_use_id: str = ""  # SDK's tool_use_id (unique per invocation)
    tool_name: str = ""    # e.g. "Bash", "Read", "Edit"
    title: str = ""        # human-readable: "Read /home/agent/foo.py"
    type: Literal["tool_call"] = "tool_call"


@dataclass
class ToolCallUpdateMessage:
    """Emitted when a tool finishes (PostToolUse / PostToolUseFailure)."""
    id: str = ""           # correlates with the query id
    tool_use_id: str = ""  # same as ToolCallMessage.tool_use_id
    status: str = ""       # "completed" | "failed"
    type: Literal["tool_call_update"] = "tool_call_update"
```

Update `WorkerMessage` union, `encode()`, and `decode_worker_message()`.

### Phase 2: SDK hooks in worker — emit tool events

**Repos:** `pykoclaw-acp` (worker), `pykoclaw` (sdk_consume — optional)
**Files:** `pykoclaw_acp/worker.py`

Register `PreToolUse`, `PostToolUse`, `PostToolUseFailure` hooks on
`ClaudeAgentOptions.hooks`:

```python
async def _pre_tool_use_hook(
    hook_input: PreToolUseHookInput,
    _: str | None,
    __: HookContext,
) -> SyncHookJSONOutput:
    title = _format_tool_title(hook_input.tool_name, hook_input.tool_input)
    write_msg(ToolCallMessage(
        id=current_query_id,
        tool_use_id=hook_input.tool_use_id,
        tool_name=hook_input.tool_name,
        title=title,
    ))
    return {"continue_": True}


async def _post_tool_use_hook(
    hook_input: PostToolUseHookInput,
    _: str | None,
    __: HookContext,
) -> SyncHookJSONOutput:
    write_msg(ToolCallUpdateMessage(
        id=current_query_id,
        tool_use_id=hook_input.tool_use_id,
        status="completed",
    ))
    return {"continue_": True}


async def _post_tool_use_failure_hook(
    hook_input: PostToolUseFailureHookInput,
    _: str | None,
    __: HookContext,
) -> SyncHookJSONOutput:
    write_msg(ToolCallUpdateMessage(
        id=current_query_id,
        tool_use_id=hook_input.tool_use_id,
        status="failed",
    ))
    return {"continue_": True}
```

#### Tool title formatting

`_format_tool_title(name, input)` generates a human-readable label:

| Tool | Input key | Example title |
|------|-----------|---------------|
| Read | `file_path` | "Read /home/agent/foo.py" |
| Write | `file_path` | "Write /home/agent/bar.py" |
| Edit | `file_path` | "Edit /home/agent/baz.py" |
| Bash | `command` (truncated) | "Bash: git status" |
| Grep | `pattern` | "Grep: ToolUseBlock" |
| Glob | `pattern` | "Glob: **/*.py" |
| WebFetch | `url` | "WebFetch: https://…" |
| WebSearch | `query` | "WebSearch: ACP protocol spec" |
| Task | `description` | "Task: explore codebase" |
| mcp__* | (tool name) | "mcp: schedule_task" |
| Other | — | tool name as-is |

Keep it simple — just the tool name + first relevant input field,
truncated to ~80 chars.

#### Thread safety

The hooks run in the worker's event loop, same process as
`_write_msg()`. Since the worker handles one query at a time (no
concurrent queries), there are no race conditions on stdout writes.

The `current_query_id` can be stored as a module-level variable (or
passed via closure) since queries are serialized by the worker's
single-threaded loop.

### Phase 3: WorkerPool — route tool events

**Repo:** `pykoclaw-acp`
**File:** `pykoclaw_acp/worker_pool.py`

The `_query()` method's read loop currently handles `TextChunkMessage`,
`WorkerResultMessage`, `ErrorMessage`, `HeartbeatMessage`. Add:

```python
elif isinstance(msg, ToolCallMessage) and msg.id == msg_id:
    if on_tool_call:
        await on_tool_call(msg.tool_use_id, msg.tool_name, msg.title)
elif isinstance(msg, ToolCallUpdateMessage) and msg.id == msg_id:
    if on_tool_call_update:
        await on_tool_call_update(msg.tool_use_id, msg.status)
```

Add `on_tool_call` and `on_tool_call_update` callback parameters to
`WorkerPool.send()`.

### Phase 4: ACP server — notifications + slash command

**Repo:** `pykoclaw-acp`
**File:** `pykoclaw_acp/server.py`

#### 4.1 — Intercept `/tools` command

In `_handle_session_prompt`, before forwarding to the worker pool:

```python
content_stripped = content.strip()
if content_stripped.startswith("/tools"):
    await self._handle_tools_command(session_id, msg_id, content_stripped)
    return
```

```python
async def _handle_tools_command(
    self, session_id: str, msg_id: Any, command: str,
) -> None:
    session = self._sessions[session_id]
    parts = command.split(maxsplit=1)
    arg = parts[1].strip().lower() if len(parts) > 1 else ""

    if arg == "on":
        session["show_tool_calls"] = True
    elif arg == "off":
        session["show_tool_calls"] = False
    else:
        # Toggle
        session["show_tool_calls"] = not session.get("show_tool_calls", False)

    state = "on" if session["show_tool_calls"] else "off"
    self._write(
        self._protocol.format_notification(
            "session/update",
            {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {
                        "type": "text",
                        "text": f"Tool call visibility: **{state}**",
                    },
                },
            },
        )
    )
    # Send prompt response so Mitto knows the turn is done
    self._write(
        self._protocol.format_response(msg_id, {"stopReason": "end_turn"})
    )
```

#### 4.2 — Conditional forwarding in `_handle_session_prompt`

Define callbacks that check the session flag:

```python
show_tools = session.get("show_tool_calls", False)

async def _send_chunk(text: str) -> None:
    # Suppress --- separators when tool calls are visible
    if show_tools and text.strip() == "---":
        return
    self._write(
        self._protocol.format_notification(
            "session/update",
            {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": text},
                },
            },
        )
    )

async def _send_tool_call(
    tool_use_id: str, tool_name: str, title: str,
) -> None:
    if not show_tools:
        return  # silently drop when tool calls are hidden
    self._write(
        self._protocol.format_notification(
            "session/update",
            {
                "sessionId": session_id,
                "update": {
                    "toolCall": {
                        "id": tool_use_id,
                        "title": title,
                        "status": "in_progress",
                    },
                },
            },
        )
    )

async def _send_tool_update(tool_use_id: str, status: str) -> None:
    if not show_tools:
        return
    self._write(
        self._protocol.format_notification(
            "session/update",
            {
                "sessionId": session_id,
                "update": {
                    "toolCallUpdate": {
                        "toolCallId": tool_use_id,
                        "status": status,
                    },
                },
            },
        )
    )
```

**ACP notification format** (matching the Go SDK's expected structure):

```json
{
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
        "sessionId": "...",
        "update": {
            "toolCall": {
                "id": "toolu_abc123",
                "title": "Read /home/agent/server.py",
                "status": "in_progress"
            }
        }
    }
}
```

```json
{
    "jsonrpc": "2.0",
    "method": "session/update",
    "params": {
        "sessionId": "...",
        "update": {
            "toolCallUpdate": {
                "toolCallId": "toolu_abc123",
                "status": "completed"
            }
        }
    }
}
```

## File inventory

### New / modified files

| File | Repo | Change |
|------|------|--------|
| `worker_protocol.py` | pykoclaw-acp | Add `ToolCallMessage`, `ToolCallUpdateMessage` |
| `worker.py` | pykoclaw-acp | Register SDK hooks, emit tool events |
| `worker_pool.py` | pykoclaw-acp | Route `ToolCallMessage` / `ToolCallUpdateMessage` via callbacks |
| `server.py` | pykoclaw-acp | `/tools` command, conditional forwarding, ACP notifications |
| `sdk_consume.py` | pykoclaw (core) | No changes needed (hooks bypass the stream) |

**Zero Mitto changes.** Everything is in `pykoclaw-acp`.

### Tests

- Unit test `ToolCallMessage` / `ToolCallUpdateMessage` serialization
  round-trips
- Unit test `_format_tool_title()` for each tool type
- Unit test `/tools` command parsing: toggle, on, off, unknown arg
- Unit test `---` suppression when `show_tool_calls` is true
- Unit test tool notifications suppressed when `show_tool_calls` is
  false
- Integration test: mock SDK with hook callbacks → verify tool events
  in protocol stream
- Integration test: full pipeline mock (server → pool → worker) →
  verify ACP notifications include tool calls

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Hooks not called for some tools | Low | Medium | Test with real agent; fall back to stream detection if needed |
| Hook callback blocks tool execution | Low | High | Always return `{"continue_": True}`; add timeout |
| stdout interleaving (hook writes during text stream) | Low | Low | Worker is single-threaded; queries are serialized |
| ACP notification format mismatch with Go SDK | Medium | Medium | Verify against `acp-go-sdk` `SessionUpdate` struct before implementation |
| `---` suppression catches intentional horizontal rules | Low | Low | Only suppress when text.strip() == "---" (full chunk); agent-authored `---` inside a larger text block is not affected |
| `/tools` command collides with user intent | Very low | Low | Distinctive prefix; user unlikely to start a real prompt with `/tools` |

## Open questions

1. **Notification format**: The ACP Go SDK uses `toolCall` (camelCase)
   as the update field name, but pykoclaw's existing updates use
   `sessionUpdate: "agent_message_chunk"` (flat key). Need to verify
   exact wire format Mitto's Go client expects. Check
   `mitto/internal/acp/client.go` switch statement.

2. **Slash command prefix**: `/tools` is proposed. Alternatives:
   `/tc`, `/toolcalls`, `/show-tools`. Could also support a general
   `/set` command for future config toggles.

3. **Persistence across restarts**: Per-session state in
   `self._sessions` is lost on ACP restart. Should the flag be
   persisted to DB? Probably not for v1 — it's a UI preference, not
   critical state. Defaulting to off after restart is fine.

## Sequencing

```
Phase 1 (worker_protocol)          ← standalone, can land first
    ↓
Phase 2 (worker hooks)             ← depends on Phase 1
    ↓
Phase 3 (worker_pool routing)      ← depends on Phase 1
    ↓
Phase 4 (server: slash cmd +       ← depends on Phase 3
         conditional forwarding)
```

All phases are in `pykoclaw-acp` (one feature branch, zero Mitto
changes).

## Streaming restore plan (buffered semantic windows)

### Confirmed diagnosis (2026-03-22)

End-to-end pipeline traced independently in two sessions; both reached the
same conclusion.

**What works correctly today:**

- `worker.py` — emits `TextChunkMessage` for every SDK partial immediately;
  `include_partial_messages=True` is set.
- `worker_pool.py` — `_query()` calls `await on_text(msg.text)` for each
  `TextChunkMessage` as they arrive; no buffering here.
- Mitto (`background_session.go`) — persists and broadcasts each
  `agent_message` event immediately as it arrives.
- Mitto (`stream_buffer.go`) — explicitly handles interleaved markdown,
  thoughts, and tool calls with its own flush logic.
- Mitto (`useWebSocket.js`) — appends/coalesces streaming `agent_message`
  events; handles `tool_call`/`tool_update` separately.
- The entire Mitto stack from `StreamBuffer` through WebSocket through frontend
  state is designed for "text, tool, more text" interleaved patterns.

**Where streaming is suppressed:**

`server.py` — `_handle_session_prompt` collects every incoming chunk into
`chunks: list[str]`, waits for `_pool.send()` to fully complete, then applies
transforms and emits exactly **one** `session/update` with
`agent_message_chunk`. This is the sole bottleneck.

**Additional suppression:** tool calls and thoughts are not forwarded to Mitto
at all — they are silently swallowed in `server.py`. This is a separate
concern; text streaming restoration can land independently of tool call
visibility.

**Zero Mitto changes required for text streaming restoration.** Everything
needed is already in place on the Mitto side.

### Goal

Restore incremental rendering for ACP responses in Mitto **without** going
back to token-by-token raw streaming and **without** waiting for the full
turn to finish. The server should forward transformed text in buffered
windows that align with semantic boundaries, keeping an unsafe tail in
memory until it becomes syntactically safe to rewrite and emit.

### Why this approach

Current ACP behavior accumulates the full response in `server.py`, applies
plugin transforms once, and emits a single `agent_message_chunk` at the end.
That preserves correctness for pykofinder link rewriting but removes the UX
benefit of streaming.

A buffered-window approach keeps the current transform contract
(`transform_response(text, ctx) -> str`) while restoring responsiveness:

- **Safer than token streaming**: incomplete markdown links, wikilinks, HTML
  `<img>` tags, and code fences stay buffered until closed.
- **Much better UX than whole-turn buffering**: prose can appear paragraph by
  paragraph or line by line at natural boundaries.
- **Smaller scope than a new plugin API**: initial implementation can live
  entirely in `pykoclaw-acp/server.py` with no mandatory plugin interface
  changes.

### Current constraints to respect

- `pykoclaw-acp` worker already produces partial text via
  `include_partial_messages=True`.
- `pykoclaw` plugin API currently supports only full-string transforms via
  `compose_transformers(load_plugins(), ctx)`.
- `pykofinder` transforms are regex-based and assume complete local syntax
  units (`[[wikilinks]]`, `[label](target)`, `![alt](target)`, `<img ...>`).
- Mitto already supports incremental `agent_message_chunk` events.

### Proposed design

#### 1. Replace whole-turn accumulation with a stream window accumulator

In `server.py`, replace the current pattern:

- accumulate all chunks in `chunks: list[str]`
- `full_text = "".join(chunks)`
- transform once
- emit once

with a persistent per-turn buffer:

- `raw_buffer: str` — all un-emitted raw text received so far
- optional bookkeeping for diagnostics:
  - `emitted_chars: int`
  - `flush_count: int`

On every incoming worker text chunk:

1. append chunk to `raw_buffer`
2. compute the largest **safe flush prefix**
3. transform only that prefix
4. emit exactly that transformed prefix as one ACP `agent_message_chunk`
5. remove the emitted raw prefix from `raw_buffer`

At end of turn:

1. transform and emit any remaining buffered tail
2. send the normal `stopReason: end_turn` response

#### 2. Define “safe flush prefix” heuristics

The flush algorithm should be conservative. If uncertain, keep more text in
buffer rather than risking malformed rewrites.

Start with these rules:

**Safe positive boundaries**
- flush after a blank line (`\n\n`) — paragraph boundary
- flush after a normal line ending when not inside an incomplete structure
- flush after a completed wikilink `]]`
- flush after a completed markdown link/image closing `)`
- flush after a completed HTML tag closing `>`
- flush after a horizontal-rule separator block if it is complete

**Unsafe tails to keep buffered**
- unmatched `[[`
- unmatched markdown link opener patterns like `[` / `](` without closing `)`
- incomplete image syntax `![...` / `![...](...`
- incomplete HTML `<img ...` tag without closing `>`
- unmatched inline backtick runs
- unclosed triple-backtick fences
- trailing partial pathish fragments immediately after `(` or `[[` where the
  next chunk may complete the reference

#### 3. Handle fenced code blocks explicitly

Code fences are the easiest place to corrupt content if we flush too early.
The accumulator should track whether the current tail is inside an unclosed
triple-backtick fence.

While inside an open code fence:
- do **not** flush on normal line boundaries
- only flush once the matching closing fence arrives
- then treat the whole fenced block as a safe unit

This keeps markdown rendering stable and avoids path-rewriting inside partial
code-fence fragments.

#### 4. Keep transforms stateless for v1

Do **not** introduce a new streaming plugin protocol initially.

Instead:
- compute the safe prefix in ACP server code
- call the existing composed full-string transformer only on that prefix
- emit the transformed result immediately

This works as long as the chosen prefix is syntactically closed enough for the
existing regex-based transforms.

#### 5. End-of-turn final flush remains authoritative

Even with streaming restored, the end-of-turn path should still:
- flush all remaining buffered raw text
- run the same transform pipeline on the final remainder
- emit it before `end_turn`

This guarantees no text is stranded in the buffer and preserves exact final
content delivery.

### Suggested implementation phases

#### Phase A — Add flush-window helper in ACP server

Create a small helper in `pykoclaw-acp` (either local to `server.py` or a new
module such as `stream_window.py`) that exposes something like:

```python
safe_prefix, tail = split_safe_flush_prefix(raw_buffer)
```

Responsibilities:
- inspect the current accumulated raw text
- identify the largest prefix that is safe to transform now
- leave a tail containing any incomplete syntax structures

Keep it pure and side-effect free so it is easy to unit test.

#### Phase B — Integrate helper into `_handle_session_prompt`

Change `_collect_chunk()` logic in `server.py` so each incoming SDK partial:
- appends into `raw_buffer`
- repeatedly flushes while a non-empty safe prefix is available
- writes each flushed transformed prefix as `agent_message_chunk`

The current whole-turn `chunks: list[str]` logic should be removed once this is
in place.

#### Phase C — Preserve compatibility with current transforms

Reuse existing:
- `TransformContext(channel_prefix="acp", native_file_extensions=frozenset())`
- `compose_transformers(load_plugins(), ctx)`

Do not change plugin registration order or semantics in this phase.

#### Phase D — Add diagnostics for real-world tuning

Add debug logging behind the existing logger to record:
- raw chunk size received
- safe prefix size emitted
- tail size retained
- whether flush was blocked by open wikilink, markdown link, HTML tag, or code fence

This will help tune heuristics after staging without changing wire behavior.

### Edge cases to design for

#### Incomplete markdown links across chunk boundaries

Example:
- chunk 1: `[Project](`
- chunk 2: `pages/Projects/Foo.md)`

Desired behavior:
- chunk 1 stays buffered
- after chunk 2 arrives, full link becomes safe
- transformed link emits once

#### Incomplete wikilinks across chunk boundaries

Example:
- chunk 1: `See [[Agent `
- chunk 2: `Commons]] for details`

Desired behavior:
- no partial emit after chunk 1 if the tail contains unmatched `[[`
- emit once `]]` arrives

#### HTML image tags split across chunks

Example:
- chunk 1: `<img src="journals/media/foo`
- chunk 2: `.png" alt="Foo">`

Desired behavior:
- keep entire tag buffered until `>`
- then normalise/transform once complete

#### Long prose without special syntax

Example:
- model emits normal paragraphs linearly

Desired behavior:
- flush at paragraph boundaries quickly
- avoid waiting for full turn

#### Code blocks containing markdown-like syntax

Example:
- fenced code contains `[[not a real wikilink]]`

Desired behavior:
- do not attempt partial semantic parsing inside an open code fence
- flush only once the fence closes

### Testing plan

#### Unit tests for safe-prefix splitting

Add focused tests for the helper with inputs such as:
- plain prose with paragraph breaks
- incomplete and complete markdown links
- incomplete and complete wikilinks
- incomplete and complete HTML `<img>` tags
- inline backticks
- fenced code blocks
- multiple flushable segments plus one unsafe tail

Each test should assert the exact returned `(safe_prefix, tail)`.

#### ACP server tests

Update/add tests in `tests/test_server.py` to verify:
- server emits multiple `agent_message_chunk` notifications during a turn when
  the mock worker yields multiple partials that cross safe boundaries
- server does **not** emit malformed partial links/tags
- server still emits final remainder before `end_turn`
- pool error path still emits `error` + `end_turn`

#### Integration tests

Revise `tests/test_integration.py` so ACP no longer enforces “exactly one
chunk per turn”. Instead assert:
- concatenated emitted chunks equal the final expected text
- chunk count is `>= 1`
- specific fixture prompts produce more than one chunk when they contain
  paragraph boundaries

### Rollout strategy

1. Implement helper + unit tests first
2. Integrate into ACP server with debug logging
3. Update ACP server/integration tests
4. Verify in staging against Mitto with a real pykoclaw-acp conversation
5. Tune flush heuristics based on actual streamed output patterns

### Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Heuristics flush too early and break rewritten links | Medium | High | Start conservative; keep unsafe tail larger; cover syntax-boundary tests thoroughly |
| Heuristics flush too late and UX still feels non-streaming | Medium | Medium | Add debug logging and tune around paragraph boundaries first |
| Regex transforms behave unexpectedly on locally complete but globally odd text | Low | Medium | Treat end-of-turn flush as final correctness backstop; stage with real prompts |
| Code-fence detection misses edge cases | Medium | Medium | Keep v1 handling simple and conservative; add explicit fence tests |
| Integration tests remain too coupled to chunk counts | Medium | Low | Assert concatenated text equality, not exact chunk count, except in targeted streaming tests |

### Open questions

1. Should a single newline in plain prose be enough to flush, or should v1
   only flush on blank lines and explicit completed constructs?
2. Do we want a maximum tail size / timeout safety valve so long unbroken prose
   still emits periodically even without paragraph breaks?
3. Should debug logging be temporary, or kept permanently at debug level for
   future streaming regressions?
4. Is there any plugin besides pykofinder that relies on full-turn context in a
   way that would invalidate safe-prefix transforms?

## Definition of done

- [ ] ACP no longer buffers the full turn before emitting text
- [ ] Incremental `agent_message_chunk` notifications resume in Mitto
- [ ] Incomplete markdown links, wikilinks, HTML image tags, and code fences do
      not leak as malformed partial output
- [ ] Existing plugin transform contract remains unchanged for v1
- [ ] Unit tests cover safe-prefix boundary detection
- [ ] ACP server tests cover multi-chunk emission and final tail flush
- [ ] Integration tests validate concatenated output equality rather than
      single-chunk semantics
- [ ] Staging verification confirms visibly incremental rendering in Mitto
- [ ] Tool-call visibility plan in this backlog item remains compatible with the
      restored streaming pipeline
