# Backlog: ACP Architecture Fragility — Process-Isolate the SDK

**Priority:** High
**Status:** Open
**Created:** 2026-02-21
**Tags:** architecture, acp, anyio, stability

## Plan (medium-term recommendation)

Move toward **process-isolated workers**: run the Claude SDK in
dedicated subprocess workers instead of in the same event loop as the
ACP server. This eliminates the anyio/asyncio impedance mismatch at
the only truly reliable boundary — process isolation.

### Implementation steps

1. Extract `ClientPool._query()` into a standalone worker script
2. Have `ClientPool` spawn/manage worker processes instead of
   in-process SDK clients
3. Communicate over a Unix socket or pipe with a simple protocol
   (could reuse JSON-RPC)
4. The ACP server becomes a pure asyncio router/proxy — boring,
   reliable

### Also: unify the message loops

Extract the "iterate SDK responses and collect text" logic into a
shared function that both `query_agent()` and `ClientPool._query()`
call. Issue #4 had to be fixed in three repos — this is a red flag.

```python
async def consume_sdk_response(client, on_text, on_result):
    """Single source of truth for SDK message consumption."""
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock) and block.text:
                    on_text(block.text)
        elif isinstance(message, ResultMessage):
            if message.result and not saw_text:
                on_text(message.result)
            on_result(message)
```

Small change with high leverage against an entire category of bugs.

---

## Context: Critical Architecture Review

*Analysis performed 2026-02-21 in Mitto "ACP Architecture Review"
session, salvaged here after the ACP process crashed (the very bug
this review describes).*

### What we've built

A three-process-deep stack: **Mitto (Go)** → **pykoclaw-acp
(Python/asyncio)** → **claude CLI subprocess (Python/anyio)**.
Communication is JSON-RPC 2.0 over stdio between Mitto and pykoclaw,
and the claude-agent-sdk's internal protocol between pykoclaw and the
claude CLI. The middle layer (pykoclaw-acp) manages long-lived Claude
sessions in a `ClientPool`, handles streaming, and bridges two async
runtime models.

---

### The honest assessment: the architecture is structurally fragile

Five distinct failure modes in six days is not a streak of bad luck —
it's the architecture telling you something.

#### 1. The anyio/asyncio impedance mismatch is not fixable — only containable

This is the deepest problem. The claude-agent-sdk uses **anyio** (task
groups, cancel scopes). Our code uses **asyncio**. These are two
different concurrency models sharing one event loop, and they have
incompatible assumptions about cancellation semantics.

Mitigations (`asyncio.shield()`, spawning disconnect in a separate
`asyncio.Task`) are clever and they work *today*. But they're
defensive patches over a boundary that should not exist in the same
process. Every time Anthropic updates the SDK, we're exposed to new
leak vectors we can't predict.

**Verdict:** This is an inherent architectural weakness. No amount of
careful coding eliminates it — we're playing whack-a-mole with cancel
scope leaks.

**Update 2026-02-21:** Confirmed. `asyncio.shield()` and
`create_task()` both failed to prevent the leak because anyio cancel
scopes target the *host task* by identity, not through the normal
asyncio cancellation path. The current workaround kills the subprocess
directly (`_kill_client`) without calling `client.disconnect()`.

#### 2. Three-process-deep is one process too many

The stack is: Mitto → pykoclaw-acp → claude CLI. Each `→` is a
subprocess boundary with lifecycle management, communication protocol,
and failure propagation. A crash at the bottom ripples up as a
different failure at the top.

The middle process exists primarily to hold long-lived
`ClaudeSDKClient` instances in memory, translate protocols, and run a
watchdog because the event loop can freeze. All five bugs lived here.

#### 3. Two SDK message loops is a design smell

`query_agent()` (WhatsApp/scheduler) and `ClientPool._query()` (ACP)
duplicate the same SDK message consumption logic. Bugs must be fixed
in both. This isn't just about DRY — it's a sign the abstraction
boundary is in the wrong place.

#### 4. The ClientPool is fighting the SDK's design

The SDK wants `async with ClaudeSDKClient(...) as client:` — a
context manager with a bounded lifetime. We keep clients alive across
calls, with idle eviction, disconnect isolation, crash-retry, etc.
This creates enormous surface area for state management bugs.

---

### Architecture options considered

#### Option A: Eliminate the middle layer

Make claude CLI speak ACP directly. **Not an option** — we don't
control the SDK or CLI.

#### Option B: Talk to Claude API directly via HTTP

Remove the claude CLI subprocess entirely. Call the Claude API with
`httpx`. Pure asyncio, no anyio contamination, deterministic shutdown,
single message loop.

**What you'd lose:** Claude Code's tool execution (Bash, Read, Write),
MCP integration, CLAUDE.md system prompt loading, session management.

**Verdict:** Only works if you don't need Claude Code's tool
execution. If pykoclaw-acp is a coding agent, you need the CLI.

#### Option C: Process-isolated workers ← RECOMMENDED

Fork dedicated worker processes for each session. Communication over
Unix socket or pipe. Worker owns the entire anyio stack. ACP server is
pure asyncio.

```
Mitto → pykoclaw-acp (pure asyncio, JSON-RPC server)
           → worker process 1 (anyio/SDK, one per session)
           → worker process 2
           ...
```

**What this buys:**
- **Complete cancellation isolation.** Anyio cancel scopes can never
  leak into the server loop.
- **Crash isolation.** SDK crash kills one worker, not the whole
  server.
- **Clean shutdown.** SIGTERM the worker, wait, SIGKILL. No
  `os._exit()` hacks.
- **Watchdog becomes trivial.** Monitor heartbeats over the pipe.
- **Single ACP server process.** No zombie chains.

#### Option D: Ditch ACP, use HTTP

Replace stdio JSON-RPC with WebSocket/SSE. Mitto connects via HTTP.
Eliminates the parent-child process relationship entirely.

**Verdict:** Good long-term option when/if Mitto adds HTTP-based agent
support.

---

### Recommendation summary

| Timeframe | Action |
| --------- | ------ |
| Short term | Current mitigations work. `_kill_client` workaround avoids the anyio leak. Stay vigilant. |
| Medium term | **Option C** — process-isolated workers. Eliminates root cause. Implement incrementally. |
| Long term | **Option D** — HTTP server. When Mitto supports it. |

---

## Issue log reference

See [ACP_ISSUES_LOG.md] in the workspace root for the detailed
chronological record of all five issues that motivated this review.

[ACP_ISSUES_LOG.md]: ../../../ACP_ISSUES_LOG.md
