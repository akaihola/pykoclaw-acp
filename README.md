# pykoclaw-acp

Agent Client Protocol plugin for
[pykoclaw]. Exposes the pykoclaw Claude
agent as an [ACP]-compatible server over JSON-RPC 2.0 on stdio, allowing any ACP
client to connect -- editors, web interfaces, and mobile apps.

## Usage

```bash
pykoclaw acp
```

This starts the ACP server. It reads JSON-RPC messages from stdin and writes
responses to stdout (logging goes to stderr). The server keeps running until
stdin is closed.

Each ACP session maps to a pykoclaw conversation named `acp-<session_id[:8]>`,
with a working directory at
`~/.local/share/pykoclaw/conversations/acp-<session_id[:8]>/`. Sessions persist
across prompts -- the underlying Claude process stays alive for multi-turn
conversations and is evicted after 10 minutes of inactivity.

## Using with Mitto

[Mitto] is a multi-interface ACP client (CLI, web browser, macOS app) that can
manage multiple AI coding agents simultaneously. To connect Mitto to pykoclaw:

### 1. Install pykoclaw with the ACP plugin

```bash
uv tool install pykoclaw@git+https://github.com/akaihola/pykoclaw.git \
    --with=pykoclaw-acp@git+https://github.com/akaihola/pykoclaw-acp.git
```

Verify the command works:

```bash
pykoclaw acp
```

(Press Ctrl+C to stop.)

### 2. Install Mitto

On macOS or Linux:

```bash
brew tap inercia/mitto
brew install mitto           # CLI only
# or
brew install --cask mitto-app  # macOS app (includes CLI)
```

Or download a release from the [Mitto releases page].

### 3. Configure Mitto

Add pykoclaw as an ACP server in your Mitto configuration.

**YAML** (`~/.mittorc`):

```yaml
acp:
  - pykoclaw:
      command: pykoclaw acp
```

**JSON** (`~/.local/share/mitto/settings.json` on Linux,
`~/Library/Application Support/Mitto/settings.json` on macOS):

```json
{
  "acp_servers": [
    { "name": "pykoclaw", "command": "pykoclaw acp" }
  ]
}
```

You can also run `mitto config create` to generate a default configuration file
and then edit it.

### 4. Start Mitto

```bash
# Interactive terminal
mitto cli --acp pykoclaw

# Web interface
mitto web --acp pykoclaw

# Or simply start Mitto and select pykoclaw from the server dropdown
mitto web
```

If pykoclaw is the only (or first) server in your config, it is used by default.

### Using alongside other agents

Mitto can manage multiple ACP agents. Add pykoclaw alongside Claude Code,
Auggie, or others:

```yaml
acp:
  - pykoclaw:
      command: pykoclaw acp
  - claude-code:
      command: npx -y @zed-industries/claude-code-acp@latest
  - auggie:
      command: auggie --acp --allow-indexing
```

Switch between them per conversation from the Mitto interface.

## Using with Toad

[Toad] is a terminal UI for ACP agents by Will McGugan (creator of Rich and
Textual). Unlike Mitto, Toad passes the command string through a shell, so
environment variables, `cd`, and pipes work as expected.

```bash
uvx toad acp "pykoclaw acp"
```

Set the data directory or working directory inline:

```bash
uvx toad acp "PYKOCLAW_DATA=/my/data pykoclaw acp"
uvx toad acp "cd /my/project && pykoclaw acp"
```

Run pykoclaw on a remote machine over SSH:

```bash
uvx toad acp "ssh myserver 'cd /my/project && pykoclaw acp'"
```

## System prompts

Two user-editable `CLAUDE.md` files control the agent's behavior:

| File                                                     | Scope                                  |
| -------------------------------------------------------- | -------------------------------------- |
| `~/.local/share/pykoclaw/CLAUDE.md`                      | Global -- applies to all conversations |
| `~/.local/share/pykoclaw/conversations/<name>/CLAUDE.md` | Per-conversation                       |

## How it works

1. The ACP client (Mitto, Toad, etc.) spawns `pykoclaw acp` as a subprocess
   and communicates via JSON-RPC 2.0 over stdio.
2. The client sends `initialize` to discover server capabilities, then
   `session/new` to create a conversation session.
3. User messages are sent via `session/prompt`. The server dispatches each
   prompt to a long-lived Claude subprocess (one per session) managed by an
   internal client pool.
4. The agent's response streams back as `session/update` notifications
   containing text chunks, followed by a final response with `stopReason:
   end_turn`.
5. Sessions are persistent -- the same Claude subprocess is reused across
   prompts, preserving full conversation context. Idle clients are evicted
   after 10 minutes.

The agent has access to standard tools (Bash, file I/O, web search) and the
built-in pykoclaw MCP tools (task scheduling).

## Protocol reference

The server implements three JSON-RPC methods:

| Method           | Purpose                           |
| ---------------- | --------------------------------- |
| `initialize`     | Handshake, returns capabilities   |
| `session/new`    | Create a new conversation session |
| `session/prompt` | Send a message, receive streaming |

Streaming responses arrive as `session/update` notifications. See the
[Agent Client Protocol specification][ACP] for the full protocol.

## Configuration

pykoclaw-acp inherits configuration from pykoclaw core:

| Variable         | Default                   | Description         |
| ---------------- | ------------------------- | ------------------- |
| `PYKOCLAW_DATA`  | `~/.local/share/pykoclaw` | Data directory      |
| `PYKOCLAW_MODEL` | `claude-opus-4-6`         | Claude model to use |

Settings are loaded from environment variables **and** `.env` files at two
locations (in order of priority):

1. `PYKOCLAW_DATA`, `PYKOCLAW_MODEL` environment variables
2. `.env` in the current working directory
3. `~/.local/share/pykoclaw/.env` (the default data directory)

If you use the default data directory, the simplest approach is to place your
settings in `~/.local/share/pykoclaw/.env`:

```bash
mkdir -p ~/.local/share/pykoclaw
cat > ~/.local/share/pykoclaw/.env << 'EOF'
PYKOCLAW_MODEL=claude-sonnet-4-20250514
EOF
```

This file is always loaded regardless of the working directory, so it works
reliably when pykoclaw is spawned by external tools like Mitto.

### Data directory with Mitto

When Mitto spawns `pykoclaw acp`, the working directory is not under your
control. This means a `.env` file in your project directory won't be found.
Two Mitto issues make this harder than it should be:

- **[Shell quoting is broken][Mitto #16]** -- commands like
  `sh -c 'PYKOCLAW_DATA=/my/dir pykoclaw acp'` fail because Mitto splits on
  whitespace without respecting quotes.
- **[No `cwd` option][Mitto #17]** -- there is no way to set the working
  directory for an ACP server in the Mitto config.

Until these are resolved, use one of these workarounds:

**Option A** -- Use the default data directory (recommended). Put your `.env` in
`~/.local/share/pykoclaw/.env`. No extra configuration needed.

**Option B** -- Export the variable in your shell profile (`~/.bashrc`,
`~/.zshrc`, etc.) before starting Mitto:

```bash
export PYKOCLAW_DATA=/path/to/my/data
```

**Option C** -- Use a wrapper script. Create e.g. `~/bin/pykoclaw-acp`:

```bash
#!/bin/sh
export PYKOCLAW_DATA=/path/to/my/data
exec pykoclaw acp
```

```bash
chmod +x ~/bin/pykoclaw-acp
```

Then in `~/.mittorc`:

```yaml
acp:
  - pykoclaw:
      command: /home/you/bin/pykoclaw-acp
```

## Installation

```bash
uv tool install pykoclaw@git+https://github.com/akaihola/pykoclaw.git \
    --with=pykoclaw-acp@git+https://github.com/akaihola/pykoclaw-acp.git
```

Or with `uv pip install`:

```bash
uv pip install pykoclaw@git+https://github.com/akaihola/pykoclaw.git
uv pip install pykoclaw-acp@git+https://github.com/akaihola/pykoclaw-acp.git
```

See the [pykoclaw README] for more details.

[ACP]: https://agentclientprotocol.com/
[Mitto]: https://github.com/inercia/mitto
[Mitto #16]: https://github.com/inercia/mitto/issues/16
[Mitto #17]: https://github.com/inercia/mitto/issues/17
[Mitto releases page]: https://github.com/inercia/mitto/releases
[Toad]: https://github.com/batrachianai/toad
[pykoclaw]: https://github.com/akaihola/pykoclaw
[pykoclaw README]: https://github.com/akaihola/pykoclaw
