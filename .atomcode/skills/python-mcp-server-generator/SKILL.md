---
name: python-mcp-server-generator
description: Scaffold and maintain custom MCP servers for the contract-harness toolchain. Generates a minimal stdio MCP server (Python + mcp SDK) with tool/resource registrations, then wires it into .mcp.json. Use when exposing harness capabilities (kb search, review, replay) as MCP tools for other Agents.
---

# Python MCP Server Generator

You generate production-ready MCP servers in Python. Follow this contract.

## 1. Scaffold

Create a new directory under `mcp_servers/<name>/` with:

```
mcp_servers/<name>/
├── __init__.py
├── server.py        # entrypoint, stdio transport
├── tools.py         # @mcp.tool() registrations
├── resources.py     # @mcp.resource() registrations
└── README.md
```

## 2. server.py template

```python
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("<name>")

# import tools/resources to register decorators
from . import tools, resources  # noqa: F401

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

## 3. tools.py template

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("<name>")

@mcp.tool()
def kb_search(query: str, top_k: int = 5) -> list[dict]:
    """Search the legal knowledge base.

    Args:
        query: natural language search query
        top_k: number of results to return

    Returns:
        List of {chunk_id, text, score, metadata} dicts
    """
    # delegate to harness.rag.knowledge_base.KnowledgeBase.query
    ...
```

## 4. Wire into .mcp.json

Append to the project root `.mcp.json` (NOT `.atomcode/mcp.json` — the loader does not read it):

```json
"<name>": {
  "command": "python",
  "args": ["-m", "mcp_servers.<name>.server"],
  "env": {
    "HARNESS_DATA_DIR": "${HARNESS_DATA_DIR}"
  }
}
```

## 5. Conventions for This Codebase

- **Delegates, doesn't reimplement** — the MCP server imports `harness.rag.knowledge_base`, `harness.replay.player`, etc. and calls their public APIs. Never duplicate logic.
- **No secrets in .mcp.json** — use `${ENV_VAR}` placeholders; real values live in `.env` (which is gitignored).
- **Stdio transport only** — matches the existing context7 + github servers. No HTTP/SSE unless the user asks.
- **Type hints required** — the MCP SDK uses them to generate JSON schema. `Any` is forbidden per the 2026-07-09 cleanup; use concrete types or `dict[str, Any]`.
- **Errors surface as tool results** — catch `HarnessError` subclasses and return structured error dicts; don't let exceptions kill the server.

## 6. Verification

```bash
# syntax + types
ruff check mcp_servers/<name>/
pyright mcp_servers/<name>/

# manual smoke test (server speaks MCP over stdio)
python -m mcp_servers.<name>.server < test_input.json
```

Restart AtomCode for `.mcp.json` changes to take effect.
