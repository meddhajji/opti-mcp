"""MCP server: exposes the solver to any MCP client or agent framework. Over stdio here
(python -m optimizer.server), or over HTTP in optimizer/web.py. Needs no API key and
never calls an LLM itself.

Two tools, kept deliberately small because an agent re-reads every tool description on
every request:

  solve(kind, data)   its description is the catalogue: one line per problem kind
  get_format(kind)    the full format of one kind, fetched only when needed

The server's instructions (kinds.GUIDANCE) tell the model when and how to use them.
Version 0.1 exposed four tools with full JSON schemas (about 3,900 tokens per request
with its guidance); this surface is about 700. See docs/benchmark.md, "Making it light".
"""
import json
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer

from . import kinds

KindName = Literal[kinds.KIND_NAMES]

server = MCPServer(name="optimizer", instructions=kinds.GUIDANCE)


@server.tool(description=kinds.catalogue(), structured_output=False)
def solve(kind: KindName, data: dict[str, Any]) -> str:
    # One line of JSON: indented output costs the agent a third more tokens for nothing.
    return json.dumps(kinds.solve(kind, data), ensure_ascii=False)


@server.tool(description="The full input format of one kind, with an example. Needed for shifts and model.",
             structured_output=False)
def get_format(kind: KindName) -> str:
    return kinds.get_format(kind)


def main():
    server.run("stdio")


if __name__ == "__main__":
    main()
