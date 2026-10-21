"""What every API-backed benchmark agent shares, so that providers differ only in
how they call their model:

  reasoning  no tools
  code       one run_python tool: runs the model's script in an empty folder with
             the system Python, like Claude Code's Bash tool
  plugin     the optimizer MCP tools, listed live from the real server
             (scripts/launch.py), plus the optimize skill as instructions

Benchmark harness only: the plugin itself never calls an LLM.
"""
import json
import os
import re
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

REPO = Path(__file__).resolve().parent.parent
MAX_TURNS = 10
TOOL_RESULT_CHARS = 6000
PYTHON_TIMEOUT_S = 60

RUN_PYTHON = {
    "name": "run_python",
    "description": "Run a Python 3 script in an empty folder and return what it prints (stdout and "
                   "stderr). Time limit 60 seconds. Only already-installed packages are available.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
}


class DailyLimitReached(RuntimeError):
    """The provider's per-day quota is used up; waiting minutes will not help."""


def env_key(name: str) -> str:
    """An API key from the environment, or from .env at the repo root (git-ignored)."""
    key = os.environ.get(name)
    env = REPO / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{name}="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise RuntimeError(f"{name} is not set: export it or add it to .env at the repo root")
    return key


def skill_text() -> str:
    text = (REPO / "skills" / "optimize" / "SKILL.md").read_text(encoding="utf-8")
    return re.sub(r"\A---.*?---\s*", "", text, flags=re.DOTALL)  # drop the YAML front matter


def deref(schema: dict) -> dict:
    """Inline $ref pointers into $defs: some models handle flat schemas better."""
    defs = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


def run_python(code: str, workdir: Path) -> str:
    script = workdir / "script.py"
    script.write_text(code, encoding="utf-8")
    try:
        proc = subprocess.run(["python", str(script)], cwd=workdir, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=PYTHON_TIMEOUT_S)
        out = proc.stdout + (f"\n[stderr]\n{proc.stderr}" if proc.stderr else "")
    except subprocess.TimeoutExpired:
        out = f"Timed out after {PYTHON_TIMEOUT_S} seconds."
    return out[:TOOL_RESULT_CHARS] or "(no output)"


@asynccontextmanager
async def condition_tools(condition: str, workdir: Path):
    """Yields (tool specs {name, description, parameters}, async call_tool(name, args) -> str)."""
    workdir.mkdir(parents=True, exist_ok=True)
    if condition == "plugin":
        params = StdioServerParameters(command="python", args=[str(REPO / "scripts" / "launch.py")], cwd=str(workdir))
        async with Client(params) as mcp:
            specs = [{"name": t.name, "description": t.description or "", "parameters": deref(t.input_schema)}
                     for t in (await mcp.list_tools()).tools]

            async def call_mcp(name, args):
                res = await mcp.call_tool(name, args)
                if res.structured_content is not None:
                    return json.dumps(res.structured_content)
                return "\n".join(getattr(c, "text", "") for c in res.content)

            yield specs, call_mcp
    elif condition == "code":
        async def call_python(name, args):
            return run_python(args.get("code", ""), workdir) if name == "run_python" else f"Unknown tool {name}"

        yield [RUN_PYTHON], call_python
    else:
        async def no_tools(name, args):
            return "No tools are available."

        yield [], no_tools


class CallLog:
    """Runs tool calls, reporting failures back to the model, and records each one."""

    def __init__(self, call_tool):
        self.call_tool, self.names, self.calls = call_tool, [], []

    async def execute(self, name: str, args) -> str:
        self.names.append(name)
        raw = args if isinstance(args, str) else json.dumps(args)
        try:
            result = await self.call_tool(name, json.loads(args or "{}") if isinstance(args, str) else (args or {}))
        except Exception as e:  # an agent tells the model what went wrong
            result = f"Tool error: {type(e).__name__}: {str(e)[:500]}"
        self.calls.append({"name": name, "arguments": raw[:2000], "result": result[:800]})
        return result[:TOOL_RESULT_CHARS]
