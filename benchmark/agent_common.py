"""What every API-backed benchmark agent shares, so that providers differ only in
how they call their model:

  reasoning  no tools
  code       one run_python tool: runs the model's script in an empty folder with
             the system Python, like Claude Code's Bash tool
  plugin     the optimizer MCP server (python -m optimizer.server), its tools listed live
             and its own instructions as the system text, as any MCP client gets them
  library    run_python, where scripts can also `import optimizer`, plus the same guidance

With answer keys given, every condition also gets a submit_answer tool with one number
per key, and the run ends when it is called: the answer arrives as structured
arguments instead of being parsed from prose. A model that stops without calling it is
reminded once.

Benchmark harness only: the plugin itself never calls an LLM.
"""
import json
import os
import re
import subprocess
import sys
import sysconfig
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

REPO = Path(__file__).resolve().parent.parent
# The plugin under test: this checkout by default, or another version checked out elsewhere
# (OPTIMIZER_PLUGIN_ROOT), so two versions can be compared in the same harness.
PLUGIN_ROOT = Path(os.environ.get("OPTIMIZER_PLUGIN_ROOT") or REPO)
MAX_TURNS = 10
TOOL_RESULT_CHARS = 6000
PYTHON_TIMEOUT_S = 60

RUN_PYTHON = {
    "name": "run_python",
    "description": "Run a Python 3 script in an empty folder and return what it prints (stdout and "
                   "stderr). Time limit 60 seconds. Only already-installed packages are available.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
}


SUBMIT = "submit_answer"
REMINDER = "You have not submitted an answer. Call submit_answer now with your final answer."


def answer_fields(answer_keys: list[str]) -> dict[str, str]:
    """{tool argument name: answer key}. Keys are sentences; argument names must be identifiers."""
    fields = {}
    for key in answer_keys:
        base = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")[:60] or "value"
        name, i = base, 2
        while name in fields:
            name, i = f"{base}_{i}", i + 1
        fields[name] = key
    return fields


def submit_spec(answer_keys: list[str]) -> dict:
    fields = answer_fields(answer_keys)
    return {
        "name": SUBMIT,
        "description": "Submit your final answer. Call it once, when you are done: the conversation ends there.",
        "parameters": {"type": "object", "required": list(fields),
                       "properties": {f: {"type": "number", "description": key} for f, key in fields.items()}},
    }


class StopRun(RuntimeError):
    """The provider cannot be used for now: stop, record nothing, and resume later."""


class DailyLimitReached(StopRun):
    """The provider's per-day quota is used up; waiting minutes will not help."""


class ProviderUnavailable(StopRun):
    """Connections keep failing (an outage): no run can say anything about the model."""


def unwrap(error: BaseException) -> BaseException:
    """The real error inside the exception groups that async task groups (the MCP client) wrap around it."""
    while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
        error = error.exceptions[0]
    return error


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


def legacy_skill() -> str | None:
    """Versions up to v0.3 shipped their guidance as a Claude Code skill rather than as server
    instructions: a pinned old version is run with its own skill, so its results stay comparable."""
    path = PLUGIN_ROOT / "skills" / "optimize" / "SKILL.md"
    if not path.exists():
        return None
    return re.sub(r"\A---.*?---\s*", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)  # drop the front matter


LIBRARY_TEXT = (
    "The Python package `optimizer` is installed. It solves optimisation problems exactly: answers are proved "
    "optimal and checked independently. Use it in run_python instead of working out the optimum yourself:\n\n"
    "    from optimizer import solve, get_format\n"
    "    print(solve(kind, data))  # a dict: status, the plan and its value\n\n"
    "{catalogue}\n\n"
    "For shifts and model, print(get_format(kind)) first. If a rule is ambiguous, say which reading you used. If the "
    "answer is INVALID_INPUT, fix what it lists and run again. If no plan exists, report the conflicting rules; never "
    "drop a rule yourself."
)


def library_text() -> str:
    from optimizer.kinds import catalogue
    return LIBRARY_TEXT.format(catalogue=catalogue())


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


# First on the code condition's path: an optimizer package whose import fails.
BLOCKED = Path(__file__).resolve().parent / "blocked"


def library_site() -> Path:
    """This harness's own environment, which holds OR-Tools."""
    return Path(sysconfig.get_paths()["purelib"])


def run_python(code: str, workdir: Path, library: bool = False) -> str:
    """Runs with the system Python, like the code condition. With library, the optimizer
    package comes first on the path and the plugin's packages last, so the system's own
    numpy and scipy are the ones scripts get, exactly as in the code condition. Without it,
    a stub shadows the optimizer even where that Python has it installed (CI, or after a pip
    install), so the code condition can never reach the tool; every other error reads as before."""
    script = (workdir / "script.py").resolve()
    script.write_text(code, encoding="utf-8")
    cmd = ["python", str(script)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(BLOCKED), os.environ.get("PYTHONPATH")]))}
    if library:
        boot = (f"import runpy, sys; sys.path.insert(0, {str(REPO)!r}); sys.path.append({str(library_site())!r}); "
                "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')")
        cmd, env = ["python", "-c", boot, str(script)], None
    try:
        proc = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=PYTHON_TIMEOUT_S)
        out = proc.stdout + (f"\n[stderr]\n{proc.stderr}" if proc.stderr else "")
    except subprocess.TimeoutExpired:
        out = f"Timed out after {PYTHON_TIMEOUT_S} seconds."
    return out[:TOOL_RESULT_CHARS] or "(no output)"


@asynccontextmanager
async def condition_tools(condition: str, workdir: Path, answer_keys: list[str] | None = None):
    """Yields (tool specs {name, description, parameters}, async call_tool(name, args) -> str,
    system text or None). With answer_keys the submit_answer spec is added; CallLog handles its calls."""
    workdir.mkdir(parents=True, exist_ok=True)
    async with _condition_tools(condition, workdir) as (specs, call_tool, system):
        yield specs + ([submit_spec(answer_keys)] if answer_keys else []), call_tool, system


@asynccontextmanager
async def _condition_tools(condition: str, workdir: Path):
    if condition == "plugin":
        # The version under test comes first on the path; its dependencies are this environment's.
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(PLUGIN_ROOT), os.environ.get("PYTHONPATH", "")])}
        params = StdioServerParameters(command=sys.executable, args=["-m", "optimizer.server"], cwd=str(workdir), env=env)
        async with Client(params) as mcp:
            specs = [{"name": t.name, "description": t.description or "", "parameters": deref(t.input_schema)}
                     for t in (await mcp.list_tools()).tools]
            system = legacy_skill() or mcp.instructions

            async def call_mcp(name, args):
                res = await mcp.call_tool(name, args)
                if res.structured_content is not None:
                    return json.dumps(res.structured_content)
                return "\n".join(getattr(c, "text", "") for c in res.content)

            yield specs, call_mcp, system
    elif condition in ("code", "library"):
        async def call_python(name, args):
            if name != "run_python":
                return f"Unknown tool {name}"
            return run_python(args.get("code", ""), workdir, library=condition == "library")

        yield [RUN_PYTHON], call_python, library_text() if condition == "library" else None
    else:
        async def no_tools(name, args):
            return "No tools are available."

        yield [], no_tools, None


class CallLog:
    """Runs tool calls, reporting failures back to the model, and records each one.
    With answer keys it also takes submit_answer calls: `answer` is set by the first valid one."""

    def __init__(self, call_tool, answer_keys: list[str] | None = None):
        self.call_tool, self.names, self.calls = call_tool, [], []
        self.fields = answer_fields(answer_keys) if answer_keys else {}
        self.answer = None

    async def execute(self, name: str, args) -> str:
        self.names.append(name)
        raw = args if isinstance(args, str) else json.dumps(args)
        try:
            parsed = json.loads(args or "{}") if isinstance(args, str) else (args or {})
            result = self.submit(parsed) if name == SUBMIT and self.fields else await self.call_tool(name, parsed)
        except Exception as e:  # an agent tells the model what went wrong
            result = f"Tool error: {type(e).__name__}: {str(e)[:500]}"
        self.calls.append({"name": name, "arguments": raw[:2000], "result": result[:800]})
        return result[:TOOL_RESULT_CHARS]

    def submit(self, args: dict) -> str:
        values, missing = {}, []
        for field, key in self.fields.items():
            try:
                value = args[field]
                values[key] = float(value.replace(",", "")) if isinstance(value, str) else float(value)
            except (KeyError, TypeError, ValueError):
                missing.append(field)
        if missing:
            return f"Not recorded: give a number for each of {missing}, then call submit_answer again."
        self.answer = values
        return "Answer recorded."
