"""A minimal tool-calling agent for models served by Groq, used to benchmark
weaker models than Claude Code can run. Benchmark harness only: the plugin
itself never calls an LLM.

Conditions mirror the Claude Code runner:
  reasoning  no tools
  code       one run_python tool: runs the model's script in an empty folder
             with the system Python, like Claude's Bash tool
  plugin     the optimizer MCP tools, listed live from the real server
             (scripts/launch.py), plus the optimize skill as the system prompt

Differences from Claude Code, stated plainly: the skill is given up front
(Claude Code shows its description and loads it on demand), and the loop is
this file's, not Claude Code's.

The API key is read from GROQ_API_KEY, or from a .env file at the repo root.
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path

import groq
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters

REPO = Path(__file__).resolve().parent.parent
MAX_TURNS = 10
# Reasoning models (gpt-oss) spend output tokens thinking before they answer; Groq's
# default cap (2,048) cut answers off mid-thought, which Claude Code runs never hit.
# Groq also counts the requested cap against the tokens-per-minute limit, so each
# request asks for whatever fits under that limit after its prompt.
MAX_COMPLETION_TOKENS = 8192
TPM_LIMIT = int(os.environ.get("GROQ_TPM_LIMIT", "8000"))  # free (on_demand) tier for gpt-oss-20b
MIN_COMPLETION_TOKENS = 256
# gpt-oss models think at length by default; under the free 8,000 tokens-per-minute limit
# the plugin condition (whose prompt carries ~4,000 tokens of tool definitions and skill)
# was left too little room to answer. "low" is applied to every condition alike, and is
# what the laptop tracker's assistant uses in production.
REASONING_EFFORT = os.environ.get("GROQ_REASONING_EFFORT", "low")
TOOL_RESULT_CHARS = 6000
PYTHON_TIMEOUT_S = 60

RUN_PYTHON = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "Run a Python 3 script in an empty folder and return what it prints (stdout and "
                       "stderr). Time limit 60 seconds. Only already-installed packages are available.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    },
}


class DailyLimitReached(RuntimeError):
    """Groq's per-day quota is used up; waiting minutes will not help."""


def api_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    env = REPO / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GROQ_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set: export it or add it to .env at the repo root")
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


def to_openai_tools(mcp_tools) -> list[dict]:
    return [{"type": "function", "function": {"name": t.name, "description": t.description or "",
                                              "parameters": deref(t.input_schema)}} for t in mcp_tools]


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


def is_daily_limit(error: Exception) -> bool:
    text = str(error)
    return "per day" in text or "(TPD)" in text or "(RPD)" in text


def retry_after_seconds(error: Exception) -> float:
    headers = getattr(getattr(error, "response", None), "headers", {}) or {}
    try:
        return float(headers.get("retry-after", 20))
    except ValueError:
        return 20.0


class RequestTooLarge(RuntimeError):
    """The prompt alone leaves no room for an answer under the per-minute token limit."""


def completion_budget(kwargs: dict) -> int:
    """Largest output cap that fits under the per-minute limit after the prompt (estimated)."""
    prompt_chars = len(json.dumps(kwargs.get("messages", []))) + len(json.dumps(kwargs.get("tools", [])))
    return max(MIN_COMPLETION_TOKENS, min(MAX_COMPLETION_TOKENS, TPM_LIMIT - int(prompt_chars / 3.2) - 300))


def overshoot(error: Exception) -> int | None:
    """From "Limit 8000, Requested 11506": by how many tokens the request was too large."""
    m = re.search(r"Limit (\d+), Requested (\d+)", str(error))
    return int(m.group(2)) - int(m.group(1)) if m else None


def chat(client, **kwargs):
    """One completion: sizes the output cap, waits out per-minute limits, stops on daily ones."""
    kwargs["max_completion_tokens"] = completion_budget(kwargs)
    for _ in range(8):
        try:
            return client.chat.completions.create(**kwargs)
        except groq.RateLimitError as e:
            if is_daily_limit(e):
                raise DailyLimitReached(str(e)[:300]) from e
            time.sleep(min(retry_after_seconds(e) + 1, 120))
        except groq.APIStatusError as e:
            excess = overshoot(e) if e.status_code == 413 else None
            if excess is None:
                raise
            smaller = kwargs["max_completion_tokens"] - excess - 200
            if smaller < MIN_COMPLETION_TOKENS:
                raise RequestTooLarge(str(e)[:300]) from e
            kwargs["max_completion_tokens"] = smaller
    raise RuntimeError("still rate-limited after 8 retries")


async def run(prompt: str, model: str, condition: str, workdir: Path, client=None) -> dict:
    """Returns {text, tools, turns, tokens, events} in the shape run.py records."""
    client = client or groq.Groq(api_key=api_key(), max_retries=0)
    workdir.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "user", "content": prompt}]
    if condition == "plugin":
        messages.insert(0, {"role": "system", "content": skill_text()})
    called, events, calls_log = [], [], []
    tokens = {"prompt": 0, "completion": 0}

    async def converse(tools, call_tool):
        for turn in range(1, MAX_TURNS + 1):
            kwargs = {"model": model, "messages": messages}
            if "gpt-oss" in model:
                kwargs["reasoning_effort"] = REASONING_EFFORT
            if tools:
                kwargs |= {"tools": tools, "tool_choice": "auto"}
            try:
                response = chat(client, **kwargs)
            except RequestTooLarge as e:
                events.append(f"turn {turn}: conversation too long for the per-minute token limit")
                return "", turn
            except groq.BadRequestError as e:  # e.g. a malformed tool call rejected by Groq
                events.append(f"turn {turn}: bad request: {str(e)[:200]}")
                messages.append({"role": "user", "content": f"Your last reply could not be processed ({str(e)[:300]}). "
                                                            "Please try again."})
                continue
            if response.usage:
                tokens["prompt"] += response.usage.prompt_tokens or 0
                tokens["completion"] += response.usage.completion_tokens or 0
            choice = response.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                events.append(f"turn {turn}: reply cut off by the output cap")
            message = choice.message
            calls = message.tool_calls or []
            messages.append({"role": "assistant", "content": message.content or "",
                             **({"tool_calls": [c.model_dump() for c in calls]} if calls else {})})
            if not calls:
                return message.content or "", turn
            for call in calls:
                called.append(call.function.name)
                try:
                    args = json.loads(call.function.arguments or "{}")
                    result = await call_tool(call.function.name, args)
                except Exception as e:  # report the failure to the model, as an agent would
                    result = f"Tool error: {type(e).__name__}: {str(e)[:500]}"
                calls_log.append({"name": call.function.name, "arguments": (call.function.arguments or "")[:2000],
                                  "result": result[:800]})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result[:TOOL_RESULT_CHARS]})
        return "", MAX_TURNS

    if condition == "plugin":
        params = StdioServerParameters(command="python", args=[str(REPO / "scripts" / "launch.py")], cwd=str(workdir))
        async with Client(params) as mcp:
            tools = to_openai_tools((await mcp.list_tools()).tools)

            async def call_tool(name, args):
                res = await mcp.call_tool(name, args)
                if res.structured_content is not None:
                    return json.dumps(res.structured_content)
                return "\n".join(getattr(c, "text", "") for c in res.content)

            text, turns = await converse(tools, call_tool)
    elif condition == "code":
        async def call_tool(name, args):
            if name != "run_python":
                return f"Unknown tool {name}"
            return run_python(args.get("code", ""), workdir)

        text, turns = await converse([RUN_PYTHON], call_tool)
    else:
        async def no_tools(name, args):
            return "No tools are available."

        text, turns = await converse([], no_tools)
    return {"text": text, "tools": called, "turns": turns, "tokens": tokens, "events": events, "calls": calls_log}
