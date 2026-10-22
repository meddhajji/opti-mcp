"""Run benchmark problems through Claude Code and score the answers.

Each run sends one problem to `claude -p` in restricted mode with an explicit
tool list, so nothing else is available: no web search (benchmark problems are
public), no sub-agents, none of the user's own plugins, settings or claude.ai
connectors (--strict-mcp-config), and a fresh empty working folder:

  reasoning  no tools at all: the model answers from its own reasoning
  code       Bash, Read, Write, Edit: the model may write and run its own code
  plugin     the optimizer skill (--plugin-dir) and its MCP server, no other tools

The plugin's MCP server is passed with --mcp-config (the same launcher command
the plugin declares) because --strict-mcp-config also drops servers that come
from --plugin-dir; the skill still loads from the plugin.

The prompt is identical in every condition and never mentions the plugin:
whether the agent chooses to use it is part of what is measured.

With --backend groq or gemini, the same problems, prompt and conditions go
through benchmark/groq_agent.py or gemini_agent.py instead (shared logic in
agent_common.py), to test models weaker than Claude Code runs. There,
--answer-tool replaces the FINAL ANSWER line with a submit_answer tool call, so
the answer is structured data rather than text to parse, and a fourth condition
is available:

  library    run_python, where scripts can also `import optimizer`, plus the
             same guidance as the skill (the plugin used as a Python library)

Results are appended to benchmark/results/<run name>.jsonl, one line per
(problem, model, condition); finished combinations are skipped, so an
interrupted run resumes where it stopped.

Example:
  python benchmark/run.py --name pilot --models haiku --conditions reasoning plugin --problems optibench-573
"""
import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BENCH))

from score import OUTCOMES, parse_final_answer, score  # noqa: E402

CLAUDE = os.environ.get("CLAUDE_BIN") or shutil.which("claude") or str(Path.home() / ".local/bin/claude.exe")
PLUGIN_TOOLS = "mcp__optimizer"  # every tool of the optimizer MCP server, as Claude Code names them
OPTIMIZER_TOOLS = {"solve", "get_format",  # raw MCP names, v0.2, then v0.1
                   "solve_model", "verify_solution", "explain_infeasibility", "schedule_shifts"}
MCP_CONFIG = json.dumps({"mcpServers": {"optimizer": {
    "command": "python", "args": [str(REPO / "scripts" / "launch.py")]}}})
CONDITIONS = ("reasoning", "code", "plugin", "library")  # library: API backends only

PROMPT = """{question}

When you are done, end your reply with one line in exactly this form:
FINAL ANSWER: {{json object}}
The JSON object must have exactly these keys, each with a number as its value:
{keys}"""

# With --answer-tool the answer is the submit_answer call's arguments, never parsed from prose.
SUBMIT_PROMPT = """{question}

When you are done, call the submit_answer tool once with your final answer: a number for each of
{keys}"""


def load_problems(ids: list[str] | None, problem_set: str | None = None) -> list[dict]:
    """The hand-built problems, or a generated set from benchmark/sets/<problem_set>/."""
    folder = BENCH / "sets" / problem_set if problem_set else BENCH / "problems"
    if not folder.is_dir():
        sys.exit(f"no problem folder {folder}")
    problems = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]
    if ids:
        known = {p["id"] for p in problems}
        unknown = set(ids) - known
        if unknown:
            sys.exit(f"unknown problem ids: {sorted(unknown)}")
        problems = [p for p in problems if p["id"] in ids]
    return problems


def command(model: str, condition: str, budget_usd: float) -> list[str]:
    cmd = [CLAUDE, "-p", "--model", model, "--output-format", "stream-json", "--verbose",
           "--restricted", "--strict-mcp-config", "--no-session-persistence", "--max-budget-usd", str(budget_usd)]
    if condition == "code":
        return cmd + ["--tools", "Bash,Read,Write,Edit", "--allowedTools", "Bash", "Read", "Write", "Edit"]
    cmd += ["--tools", ""]
    if condition == "plugin":
        cmd += ["--plugin-dir", str(REPO), "--mcp-config", MCP_CONFIG, "--allowedTools", PLUGIN_TOOLS]
    return cmd


def parse_stream(stdout: str) -> dict:
    """Final text, tools called, turns, cost and usage from stream-json output."""
    tools, final, result = [], "", {}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    tools.append(block.get("name"))
                elif block.get("type") == "text":
                    final = block.get("text", "")  # the last text block holds the answer
        elif event.get("type") == "result":
            result = event
    return {
        "text": result.get("result") or final,
        "tools": tools,
        "turns": result.get("num_turns"),
        "cost_usd": result.get("total_cost_usd"),
        "is_error": result.get("is_error"),
        "stop": result.get("subtype"),
    }


def run_claude(prompt: str, model: str, condition: str, budget_usd: float, timeout_s: int, workdir: Path):
    try:
        proc = subprocess.run(command(model, condition, budget_usd), input=prompt, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", cwd=workdir, timeout=timeout_s)
        stream = parse_stream(proc.stdout)
        error = None if proc.returncode == 0 else (proc.stderr.strip()[-500:] or f"exit code {proc.returncode}")
    except subprocess.TimeoutExpired:
        stream, error = {"text": "", "tools": [], "turns": None, "cost_usd": None, "is_error": True, "stop": "timeout"}, \
            f"timed out after {timeout_s} s"
    return stream, error


def run_api(backend: str, prompt: str, model: str, condition: str, timeout_s: int, workdir: Path,
            answer_keys: list[str] | None = None):
    """Groq or Gemini models through this repo's own agent loop (imported lazily: optional SDKs)."""
    import importlib

    from agent_common import DailyLimitReached

    agent = importlib.import_module(f"{backend}_agent")
    empty = {"text": "", "tools": [], "turns": None, "cost_usd": None}
    try:
        out = asyncio.run(asyncio.wait_for(agent.run(prompt, model, condition, workdir, answer_keys=answer_keys),
                                           timeout_s))
    except TimeoutError:
        return empty | {"stop": "timeout"}, f"timed out after {timeout_s} s"
    except DailyLimitReached:
        raise
    except Exception as e:  # one failed run is recorded, never fatal for the whole benchmark
        return empty | {"stop": "error"}, f"{type(e).__name__}: {str(e)[:400]}"
    stream = {"text": out["text"], "tools": out["tools"], "turns": out["turns"], "cost_usd": None,
              "tokens": out["tokens"], "stop": "; ".join(out["events"]) or "success", "calls": out["calls"],
              "answer": out.get("answer")}
    return stream, None


def run_one(problem: dict, model: str, condition: str, budget_usd: float, timeout_s: int, workdir: Path,
            backend: str = "claude", answer_tool: bool = False) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    keys = problem["answer_keys"]
    prompt = (SUBMIT_PROMPT if answer_tool else PROMPT).format(
        question=problem["question"], keys=json.dumps(keys, ensure_ascii=False))
    started = time.monotonic()
    if backend in ("groq", "gemini"):  # DailyLimitReached propagates
        stream, error = run_api(backend, prompt, model, condition, timeout_s, workdir, keys if answer_tool else None)
    elif answer_tool:
        raise ValueError("submit_answer is only implemented for the groq and gemini backends")
    else:
        stream, error = run_claude(prompt, model, condition, budget_usd, timeout_s, workdir)
    answer = stream.get("answer") if answer_tool else parse_final_answer(stream["text"])
    verdict = score(problem, answer)
    if answer_tool and answer is None:
        verdict["detail"] = "submit_answer was never called with valid numbers"
    return {
        "problem": problem["id"], "model": model, "condition": condition, "backend": backend,
        **({"protocol": "submit_answer"} if answer_tool else {}),
        **{k: problem[k] for k in ("family", "level", "noise", "seed") if k in problem},
        "outcome": verdict["outcome"], "detail": verdict["detail"], "answer": answer,
        "tools": Counter(stream["tools"]),
        "plugin_used": any(t.startswith(PLUGIN_TOOLS) or t in OPTIMIZER_TOOLS for t in stream["tools"])
                       or any("optimizer" in c["arguments"] for c in stream.get("calls") or [] if c["name"] == "run_python"),
        "turns": stream["turns"], "cost_usd": stream["cost_usd"], "tokens": stream.get("tokens"),
        "seconds": round(time.monotonic() - started, 1),
        "stop": stream["stop"], "error": error, "text_tail": stream["text"][-600:],
        **({"calls": stream["calls"]} if stream.get("calls") else {}),
    }


def summarise(rows: list[dict]) -> None:
    groups: dict[tuple, Counter] = {}
    for r in rows:
        groups.setdefault((r["model"], r["condition"]), Counter())[r["outcome"]] += 1
    shown = [o for o in OUTCOMES if any(c[o] for c in groups.values())]
    print(f"\n{'model':10} {'condition':10} {'runs':>4} " + " ".join(f"{o:>12}" for o in shown))
    for (model, condition), c in sorted(groups.items()):
        n = sum(c.values())
        print(f"{model:10} {condition:10} {n:>4} " + " ".join(f"{c[o]:>12}" for o in shown)
              + f"   optimal {c['optimal'] / n:.0%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run name: results go to benchmark/results/<name>.jsonl")
    ap.add_argument("--backend", choices=("claude", "groq", "gemini"), default="claude")
    ap.add_argument("--models", nargs="+", default=["haiku"])
    ap.add_argument("--conditions", nargs="+", default=["reasoning", "code", "plugin"], choices=CONDITIONS)
    ap.add_argument("--set", dest="problem_set", help="generated set in benchmark/sets/ (default: the hand-built problems)")
    ap.add_argument("--problems", nargs="*", help="problem ids (default: all)")
    ap.add_argument("--budget-usd", type=float, default=1.0, help="spending cap per run")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per run")
    ap.add_argument("--answer-tool", action="store_true",
                    help="answers come through a submit_answer tool call (groq and gemini backends)")
    args = ap.parse_args()
    if args.backend == "claude" and (args.answer_tool or "library" in args.conditions):
        sys.exit("--answer-tool and the library condition need --backend groq or gemini")

    out = BENCH / "results" / f"{args.name}.jsonl"
    out.parent.mkdir(exist_ok=True)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()] if out.exists() else []
    done = {(r["problem"], r["model"], r["condition"]) for r in rows}
    problems = load_problems(args.problems, args.problem_set)
    todo = [(p, m, c) for p in problems for m in args.models for c in args.conditions if (p["id"], m, c) not in done]
    print(f"{len(todo)} runs to do ({len(done)} already in {out.name}), backend = {args.backend}")

    for i, (problem, model, condition) in enumerate(todo, 1):
        workdir = BENCH / "runs" / args.name / f"{problem['id']}-{model.replace('/', '_')}-{condition}"
        try:
            row = run_one(problem, model, condition, args.budget_usd, args.timeout, workdir, args.backend,
                          args.answer_tool)
        except Exception as e:
            if type(e).__name__ == "DailyLimitReached":
                print(f"Stopped: the daily quota for {model} is used up. Re-run the same command later to resume.")
                break
            raise
        rows.append(row)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(todo)}] {problem['id']:32} {model:8} {condition:9} -> {row['outcome']:12} "
              f"{row['seconds']:>6}s  tools={dict(row['tools'])}  {row['detail'][:70]}"
              + (f"  ERROR: {row['error'][:120]}" if row["error"] else ""))
    summarise(rows)


if __name__ == "__main__":
    main()
