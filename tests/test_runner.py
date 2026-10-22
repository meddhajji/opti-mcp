import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))

from run import PLUGIN_TOOLS, PROMPT, SUBMIT_PROMPT, command, load_problems, parse_stream  # noqa: E402


def test_conditions_differ_only_in_what_they_allow():
    reasoning, code, plugin = (command("haiku", c, 1.0) for c in ("reasoning", "code", "plugin"))
    for cmd in (reasoning, code, plugin):  # shared isolation flags
        assert {"-p", "--restricted", "--strict-mcp-config", "--no-session-persistence"} <= set(cmd)
        assert cmd[cmd.index("--model") + 1] == "haiku"
    tools = {name: cmd[cmd.index("--tools") + 1] for name, cmd in (("r", reasoning), ("c", code), ("p", plugin))}
    assert tools == {"r": "", "c": "Bash,Read,Write,Edit", "p": ""}  # explicit lists: no web search, no sub-agents
    assert "--plugin-dir" not in reasoning and "--plugin-dir" not in code
    assert "--plugin-dir" in plugin and "--mcp-config" in plugin and PLUGIN_TOOLS in plugin
    assert "launch.py" in plugin[plugin.index("--mcp-config") + 1]


def test_prompt_never_mentions_the_plugin():
    for prompt in (PROMPT, SUBMIT_PROMPT):
        text = prompt.format(question="Q", keys='["a"]').lower()
        assert "plugin" not in text and "optimizer" not in text and "solver" not in text


def test_answer_tool_takes_the_answer_from_the_call_not_the_text(tmp_path, monkeypatch):
    import run

    problem = load_problems(["optibench-573"])[0]
    seen = {}

    def fake_api(backend, prompt, model, condition, timeout_s, workdir, answer_keys=None):
        seen.update(prompt=prompt, keys=answer_keys)
        answer = {k: problem["expected"][k] for k in problem["answer_keys"]}
        return {"text": "FINAL ANSWER: {}", "tools": ["submit_answer"], "turns": 1, "cost_usd": None,
                "stop": "success", "answer": answer}, None

    monkeypatch.setattr(run, "run_api", fake_api)
    row = run.run_one(problem, "m", "code", 1.0, 60, tmp_path, "groq", answer_tool=True)
    assert row["outcome"] == "optimal" and row["protocol"] == "submit_answer"
    assert seen["keys"] == problem["answer_keys"] and "submit_answer" in seen["prompt"]


def test_parse_stream_collects_tools_and_final_text():
    events = [
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Let me model this."},
            {"type": "tool_use", "name": f"{PLUGIN_TOOLS}__solve_model", "input": {}},
        ]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": "{}"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": 'FINAL ANSWER: {"x": 1}'}]}},
        {"type": "result", "subtype": "success", "result": 'FINAL ANSWER: {"x": 1}', "num_turns": 3,
         "total_cost_usd": 0.012, "is_error": False},
    ]
    out = parse_stream("\n".join(json.dumps(e) for e in events) + "\nnot json\n")
    assert out["tools"] == [f"{PLUGIN_TOOLS}__solve_model"]
    assert out["text"] == 'FINAL ANSWER: {"x": 1}'
    assert (out["turns"], out["cost_usd"], out["stop"]) == (3, 0.012, "success")
