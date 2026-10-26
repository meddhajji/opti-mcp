"""The Gemini harness with a scripted fake client built from the SDK's real types:
no API key, no quota. The optimizer MCP server is real."""
import asyncio
import sys
from pathlib import Path

from google.genai import types

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))

import gemini_agent  # noqa: E402
from gemini_agent import is_daily_limit, retry_delay_seconds  # noqa: E402

KNAPSACK = {
    "variables": [{"name": n, "type": "boolean"} for n in ("a", "b", "c")],
    "constraints": [{"name": "budget", "terms": {"a": 3, "b": 4, "c": 5}, "sense": "<=", "rhs": 8}],
    "objective": {"sense": "maximize", "terms": {"a": 4, "b": 5, "c": 7}},
}


def response(parts, finish=types.FinishReason.STOP):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts), finish_reason=finish)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100, candidates_token_count=20, thoughts_token_count=5))


class FakeGemini:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.models = self

    def generate_content(self, model, contents, config):
        self.requests.append({"model": model, "contents": list(contents), "config": config})
        return self.replies.pop(0)


def test_plugin_condition_with_the_real_mcp_server(tmp_path):
    fake = FakeGemini([
        response([types.Part(function_call=types.FunctionCall(name="solve", args={"kind": "model", "data": KNAPSACK}))]),
        response([types.Part(text='FINAL ANSWER: {"value": 12}')]),
    ])
    out = asyncio.run(gemini_agent.run("Q", "gemini-3.5-flash-lite", "plugin", tmp_path, client=fake))
    assert out["tools"] == ["solve"] and out["turns"] == 2
    assert out["text"] == 'FINAL ANSWER: {"value": 12}'
    first, second = fake.requests
    assert "even when the problem looks small" in first["config"].system_instruction  # the server's instructions
    names = {d.name for d in first["config"].tools[0].function_declarations}
    assert names == {"solve", "get_format"}
    function_response = second["contents"][-1].parts[0].function_response
    assert function_response.name == "solve" and '"OPTIMAL"' in function_response.response["result"]
    assert out["tokens"] == {"prompt": 200, "completion": 50, "cached": 0}  # thinking tokens counted as output


def test_gemma_gets_the_guidance_in_the_user_message(tmp_path):
    fake = FakeGemini([response([types.Part(text="FINAL ANSWER: {}")])])
    asyncio.run(gemini_agent.run("THE QUESTION", "gemma-4-26b-a4b-it", "plugin", tmp_path, client=fake))
    request = fake.requests[0]
    assert request["config"].system_instruction is None
    text = request["contents"][0].parts[0].text
    assert "even when the problem looks small" in text and text.endswith("THE QUESTION")


def test_code_and_reasoning_conditions(tmp_path):
    fake = FakeGemini([
        response([types.Part(function_call=types.FunctionCall(name="run_python", args={"code": "print(6 * 7)"}))]),
        response([types.Part(text="FINAL ANSWER: {}")]),
    ])
    out = asyncio.run(gemini_agent.run("Q", "gemini-3.5-flash-lite", "code", tmp_path / "c", client=fake))
    assert out["tools"] == ["run_python"]
    assert fake.requests[1]["contents"][-1].parts[0].function_response.response["result"].strip() == "42"

    fake = FakeGemini([response([types.Part(text="FINAL ANSWER: {}")])])
    asyncio.run(gemini_agent.run("Q", "gemini-3.5-flash-lite", "reasoning", tmp_path / "r", client=fake))
    assert fake.requests[0]["config"].tools is None


def test_malformed_tool_call_is_retried(tmp_path):
    fake = FakeGemini([
        types.GenerateContentResponse(candidates=[types.Candidate(finish_reason=types.FinishReason.MALFORMED_FUNCTION_CALL)]),
        response([types.Part(text="FINAL ANSWER: {}")]),
    ])
    out = asyncio.run(gemini_agent.run("Q", "gemini-3.5-flash-lite", "code", tmp_path, client=fake))
    assert out["turns"] == 2 and "malformed" in fake.requests[1]["contents"][-1].parts[0].text
    assert any("MALFORMED" in e for e in out["events"])


def test_submit_answer_ends_the_run(tmp_path):
    submit = types.FunctionCall(name="submit_answer", args={"the_value": 12})
    fake = FakeGemini([response([types.Part(text="Done.")]), response([types.Part(function_call=submit)])])
    out = asyncio.run(gemini_agent.run("Q", "gemini-3.5-flash-lite", "reasoning", tmp_path, client=fake,
                                       answer_keys=["The value"]))
    assert out["answer"] == {"The value": 12.0} and out["turns"] == 2
    assert "submit_answer" in fake.requests[1]["contents"][-1].parts[0].text  # the reminder
    assert [d.name for d in fake.requests[0]["config"].tools[0].function_declarations] == ["submit_answer"]


def test_quota_errors_are_classified():
    assert is_daily_limit(Exception("quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'"))
    assert not is_daily_limit(Exception("quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'"))
    assert retry_delay_seconds(Exception("'retryDelay': '23s'")) == 23.0
    assert retry_delay_seconds(Exception("no hint")) == 30.0


def test_dropped_connections_are_retried_then_stop_the_run(monkeypatch):
    import httpx

    monkeypatch.setattr(gemini_agent.time, "sleep", lambda s: None)
    failures = [httpx.ConnectError("no route"), httpx.ReadTimeout("no reply")]

    class Flaky:
        def __init__(self, always):
            self.always, self.models = always, self

        def generate_content(self, **kwargs):
            if self.always or failures:
                raise failures.pop(0) if failures else httpx.ConnectError("still down")
            return "ok"

    assert gemini_agent.generate(Flaky(always=False), model="m") == "ok"
    try:
        gemini_agent.generate(Flaky(always=True), model="m")
        raise AssertionError("expected ProviderUnavailable")
    except gemini_agent.ProviderUnavailable:
        pass
