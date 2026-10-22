"""The Groq harness, tested with a scripted fake client: no API key, no quota.
Everything else is real, including the optimizer MCP server for the plugin condition."""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))

import groq_agent  # noqa: E402
from groq_agent import deref, is_daily_limit  # noqa: E402


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def model_dump(self):
        return {k: v.model_dump() if isinstance(v, Obj) else v for k, v in self.__dict__.items()}


def tool_call(name, args, cid="call_1"):
    return Obj(id=cid, type="function", function=Obj(name=name, arguments=json.dumps(args)))


def reply(content="", calls=None):
    return Obj(choices=[Obj(message=Obj(content=content, tool_calls=calls))],
               usage=Obj(prompt_tokens=100, completion_tokens=20))


class FakeGroq:
    """Replays scripted replies and records every request."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.chat = Obj(completions=Obj(create=self.create))

    def create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs, default=str)))
        return self.replies.pop(0)


JACKETS_MODEL = {
    "variables": [{"name": "f", "type": "integer", "lb": 0, "ub": 30}, {"name": "d", "type": "integer", "lb": 0, "ub": 30}],
    "constraints": [{"name": "flight rate", "terms": {"f": 1}, "sense": "<=", "rhs": 10},
                    {"name": "denim rate", "terms": {"d": 1}, "sense": "<=", "rhs": 25},
                    {"name": "sewing", "terms": {"f": 1, "d": 1}, "sense": "<=", "rhs": 30}],
    "objective": {"sense": "maximize", "terms": {"f": 70, "d": 100}},
}


def test_plugin_condition_calls_the_real_mcp_server(tmp_path):
    fake = FakeGroq([
        reply(calls=[tool_call("solve", {"kind": "model", "data": JACKETS_MODEL})]),
        reply('FINAL ANSWER: {"profit": 2850}'),
    ])
    out = asyncio.run(groq_agent.run("Q", "fake-model", "plugin", tmp_path, client=fake))
    assert out["tools"] == ["solve"] and out["turns"] == 2
    assert out["text"].endswith('{"profit": 2850}')
    first, second = fake.requests
    assert first["messages"][0]["role"] == "system" and "Optimize" in first["messages"][0]["content"]
    assert {t["function"]["name"] for t in first["tools"]} == {"solve", "get_format"}
    tool_message = second["messages"][-1]
    assert tool_message["role"] == "tool" and '"OPTIMAL"' in tool_message["content"]
    assert out["tokens"] == {"prompt": 200, "completion": 40}


def test_code_condition_runs_the_models_python(tmp_path):
    fake = FakeGroq([reply(calls=[tool_call("run_python", {"code": "print(6 * 7)"})]), reply("FINAL ANSWER: {}")])
    out = asyncio.run(groq_agent.run("Q", "fake-model", "code", tmp_path, client=fake))
    assert out["tools"] == ["run_python"]
    assert fake.requests[1]["messages"][-1]["content"].strip() == "42"
    assert [t["function"]["name"] for t in fake.requests[0]["tools"]] == ["run_python"]


def test_library_condition_can_import_the_optimizer(tmp_path):
    code = ("from optimizer import solve\n"
            "print(solve('selection', {'items': [{'name': 'a', 'cost': 3, 'value': 4}], 'budget': 5})['total_value'])")
    fake = FakeGroq([reply(calls=[tool_call("run_python", {"code": code})]), reply("FINAL ANSWER: {}")])
    asyncio.run(groq_agent.run("Q", "fake-model", "library", tmp_path, client=fake))
    system = fake.requests[0]["messages"][0]
    assert system["role"] == "system" and "from optimizer import solve" in system["content"] and "cutting:" in system["content"]
    assert fake.requests[1]["messages"][-1]["content"].strip() == "4"
    fake = FakeGroq([reply(calls=[tool_call("run_python", {"code": "import optimizer"})]), reply("FINAL ANSWER: {}")])
    asyncio.run(groq_agent.run("Q", "fake-model", "code", tmp_path / "c", client=fake))
    assert "ModuleNotFoundError" in fake.requests[1]["messages"][-1]["content"]  # plain code has no optimizer


def test_reasoning_condition_sends_no_tools(tmp_path):
    fake = FakeGroq([reply("FINAL ANSWER: {}")])
    out = asyncio.run(groq_agent.run("Q", "fake-model", "reasoning", tmp_path, client=fake))
    assert "tools" not in fake.requests[0] and out["tools"] == [] and out["turns"] == 1


def test_bad_tool_arguments_are_reported_back_to_the_model(tmp_path):
    bad = Obj(id="c1", type="function", function=Obj(name="run_python", arguments="{not json"))
    fake = FakeGroq([reply(calls=[bad]), reply("FINAL ANSWER: {}")])
    asyncio.run(groq_agent.run("Q", "fake-model", "code", tmp_path, client=fake))
    assert fake.requests[1]["messages"][-1]["content"].startswith("Tool error: JSONDecodeError")


KEYS = ["The maximum profit in dollars"]
FIELD = "the_maximum_profit_in_dollars"


def test_submit_answer_ends_the_run_with_structured_numbers(tmp_path):
    fake = FakeGroq([
        reply(calls=[tool_call("solve", {"kind": "model", "data": JACKETS_MODEL})]),
        reply(calls=[tool_call("submit_answer", {FIELD: 2850})]),
        reply("never requested"),
    ])
    out = asyncio.run(groq_agent.run("Q", "fake-model", "plugin", tmp_path, client=fake, answer_keys=KEYS))
    assert out["answer"] == {"The maximum profit in dollars": 2850.0} and out["turns"] == 2
    assert len(fake.requests) == 2  # nothing is asked after the answer
    submit = next(t["function"] for t in fake.requests[0]["tools"] if t["function"]["name"] == "submit_answer")
    assert submit["parameters"]["required"] == [FIELD]
    assert submit["parameters"]["properties"][FIELD]["description"] == KEYS[0]


def test_a_model_that_stops_without_submitting_is_reminded_once(tmp_path):
    fake = FakeGroq([reply("The profit is 2850."), reply(calls=[tool_call("submit_answer", {FIELD: "2,850"})])])
    out = asyncio.run(groq_agent.run("Q", "fake-model", "code", tmp_path, client=fake, answer_keys=KEYS))
    assert out["answer"] == {KEYS[0]: 2850.0}
    assert "submit_answer" in fake.requests[1]["messages"][-1]["content"]

    fake = FakeGroq([reply("2850"), reply("I said 2850.")])
    out = asyncio.run(groq_agent.run("Q", "fake-model", "code", tmp_path, client=fake, answer_keys=KEYS))
    assert out["answer"] is None and len(fake.requests) == 2


def test_an_invalid_submission_is_sent_back(tmp_path):
    fake = FakeGroq([reply(calls=[tool_call("submit_answer", {FIELD: "a lot"})]),
                     reply(calls=[tool_call("submit_answer", {FIELD: 2850}, cid="call_2")])])
    out = asyncio.run(groq_agent.run("Q", "fake-model", "reasoning", tmp_path, client=fake, answer_keys=KEYS))
    assert fake.requests[1]["messages"][-1]["content"].startswith("Not recorded")
    assert out["answer"] == {KEYS[0]: 2850.0}
    assert [t["function"]["name"] for t in fake.requests[0]["tools"]] == ["submit_answer"]  # reasoning: nothing else


def test_deref_inlines_definitions():
    schema = {"type": "object", "properties": {"m": {"$ref": "#/$defs/M"}},
              "$defs": {"M": {"type": "object", "properties": {"v": {"$ref": "#/$defs/V"}}}, "V": {"type": "integer"}}}
    assert deref(schema) == {"type": "object", "properties": {"m": {"type": "object", "properties": {"v": {"type": "integer"}}}}}


def test_daily_limits_are_told_apart_from_per_minute_ones():
    assert is_daily_limit(Exception("Rate limit reached for model on tokens per day (TPD): Limit 200000"))
    assert not is_daily_limit(Exception("Rate limit reached on tokens per minute (TPM): Limit 8000"))
