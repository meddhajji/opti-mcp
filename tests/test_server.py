"""The MCP tools as an agent sees them: what they cost to read, results and error messages."""
import asyncio
import json
import sys
from pathlib import Path

from mcp.client import Client

from optimizer.server import server

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))

from optimizer import GUIDANCE  # noqa: E402

# What an agent reads on every request (tools and the server's instructions), at about 4 characters
# per token. Version 0.1 cost 3,904 tokens (four tools with full schemas, 1,000 tokens of guidance);
# this guards the light design.
TOKEN_BUDGET = 700


def tools() -> list:
    async def go():
        async with Client(server) as c:
            return (await c.list_tools()).tools
    return asyncio.run(go())


def call(tool: str, args: dict):
    async def go():
        async with Client(server) as c:
            res = await c.call_tool(tool, args)
            return "".join(getattr(x, "text", "") for x in res.content)
    return asyncio.run(go())


def test_two_tools_within_the_token_budget():
    specs = [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
             for t in tools()]
    assert [s["function"]["name"] for s in specs] == ["solve", "get_format"]
    tokens = (len(json.dumps(specs)) + len(GUIDANCE)) / 4
    assert tokens <= TOKEN_BUDGET, f"the server now costs ~{tokens:.0f} tokens per request"


def test_the_guidance_reaches_every_client():
    async def go():
        async with Client(server) as c:
            return c.instructions
    assert asyncio.run(go()) == GUIDANCE and "even when the problem looks small" in GUIDANCE


def test_answers_are_one_line_of_json():
    text = call("solve", {"kind": "selection", "data": {"items": [{"name": "a", "cost": 3, "value": 4}], "budget": 5}})
    assert "\n" not in text
    assert json.loads(text) == {"status": "OPTIMAL", "meaning": "Best possible solution, proved optimal.",
                                "total_value": 4, "plan": {"chosen": ["a"], "spent": 3}, "checked": True}


def test_get_format_is_plain_text():
    assert call("get_format", {"kind": "cutting"}).startswith("cutting: Cut ordered pieces")


def test_unbounded_integer_is_rejected_not_capped_at_one():
    # This exact request used to come back OPTIMAL with x = 1 and a passing verification.
    result = json.loads(call("solve", {"kind": "model", "data": {
        "variables": [{"name": "x", "type": "integer", "lb": 0}],
        "objective": {"sense": "maximize", "terms": {"x": 5}}}}))
    assert result["status"] == "INVALID_INPUT"
    assert result["problems"] == ["variables[0]: integer variable 'x' needs an upper bound 'ub': use the largest "
                                  "value that makes sense, for example what the available resources allow"]


def test_general_model_flags_answers_capped_by_a_bound():
    result = json.loads(call("solve", {"kind": "model", "data": {
        "variables": [{"name": "x", "type": "integer", "ub": 10}, {"name": "b", "type": "boolean"}],
        "objective": {"sense": "maximize", "terms": {"x": 5, "b": 1}}}}))
    assert result["status"] == "OPTIMAL" and result["at_upper_bound"] == ["x"] and "raise it" in result["warning"]

    result = json.loads(call("solve", {"kind": "model", "data": {
        "variables": [{"name": "x", "type": "integer", "ub": 100}, {"name": "y", "type": "integer", "ub": 100}],
        "constraints": [{"name": "wood", "terms": {"x": 2}, "sense": "<=", "rhs": 14}],
        "objective": {"sense": "maximize", "terms": {"x": 5}}}}))
    assert result["values"] == {"x": 7} and result["note"] == "variables not listed are 0"
    assert "at_upper_bound" not in result


def test_unknown_kind_is_refused_by_the_schema():
    text = call("solve", {"kind": "knapsack", "data": {}})
    assert "knapsack" in text or "Input should be" in text
