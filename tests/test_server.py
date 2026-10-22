"""The MCP tools as an agent sees them: schemas, results and error messages."""
import asyncio
import json

import pytest
from mcp.client import Client

from optimizer.model import Model
from optimizer.server import server


def call(tool: str, args: dict) -> dict:
    async def go():
        async with Client(server) as c:
            res = await c.call_tool(tool, args)
            return json.loads("".join(getattr(x, "text", "") for x in res.content))
    return asyncio.run(go())


def test_unbounded_integer_is_rejected_not_capped_at_one():
    # This exact request used to come back OPTIMAL with x = 1 and a passing verification.
    result = call("solve_model", {"model": {
        "variables": [{"name": "x", "type": "integer", "lb": 0}],
        "objective": {"sense": "maximize", "terms": {"x": 5}}}})
    assert result["status"] == "INVALID_MODEL"
    assert result["problems"] == ["variables[0]: integer variable 'x' needs an upper bound 'ub': use the largest "
                                  "value that makes sense, for example what the available resources allow"]


def test_mistakes_are_listed_in_plain_words():
    result = call("solve_model", {"model": {
        "variables": [{"name": "x", "type": "integer", "lb": 0, "ub": 9}],
        "constraints": [{"name": "c", "terms": {"y": 1}, "sense": "<=", "rhs": 3}]}})
    assert result["status"] == "INVALID_MODEL"
    assert result["problems"] == ["constraint c uses unknown variables ['y']"]
    assert "pydantic" not in json.dumps(result)


def test_valid_model_still_solves():
    result = call("solve_model", {"model": {
        "variables": [{"name": "x", "type": "integer", "lb": 0, "ub": 9}],
        "objective": {"sense": "maximize", "terms": {"x": 5}}}})
    assert result["status"] == "OPTIMAL" and result["values"] == {"x": 9}


def test_answers_capped_by_a_bound_are_flagged():
    # An invented ub can cap the answer just like the old default did; OPTIMAL alone does not show it.
    result = call("solve_model", {"model": {
        "variables": [{"name": "x", "type": "integer", "ub": 10}, {"name": "b", "type": "boolean"}],
        "objective": {"sense": "maximize", "terms": {"x": 5, "b": 1}}}})
    assert result["status"] == "OPTIMAL" and result["at_upper_bound"] == ["x"]
    assert "raise it" in result["warning"]


def test_no_flag_when_a_constraint_is_the_limit():
    result = call("solve_model", {"model": {
        "variables": [{"name": "x", "type": "integer", "ub": 100}],
        "constraints": [{"name": "wood", "terms": {"x": 2}, "sense": "<=", "rhs": 14}],
        "objective": {"sense": "maximize", "terms": {"x": 5}}}})
    assert result["values"] == {"x": 7}
    assert "at_upper_bound" not in result and "warning" not in result


def test_every_tool_validates_its_input():
    bad = {"variables": [{"name": "x"}]}  # integer by default, no ub
    for tool, args in (("verify_solution", {"model": bad, "values": {"x": 1}}),
                       ("explain_infeasibility", {"model": bad})):
        assert call(tool, args)["status"] == "INVALID_MODEL"
    assert call("schedule_shifts", {"request": {"days": ["Mon"]}})["status"] == "INVALID_MODEL"


def test_schemas_are_flat_for_small_models():
    async def go():
        async with Client(server) as c:
            return {t.name: t.input_schema for t in (await c.list_tools()).tools}
    schemas = asyncio.run(go())
    for name, schema in schemas.items():
        assert "$ref" not in json.dumps(schema), name
    variable = schemas["solve_model"]["properties"]["model"]["properties"]["variables"]["items"]
    assert "required for integer variables" in json.dumps(variable)


def test_model_class_requires_ub_for_integers_only():
    with pytest.raises(ValueError, match="needs an upper bound"):
        Model.model_validate({"variables": [{"name": "x", "type": "integer"}]})
    m = Model.model_validate({"variables": [{"name": "b", "type": "boolean"}]})
    assert (m.variables[0].lb, m.variables[0].ub) == (0, 1)
