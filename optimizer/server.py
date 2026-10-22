"""MCP server: exposes the solver tools to any MCP-capable agent
(Claude Code, Codex, Cursor, opencode...). Runs locally over stdio, needs
no API key, and never calls an LLM itself.

Tools receive their input as plain JSON and check it themselves, so a mistake
comes back as status INVALID_MODEL with a short list of problems in plain
words, which a small model can act on, rather than raw validation output.
The schemas agents see are flattened (no $ref), which small models read better.
"""
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, ValidationError, WithJsonSchema

from .explain import explain_infeasible
from .model import Model
from .scheduling import ScheduleRequest, schedule_shifts as _schedule_shifts
from .solve import solve
from .verify import verify


def flat_schema(cls: type[BaseModel]) -> dict:
    """The model's JSON schema with every $ref inlined."""
    schema = cls.model_json_schema()
    defs = schema.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(defs[node["$ref"].split("/")[-1]])
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


ModelInput = Annotated[dict[str, Any], WithJsonSchema(flat_schema(Model))]
ScheduleInput = Annotated[dict[str, Any], WithJsonSchema(flat_schema(ScheduleRequest))]


def parse(cls: type[BaseModel], data: dict):
    """(parsed object, None) or (None, an INVALID_MODEL result listing each problem in plain words)."""
    try:
        return cls.model_validate(data), None
    except ValidationError as e:
        problems = []
        for err in e.errors():
            where = "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in err["loc"]).lstrip(".")
            message = err["msg"].removeprefix("Value error, ")
            problems.append(f"{where}: {message}" if where else message)
        return None, {
            "status": "INVALID_MODEL",
            "meaning": "The input could not be read. Fix these problems and call the tool again.",
            "problems": problems,
        }

server = MCPServer(
    name="optimizer",
    instructions=(
        "Exact optimisation tools. Use them instead of reasoning out or hand-coding a solution: "
        "the solver proves optimality and every answer is checked independently. "
        "Prefer a specific tool (schedule_shifts) when the problem matches it; use solve_model otherwise. "
        "Before solving, restate the user's rules back to them; after solving, report the status "
        "meaning word for word, and on INFEASIBLE show the conflicting rules rather than guessing a fix."
    ),
)


@server.tool()
def schedule_shifts(request: ScheduleInput, time_limit_s: float = 20.0) -> dict:
    """Build an optimal staff rota: who works which shift on which day.

    Hard rules: people needed per shift (min_staff, optional max_staff; day "*" means every day,
    a specific day overrides it), unavailability, max/min shifts per person, one shift per day,
    rest between the last shift of a day and the first of the next, max consecutive days.
    Goals, strictly in this order: no shift beyond what is needed, then the smallest gap between
    the most and least loaded person, then respecting prefers_off wishes.
    Returns the rota, per-person totals, an independent verification, or, when no rota exists,
    the smallest set of rules that conflict.
    """
    parsed, error = parse(ScheduleRequest, request)
    return error or _schedule_shifts(parsed, time_limit_s)


@server.tool()
def solve_model(model: ModelInput, time_limit_s: float = 10.0) -> dict:
    """Solve a general integer optimisation model (any problem with whole-number decisions).

    Write the model as: variables (integer with lb and a required ub, or boolean), linear constraints
    sum(coef * var) <=|>=|== rhs with integer coefficients, and an optional linear objective to
    minimize or maximize. Scale decimals to integers first (cents, grams). Give every constraint
    a human-readable name: conflict explanations quote it.
    Returns status (OPTIMAL = proved best; FEASIBLE = valid but not proved best), variable values,
    the objective, the best bound, and an independent verification. On INFEASIBLE it also returns
    a minimal set of conflicting constraints. Integer variables that end at their ub are listed
    in at_upper_bound: raise any bound that is not a real limit and solve again.
    """
    parsed, error = parse(Model, model)
    if error:
        return error
    result = solve(parsed, time_limit_s=time_limit_s)
    if result["status"] == "INFEASIBLE":
        result["conflict"] = explain_infeasible(parsed)
    elif "values" in result and parsed.objective:
        # OPTIMAL only means best within the bounds given; an invented ub can still cap the answer.
        capped = [v.name for v in parsed.variables if v.type == "integer" and result["values"][v.name] == v.ub]
        if capped:
            result["at_upper_bound"] = capped
            result["warning"] = ("These variables sit at their upper bound 'ub'. If a bound is not a real limit "
                                 "from the problem, raise it and solve again: a better solution may lie beyond it.")
    return result


@server.tool()
def verify_solution(model: ModelInput, values: dict[str, int]) -> dict:
    """Check any proposed solution (yours, a user's, another tool's) against a model.

    Recomputes every bound and constraint with plain arithmetic and lists each violation,
    plus the objective value. Use it before presenting any solution not produced by solve_model.
    """
    parsed, error = parse(Model, model)
    return error or verify(parsed, values)


@server.tool()
def explain_infeasibility(model: ModelInput, time_limit_s: float = 30.0) -> dict:
    """Find a minimal set of constraints that cannot all hold together.

    Removing any single constraint from the returned set removes that conflict. Constraints
    marked definition=true (bookkeeping equations) are never blamed.
    """
    parsed, error = parse(Model, model)
    return error or explain_infeasible(parsed, time_limit_s)


def main():
    server.run("stdio")


if __name__ == "__main__":
    main()
