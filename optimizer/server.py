"""MCP server: exposes the solver tools to any MCP-capable agent
(Claude Code, Codex, Cursor, opencode...). Runs locally over stdio, needs
no API key, and never calls an LLM itself.
"""
from mcp.server.mcpserver import MCPServer

from .explain import explain_infeasible
from .model import Model
from .scheduling import ScheduleRequest, schedule_shifts as _schedule_shifts
from .solve import solve
from .verify import verify

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
def schedule_shifts(request: ScheduleRequest, time_limit_s: float = 20.0) -> dict:
    """Build an optimal staff rota: who works which shift on which day.

    Hard rules: people needed per shift (min_staff, optional max_staff; day "*" means every day,
    a specific day overrides it), unavailability, max/min shifts per person, one shift per day,
    rest between the last shift of a day and the first of the next, max consecutive days.
    Goals, strictly in this order: no shift beyond what is needed, then the smallest gap between
    the most and least loaded person, then respecting prefers_off wishes.
    Returns the rota, per-person totals, an independent verification, or, when no rota exists,
    the smallest set of rules that conflict.
    """
    return _schedule_shifts(request, time_limit_s)


@server.tool()
def solve_model(model: Model, time_limit_s: float = 10.0) -> dict:
    """Solve a general integer optimisation model (any problem with whole-number decisions).

    Write the model as: variables (integer with finite lb/ub, or boolean), linear constraints
    sum(coef * var) <=|>=|== rhs with integer coefficients, and an optional linear objective to
    minimize or maximize. Scale decimals to integers first (cents, grams). Give every constraint
    a human-readable name: conflict explanations quote it.
    Returns status (OPTIMAL = proved best; FEASIBLE = valid but not proved best), variable values,
    the objective, the best bound, and an independent verification. On INFEASIBLE it also returns
    a minimal set of conflicting constraints.
    """
    result = solve(model, time_limit_s=time_limit_s)
    if result["status"] == "INFEASIBLE":
        result["conflict"] = explain_infeasible(model)
    return result


@server.tool()
def verify_solution(model: Model, values: dict[str, int]) -> dict:
    """Check any proposed solution (yours, a user's, another tool's) against a model.

    Recomputes every bound and constraint with plain arithmetic and lists each violation,
    plus the objective value. Use it before presenting any solution not produced by solve_model.
    """
    return verify(model, values)


@server.tool()
def explain_infeasibility(model: Model, time_limit_s: float = 30.0) -> dict:
    """Find a minimal set of constraints that cannot all hold together.

    Removing any single constraint from the returned set removes that conflict. Constraints
    marked definition=true (bookkeeping equations) are never blamed.
    """
    return explain_infeasible(model, time_limit_s)


def main():
    server.run("stdio")


if __name__ == "__main__":
    main()
