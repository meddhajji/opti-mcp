"""Solve a general model with Google OR-Tools CP-SAT.

CP-SAT searches the space of whole-number assignments without listing it:
constraint propagation narrows each variable's possible values ("if Karim
works Monday night he cannot work Tuesday morning"), and every dead end
teaches it a new rule so the same mistake is never explored twice. When it
reports OPTIMAL it has also proved that no better solution exists.
"""
from ortools.sat.python import cp_model

from .model import Model
from .verify import verify

SENSE = {
    "<=": lambda expr, rhs: expr <= rhs,
    ">=": lambda expr, rhs: expr >= rhs,
    "==": lambda expr, rhs: expr == rhs,
}

# What each solver status means, in words the LLM can pass on unchanged.
MEANING = {
    "OPTIMAL": "Best possible solution, proved optimal.",
    "FEASIBLE": "A valid solution, but the time limit stopped the search before proving it is the best.",
    "INFEASIBLE": "Proved that no solution satisfies all the rules together.",
    "UNKNOWN": "The time limit was reached before any solution was found or ruled out.",
    "MODEL_INVALID": "The model is malformed (for example a bound or coefficient out of range).",
}


def build(model: Model) -> tuple[cp_model.CpModel, dict]:
    cp = cp_model.CpModel()
    var = {}
    for v in model.variables:
        var[v.name] = cp.new_bool_var(v.name) if v.type == "boolean" else cp.new_int_var(v.lb, v.ub, v.name)
    for c in model.constraints:
        expr = sum(coef * var[name] for name, coef in c.terms.items())
        cp.add(SENSE[c.sense](expr, c.rhs)).with_name(c.name)
    if model.objective:
        expr = model.objective.constant + sum(coef * var[n] for n, coef in model.objective.terms.items())
        (cp.minimize if model.objective.sense == "minimize" else cp.maximize)(expr)
    return cp, var


def solve(model: Model, time_limit_s: float = 10.0, workers: int = 8) -> dict:
    cp, var = build(model)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = workers
    status = solver.status_name(solver.solve(cp))

    result = {"status": status, "meaning": MEANING.get(status, status), "solve_seconds": round(solver.wall_time, 3)}
    if status in ("OPTIMAL", "FEASIBLE"):
        values = {name: int(solver.value(v)) for name, v in var.items()}
        result["values"] = values
        if model.objective:
            result["objective"] = int(solver.objective_value)
            result["best_bound"] = int(solver.best_objective_bound)  # no solution can beat this
        # Never hand back an answer that has not been checked independently.
        result["verification"] = verify(model, values)
    return result
