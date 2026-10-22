"""Solve a general model with Google OR-Tools CP-SAT.

CP-SAT searches the space of whole-number assignments without listing it:
constraint propagation narrows each variable's possible values ("if Karim
works Monday night he cannot work Tuesday morning"), and every dead end
teaches it a new rule so the same mistake is never explored twice. When it
reports OPTIMAL it has also proved that no better solution exists.

CP-SAT only takes whole numbers. Decimals are handled exactly: each rule is
multiplied by the power of ten that makes all its numbers whole (0.25 x + 1.5 y
<= 10 becomes 25 x + 150 y <= 1000), which changes nothing about which plans
satisfy it. The objective is scaled the same way and divided back when reported.
"""
from decimal import Decimal
from fractions import Fraction

from ortools.sat.python import cp_model

from .model import Model
from .verify import shown, verify

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


def scale_of(numbers) -> int:
    """The power of ten that makes every number whole: 1 for integers, 100 for 0.25."""
    places = [-Decimal(str(x)).normalize().as_tuple().exponent for x in numbers]
    return 10 ** max([0, *places])


def whole(x, scale: int) -> int:
    return int(Fraction(str(x)) * scale)


def add_rule(cp: cp_model.CpModel, var: dict, c):
    s = scale_of([*c.terms.values(), c.rhs])
    expr = sum(whole(coef, s) * var[name] for name, coef in c.terms.items())
    return cp.add(SENSE[c.sense](expr, whole(c.rhs, s))).with_name(c.name)


def new_vars(cp: cp_model.CpModel, model: Model) -> dict:
    return {v.name: cp.new_bool_var(v.name) if v.type == "boolean" else cp.new_int_var(v.lb, v.ub, v.name)
            for v in model.variables}


def build(model: Model) -> tuple[cp_model.CpModel, dict, int]:
    """The CP-SAT model, its variables, and the factor the objective was multiplied by."""
    cp = cp_model.CpModel()
    var = new_vars(cp, model)
    for c in model.constraints:
        add_rule(cp, var, c)
    s = 1
    if model.objective:
        o = model.objective
        s = scale_of([*o.terms.values(), o.constant])
        expr = whole(o.constant, s) + sum(whole(coef, s) * var[n] for n, coef in o.terms.items())
        (cp.minimize if o.sense == "minimize" else cp.maximize)(expr)
    return cp, var, s


def solve_model(model: Model, time_limit_s: float = 10.0, workers: int = 8) -> dict:
    cp, var, s = build(model)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = workers
    status = solver.status_name(solver.solve(cp))

    result = {"status": status, "meaning": MEANING.get(status, status), "solve_seconds": round(solver.wall_time, 3)}
    if status in ("OPTIMAL", "FEASIBLE"):
        values = {name: int(solver.value(v)) for name, v in var.items()}
        result["values"] = values
        if model.objective:
            result["objective"] = shown(Fraction(round(solver.objective_value)) / s)
            result["best_bound"] = shown(Fraction(round(solver.best_objective_bound)) / s)  # no solution can beat this
        # Never hand back an answer that has not been checked independently.
        result["verification"] = verify(model, values)
    return result
