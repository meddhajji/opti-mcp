"""Independent check of a solution against a model.

Deliberately plain Python arithmetic, with no solver involved: if the solver
(or whoever produced the values) made a mistake, this is what catches it.
Decimal coefficients are compared as exact fractions, never as rounded floats.
"""
from fractions import Fraction

from .model import Model

OPS = {
    "<=": lambda lhs, rhs: lhs <= rhs,
    ">=": lambda lhs, rhs: lhs >= rhs,
    "==": lambda lhs, rhs: lhs == rhs,
}


def exact(x) -> Fraction:
    return Fraction(str(x))


def shown(x: Fraction):
    """An exact value as JSON shows it best: an int when whole."""
    return int(x) if x.denominator == 1 else float(x)


def evaluate_objective(model: Model, values: dict[str, int]):
    if model.objective is None:
        return None
    o = model.objective
    return shown(exact(o.constant) + sum(exact(c) * values[v] for v, c in o.terms.items()))


def verify(model: Model, values: dict[str, int]) -> dict:
    """Return {"valid": bool, "violations": [...], "objective": recomputed value}."""
    violations = []
    for v in model.variables:
        if v.name not in values:
            violations.append({"rule": f"variable {v.name}", "problem": "no value given"})
            continue
        x = values[v.name]
        if not isinstance(x, int) or isinstance(x, bool):
            violations.append({"rule": f"variable {v.name}", "problem": f"value {x!r} is not an integer"})
        elif not v.lb <= x <= v.ub:
            violations.append({"rule": f"bounds of {v.name}", "problem": f"{x} outside [{v.lb}, {v.ub}]"})
    if violations:  # cannot evaluate constraints without valid values
        return {"valid": False, "violations": violations, "objective": None}

    for c in model.constraints:
        lhs = sum(exact(coef) * values[name] for name, coef in c.terms.items())
        if not OPS[c.sense](lhs, exact(c.rhs)):
            violations.append({"rule": c.name, "problem": f"left side is {shown(lhs)}, needs {c.sense} {c.rhs}"})
    return {"valid": not violations, "violations": violations, "objective": evaluate_objective(model, values)}
