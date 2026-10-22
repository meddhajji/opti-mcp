"""Explain why a model has no solution: find a minimal set of conflicting rules.

Every explainable constraint gets an on/off switch (an enforcement literal):
"this rule applies only if its switch is on". The solver is asked to keep
all switches on (assumptions); when that is impossible it names a subset
of switches responsible. That subset is then shrunk one rule at a time
(the deletion filter): if the problem stays impossible without a rule,
that rule was not needed to explain the conflict and is dropped for good.

What remains is an irreducible infeasible subset (IIS): the rules cannot
all hold together, but removing any single one of them makes the problem
solvable. Bookkeeping equations marked `definition` are never blamed.
"""
from ortools.sat.python import cp_model

from .model import Model
from .engine import add_rule, new_vars


def _build_with_switches(model: Model):
    cp = cp_model.CpModel()
    var = new_vars(cp, model)
    switches = {}
    for c in model.constraints:
        ct = add_rule(cp, var, c)
        if not c.definition:
            switches[c.name] = cp.new_bool_var(f"on[{c.name}]")
            ct.only_enforce_if(switches[c.name])
    return cp, switches


def _feasible(cp, switches, active: list[str], time_limit_s: float) -> bool | None:
    """True / False when decided, None when the time limit was hit."""
    cp.clear_assumptions()
    cp.add_assumptions([switches[name] for name in active])
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 1  # assumption cores are reported reliably with one worker
    status = solver.solve(cp)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return True
    if status == cp_model.INFEASIBLE:
        return False
    return None


def explain_infeasible(model: Model, time_limit_s: float = 30.0) -> dict:
    cp, switches = _build_with_switches(model)
    names = list(switches)

    cp.clear_assumptions()
    cp.add_assumptions([switches[n] for n in names])
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = 1
    status = solver.solve(cp)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"infeasible": False, "message": "The model has a solution: there is no conflict to explain."}
    if status != cp_model.INFEASIBLE:
        return {"infeasible": None, "message": "Time limit reached before the solver could decide."}

    # Start from the solver's own core (often already small), then shrink it.
    index_to_name = {switches[n].index: n for n in names}
    core = [index_to_name[i] for i in solver.sufficient_assumptions_for_infeasibility() if i in index_to_name]
    if not core or _feasible(cp, switches, core, time_limit_s) is not False:
        core = names  # fall back to every rule; the filter below still minimises it

    if not core:
        return {
            "infeasible": True,
            "conflict": [],
            "message": "No solution even with every rule switched off: the variable bounds "
                       "or the bookkeeping definitions contradict each other.",
        }

    conflict = list(core)
    for name in list(conflict):
        trial = [n for n in conflict if n != name]
        if _feasible(cp, switches, trial, time_limit_s) is False:
            conflict = trial  # still impossible without it: not part of the explanation
    by_name = {c.name: c for c in model.constraints}
    return {
        "infeasible": True,
        "conflict": [
            {"rule": n, "terms": by_name[n].terms, "sense": by_name[n].sense, "rhs": by_name[n].rhs}
            for n in conflict
        ],
        "message": f"These {len(conflict)} rules cannot all hold together; relaxing any one of them "
                   "removes this conflict (other conflicts may remain).",
    }
