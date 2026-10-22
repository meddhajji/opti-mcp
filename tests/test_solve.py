import itertools
import random

import pytest

from optimizer.engine import solve_model
from optimizer.explain import explain_infeasible
from optimizer.model import Model
from optimizer.verify import verify

FURNITURE = {
    "variables": [
        {"name": "chairs", "type": "integer", "lb": 0, "ub": 100},
        {"name": "tables", "type": "integer", "lb": 0, "ub": 100},
    ],
    "constraints": [
        {"name": "wood", "terms": {"chairs": 2, "tables": 5}, "sense": "<=", "rhs": 200},
        {"name": "hours", "terms": {"chairs": 1, "tables": 3}, "sense": "<=", "rhs": 90},
    ],
    "objective": {"sense": "maximize", "terms": {"chairs": 30, "tables": 80}},
}


def test_furniture_example_is_optimal_and_verified():
    r = solve_model(Model.model_validate(FURNITURE))
    assert r["status"] == "OPTIMAL"
    assert r["objective"] == r["best_bound"] == 2700
    assert r["verification"] == {"valid": True, "violations": [], "objective": 2700}


def random_model(rng: random.Random) -> dict:
    n = rng.randint(2, 3)
    names = [f"x{i}" for i in range(n)]
    variables = [{"name": v, "type": "integer", "lb": rng.randint(-2, 0), "ub": rng.randint(1, 4)} for v in names]
    constraints = [
        {
            "name": f"c{j}",
            "terms": {v: rng.randint(-3, 5) for v in names},
            "sense": rng.choice(["<=", ">=", "=="]) if rng.random() < 0.15 else rng.choice(["<=", ">="]),
            "rhs": rng.randint(-4, 10),
        }
        for j in range(rng.randint(1, 4))
    ]
    objective = {"sense": rng.choice(["minimize", "maximize"]), "terms": {v: rng.randint(-5, 5) for v in names}}
    return {"variables": variables, "constraints": constraints, "objective": objective}


def brute_force(m: dict) -> int | None:
    """Best objective by enumerating every assignment (None if none is valid)."""
    model = Model.model_validate(m)
    domains = [range(v.lb, v.ub + 1) for v in model.variables]
    names = [v.name for v in model.variables]
    best = None
    for combo in itertools.product(*domains):
        values = dict(zip(names, combo))
        check = verify(model, values)
        if check["valid"]:
            obj = check["objective"]
            if best is None or (obj < best if model.objective.sense == "minimize" else obj > best):
                best = obj
    return best


@pytest.mark.parametrize("seed", range(60))
def test_solver_matches_brute_force(seed):
    m = random_model(random.Random(seed))
    expected = brute_force(m)
    r = solve_model(Model.model_validate(m), workers=1)
    if expected is None:
        assert r["status"] == "INFEASIBLE"
    else:
        assert r["status"] == "OPTIMAL"
        assert r["objective"] == expected
        assert r["verification"]["valid"]


def test_verify_catches_violations_and_bad_values():
    model = Model.model_validate(FURNITURE)
    bad = verify(model, {"chairs": 100, "tables": 0})
    assert not bad["valid"] and [v["rule"] for v in bad["violations"]] == ["hours"]
    out_of_bounds = verify(model, {"chairs": 101, "tables": 0})
    assert out_of_bounds["violations"][0]["rule"] == "bounds of chairs"
    missing = verify(model, {"chairs": 1})
    assert missing["violations"][0]["problem"] == "no value given"


CONFLICT = {
    "variables": [{"name": n, "type": "integer", "lb": 0, "ub": 10} for n in "abc"],
    "constraints": [
        {"name": "a+b at least 8", "terms": {"a": 1, "b": 1}, "sense": ">=", "rhs": 8},
        {"name": "a at most 3", "terms": {"a": 1}, "sense": "<=", "rhs": 3},
        {"name": "b at most 4", "terms": {"b": 1}, "sense": "<=", "rhs": 4},
        {"name": "c at least 2", "terms": {"c": 1}, "sense": ">=", "rhs": 2},
        {"name": "a+c at most 9", "terms": {"a": 1, "c": 1}, "sense": "<=", "rhs": 9},
    ],
}


def test_conflict_is_found_and_minimal():
    model = Model.model_validate(CONFLICT)
    result = explain_infeasible(model)
    rules = {c["rule"] for c in result["conflict"]}
    assert rules == {"a+b at least 8", "a at most 3", "b at most 4"}
    # irreducible: dropping any one rule of the conflict makes the rest solvable
    for rule in rules:
        kept = [c for c in CONFLICT["constraints"] if c["name"] in rules - {rule}]
        assert solve_model(Model.model_validate({**CONFLICT, "constraints": kept}))["status"] in ("OPTIMAL", "FEASIBLE")


def test_feasible_model_has_nothing_to_explain():
    assert explain_infeasible(Model.model_validate(FURNITURE))["infeasible"] is False


@pytest.mark.parametrize("bad, message", [
    ({"variables": [{"name": "x", "ub": 1}, {"name": "x", "ub": 1}]}, "duplicate variable"),
    ({"variables": [{"name": "x", "ub": 1}], "constraints": [{"name": "c", "terms": {"y": 1}, "sense": "<=", "rhs": 1}]}, "unknown variables"),
    ({"variables": [{"name": "x", "lb": 5, "ub": 1}]}, "lb 5 > ub 1"),
])
def test_invalid_models_are_rejected_with_a_reason(bad, message):
    with pytest.raises(ValueError, match=message):
        Model.model_validate(bad)


def test_decimals_are_solved_exactly():
    # 0.25 lb of cashews per regular box: scaled to whole numbers inside, never rounded
    m = Model.model_validate({
        "variables": [{"name": n, "type": "integer", "lb": 0, "ub": 1000} for n in ("regular", "deluxe")],
        "constraints": [{"name": "cashews", "terms": {"regular": 0.25, "deluxe": 0.5}, "sense": "<=", "rhs": 120.5}],
        "objective": {"sense": "maximize", "terms": {"regular": 0.8, "deluxe": 0.9}, "constant": 0.05}})
    r = solve_model(m)
    assert r["values"] == {"regular": 482, "deluxe": 0} and r["objective"] == 385.65
    assert r["verification"] == {"valid": True, "violations": [], "objective": 385.65}


def test_verification_compares_decimals_exactly():
    m = Model.model_validate({"variables": [{"name": "x", "type": "integer", "ub": 10}],
                              "constraints": [{"name": "c", "terms": {"x": 0.1}, "sense": "==", "rhs": 0.3}]})
    assert verify(m, {"x": 3})["valid"]  # 3 * 0.1 == 0.3 exactly, which floats get wrong


def test_too_many_decimals_and_empty_rules_are_refused():
    with pytest.raises(ValueError, match="more than 6 decimal places"):
        Model.model_validate({"variables": [{"name": "x", "ub": 1}],
                              "constraints": [{"name": "c", "terms": {"x": 0.1234567}, "sense": "<=", "rhs": 1}]})
    with pytest.raises(ValueError, match="has no terms"):
        Model.model_validate({"variables": [{"name": "x", "ub": 1}], "constraints": [{"name": "c", "terms": {}, "sense": "<=", "rhs": 1}]})
