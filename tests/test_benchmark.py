import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))

from score import parse_final_answer, score  # noqa: E402

PROBLEMS = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "benchmark" / "problems").glob("*.json")}


def test_problem_set_has_twenty_problems():
    assert len(PROBLEMS) == 20
    assert sum(p["source"].startswith("OptiBench") for p in PROBLEMS.values()) == 15


@pytest.mark.parametrize("pid", sorted(PROBLEMS))
def test_every_expected_answer_scores_optimal(pid):
    """Self-consistency: the reference answers must pass the scorer."""
    assert score(PROBLEMS[pid], PROBLEMS[pid]["expected"])["outcome"] == "optimal"


HELDOUT = {p.stem: json.loads(p.read_text(encoding="utf-8"))
           for p in (ROOT / "benchmark" / "sets" / "optibench-heldout-v1").glob("*.json")}


@pytest.mark.parametrize("pid", sorted(HELDOUT))
def test_heldout_labels_agree_with_the_solver_and_score_optimal(pid):
    p = HELDOUT[pid]
    assert p["label_agrees_with_solver"] and score(p, p["expected"])["outcome"] == "optimal"


@pytest.mark.parametrize("text, expected", [
    ('Reasoning...\nFINAL ANSWER: {"a": 1, "b": 2}', {"a": 1, "b": 2}),
    ('FINAL ANSWER:\n```json\n{"a": 3}\n```', {"a": 3}),
    ('final answer {"a": 1}\nFINAL ANSWER: {"a": 2}', {"a": 2}),  # the last one wins
    ("I think the answer is 42.", None),
    ("FINAL ANSWER: {not json}", None),
])
def test_parse_final_answer(text, expected):
    assert parse_final_answer(text) == expected


JACKETS = "optibench-573"  # max 70 f + 100 d, f <= 10, d <= 25, f + d <= 30; optimum 2850 at (5, 25)
F, D, P = ("The number of flight jackets to produce", "The number of denim jackets to produce", "The maximum profit")


@pytest.mark.parametrize("answer, outcome", [
    ({F: 5, D: 25, P: 2850}, "optimal"),
    ({F: "5", D: "25.0", P: "$2,850"}, "optimal"),  # numbers written as text are accepted
    ({F: 0, D: 25, P: 2500}, "suboptimal"),
    ({F: 11, D: 19, P: 2670}, "infeasible"),  # breaks the 10-per-day flight team rate
    ({F: 5.5, D: 24.5, P: 2835}, "infeasible"),  # fractions of jackets
    ({F: 5, D: 25, P: 3000}, "inconsistent"),  # this plan earns 2850, not 3000
    ({F: 5, D: 25}, "incomplete"),
    (None, "no_answer"),
])
def test_jackets_verdicts(answer, outcome):
    assert score(PROBLEMS[JACKETS], answer)["outcome"] == outcome


def test_infeasible_plan_names_the_broken_rule():
    result = score(PROBLEMS[JACKETS], {F: 11, D: 19, P: 2670})
    assert "flight team rate" in result["detail"]


PRICE = "optibench-0"  # objective-only; the optimum 3600 is reached at two prices


@pytest.mark.parametrize("claimed, outcome", [(3600, "optimal"), (3400, "suboptimal"), (4000, "impossible")])
def test_objective_only_verdicts(claimed, outcome):
    assert score(PROBLEMS[PRICE], {"The maximum profit you can get": claimed})["outcome"] == outcome


def test_alternative_optimal_plan_is_accepted():
    # cashier staffing: shift 8-14h people from 10h to 8h is a different plan with the same total
    p = PROBLEMS["industrial-cashier-staffing"]
    plan = dict(p["expected"])
    keys = [k for k in p["answer_keys"] if k != p["objective_key"]]
    found = False
    for a in keys:
        for b in keys:
            if a != b and plan[a] > 0:
                trial = dict(plan, **{a: plan[a] - 1, b: plan[b] + 1})
                if score(p, trial)["outcome"] == "optimal" and trial != plan:
                    found = True
    assert found, "expected at least one alternative optimal staffing plan"


def test_objective_only_industrial_case():
    # production planning asks only for the total cost, so only the value is judged
    p = PROBLEMS["industrial-production-planning"]
    assert score(p, {"The minimum total cost in dollars": 25700})["outcome"] == "optimal"
    assert score(p, {"The minimum total cost in dollars": 25000})["outcome"] == "impossible"
