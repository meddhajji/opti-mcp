"""Every problem kind checked against a method that does not use the solver: brute force
over every plan, or dynamic programming. Plus the generator's own instances: each kind
must reproduce the proven optimum from the data a perfect reader would pass."""
import itertools
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))

from generate import FAMILIES, bars_by_assignment, build, production_dp, solved_spec  # noqa: E402

from optimizer import KIND_NAMES, get_format, solve  # noqa: E402

VALUE = {"assignment": "total", "selection": "total_value", "cutting": "bars", "production": "total_cost",
         "product_mix": "total_profit", "transport": "total_cost"}


@pytest.mark.parametrize("path", sorted((ROOT / "benchmark" / "sets" / "scale-v1").glob("*.json")), ids=lambda p: p.stem)
def test_scale_set_optimum_from_the_kind(path):
    problem = json.loads(path.read_text(encoding="utf-8"))
    kind = problem["reference"]["kind"]
    out = solve(kind["kind"], kind["data"])
    assert out["status"] == "OPTIMAL" and out["checked"]
    assert out[VALUE[kind["kind"]]] == problem["optimum"]


@pytest.mark.parametrize("family", FAMILIES)
def test_generated_instances_optimum_from_the_kind(family):
    for level, seed in itertools.product((1, 2, 3), (1, 2)):
        spec = solved_spec(family, level, seed)
        kind, data = spec["kind"]
        out = solve(kind, json.loads(json.dumps(data)))  # as an agent would send it: plain JSON
        assert out[VALUE[kind]] == spec["optimum"], (family, level, seed)
        assert build(family, level, 0, seed, spec)["reference"]["kind"]["kind"] == kind


def test_assignment_matches_brute_force():
    rng = random.Random(1)
    for _ in range(40):
        n_workers, n_jobs = rng.randint(1, 4), rng.randint(1, 5)
        per = rng.choice([1, 2])
        cost = [[rng.choice([None] + list(range(1, 20))) for _ in range(n_jobs)] for _ in range(n_workers)]
        goal = rng.choice(["min", "max"])
        data = {"workers": [f"W{i}" for i in range(n_workers)], "jobs": [f"J{j}" for j in range(n_jobs)],
                "cost": cost, "goal": goal, "jobs_per_worker": per}
        best = None
        for who in itertools.product(range(n_workers), repeat=n_jobs):  # who[j] does job j
            if any(cost[w][j] is None for j, w in enumerate(who)) or any(who.count(w) > per for w in range(n_workers)):
                continue
            value = sum(cost[w][j] for j, w in enumerate(who))
            best = value if best is None else (min if goal == "min" else max)(best, value)
        out = solve("assignment", data)
        if out["status"] == "INVALID_INPUT":  # a job nobody may do
            assert any(all(row[j] is None for row in cost) for j in range(n_jobs))
            continue
        assert out["status"] == ("INFEASIBLE" if best is None else "OPTIMAL")
        if best is not None:
            assert out["total"] == best


def test_selection_matches_brute_force():
    rng = random.Random(2)
    for _ in range(40):
        n = rng.randint(1, 8)
        items = [{"name": f"P{i}", "cost": rng.randint(1, 30), "value": rng.randint(1, 40)} for i in range(n)]
        budget = rng.randint(10, 80)
        names = [i["name"] for i in items]
        exclusive = [rng.sample(names, 2)] if n >= 2 and rng.random() < 0.5 else []
        requires = [rng.sample(names, 2)] if n >= 2 and rng.random() < 0.5 else []
        max_items = rng.choice([None, 2, 3])
        best = max(sum(i["value"] for i, t in zip(items, take) if t)
                   for take in itertools.product((0, 1), repeat=n)
                   if sum(i["cost"] for i, t in zip(items, take) if t) <= budget
                   and (max_items is None or sum(take) <= max_items)
                   and all(sum(take[names.index(x)] for x in g) <= 1 for g in exclusive)
                   and all(take[names.index(a)] <= take[names.index(b)] for a, b in requires))
        out = solve("selection", {"items": items, "budget": budget, "max_items": max_items,
                                  "exclusive": exclusive, "requires": requires})
        assert out["total_value"] == best and out["plan"]["spent"] <= budget


def test_selection_with_several_budgets():
    items = [{"name": "A", "cost": {"y1": 5, "y2": 1}, "value": 10}, {"name": "B", "cost": {"y1": 1, "y2": 5}, "value": 10},
             {"name": "C", "cost": {"y1": 3, "y2": 3}, "value": 12}]
    out = solve("selection", {"items": items, "budget": {"y1": 6, "y2": 6}})
    assert out["total_value"] == 20 and sorted(out["plan"]["chosen"]) == ["A", "B"]


def test_cutting_matches_a_second_formulation():
    rng = random.Random(3)
    for _ in range(12):
        lengths = rng.sample(range(120, 420, 10), rng.randint(1, 3))
        pieces = {L: rng.randint(1, 4) for L in lengths}
        out = solve("cutting", {"bar_length": 600, "pieces": [{"length": L, "count": c} for L, c in pieces.items()]})
        bars = out["bars"]
        assert bars >= -(-sum(L * c for L, c in pieces.items()) // 600)  # never below the length bound
        assert bars_by_assignment(pieces, bars) and (bars == 1 or not bars_by_assignment(pieces, bars - 1))
        made = {}
        for row in out["plan"]["cuts"]:
            for part in row["cut"].split(" + "):
                k, L = part.split(" x ")
                made[int(L)] = made.get(int(L), 0) + int(k) * row["bars"]
        assert all(made[L] >= c for L, c in pieces.items())


def test_cutting_in_metres_equals_centimetres():
    cm = solve("cutting", {"bar_length": 600, "pieces": [{"length": 245, "count": 5}, {"length": 130, "count": 6}]})
    m = solve("cutting", {"bar_length": 6, "pieces": [{"length": 2.45, "count": 5}, {"length": 1.3, "count": 6}]})
    assert cm["bars"] == m["bars"] and m["plan"]["cuts"][0]["cut"].count(".") >= 1


def test_production_matches_dynamic_programming():
    rng = random.Random(4)
    for _ in range(25):
        weeks = rng.randint(1, 5)
        reg_cap, ot_cap = rng.randint(20, 40), rng.randint(0, 15)
        reg_cost, ot_cost, hold = rng.randint(5, 9), rng.randint(10, 15), rng.randint(1, 4)
        store, stock0 = rng.randint(5, 40), rng.randint(0, 10)
        demand = [rng.randint(10, 60) for _ in range(weeks)]
        best = production_dp(demand, reg_cap, ot_cap, reg_cost, ot_cost, hold, store, stock0)
        out = solve("production", {"demand": demand, "regular": {"capacity": reg_cap, "cost": reg_cost},
                                   "overtime": {"capacity": ot_cap, "cost": ot_cost}, "holding_cost": hold,
                                   "storage": store, "start_stock": stock0})
        if best is None:
            assert out["status"] == "INFEASIBLE" and out["conflicting_rules"]
        else:
            assert out["total_cost"] == best


def test_product_mix_with_decimals_matches_brute_force():
    rng = random.Random(5)
    for _ in range(25):
        products = [{"name": f"p{i}", "profit": round(rng.uniform(0.5, 9.5), 2),
                     "uses": {"wood": round(rng.uniform(0.25, 3), 2), "hours": rng.choice([0.5, 1, 1.5, 2])}}
                    for i in range(rng.randint(1, 3))]
        available = {"wood": rng.randint(5, 25), "hours": rng.randint(4, 15)}
        ranges = [range(int(min(available[r] / p["uses"][r] for r in available)) + 1) for p in products]
        best = max(round(sum(p["profit"] * q for p, q in zip(products, qty)), 6) for qty in itertools.product(*ranges)
                   if all(sum(p["uses"][r] * q for p, q in zip(products, qty)) <= available[r] + 1e-9 for r in available))
        out = solve("product_mix", {"products": products, "available": available})
        assert abs(out["total_profit"] - best) < 1e-6 and out["checked"]


def test_transport_matches_brute_force():
    rng = random.Random(6)
    for _ in range(15):
        supply = {"A": rng.randint(2, 6), "B": rng.randint(2, 6)}
        demand = {"X": rng.randint(1, 4), "Y": rng.randint(1, 4)}
        cost = {s: {d: rng.randint(1, 9) for d in demand} for s in supply}
        best = None
        for ax in range(demand["X"] + 1):
            for ay in range(demand["Y"] + 1):
                bx, by = demand["X"] - ax, demand["Y"] - ay
                if ax + ay <= supply["A"] and bx + by <= supply["B"]:
                    c = ax * cost["A"]["X"] + ay * cost["A"]["Y"] + bx * cost["B"]["X"] + by * cost["B"]["Y"]
                    best = c if best is None else min(best, c)
        out = solve("transport", {"supply": supply, "demand": demand, "cost": cost})
        assert out["status"] == ("INFEASIBLE" if best is None else "OPTIMAL")
        if best is not None:
            assert out["total_cost"] == best


def test_conflicts_quote_the_real_limits():
    out = solve("production", {"demand": [100, 300], "regular": {"capacity": 120, "cost": 40}, "storage": 50})
    assert out["status"] == "INFEASIBLE"
    assert out["conflicting_rules"] == ["period 1: storage holds at most 50", "period 2: demand of 300 met on time",
                                        "period 2: regular at most 120"]
    out = solve("transport", {"supply": {"A": 20}, "demand": {"X": 30}, "cost": {"A": {"X": 4}}})
    assert sorted(out["conflicting_rules"]) == ["A ships at most 20", "X receives 30"]


def test_mistakes_come_back_in_plain_words():
    assert solve("knapsack", {})["problems"] == [f"unknown kind 'knapsack': choose one of {', '.join(KIND_NAMES)}"]
    out = solve("selection", {"items": [{"name": "A", "cost": 3, "value": 4}], "budgett": 5})
    assert out["status"] == "INVALID_INPUT" and "get_format" in out["help"]
    assert any(p.startswith("budget:") for p in out["problems"]) and any("budgett" in p for p in out["problems"])
    out = solve("cutting", '{"bar_length": 6, "pieces": [{"length": 7, "count": 1}]}')  # a JSON string is accepted
    assert out["problems"] == ["pieces longer than the bar: [7]"]


def test_every_kind_has_a_format_with_an_example():
    for kind in KIND_NAMES:
        text = get_format(kind)
        assert text.startswith(f"{kind}: ") and "Example" in text
        example = json.loads(text.split("Example: ")[1].split("\n")[0])
        assert solve(kind, example)["status"] == "OPTIMAL", kind
