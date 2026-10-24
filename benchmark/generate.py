"""Size-controlled generators of optimisation problems with proven answers.

Each family turns (size level, seed) into a word problem, a reference model and its
optimum, proven by the solver and, wherever one is computable, confirmed by an
independent method (dynamic programming, brute force, a lower bound or a second
formulation). Numbers come from code, never from an LLM, so every expected answer is
exact.

  family               size levels 1..5                      independent check
  product_mix          2, 3, 5, 8, 12 products               brute force (levels 1-2)
  project_selection    5, 8, 12, 20, 35 projects             dynamic programming (all)
  assignment           3, 4, 6, 8, 12 workers                dynamic programming (all)
  cutting_stock        2, 3, 4, 6, 8 piece lengths           lower bound, or a second model
  production_planning  3, 4, 6, 9, 12 weeks                  dynamic programming (all)

Noise level 0 states the problem plainly; level 1 tells the same instance as a longer
story with irrelevant facts mixed in. Both levels share exactly the same numbers, so
any difference in accuracy comes from the noise alone.

An instance's id is family-L<level>-N<noise>-<seed> and regenerating it gives the
same problem.

Run: python benchmark/generate.py --set sweep-v1 --levels 1 2 3 4 5 --noise 0 1 --seeds 3
"""
import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH.parent))

from optimizer.model import Model  # noqa: E402
from optimizer.engine import solve_model  # noqa: E402

FAMILIES = ("product_mix", "project_selection", "assignment", "cutting_stock", "production_planning")
NAMES = ["Amine", "Salma", "Youssef", "Imane", "Omar", "Khadija", "Hamza", "Sara", "Mehdi", "Nour",
         "Driss", "Lina", "Karim", "Aya", "Rachid", "Hiba", "Anas", "Meryem", "Ilyas", "Zineb"]
CITIES = ["Casablanca", "Rabat", "Tangier", "Marrakech", "Fes", "Agadir", "Kenitra", "Oujda"]


def money(x: int) -> str:
    return f"{x:,}"


def model(variables, constraints, sense, objective) -> dict:
    """variables {name: (lb, ub) or "bool"}, constraints [(name, terms, sense, rhs)]."""
    return {
        "variables": [{"name": n, "type": "boolean"} if b == "bool" else
                      {"name": n, "type": "integer", "lb": b[0], "ub": b[1]} for n, b in variables.items()],
        "constraints": [{"name": n, "terms": {k: v for k, v in t.items() if v}, "sense": s, "rhs": r}
                        for n, t, s, r in constraints],
        "objective": {"sense": sense, "terms": objective},
    }


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


# ---------------------------------------------------------------- product mix
PRODUCTS = ["chairs", "tables", "shelves", "desks", "cabinets", "benches", "stools", "bookcases",
            "beds", "wardrobes", "doors", "frames"]
RESOURCES = [("wood", "units of wood"), ("machine time", "machine hours"), ("labour", "labour hours"),
             ("paint", "litres of paint"), ("packing", "packing hours")]


def product_mix(rng: random.Random, level: int) -> dict:
    n, m = [2, 3, 5, 8, 12][level - 1], [2, 2, 3, 4, 5][level - 1]
    products, resources = rng.sample(PRODUCTS, n), RESOURCES[:m]
    profit = {p: rng.randint(20, 120) for p in products}
    use = {(p, r[0]): rng.randint(1, 9) if (k == 0 or rng.random() < 0.8) else 0
           for p in products for k, r in enumerate(resources)}
    cap = {r[0]: int(sum(use[p, r[0]] for p in products) * rng.uniform(4, 9)) for r in resources}
    ub = {p: min(cap[r[0]] // use[p, r[0]] for r in resources if use[p, r[0]]) for p in products}
    minimum = {p: rng.randint(1, max(1, ub[p] // 5)) for p in rng.sample(products, (level >= 3) + (level >= 5))}

    cons = [(f"{r[1]} available", {p: use[p, r[0]] for p in products}, "<=", cap[r[0]]) for r in resources]
    cons += [(f"at least {k} {p}", {p: 1}, ">=", k) for p, k in minimum.items()]
    ref = model({p: (0, ub[p]) for p in products}, cons, "maximize", profit)

    check = None
    if math.prod(ub[p] + 1 for p in products) <= 2_000_000:
        best = max(sum(profit[p] * q for p, q in zip(products, qty))
                   for qty in itertools.product(*(range(ub[p] + 1) for p in products))
                   if all(sum(use[p, r[0]] * q for p, q in zip(products, qty)) <= cap[r[0]] for r in resources)
                   and all(qty[products.index(p)] >= k for p, k in minimum.items()))
        check = (best, "brute force over every production plan")

    facts = [f"The workshop makes {', '.join(products[:-1])} and {products[-1]}." if n > 1 else ""]
    if level <= 2:
        for p in products:
            needs = ", ".join(f"{use[p, r[0]]} {r[1]}" for r in resources if use[p, r[0]])
            facts.append(f"Each of the {p} needs {needs}, and earns a profit of {profit[p]} dollars.")
    else:
        facts.append("Resources needed per unit and profit per unit:\n\n" + table(
            ["Product"] + [r[1].capitalize() for r in resources] + ["Profit (dollars)"],
            [[p] + [use[p, r[0]] for r in resources] + [profit[p]] for p in products]) + "\n")
    amounts = [f"{money(cap[r[0]])} {r[1]}" for r in resources]
    facts.append("This week the workshop has " + ", ".join(amounts[:-1]) + f" and {amounts[-1]} available.")
    for p, k in minimum.items():
        facts.append(f"A customer order requires at least {k} {p}.")
    facts.append("Everything produced will be sold, and only whole units can be made.")
    kind = ("product_mix", {
        "products": [{"name": p, "profit": profit[p], "uses": {r[0]: use[p, r[0]] for r in resources if use[p, r[0]]},
                      **({"min": minimum[p]} if p in minimum else {})} for p in products],
        "available": {r[0]: cap[r[0]] for r in resources}})
    return {"facts": [f for f in facts if f], "ask": "How many of each product should be made to maximise total profit?",
            "objective_key": "The maximum total profit in dollars", "model": ref, "check": check,
            "setting": "a furniture workshop", "kind": kind}


# ---------------------------------------------------------------- project selection
def project_selection(rng: random.Random, level: int) -> dict:
    n = [5, 8, 12, 20, 35][level - 1]
    names = [f"P{i + 1}" for i in range(n)]
    cost = {p: rng.randint(10, 90) for p in names}  # thousands of dollars
    gain = {p: max(5, int(cost[p] * rng.uniform(0.8, 2.2))) for p in names}
    budget = int(sum(cost.values()) * rng.uniform(0.35, 0.55))
    ref = model({p: "bool" for p in names},
                [("budget", {p: cost[p] * 1000 for p in names}, "<=", budget * 1000)],
                "maximize", {p: gain[p] * 1000 for p in names})
    best = [0] * (budget + 1)  # 0/1 knapsack by dynamic programming over the budget
    for p in names:
        for b in range(budget, cost[p] - 1, -1):
            best[b] = max(best[b], best[b - cost[p]] + gain[p])
    facts = [f"The company is choosing among {n} improvement projects, labelled P1 to P{n}.",
             "Each project is either done in full or not at all. Cost and expected yearly benefit:\n\n"
             + table(["Project", "Cost (dollars)", "Yearly benefit (dollars)"],
                     [[p, money(cost[p] * 1000), money(gain[p] * 1000)] for p in names]) + "\n",
             f"The total budget is {money(budget * 1000)} dollars."]
    return {"facts": facts, "ask": "Which projects should be funded to maximise the total yearly benefit?",
            "objective_key": "The maximum total yearly benefit in dollars", "model": ref,
            "check": (best[budget] * 1000, "dynamic programming over the budget"), "setting": "a manufacturing company",
            "kind": ("selection", {"items": [{"name": p, "cost": cost[p] * 1000, "value": gain[p] * 1000} for p in names],
                                   "budget": budget * 1000})}


# ---------------------------------------------------------------- assignment
def assignment(rng: random.Random, level: int) -> dict:
    n = [3, 4, 6, 8, 12][level - 1]
    workers, jobs = rng.sample(NAMES, n), [f"J{j + 1}" for j in range(n)]
    t = [[rng.randint(5, 60) for _ in jobs] for _ in workers]
    x = {(i, j): f"assign_{workers[i]}_{jobs[j]}" for i in range(n) for j in range(n)}
    ref = model({v: "bool" for v in x.values()},
                [(f"{workers[i]} does exactly one job", {x[i, j]: 1 for j in range(n)}, "==", 1) for i in range(n)]
                + [(f"{jobs[j]} done by exactly one worker", {x[i, j]: 1 for i in range(n)}, "==", 1) for j in range(n)],
                "minimize", {x[i, j]: t[i][j] for i in range(n) for j in range(n)})
    dp = {0: 0}  # minimum time to give the first k workers the jobs in bitmask, k = popcount
    for i in range(n):
        nxt = {}
        for mask, cost in dp.items():
            for j in range(n):
                if not mask >> j & 1:
                    key = mask | 1 << j
                    nxt[key] = min(nxt.get(key, 10 ** 9), cost + t[i][j])
        dp = nxt
    facts = [f"A team leader must give each of {n} workers exactly one of {n} jobs ({', '.join(jobs)}), "
             "and every job must be done by exactly one worker.",
             "Time in minutes each worker needs for each job:\n\n"
             + table(["Worker"] + jobs, [[w] + row for w, row in zip(workers, t)]) + "\n"]
    return {"facts": facts, "ask": "Which worker should do which job so that the total time is as small as possible?",
            "objective_key": "The minimum total time in minutes", "model": ref,
            "check": (dp[(1 << n) - 1], "dynamic programming over subsets of jobs"), "setting": "a maintenance team",
            "kind": ("assignment", {"workers": workers, "jobs": jobs, "cost": t})}


# ---------------------------------------------------------------- cutting stock
BAR = 600  # cm


def cutting_stock(rng: random.Random, level: int) -> dict:
    k = [2, 3, 4, 6, 8][level - 1]
    lengths = sorted(rng.sample(range(95, 325, 5), k), reverse=True)
    demand = {L: rng.randint(3, [8, 10, 14, 18, 20][level - 1]) for L in lengths}
    patterns = [c for c in fitting_patterns(lengths, BAR) if any(c)]
    var = {c: "cut_" + "_".join(f"{n}x{L}" for n, L in zip(c, lengths)) for c in patterns}
    most = sum(demand.values())
    ref = model({v: (0, most) for v in var.values()},
                [(f"{demand[L]} pieces of {L} cm", {var[c]: c[i] for c in patterns}, ">=", demand[L])
                 for i, L in enumerate(lengths)], "minimize", {v: 1 for v in var.values()})
    bound = -(-sum(L * n for L, n in demand.items()) // BAR)
    check = ("lower_bound", bound)  # resolved after solving: confirms only if the optimum reaches it
    facts = [f"A workshop cuts pieces from bars {BAR // 100} metres long. Cuts lose no material and offcuts cannot be joined.",
             "Pieces ordered: " + ", ".join(f"{demand[L]} pieces of {L / 100:.2f} m" for L in lengths) + "."]
    return {"facts": facts, "ask": f"What is the minimum number of {BAR // 100}-metre bars needed to cut the whole order?",
            "objective_key": f"The minimum number of {BAR // 100}-metre bars", "model": ref, "check": check,
            "setting": "a metal workshop", "pieces": demand,
            "kind": ("cutting", {"bar_length": BAR / 100, "pieces": [{"length": L / 100, "count": demand[L]} for L in lengths]})}


def fitting_patterns(lengths: list[int], room: int) -> list[tuple]:
    """Every way to cut one bar: counts per length whose total fits (recursive, stops when full)."""
    if not lengths:
        return [()]
    head, rest = lengths[0], lengths[1:]
    return [(n,) + tail for n in range(room // head + 1) for tail in fitting_patterns(rest, room - n * head)]


def bars_by_assignment(demand: dict[int, int], bars: int) -> bool:
    """Second formulation: can the pieces be packed into `bars` bars? (piece-to-bar assignment)"""
    pieces = [L for L, n in demand.items() for _ in range(n)]
    x = {(p, b): f"x{p}_{b}" for p in range(len(pieces)) for b in range(bars)}
    m = model({v: "bool" for v in x.values()},
              [(f"piece {p} cut once", {x[p, b]: 1 for b in range(bars)}, "==", 1) for p in range(len(pieces))]
              + [(f"bar {b} length", {x[p, b]: pieces[p] for p in range(len(pieces))}, "<=", BAR) for b in range(bars)],
              "minimize", {})
    return solve_model(Model.model_validate(m), time_limit_s=60)["status"] in ("OPTIMAL", "FEASIBLE")


# ---------------------------------------------------------------- production planning
def production_planning(rng: random.Random, level: int) -> dict:
    weeks = [3, 4, 6, 9, 12][level - 1]
    while True:  # draw until demand can be met on time
        reg_cap, ot_cap = rng.randint(120, 180), rng.randint(20, 60)
        reg_cost = rng.randint(30, 50)
        ot_cost, hold = reg_cost + rng.randint(10, 25), rng.randint(2, 8)
        store, stock0 = rng.randint(80, 200), rng.randint(0, 40)
        demand = [rng.randint(int(reg_cap * 0.6), reg_cap + ot_cap + 30) for _ in range(weeks)]
        best = production_dp(demand, reg_cap, ot_cap, reg_cost, ot_cost, hold, store, stock0)
        if best is not None:
            break
    variables, cons, obj = {}, [], {}
    for t in range(weeks):
        w = t + 1
        variables |= {f"regular_w{w}": (0, reg_cap), f"overtime_w{w}": (0, ot_cap), f"stock_w{w}": (0, store)}
        terms = {f"regular_w{w}": 1, f"overtime_w{w}": 1, f"stock_w{w}": -1}
        if t:
            terms[f"stock_w{t}"] = 1
        cons.append((f"stock balance week {w}", terms, "==", demand[t] - (stock0 if t == 0 else 0)))
        obj |= {f"regular_w{w}": reg_cost, f"overtime_w{w}": ot_cost, f"stock_w{w}": hold}
    ref = model(variables, cons, "minimize", obj)
    facts = [f"A factory plans production for the next {weeks} weeks.",
             "Weekly demand, which must be met on time with no backorders: "
             + ", ".join(f"week {t + 1}: {d}" for t, d in enumerate(demand)) + " units.",
             f"Each week it can make up to {reg_cap} units in regular time at {reg_cost} dollars per unit, "
             f"plus up to {ot_cap} units in overtime at {ot_cost} dollars per unit.",
             f"Units kept in stock at the end of a week cost {hold} dollars each, and the warehouse holds at most {store} units.",
             f"There are {stock0} units in stock at the start."]
    return {"facts": facts, "ask": f"What is the minimum total cost of meeting demand over the {weeks} weeks?",
            "objective_key": "The minimum total cost in dollars", "model": ref,
            "check": (best, "dynamic programming over weekly stock levels"), "setting": "a factory",
            "kind": ("production", {"demand": demand, "regular": {"capacity": reg_cap, "cost": reg_cost},
                                    "overtime": {"capacity": ot_cap, "cost": ot_cost}, "holding_cost": hold,
                                    "storage": store, "start_stock": stock0})}


def production_dp(demand, reg_cap, ot_cap, reg_cost, ot_cost, hold, store, stock0) -> int | None:
    best = {stock0: 0}
    for d in demand:
        nxt = {}
        for stock, cost in best.items():
            for made in range(reg_cap + ot_cap + 1):
                end = stock + made - d
                if 0 <= end <= store:
                    c = cost + reg_cost * min(made, reg_cap) + ot_cost * max(0, made - reg_cap) + hold * end
                    if c < nxt.get(end, 10 ** 12):
                        nxt[end] = c
        if not nxt:
            return None
        best = nxt
    return min(best.values())


# ---------------------------------------------------------------- noise
def distractors(rng: random.Random, setting: str) -> list[str]:
    """Irrelevant facts: true-sounding context that changes nothing in the optimum."""
    pool = [
        f"The business was founded in {rng.randint(1975, 2015)} in {rng.choice(CITIES)} and now employs {rng.randint(18, 240)} people.",
        f"Last year's total revenue was {money(rng.randint(400, 9000) * 1000)} dollars.",
        f"Office rent is {money(rng.randint(30, 120) * 100)} dollars per month, paid whatever the plan.",
        f"{rng.choice(NAMES)}, who manages the site, has {rng.randint(3, 25)} years of experience.",
        f"A competitor opened a new site in {rng.choice(CITIES)} {rng.randint(2, 11)} months ago.",
        f"The site's internet subscription costs {rng.randint(40, 150)} dollars per month.",
        f"Around {rng.randint(60, 98)}% of customers said they were satisfied in the last survey.",
        f"The parking lot has {rng.randint(12, 80)} spaces, {rng.randint(2, 6)} of them reserved for visitors.",
        f"Delivery trucks usually arrive around {rng.randint(6, 10)}:{rng.choice(['00', '15', '30', '45'])} in the morning.",
        f"The annual safety inspection last took place {rng.randint(2, 10)} months ago and found no issues.",
    ]
    return rng.sample(pool, 4)


def compose(spec: dict, noise: int, rng: random.Random) -> str:
    facts = list(spec["facts"])
    if noise:
        intro = (f"You are helping {rng.choice(NAMES)}, who runs {spec['setting']} near {rng.choice(CITIES)}. "
                 "Here is what they told you, in their own order.")
        for fact in distractors(rng, spec["setting"]):
            facts.insert(rng.randint(1, len(facts)), fact)
        facts.insert(0, intro)
    return "\n".join(facts) + "\n" + spec["ask"]


# ---------------------------------------------------------------- build
BUILDERS = {f: globals()[f] for f in FAMILIES}


def solved_spec(family: str, level: int, seed: int) -> dict:
    """The instance's numbers, reference model and proven optimum: shared by every noise level."""
    spec = BUILDERS[family](random.Random(f"{family}-{level}-{seed}"), level)
    result = solve_model(Model.model_validate(spec["model"]), time_limit_s=120, workers=8)
    if result["status"] != "OPTIMAL":
        raise RuntimeError(f"{family}-L{level}-{seed}: reference solve ended {result['status']}")
    return spec | {"optimum": result["objective"], "solve_seconds": result["solve_seconds"]}


def build(family: str, level: int, noise: int, seed: int, spec: dict | None = None) -> dict:
    pid = f"{family}-L{level}-N{noise}-{seed}"
    spec = spec or solved_spec(family, level, seed)
    question = compose(spec, noise, random.Random(f"{pid}-story"))
    ref = Model.model_validate(spec["model"])
    optimum = spec["optimum"]

    check = spec["check"]
    if family == "cutting_stock":
        bound = check[1]
        if optimum == bound:
            check = (optimum, "lower bound: total length / bar length, rounded up, is reached")
        elif sum(spec["pieces"].values()) <= 40 and not bars_by_assignment(spec["pieces"], optimum - 1):
            check = (optimum, "second formulation (piece-to-bar assignment) proves one bar fewer is impossible")
        else:
            check = None
    if check is not None and check[0] != optimum:
        raise RuntimeError(f"{pid}: solver {optimum} disagrees with {check[1]} ({check[0]})")

    key = spec["objective_key"]
    return {
        "id": pid, "source": "Generated (benchmark/generate.py)", "kind": "generated",
        "family": family, "level": level, "noise": noise, "seed": seed,
        "question": question, "answer_keys": [key], "objective_key": key,
        "objective_sense": ref.objective.sense, "expected": {key: float(optimum)}, "optimum": float(optimum),
        "independent_check": check[1] if check else "none available at this size (solver proof only)",
        "reference": {"key_to_variable": {key: "objective"}, "objective_divisor": 1, "model": spec["model"],
                      # the data a perfect reader would pass to the matching kind (tests and error analysis only)
                      "kind": {"kind": spec["kind"][0], "data": spec["kind"][1]}},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="output: benchmark/sets/<set>/")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--noise", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--seeds", type=int, default=3, help="instances per family and level")
    ap.add_argument("--first-seed", type=int, default=1, help="first seed; a fresh range gives unseen instances")
    args = ap.parse_args()
    out = BENCH / "sets" / args.set
    out.mkdir(parents=True, exist_ok=True)
    for family in args.families:
        for level in args.levels:
            for seed in range(args.first_seed, args.first_seed + args.seeds):
                spec = solved_spec(family, level, seed)
                for noise in args.noise:
                    p = build(family, level, noise, seed, spec)
                    (out / f"{p['id']}.json").write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"{family:20} L{level} seed {seed}: optimum {p['optimum']:>12,.0f}  ({p['independent_check']})")


if __name__ == "__main__":
    main()
