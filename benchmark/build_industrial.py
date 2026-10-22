"""Five industrial cases written for this benchmark: the routine questions an
operations team faces. They have no external label, so every expected answer
is the solver's proven optimum AND is confirmed by an independent method that
does not use the model: exhaustive search, a lower bound, or dynamic
programming. Both must agree or the build fails.

Run: python benchmark/build_industrial.py
"""
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from optimizer.model import Model  # noqa: E402
from optimizer.engine import solve_model  # noqa: E402

SOURCE = "Written for this benchmark (industrial operations)"


def m(variables, constraints, sense, objective):
    return {
        "variables": [
            {"name": n, "type": "boolean"} if b == "bool" else {"name": n, "type": "integer", "lb": b[0], "ub": b[1]}
            for n, b in variables.items()
        ],
        "constraints": [{"name": n, "terms": t, "sense": s, "rhs": r} for n, t, s, r in constraints],
        "objective": {"sense": sense, "terms": objective},
    }


# ---------------------------------------------------------------- 1. assignment
OPERATORS = ["Amine", "Brahim", "Chaimae", "Driss", "Fatima"]
TIMES = [  # changeover minutes, operator x machine M1..M5
    [14, 5, 8, 7, 15],
    [2, 12, 6, 5, 9],
    [7, 8, 3, 9, 6],
    [2, 4, 6, 10, 11],
    [9, 6, 7, 8, 4],
]


def assignment():
    q = ("A workshop has five operators (Amine, Brahim, Chaimae, Driss, Fatima) and five machines "
         "(M1 to M5). Each operator must be assigned to exactly one machine and each machine needs exactly "
         "one operator. The changeover time in minutes for each operator on each machine is:\n"
         + "\n".join(f"{o}: " + ", ".join(f"M{j + 1} {t}" for j, t in enumerate(row)) for o, row in zip(OPERATORS, TIMES))
         + "\nAssign operators to machines to minimise the total changeover time.")
    x = {(i, j): f"assign_{OPERATORS[i]}_M{j + 1}" for i in range(5) for j in range(5)}
    model = m({v: "bool" for v in x.values()},
              [(f"{OPERATORS[i]} on exactly one machine", {x[i, j]: 1 for j in range(5)}, "==", 1) for i in range(5)]
              + [(f"M{j + 1} has exactly one operator", {x[i, j]: 1 for i in range(5)}, "==", 1) for j in range(5)],
              "minimize", {x[i, j]: TIMES[i][j] for i in range(5) for j in range(5)})
    brute = min(sum(TIMES[i][p[i]] for i in range(5)) for p in itertools.permutations(range(5)))
    return "industrial-assignment", q, model, {"The minimum total changeover time in minutes": "objective"}, brute, \
        "exhaustive search over all 120 assignments"


# ---------------------------------------------------------------- 2. cutting stock
PIECES = {250: 4, 180: 10, 120: 15}  # length cm -> pieces ordered
BAR = 600


def cutting_stock():
    q = ("A carpentry shop buys timber in 6-metre bars. A customer orders 4 pieces of 2.50 m, 10 pieces of 1.80 m "
         "and 15 pieces of 1.20 m. Saw cuts lose no material, and leftover offcuts cannot be joined. "
         "What is the minimum number of 6-metre bars needed to cut the whole order?")
    lengths = sorted(PIECES, reverse=True)
    patterns = [c for c in itertools.product(*(range(BAR // L + 1) for L in lengths))
                if 0 < sum(n * L for n, L in zip(c, lengths)) <= BAR]
    var = {c: "pattern_" + "_".join(f"{n}x{L}" for n, L in zip(c, lengths)) for c in patterns}
    model = m({v: (0, 30) for v in var.values()},
              [(f"{PIECES[L]} pieces of {L} cm", {var[c]: c[k] for c in patterns if c[k]}, ">=", PIECES[L])
               for k, L in enumerate(lengths)],
              "minimize", {v: 1 for v in var.values()})
    bound = -(-sum(L * n for L, n in PIECES.items()) // BAR)  # ceil(total length / bar length)
    return "industrial-cutting-stock", q, model, {"The minimum number of 6-metre bars": "objective"}, bound, \
        "lower bound: total ordered length / 6 m, rounded up (the optimum reaches it)"


# ---------------------------------------------------------------- 3. production planning
DEMAND = [120, 160, 220, 140]
REG_CAP, OT_CAP, REG_COST, OT_COST, HOLD, STOCK0, STORE = 150, 40, 40, 55, 5, 20, 100


def production_planning():
    q = ("A factory plans production for the next 4 weeks. Demand is 120, 160, 220 and 140 units, and must be met "
         "on time (no backorders). Each week it can make up to 150 units in regular time at 40 dollars per unit, "
         "plus up to 40 units in overtime at 55 dollars per unit. Units kept in stock at the end of a week cost "
         "5 dollars each, and the warehouse holds at most 100 units. There are 20 units in stock at the start. "
         "What is the minimum total cost of meeting demand over the 4 weeks?")
    weeks = range(4)
    variables = {}
    for t in weeks:
        variables |= {f"regular_w{t + 1}": (0, REG_CAP), f"overtime_w{t + 1}": (0, OT_CAP), f"stock_w{t + 1}": (0, STORE)}
    cons = []
    for t in weeks:  # stock balance: previous stock + production - demand = stock
        terms = {f"regular_w{t + 1}": 1, f"overtime_w{t + 1}": 1, f"stock_w{t + 1}": -1}
        rhs = DEMAND[t]
        if t == 0:
            rhs -= STOCK0
        else:
            terms[f"stock_w{t}"] = 1
        cons.append((f"stock balance week {t + 1}", terms, "==", rhs))
    objective = {}
    for t in weeks:
        objective |= {f"regular_w{t + 1}": REG_COST, f"overtime_w{t + 1}": OT_COST, f"stock_w{t + 1}": HOLD}
    model = m(variables, cons, "minimize", objective)

    # independent check: dynamic programming over end-of-week stock levels
    best = {STOCK0: 0}
    for t in weeks:
        nxt = {}
        for stock, cost in best.items():
            for made in range(REG_CAP + OT_CAP + 1):
                end = stock + made - DEMAND[t]
                if 0 <= end <= STORE:
                    c = cost + REG_COST * min(made, REG_CAP) + OT_COST * max(0, made - REG_CAP) + HOLD * end
                    if c < nxt.get(end, float("inf")):
                        nxt[end] = c
        best = nxt
    return "industrial-production-planning", q, model, {"The minimum total cost in dollars": "objective"}, \
        min(best.values()), "dynamic programming over weekly stock levels"


# ---------------------------------------------------------------- 4. exam timetable
EXAMS = ["Statistics", "Machine Learning", "Operations Research", "Logistics",
         "Cost Evaluation", "Lean Management", "Cryptography", "Control Systems"]
CONFLICTS = [("Statistics", "Machine Learning"), ("Statistics", "Operations Research"), ("Statistics", "Logistics"),
             ("Machine Learning", "Operations Research"), ("Machine Learning", "Cryptography"),
             ("Machine Learning", "Control Systems"), ("Operations Research", "Logistics"),
             ("Operations Research", "Cost Evaluation"), ("Logistics", "Lean Management"),
             ("Cost Evaluation", "Lean Management"), ("Cost Evaluation", "Control Systems"),
             ("Lean Management", "Control Systems"), ("Cryptography", "Control Systems"),
             ("Statistics", "Control Systems")]


def exam_timetable():
    q = ("An engineering school must schedule 8 exams: " + ", ".join(EXAMS) + ". Two exams that share at least "
         "one student cannot take place in the same time slot. These pairs share students: "
         + "; ".join(f"{a} and {b}" for a, b in CONFLICTS)
         + ". What is the minimum number of time slots needed so that no student has two exams at the same time?")
    slots = range(len(EXAMS))
    x = {(e, s): f"slot{s + 1}_{e.replace(' ', '_')}" for e in EXAMS for s in slots}
    used = {s: f"slot{s + 1}_used" for s in slots}
    cons = [(f"{e} in exactly one slot", {x[e, s]: 1 for s in slots}, "==", 1) for e in EXAMS]
    cons += [(f"{a} and {b} apart (slot {s + 1})", {x[a, s]: 1, x[b, s]: 1}, "<=", 1) for a, b in CONFLICTS for s in slots]
    cons += [(f"slot {s + 1} counted if used by {e}", {x[e, s]: 1, used[s]: -1}, "<=", 0) for e in EXAMS for s in slots]
    model = m({**{v: "bool" for v in x.values()}, **{v: "bool" for v in used.values()}}, cons,
              "minimize", {v: 1 for v in used.values()})

    idx = {e: i for i, e in enumerate(EXAMS)}
    edges = [(idx[a], idx[b]) for a, b in CONFLICTS]
    brute = next(k for k in range(1, len(EXAMS) + 1)
                 if any(all(c[a] != c[b] for a, b in edges) for c in itertools.product(range(k), repeat=len(EXAMS))))
    return "industrial-exam-timetable", q, model, {"The minimum number of time slots": "objective"}, brute, \
        "exhaustive search over slot assignments for each number of slots"


# ---------------------------------------------------------------- 5. cashier staffing
BLOCKS = ["8-10", "10-12", "12-14", "14-16", "16-18", "18-20", "20-22"]
NEEDED = [3, 5, 7, 6, 5, 8, 4]
STARTS = [8, 10, 12, 14, 16]  # 6-hour shifts cover 3 consecutive 2-hour blocks


def staffing():
    q = ("A supermarket is open from 8:00 to 22:00. The number of cashiers needed is: "
         + ", ".join(f"{n} from {b.split('-')[0]}:00 to {b.split('-')[1]}:00" for b, n in zip(BLOCKS, NEEDED))
         + ". Each cashier works one 6-hour shift starting at 8:00, 10:00, 12:00, 14:00 or 16:00. "
           "How many cashiers should start at each time to cover every period with the fewest cashiers in total?")
    var = {s: f"start_{s}h" for s in STARTS}
    cover = {b: [s for k, s in enumerate(STARTS) if k <= i <= k + 2] for i, b in enumerate(BLOCKS)}
    model = m({v: (0, 20) for v in var.values()},
              [(f"{n} cashiers {b}", {var[s]: 1 for s in cover[b]}, ">=", n) for b, n in zip(BLOCKS, NEEDED)],
              "minimize", {v: 1 for v in var.values()})
    brute = min(sum(c) for c in itertools.product(range(11), repeat=len(STARTS))
                if all(sum(c[STARTS.index(s)] for s in cover[b]) >= n for b, n in zip(BLOCKS, NEEDED)))
    keys = {f"The number of cashiers starting at {s}:00": var[s] for s in STARTS}
    return "industrial-cashier-staffing", q, model, keys | {"The minimum total number of cashiers": "objective"}, \
        brute, "exhaustive search over shift counts 0 to 10 at each start time (161,051 combinations)"


def main():
    out_dir = ROOT / "problems"
    out_dir.mkdir(exist_ok=True)
    for build in (assignment, cutting_stock, production_planning, exam_timetable, staffing):
        pid, question, model_dict, keymap, independent, method = build()
        model = Model.model_validate(model_dict)
        result = solve_model(model, workers=8)
        assert result["status"] == "OPTIMAL", f"{pid}: {result['status']}"
        assert result["objective"] == independent, f"{pid}: solver {result['objective']} vs check {independent}"
        values = result["values"]
        expected = {k: float(result["objective"] if v == "objective" else values[v]) for k, v in keymap.items()}
        objective_key = next(k for k, v in keymap.items() if v == "objective")
        print(f"{pid:32} optimum {result['objective']:>7}  confirmed by {method}")
        problem = {
            "id": pid,
            "source": SOURCE,
            "kind": "industrial",
            "question": question,
            "answer_keys": list(keymap),
            "objective_key": objective_key,
            "objective_sense": model.objective.sense,
            "expected": expected,  # decision values are one optimal plan; others may exist
            "optimum": float(result["objective"]),
            "independent_check": method,
            "reference": {"key_to_variable": keymap, "objective_divisor": 1, "model": model_dict},
        }
        (out_dir / f"{pid}.json").write_text(json.dumps(problem, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
