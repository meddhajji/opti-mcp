"""Write a reference model for each selected OptiBench problem, solve it, and
check the dataset's expected answer against the solver's proven optimum.

Output: one benchmark/problems/optibench-<index>.json per problem, holding the
question, the expected answer, the reference model and how the answer keys map
to model variables (so an agent's decisions can be checked for feasibility, and
an alternative optimal plan is not marked wrong).

Run: python benchmark/build_optibench.py            (the 15 problems in benchmark/problems/)
     python benchmark/build_optibench.py --heldout  (10 held-out ones in benchmark/sets/optibench-heldout-v1/)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from optimizer.model import Model  # noqa: E402
from optimizer.engine import solve_model  # noqa: E402


def m(variables, constraints, sense, objective):
    """Compact model writer: variables {name: (lb, ub) or "bool"}, constraints [(name, terms, sense, rhs)]."""
    return {
        "variables": [
            {"name": n, "type": "boolean"} if b == "bool" else {"name": n, "type": "integer", "lb": b[0], "ub": b[1]}
            for n, b in variables.items()
        ],
        "constraints": [{"name": n, "terms": t, "sense": s, "rhs": r} for n, t, s, r in constraints],
        "objective": {"sense": sense, "terms": objective},
    }


# Each entry: index -> (model, {answer key: variable name or "objective"}, objective divisor)
MODELS = {
    539: (m({"in_vivo": (0, 20), "ex_vivo": (0, 20)},
            [("preparation minutes", {"in_vivo": 30, "ex_vivo": 45}, "<=", 400),
             ("execution minutes", {"in_vivo": 60, "ex_vivo": 30}, "<=", 500)],
            "minimize", {"in_vivo": 2, "ex_vivo": 3}),
          {"The number of in-vivo experiments": "in_vivo", "The number of ex-vivo experiments": "ex_vivo",
           "Total radiation received": "objective"}, 1),

    386: (m({"labradors": (0, 300), "goldens": (0, 300)},
            [("bone treats", {"labradors": 5, "goldens": 6}, "<=", 1500),
             ("at least 50 golden retrievers", {"goldens": 1}, ">=", 50),
             # labradors <= 60% of all dogs  <=>  0.4 L <= 0.6 G  <=>  2 L - 3 G <= 0
             ("at most 60% labradors", {"labradors": 2, "goldens": -3}, "<=", 0)],
            "maximize", {"labradors": 7, "goldens": 10}),
          {"The number of labradors": "labradors", "The number of golden retrievers": "goldens",
           "The number of newspapers delivered": "objective"}, 1),

    573: (m({"flight": (0, 30), "denim": (0, 30)},
            [("flight team rate", {"flight": 1}, "<=", 10),
             ("denim team rate", {"denim": 1}, "<=", 25),
             ("sewing machine", {"flight": 1, "denim": 1}, "<=", 30)],
            "maximize", {"flight": 70, "denim": 100}),
          {"The number of flight jackets to produce": "flight", "The number of denim jackets to produce": "denim",
           "The maximum profit": "objective"}, 1),

    10: (m({c: "bool" for c in ("calculus", "op_research", "data_structures", "business_stats",
                                "computer_sim", "intro_programming", "forecasting")},
           [("math requirement", {"calculus": 1, "op_research": 1, "data_structures": 1, "business_stats": 1,
                                  "forecasting": 1}, ">=", 2),
            ("OR requirement", {"op_research": 1, "business_stats": 1, "computer_sim": 1, "forecasting": 1}, ">=", 2),
            ("computer requirement", {"data_structures": 1, "computer_sim": 1, "intro_programming": 1}, ">=", 2),
            ("calculus before business statistics", {"business_stats": 1, "calculus": -1}, "<=", 0),
            ("programming before simulation", {"computer_sim": 1, "intro_programming": -1}, "<=", 0),
            ("programming before data structures", {"data_structures": 1, "intro_programming": -1}, "<=", 0),
            ("business statistics before forecasting", {"forecasting": 1, "business_stats": -1}, "<=", 0)],
           "minimize", {c: 1 for c in ("calculus", "op_research", "data_structures", "business_stats",
                                       "computer_sim", "intro_programming", "forecasting")}),
         {"Calculus": "calculus", "Operations Research": "op_research", "Data Structures": "data_structures",
          "Business Statistics": "business_stats", "Computer Simulation": "computer_sim",
          "Introduction to Computer Programming": "intro_programming", "Forecasting": "forecasting",
          "The minimum number of courses needed": "objective"}, 1),

    379: (m({"bike": (0, 40), "scooter": (0, 40)},
            [("shifts available", {"bike": 1, "scooter": 1}, "<=", 40),
             ("energy", {"bike": 5, "scooter": 6}, "<=", 230),
             ("orders", {"bike": 10, "scooter": 7}, ">=", 320),
             ("at least 5 scooter shifts", {"scooter": 1}, ">=", 5)],
            "maximize", {"bike": 50, "scooter": 43}),
          {"The number of bike shifts": "bike", "The number of scooter shifts": "scooter",
           "The total tips received": "objective"}, 1),

    404: (m({"hams": (0, 2000), "ribs": (0, 2000)},
            [("meat slicer hours", {"hams": 4, "ribs": 2}, "<=", 4000),
             # 2.5 h + 3.5 r <= 4000, doubled to stay in whole numbers
             ("meat packer half-hours", {"hams": 5, "ribs": 7}, "<=", 8000)],
            "maximize", {"hams": 150, "ribs": 300}),
          {"The number of batches of hams": "hams", "The number of batches of pork ribs": "ribs",
           "The total profit": "objective"}, 1),

    316: (m({"kayak": (0, 200), "motorboat": (0, 200)},
            [("people moved", {"kayak": 4, "motorboat": 5}, ">=", 550),
             ("at most 25 motorboat trips", {"motorboat": 1}, "<=", 25),
             # kayak >= 75% of trips  <=>  0.25 k >= 0.75 m  <=>  k - 3 m >= 0
             ("at least 75% kayak trips", {"kayak": 1, "motorboat": -3}, ">=", 0)],
            "minimize", {"kayak": 5, "motorboat": 3}),
          {"The number of kayak trips": "kayak", "The number of motorboat trips": "motorboat",
           "The total time needed to transport all the locals": "objective"}, 1),

    239: (m({f"item{i}": "bool" for i in range(1, 5)},
            [("knapsack 12 kg", {"item1": 3, "item2": 6, "item3": 5, "item4": 5}, "<=", 12)],
            "maximize", {"item1": 15, "item2": 25, "item3": 12, "item4": 10}),
          {f"The decision to put item {i} in the knapsack": f"item{i}" for i in range(1, 5)}
          | {"The maximum total value of the knapsack": "objective"}, 1),

    473: (m({"shelves": (0, 40), "stands": (0, 40)},
            [("carving minutes", {"shelves": 25, "stands": 20}, "<=", 350),
             ("polishing minutes", {"shelves": 20, "stands": 10}, "<=", 600)],
            "maximize", {"shelves": 55, "stands": 45}),
          {"The number of display shelves to manufacture": "shelves",
           "The number of plant stands to manufacture": "stands", "Total profit": "objective"}, 1),

    401: (m({"boat": (0, 100), "canoe": (0, 100)},
            [("ducks moved", {"boat": 10, "canoe": 8}, ">=", 300),
             ("at most 12 boat trips", {"boat": 1}, "<=", 12),
             # canoe >= 60% of trips  <=>  0.4 c >= 0.6 b  <=>  2 c - 3 b >= 0
             ("at least 60% canoe trips", {"canoe": 2, "boat": -3}, ">=", 0)],
            "minimize", {"boat": 20, "canoe": 40}),
          {"The number of boat trips": "boat", "The number of canoe trips": "canoe",
           "The total amount of time": "objective"}, 1),

    # Profit is a curve in the price, not a line: list the profit of every allowed price
    # (k cuts of 10 cents from 1.50) and let the solver pick exactly one.
    # profit(k) = (1.00 - 0.10 k)(5000 + 1000 k) - 2000 = (10 - k)(500 + 100 k) - 2000 dollars
    0: (m({f"cut_{k}": "bool" for k in range(11)},
          [("exactly one price", {f"cut_{k}": 1 for k in range(11)}, "==", 1)],
          "maximize", {f"cut_{k}": (10 - k) * (500 + 100 * k) - 2000 for k in range(11)}),
        {"The maximum profit you can get": "objective"}, 1),

    21: (m({"butter": (0, 200), "sauce": (0, 200), "jelly": (0, 200)},
           # hours given to one decimal: multiplied by 10
           [("cooking tenths of hours", {"butter": 35, "sauce": 52, "jelly": 28}, "<=", 5000),
            ("labor tenths of hours", {"butter": 12, "sauce": 8, "jelly": 15}, "<=", 2400),
            ("apples", {"butter": 40, "sauce": 55, "jelly": 20}, "<=", 6500)],
           "maximize", {"butter": 190, "sauce": 170, "jelly": 155}),
         {"Number of Apple Butter batches to produce": "butter", "Number of Apple Sauce batches to produce": "sauce",
          "Number of Apple Jelly batches to produce": "jelly", "Total Revenue": "objective"}, 1),

    56: (m({"four_color": (0, 20), "two_color": (0, 20)},
           [("steel rollers", {"four_color": 16, "two_color": 8}, "<=", 100),
            ("gear cutting hours", {"four_color": 30, "two_color": 12}, "<=", 160),
            ("polishing hours", {"four_color": 8, "two_color": 3}, "<=", 40),
            ("at least 2 four-color", {"four_color": 1}, ">=", 2),
            ("at least 2 two-color", {"two_color": 1}, ">=", 2)],
           "maximize", {"four_color": 24000, "two_color": 10000}),
         {"The number of four-color litho to produce": "four_color", "The number of two-color litho to produce": "two_color",
          "The maximum profit": "objective"}, 1),

    # A regular box is 1/4 lb of each ingredient, a deluxe box 1/2 lb cashews + 1/2 lb chocolate.
    # Bin capacities multiplied by 4; profit in cents (divisor 100 gives dollars).
    400: (m({p: (0, 1000) for p in ("regular", "deluxe", "cashews", "raisins", "caramels", "chocolates")},
            [("cashew bin", {"regular": 1, "deluxe": 2, "cashews": 4}, "<=", 480),
             ("raisin bin", {"regular": 1, "raisins": 4}, "<=", 800),
             ("caramel bin", {"regular": 1, "caramels": 4}, "<=", 400),
             ("chocolate bin", {"regular": 1, "deluxe": 2, "chocolates": 4}, "<=", 640)]
            + [(f"at least 20 {p} boxes", {p: 1}, ">=", 20)
               for p in ("regular", "deluxe", "cashews", "raisins", "caramels", "chocolates")],
            "maximize", {"regular": 80, "deluxe": 90, "cashews": 70, "raisins": 60, "caramels": 50, "chocolates": 75}),
          {"The number of regular mix boxes": "regular", "The number of deluxe mix boxes": "deluxe",
           "The number of cashew boxes": "cashews", "The number of raisin boxes": "raisins",
           "The number of caramel boxes": "caramels", "The number of chocolate boxes": "chocolates",
           "The maximum profit": "objective"}, 100),

    # Fixed charge: a garment can only be made if its machinery is rented (big-M links).
    75: (m({"shirts": (0, 40), "shorts": (0, 53), "pants": (0, 25),
            "rent_shirt": "bool", "rent_shorts": "bool", "rent_pants": "bool"},
           [("labor hours", {"shirts": 3, "shorts": 2, "pants": 6}, "<=", 150),
            ("cloth sq yd", {"shirts": 4, "shorts": 3, "pants": 4}, "<=", 160),
            ("shirts need shirt machinery", {"shirts": 1, "rent_shirt": -40}, "<=", 0),
            ("shorts need shorts machinery", {"shorts": 1, "rent_shorts": -53}, "<=", 0),
            ("pants need pants machinery", {"pants": 1, "rent_pants": -25}, "<=", 0)],
           "maximize", {"shirts": 6, "shorts": 4, "pants": 7,
                        "rent_shirt": -200, "rent_shorts": -150, "rent_pants": -100}),
         {"The number of shirts produced each week": "shirts", "The number of shorts produced each week": "shorts",
          "The number of pants produced each week": "pants", "Shirt machinery rented": "rent_shirt",
          "Shorts machinery rented": "rent_shorts", "Pants machinery rented": "rent_pants",
          "The maximum weekly profit": "objective"}, 1),
}

# Held-out set for the v0.3 tool-adoption test (optibench/heldout.json): drawn after v0.3 was designed.
# Decimals are written as given, since the engine scales them exactly.
HELDOUT_MODELS = {
    # The question asks for the fewest vans, so the vans answer is the objective.
    493: (m({"vans": (0, 40), "trucks": (0, 20)},
            [("at least 2000 pairs supplied", {"vans": 50, "trucks": 100}, ">=", 2000),
             ("trucks at most vans", {"trucks": 1, "vans": -1}, "<=", 0)],
            "minimize", {"vans": 1}),
          {"The number of vans used": "objective", "The number of trucks used": "trucks"}, 1),

    151: (m({"oil": (0, 833), "auto": (0, 1923), "pharma": (0, 5555)},
            [("budget $100,000", {"oil": 120, "auto": 52, "pharma": 18}, "<=", 100000),
             ("at most 40% in oil", {"oil": 120}, "<=", 40000),
             ("at most 40% in auto", {"auto": 52}, "<=", 40000),
             ("at most 40% in pharmaceutical", {"pharma": 18}, "<=", 40000),
             ("at least $10,000 in oil", {"oil": 120}, ">=", 10000)],
            "maximize", {"oil": 11, "auto": 4, "pharma": 2}),
          {"The number of oil stock shares": "oil", "The number of auto stock shares": "auto",
           "The number of pharmaceutical stock shares": "pharma", "The maximum annual income": "objective"}, 1),

    69: (m({"pill": (0, 1000), "shot": (0, 500)},
           [("10,000 minutes", {"pill": 10, "shot": 20}, "<=", 10000),
            ("at least 3 shots per pill", {"shot": 1, "pill": -3}, ">=", 0),
            ("at least 30 pills", {"pill": 1}, ">=", 30)],
           "maximize", {"pill": 1, "shot": 1}),
         {"The number of pill vaccines": "pill", "The number of shot vaccines": "shot",
          "The total number of patients vaccinated": "objective"}, 1),

    # Set covering. The dataset labels the sixth row "City 5"; it is city 6 (cities 1-6, one row each).
    20: (m({f"city{i}": "bool" for i in range(1, 7)},
           [(f"city {j} within 15 minutes of a station", {f"city{i}": 1 for i in near}, ">=", 1)
            for j, near in {1: (1, 2), 2: (1, 2, 6), 3: (3, 4), 4: (3, 4, 5), 5: (4, 5, 6), 6: (2, 5, 6)}.items()],
           "minimize", {f"city{i}": 1 for i in range(1, 7)}),
         {"The optimal number of fire stations to be built is": "objective",
          **{f"Whether a fire stations should be built in City {i}": f"city{i}" for i in range(1, 7)}}, 1),

    163: (m({"suv": (0, 100), "sedan": (0, 133)},
            [("manufacturing minutes", {"suv": 200, "sedan": 150}, "<=", 20000),
             ("testing minutes", {"suv": 120, "sedan": 100}, "<=", 10000)],
            "maximize", {"suv": 10000, "sedan": 9000}),
          {"The number of SUVs to produce": "suv", "The number of sedans to produce": "sedan",
           "The maximum total profit": "objective"}, 1),

    442: (m({"fish": (0, 400), "shrimp": (0, 800)},
            [("weight checking minutes", {"fish": 3, "shrimp": 1.5}, "<=", 1200),
             ("packaging inspection minutes", {"fish": 15, "shrimp": 7}, "<=", 1200)],
            "maximize", {"fish": 7, "shrimp": 3}),
          {"The number of packages of fish meat": "fish", "The number of packages of shrimp meat": "shrimp",
           "The maximum profit": "objective"}, 1),

    # Dollars invested; condos >= 20% of the total  <=>  0.8 condos - 0.2 detached >= 0.
    62: (m({"condos": (0, 760000), "detached": (0, 760000)},
           [("budget $760,000", {"condos": 1, "detached": 1}, "<=", 760000),
            ("at least 20% in condos", {"condos": 0.8, "detached": -0.2}, ">=", 0),
            ("at least $20,000 in detached houses", {"detached": 1}, ">=", 20000)],
           "maximize", {"condos": 0.5, "detached": 1}),
         {"The amount invested in condos": "condos", "The amount invested in detached houses": "detached",
          "The total profit from the investment": "objective"}, 1),

    549: (m({f"item{i}": "bool" for i in range(1, 5)},
            [("12 kg capacity", {"item1": 3, "item2": 6, "item3": 5, "item4": 5}, "<=", 12)],
            "maximize", {"item1": 15, "item2": 25, "item3": 12, "item4": 10}),
          {**{f"The decision to put item {i} in the knapsack": f"item{i}" for i in range(1, 5)},
           "The maximum total value of the knapsack": "objective"}, 1),

    468: (m({"helicopter": (0, 27), "truck": (0, 12)},
            [("80 cows moved", {"helicopter": 3, "truck": 7}, ">=", 80),
             ("at most 8 truck trips", {"truck": 1}, "<=", 8)],
            "minimize", {"helicopter": 5, "truck": 10}),
          {"Number of helicopter trips": "helicopter", "Number of truck trips": "truck",
           "Total amount of pollution produced": "objective"}, 1),

    81: (m({"method_a": (0, 58), "method_b": (0, 53)},
           [("at least 1400 fabric", {"method_a": 25, "method_b": 45}, ">=", 1400),
            ("at least 1000 plastic", {"method_a": 14, "method_b": 25}, ">=", 1000),
            ("3500 units of special element", {"method_a": 60, "method_b": 65}, "<=", 3500)],
           "minimize", {"method_a": 1, "method_b": 1}),
         {"The number of hours using method A": "method_a", "The number of hours using method B": "method_b",
          "The total time needed": "objective"}, 1),
}


def main():
    heldout = "--heldout" in sys.argv[1:]
    selection = json.loads((ROOT / "optibench" / ("heldout.json" if heldout else "selection.json")).read_text(encoding="utf-8"))
    models = HELDOUT_MODELS if heldout else MODELS
    out_dir = ROOT / ("sets/optibench-heldout-v1" if heldout else "problems")
    out_dir.mkdir(parents=True, exist_ok=True)
    disagreements = 0
    for p in selection["problems"]:
        model_dict, keymap, divisor = models[p["index"]]
        model = Model.model_validate(model_dict)
        expected = {k: float(v) for k, v in p["results"].items()}
        assert set(keymap) == set(expected), f"#{p['index']}: answer keys and key map differ"
        objective_key = next(k for k, v in keymap.items() if v == "objective")

        result = solve_model(model, workers=8)
        assert result["status"] == "OPTIMAL", f"#{p['index']}: {result['status']}"
        optimum = result["objective"] / divisor
        agrees = abs(optimum - expected[objective_key]) < 1e-6
        disagreements += not agrees
        print(f"#{p['index']:>3} {p['type']:15} expected {expected[objective_key]:>10} "
              f"solver {optimum:>10}  {'agrees' if agrees else 'DISAGREES'}")

        problem = {
            "id": f"optibench-{p['index']}",
            "source": "OptiBench (Yang et al., ICLR 2025), Apache-2.0",
            "kind": p["type"],
            "question": p["question"],
            "answer_keys": list(expected),
            "objective_key": objective_key,
            "objective_sense": model.objective.sense,
            "expected": expected,
            "optimum": optimum,
            "label_agrees_with_solver": agrees,
            "reference": {"key_to_variable": keymap, "objective_divisor": divisor, "model": model_dict},
        }
        (out_dir / f"{problem['id']}.json").write_text(json.dumps(problem, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(selection['problems'])} problems written, {disagreements} where the label disagrees with the solver")


if __name__ == "__main__":
    main()
