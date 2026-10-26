"""Problem kinds: the agent passes the data of a standard problem, the tool writes the model.

Each kind turns plain data (a cost table, a list of items, weekly demand...) into the
general integer model, solves it with the engine and answers in the problem's own
words: the plan, its value, the solver status, and whether the independent check
passed. When no plan exists it names the rules that conflict.

This is what keeps the server light for small models. The agent reads one line per
kind (CATALOGUE), sends only the numbers it has already read, and gets a short answer
back. Loops, variables and the exact scaling of decimals happen here, in code tested
against independent methods (brute force, dynamic programming), not in the agent's
head. `solve` and `get_format` are both the MCP tools and the Python library.
"""
import json
import math
from fractions import Fraction
from typing import Any, Callable, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .engine import scale_of, solve_model
from .explain import explain_infeasible
from .model import Constraint, Model, Number, Objective, Variable
from .scheduling import ScheduleRequest, schedule_shifts
from .verify import exact, shown


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a misspelt field is reported, never silently ignored


def parse(cls: type[BaseModel], data) -> tuple[Any, dict | None]:
    """(parsed input, None) or (None, an INVALID_INPUT answer listing each problem in plain words)."""
    if isinstance(data, str):  # some models send the data object as a JSON string
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return None, invalid(["data must be a JSON object"])
    try:
        return cls.model_validate(data), None
    except ValidationError as e:
        problems = []
        for err in e.errors():
            where = "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in err["loc"]).lstrip(".")
            message = err["msg"].removeprefix("Value error, ")
            problems.append(f"{where}: {message}" if where else message)
        return None, invalid(problems)


def invalid(problems: list[str]) -> dict:
    return {"status": "INVALID_INPUT", "meaning": "The input could not be read. Fix these problems and call again.",
            "problems": problems}


def answer(model: Model, value_name: str, plan: Callable[[dict], Any], time_limit_s: float = 30.0) -> dict:
    """Solve, then report in the kind's own terms; the full model never leaves this function."""
    r = solve_model(model, time_limit_s=time_limit_s)
    out = {"status": r["status"], "meaning": r["meaning"]}
    if "values" in r:
        out[value_name] = r["objective"]
        if r["status"] == "FEASIBLE":
            out["best_possible"] = r["best_bound"]
        out["plan"] = plan(r["values"])
        out["checked"] = r["verification"]["valid"]
        if not out["checked"]:
            out["violations"] = r["verification"]["violations"]
    elif r["status"] == "INFEASIBLE":
        out["conflicting_rules"] = [c["rule"] for c in explain_infeasible(model).get("conflict", [])]
    return out


def total(numbers) -> Fraction:
    return sum((exact(x) for x in numbers), Fraction(0))


# ---------------------------------------------------------------- assignment
class Assignment(Strict):
    workers: list[str] = Field(min_length=1)
    jobs: list[str] = Field(min_length=1)
    cost: list[list[Number | None]] = Field(description="One row per worker, one value per job; null = not allowed.")
    goal: Literal["min", "max"] = "min"
    jobs_per_worker: int = Field(1, ge=1)

    @model_validator(mode="after")
    def _shape(self):
        for names, what in ((self.workers, "workers"), (self.jobs, "jobs")):
            if len(set(names)) != len(names):
                raise ValueError(f"{what} must have different names")
        if len(self.cost) != len(self.workers) or any(len(row) != len(self.jobs) for row in self.cost):
            raise ValueError(f"cost needs {len(self.workers)} rows (one per worker) of {len(self.jobs)} values (one per job)")
        for j, job in enumerate(self.jobs):
            if all(row[j] is None for row in self.cost):
                raise ValueError(f"no worker is allowed to do {job}")
        return self


def assignment(a: Assignment) -> dict:
    x = {(w, j): f"{w} -> {j}" for i, w in enumerate(a.workers) for k, j in enumerate(a.jobs) if a.cost[i][k] is not None}
    cost = {x[w, j]: a.cost[i][k] for i, w in enumerate(a.workers) for k, j in enumerate(a.jobs) if (w, j) in x}
    per = a.jobs_per_worker
    cons = [Constraint(name=f"{j} is done by exactly one worker", terms={x[w, j]: 1 for w in a.workers if (w, j) in x},
                       sense="==", rhs=1) for j in a.jobs]
    cons += [Constraint(name=f"{w} does at most {per} job{'s' if per > 1 else ''}",
                        terms={x[w, j]: 1 for j in a.jobs if (w, j) in x}, sense="<=", rhs=per)
             for w in a.workers if any((w, j) in x for j in a.jobs)]
    model = Model(variables=[Variable(name=v, type="boolean") for v in x.values()], constraints=cons,
                  objective=Objective(sense="minimize" if a.goal == "min" else "maximize", terms=cost))

    def plan(v):
        given = {w: [j for j in a.jobs if v.get(x.get((w, j)), 0)] for w in a.workers}
        return {w: (js[0] if per == 1 else js) for w, js in given.items() if js}
    return answer(model, "total", plan)


# ---------------------------------------------------------------- selection
class Item(Strict):
    name: str
    cost: Number | dict[str, Number]
    value: Number


class Selection(Strict):
    items: list[Item] = Field(min_length=1)
    budget: Number | dict[str, Number]
    max_items: int | None = None
    min_items: int | None = None
    exclusive: list[list[str]] = Field(default_factory=list, description="Groups of items: at most one of each group.")
    requires: list[list[str]] = Field(default_factory=list, description="Pairs [a, b]: a can only be chosen with b.")

    @model_validator(mode="after")
    def _names(self):
        names = [i.name for i in self.items]
        if len(set(names)) != len(names):
            raise ValueError("items must have different names")
        for group in self.exclusive + self.requires:
            unknown = set(group) - set(names)
            if unknown:
                raise ValueError(f"unknown items {sorted(unknown)}")
        if any(len(pair) != 2 for pair in self.requires):
            raise ValueError("each requires entry is a pair [item, item it needs]")
        several = isinstance(self.budget, dict)
        for i in self.items:
            if isinstance(i.cost, dict) != several:
                raise ValueError(f"{i.name}: cost and budget must both be numbers, or both {{resource: amount}}")
            if several and set(i.cost) - set(self.budget):
                raise ValueError(f"{i.name}: cost uses resources missing from budget: {sorted(set(i.cost) - set(self.budget))}")
        return self


def selection(s: Selection) -> dict:
    names = [i.name for i in s.items]
    budgets = s.budget if isinstance(s.budget, dict) else {"budget": s.budget}
    costs = {i.name: i.cost if isinstance(i.cost, dict) else {"budget": i.cost} for i in s.items}
    cons = [Constraint(name=f"{r}: at most {cap}" if r != "budget" else f"budget: at most {cap}",
                       terms={n: costs[n].get(r, 0) for n in names}, sense="<=", rhs=cap) for r, cap in budgets.items()]
    if s.max_items is not None:
        cons.append(Constraint(name=f"at most {s.max_items} items", terms={n: 1 for n in names}, sense="<=", rhs=s.max_items))
    if s.min_items is not None:
        cons.append(Constraint(name=f"at least {s.min_items} items", terms={n: 1 for n in names}, sense=">=", rhs=s.min_items))
    groups = list(dict.fromkeys(tuple(g) for g in s.exclusive))  # a repeated group is one rule
    cons += [Constraint(name=f"at most one of {', '.join(g)}", terms={n: 1 for n in g}, sense="<=", rhs=1) for g in groups]
    cons += [Constraint(name=f"{a} requires {b}", terms={a: 1, b: -1}, sense="<=", rhs=0)
             for a, b in dict.fromkeys(tuple(p) for p in s.requires)]
    model = Model(variables=[Variable(name=n, type="boolean") for n in names], constraints=cons,
                  objective=Objective(sense="maximize", terms={i.name: i.value for i in s.items}))

    def plan(v):
        chosen = [n for n in names if v[n]]
        spent = {r: shown(total(costs[n].get(r, 0) for n in chosen)) for r in budgets}
        return {"chosen": chosen, "spent": spent if isinstance(s.budget, dict) else spent["budget"]}
    return answer(model, "total_value", plan)


# ---------------------------------------------------------------- cutting stock
MAX_PATTERNS = 20_000


class Piece(Strict):
    length: Number = Field(gt=0)
    count: int = Field(ge=1)


class Cutting(Strict):
    bar_length: Number = Field(gt=0)
    pieces: list[Piece] = Field(min_length=1)

    @model_validator(mode="after")
    def _fits(self):
        lengths = [p.length for p in self.pieces]
        if len(set(lengths)) != len(lengths):
            raise ValueError("list each piece length once, with its total count")
        too_long = [p.length for p in self.pieces if p.length > self.bar_length]
        if too_long:
            raise ValueError(f"pieces longer than the bar: {too_long}")
        return self


def cut_patterns(lengths: list[int], counts: list[int], room: int) -> list[tuple[int, ...]]:
    """Every way to cut one bar that cannot take one more needed piece. A bar never needs
    more of a length than is ordered, and any other cut is matched by one of these with
    some pieces added, so these patterns are enough for the optimum."""
    out = []

    def walk(i, left, chosen):
        if len(out) > MAX_PATTERNS:
            return
        if i == len(lengths):
            if any(chosen) and all(chosen[k] == counts[k] or lengths[k] > left for k in range(len(lengths))):
                out.append(tuple(chosen))
            return
        for n in range(min(counts[i], left // lengths[i]), -1, -1):
            walk(i + 1, left - n * lengths[i], chosen + [n])
    walk(0, room, [])
    return out


def cutting(c: Cutting) -> dict:
    s = scale_of([c.bar_length, *(p.length for p in c.pieces)])
    pieces = sorted(c.pieces, key=lambda p: -p.length)
    lengths = [int(exact(p.length) * s) for p in pieces]
    counts = [p.count for p in pieces]
    patterns = cut_patterns(lengths, counts, int(exact(c.bar_length) * s))
    if len(patterns) > MAX_PATTERNS:
        return {"status": "TOO_LARGE", "meaning": f"More than {MAX_PATTERNS:,} ways to cut one bar: too many to search "
                                                  "exactly. Split the order, or describe it with kind model."}
    label = {pat: " + ".join(f"{n} x {p.length:g}" for n, p in zip(pat, pieces) if n) for pat in patterns}
    variables = [Variable(name=label[pat], lb=0, ub=max(math.ceil(k / n) for k, n in zip(counts, pat) if n))
                 for pat in patterns]
    cons = [Constraint(name=f"{p.count} pieces of {p.length:g}", terms={label[pat]: pat[i] for pat in patterns if pat[i]},
                       sense=">=", rhs=p.count) for i, p in enumerate(pieces)]
    model = Model(variables=variables, constraints=cons,
                  objective=Objective(sense="minimize", terms={v.name: 1 for v in variables}))

    def plan(v):
        used = [pat for pat in patterns if v[label[pat]]]
        rows = [{"cut": label[pat], "bars": v[label[pat]],
                 "waste": shown(exact(c.bar_length) - total(n * exact(p.length) for n, p in zip(pat, pieces)))}
                for pat in used]
        made = [sum(v[label[pat]] * pat[i] for pat in used) for i in range(len(pieces))]
        extra = {f"{p.length:g}": m - p.count for p, m in zip(pieces, made) if m > p.count}
        return {"cuts": rows, **({"extra_pieces": extra} if extra else {})}
    return answer(model, "bars", plan)


# ---------------------------------------------------------------- production planning
class Capacity(Strict):
    capacity: int | list[int] = Field(description="Units per period: one number, or one per period.")
    cost: Number | list[Number] = Field(description="Cost per unit: one number, or one per period.")


class Production(Strict):
    demand: list[int] = Field(min_length=1, description="Units needed in each period, met on time.")
    regular: Capacity
    overtime: Capacity | None = None
    holding_cost: Number | list[Number] = Field(0, description="Cost per unit in stock at the end of a period.")
    storage: int | None = Field(None, description="Most units in stock at the end of a period.")
    start_stock: int = 0
    end_stock: int = Field(0, description="Units that must be left at the end.")
    periods: list[str] | None = Field(None, description='Period names, e.g. ["week 1", ...]; default "period 1"...')

    @model_validator(mode="after")
    def _lengths(self):
        n = len(self.demand)
        lists = {"regular.capacity": self.regular.capacity, "regular.cost": self.regular.cost,
                 "holding_cost": self.holding_cost, "periods": self.periods}
        if self.overtime:
            lists |= {"overtime.capacity": self.overtime.capacity, "overtime.cost": self.overtime.cost}
        for name, value in lists.items():
            if isinstance(value, list) and len(value) != n:
                raise ValueError(f"{name} has {len(value)} values, demand has {n} periods")
        if self.periods and len(set(self.periods)) != n:
            raise ValueError("periods must have different names")
        return self


def per_period(value, n: int) -> list:
    return value if isinstance(value, list) else [value] * n


def production(p: Production) -> dict:
    n = len(p.demand)
    names = p.periods or [f"period {t + 1}" for t in range(n)]
    most = max(0, sum(p.demand) + p.end_stock - p.start_stock)
    lines = {"regular": p.regular, **({"overtime": p.overtime} if p.overtime else {})}
    make = {(k, t): f"{k}, {names[t]}" for k in lines for t in range(n)}
    stock = [f"stock after {names[t]}" for t in range(n)]
    # Bounds stay loose on purpose: limits live in named rules, so a conflict can quote them.
    variables = [Variable(name=v, lb=0, ub=most) for v in make.values()]
    variables += [Variable(name=v, lb=0, ub=most + p.start_stock) for v in stock]
    cons, cost = [], {}
    for t in range(n):
        terms = {make[k, t]: 1 for k in lines} | {stock[t]: -1}
        if t:
            terms[stock[t - 1]] = 1
        cons.append(Constraint(name=f"{names[t]}: demand of {p.demand[t]} met on time", terms=terms, sense="==",
                               rhs=p.demand[t] - (p.start_stock if t == 0 else 0)))
        for k, line in lines.items():
            cap = per_period(line.capacity, n)[t]
            cons.append(Constraint(name=f"{names[t]}: {k} at most {cap}", terms={make[k, t]: 1}, sense="<=", rhs=cap))
            cost[make[k, t]] = per_period(line.cost, n)[t]
        if p.storage is not None:
            cons.append(Constraint(name=f"{names[t]}: storage holds at most {p.storage}", terms={stock[t]: 1},
                                   sense="<=", rhs=p.storage))
        cost[stock[t]] = per_period(p.holding_cost, n)[t]
    if p.end_stock:
        cons.append(Constraint(name=f"at least {p.end_stock} units left at the end", terms={stock[-1]: 1}, sense=">=",
                               rhs=p.end_stock))
    model = Model(variables=variables, constraints=cons, objective=Objective(sense="minimize", terms=cost))

    def plan(v):
        return {names[t]: {**{k: v[make[k, t]] for k in lines}, "end_stock": v[stock[t]]} for t in range(n)}
    return answer(model, "total_cost", plan)


# ---------------------------------------------------------------- product mix
class Product(Strict):
    name: str
    profit: Number
    uses: dict[str, Number] = Field(default_factory=dict, description="Resource used per unit.")
    min: int = 0
    max: int | None = None


class ProductMix(Strict):
    products: list[Product] = Field(min_length=1)
    available: dict[str, Number] = Field(description="Amount of each resource available.")

    @model_validator(mode="after")
    def _known(self):
        names = [p.name for p in self.products]
        if len(set(names)) != len(names):
            raise ValueError("products must have different names")
        for p in self.products:
            unknown = set(p.uses) - set(self.available)
            if unknown:
                raise ValueError(f"{p.name} uses resources missing from available: {sorted(unknown)}")
            if p.max is None and not any(a > 0 for a in p.uses.values()):
                raise ValueError(f"{p.name} uses no limited resource: give its max")
        return self


def product_mix(m: ProductMix) -> dict:
    def most(p: Product) -> int:
        limits = [math.floor(exact(m.available[r]) / exact(a)) for r, a in p.uses.items() if a > 0]
        return max(0, min(limits + ([p.max] if p.max is not None else [])))
    variables = [Variable(name=p.name, lb=0, ub=max(most(p), p.min)) for p in m.products]
    cons = [Constraint(name=f"{r}: at most {cap} available", terms={p.name: p.uses[r] for p in m.products if r in p.uses},
                       sense="<=", rhs=cap) for r, cap in m.available.items() if any(r in p.uses for p in m.products)]
    cons += [Constraint(name=f"at least {p.min} {p.name}", terms={p.name: 1}, sense=">=", rhs=p.min) for p in m.products if p.min]
    cons += [Constraint(name=f"at most {p.max} {p.name}", terms={p.name: 1}, sense="<=", rhs=p.max)
             for p in m.products if p.max is not None]
    model = Model(variables=variables, constraints=cons,
                  objective=Objective(sense="maximize", terms={p.name: p.profit for p in m.products}))
    return answer(model, "total_profit", lambda v: {p.name: v[p.name] for p in m.products})


# ---------------------------------------------------------------- transport
class Transport(Strict):
    supply: dict[str, int] = Field(description="Units available at each source.")
    demand: dict[str, int] = Field(description="Units needed at each destination.")
    cost: dict[str, dict[str, Number | None]] = Field(description="cost[source][destination] per unit; missing or null = no route.")

    @model_validator(mode="after")
    def _routes(self):
        unknown = (set(self.cost) - set(self.supply)) | {d for row in self.cost.values() for d in row} - set(self.demand)
        if unknown:
            raise ValueError(f"cost names unknown sources or destinations: {sorted(unknown)}")
        for d in self.demand:
            if not any(row.get(d) is not None for row in self.cost.values()):
                raise ValueError(f"no route reaches {d}")
        return self


def transport(t: Transport) -> dict:
    x = {(s, d): f"{s} -> {d}" for s, row in t.cost.items() for d, c in row.items() if c is not None}
    variables = [Variable(name=v, lb=0, ub=t.demand[d]) for (s, d), v in x.items()]  # supply stays a named rule
    cons = [Constraint(name=f"{s} ships at most {cap}", terms={x[s, d]: 1 for d in t.demand if (s, d) in x}, sense="<=", rhs=cap)
            for s, cap in t.supply.items() if any((s, d) in x for d in t.demand)]
    cons += [Constraint(name=f"{d} receives {need}", terms={x[s, d]: 1 for s in t.supply if (s, d) in x}, sense="==", rhs=need)
             for d, need in t.demand.items()]
    model = Model(variables=variables, constraints=cons,
                  objective=Objective(sense="minimize", terms={v: t.cost[s][d] for (s, d), v in x.items()}))
    return answer(model, "total_cost", lambda v: {name: v[name] for name in x.values() if v[name]})


# ---------------------------------------------------------------- general model and shifts
def general(model: Model) -> dict:
    r = solve_model(model, time_limit_s=30.0)
    out = {"status": r["status"], "meaning": r["meaning"]}
    if "values" in r:
        if model.objective:
            out["objective"] = r["objective"]
        if r["status"] == "FEASIBLE" and model.objective:
            out["best_possible"] = r["best_bound"]
        nonzero = {k: v for k, v in r["values"].items() if v}
        out["values"] = nonzero
        if len(nonzero) < len(r["values"]):
            out["note"] = "variables not listed are 0"
        out["checked"] = r["verification"]["valid"]
        # OPTIMAL only means best within the bounds given; an invented ub can still cap the answer.
        capped = [v.name for v in model.variables if v.type == "integer" and r["values"][v.name] == v.ub]
        if capped and model.objective:
            out["at_upper_bound"] = capped
            out["warning"] = ("These variables sit at their upper bound 'ub'. If a bound is not a real limit "
                              "from the problem, raise it and solve again: a better solution may lie beyond it.")
    elif r["status"] == "INFEASIBLE":
        out["conflicting_rules"] = [c["rule"] for c in explain_infeasible(model).get("conflict", [])]
    return out


def shifts(request: ScheduleRequest) -> dict:
    return schedule_shifts(request)


# ---------------------------------------------------------------- catalogue
class Kind(NamedTuple):
    input: type[BaseModel]
    run: Callable[[Any], dict]
    line: str  # the one-line format every agent sees
    format: str  # the full format, sent only when asked for


KINDS: dict[str, Kind] = {
    "assignment": Kind(Assignment, assignment,
        '{workers: [names], jobs: [names], cost: [[one row per worker; null = not allowed]], goal?: "min"|"max"}',
        "Give each job to exactly one worker, minimising the total (or maximising it with goal \"max\").\n"
        "workers, jobs: names. cost: one row per worker, one value per job (time, cost or score); null where that worker "
        "cannot do that job. jobs_per_worker: most jobs one worker may take (default 1).\n"
        'Example: {"workers": ["Ana", "Ben"], "jobs": ["J1", "J2"], "cost": [[4, 7], [6, 3]]}\n'
        "Returns total and plan {worker: job}."),
    "selection": Kind(Selection, selection,
        "{items: [{name, cost, value}], budget, max_items?, exclusive?: [[names]], requires?: [[item, needed item]]}",
        "Choose items (projects, investments) to maximise total value within a budget. Each item is taken whole or not "
        "at all.\nitems: [{name, cost, value}]. budget: a number, or {resource: amount} with each cost also "
        "{resource: amount} (e.g. a budget per year). max_items / min_items: limits on how many. exclusive: groups of "
        "which at most one may be chosen. requires: pairs [a, b] where a needs b.\n"
        'Example: {"items": [{"name": "P1", "cost": 40, "value": 65}, {"name": "P2", "cost": 25, "value": 30}], "budget": 50}\n'
        "Returns total_value and plan {chosen, spent}."),
    "cutting": Kind(Cutting, cutting,
        "{bar_length, pieces: [{length, count}]}",
        "Cut ordered pieces from stock bars (or rolls, boards) of one length, using as few bars as possible. Cuts lose no "
        "material and offcuts cannot be joined. Use one unit for every length (all metres or all cm).\n"
        "bar_length: length of a stock bar. pieces: [{length, count}], each length listed once.\n"
        'Example: {"bar_length": 6, "pieces": [{"length": 2.5, "count": 4}, {"length": 1.2, "count": 7}]}\n'
        "Returns bars and plan {cuts: [{cut, bars, waste}]}."),
    "production": Kind(Production, production,
        "{demand: [per period], regular: {capacity, cost}, overtime?: {capacity, cost}, holding_cost, storage?, start_stock?}",
        "Plan how much to make in each period at minimum cost, meeting every period's demand on time (no backorders); "
        "units made early wait in stock.\ndemand: units per period. regular, overtime: {capacity, cost per unit}, each a "
        "number or one per period. holding_cost: per unit left in stock at the end of a period. storage: most units in "
        "stock. start_stock, end_stock: units at the start, units required at the end. periods: optional names.\n"
        'Example: {"demand": [100, 150], "regular": {"capacity": 120, "cost": 40}, "overtime": {"capacity": 30, "cost": 55}, '
        '"holding_cost": 5}\nReturns total_cost and plan {period: {regular, overtime, end_stock}}.'),
    "product_mix": Kind(ProductMix, product_mix,
        "{products: [{name, profit, uses: {resource: amount}, min?, max?}], available: {resource: amount}}",
        "Decide how many whole units of each product to make to maximise total profit within limited resources.\n"
        "products: [{name, profit per unit, uses: {resource: amount per unit}, min?, max?}]. available: {resource: amount}. "
        "Decimals are fine (0.25 lb per box).\n"
        'Example: {"products": [{"name": "chairs", "profit": 30, "uses": {"wood": 2, "hours": 1}}, {"name": "tables", '
        '"profit": 80, "uses": {"wood": 5, "hours": 3}}], "available": {"wood": 200, "hours": 90}}\n'
        "Returns total_profit and plan {product: units}."),
    "transport": Kind(Transport, transport,
        "{supply: {source: units}, demand: {destination: units}, cost: {source: {destination: unit cost}}}",
        "Ship units from sources to destinations at minimum total cost. Each destination receives exactly its demand; "
        "each source ships at most its supply.\nsupply, demand: {name: units}. cost: {source: {destination: cost per unit}}; "
        "leave out (or null) routes that do not exist.\n"
        'Example: {"supply": {"A": 50, "B": 40}, "demand": {"X": 30, "Y": 60}, "cost": {"A": {"X": 4, "Y": 6}, "B": {"X": 5, "Y": 3}}}\n'
        "Returns total_cost and plan {\"source -> destination\": units}."),
    "shifts": Kind(ScheduleRequest, shifts,
        "a staff rota (who works which shift on which day); call get_format first",
        "Build a staff rota. Hard rules: people needed per shift, unavailability, max/min shifts per person, one shift "
        "per day, rest between the last shift of a day and the first of the next, max consecutive days. Goals in order: "
        "no shift beyond what is needed, then the fairest split, then prefers_off wishes.\n"
        "days, shifts: names in order. staff: [{name, max_shifts?, min_shifts?, unavailable?: [{day, shift}], "
        "prefers_off?: [{day, shift}]}] (shift \"*\" = whole day). needs: [{day, shift, min_staff, max_staff?}] "
        "(day \"*\" = every day; a specific day overrides it). one_shift_per_day, rest_between_days: true by default. "
        "max_consecutive_days: optional.\n"
        'Example: {"days": ["Mon", "Tue"], "shifts": ["am", "pm"], "staff": [{"name": "Sara"}, {"name": "Omar", '
        '"unavailable": [{"day": "Tue", "shift": "*"}]}], "needs": [{"day": "*", "shift": "am", "min_staff": 1}]}'),
    "model": Kind(Model, general,
        # its format in one line, so a small problem outside the families still takes one call (v0.3)
        '{variables: [{name, ub, type?: "integer"|"boolean"}], constraints: [{name, terms: {variable: coefficient}, '
        'sense: "<="|">="|"==", rhs}], objective: {sense: "maximize"|"minimize", terms}}',
        "A general integer model. variables: [{name, type: \"integer\" (needs lb and ub, the largest sensible value) or "
        "\"boolean\"}]. constraints: [{name in plain words, terms: {variable: coefficient}, sense: \"<=\"|\">=\"|\"==\", rhs}]. "
        "objective: {sense: \"minimize\"|\"maximize\", terms: {variable: coefficient}}. Decimals are fine.\n"
        'Example: {"variables": [{"name": "x", "type": "integer", "lb": 0, "ub": 50}, {"name": "y", "type": "integer", '
        '"lb": 0, "ub": 50}], "constraints": [{"name": "wood", "terms": {"x": 2, "y": 5}, "sense": "<=", "rhs": 200}], '
        '"objective": {"sense": "maximize", "terms": {"x": 30, "y": 80}}}\n'
        "Returns objective and values (zeros left out)."),
}
KIND_NAMES = tuple(KINDS)

# The first sentence is a usage rule, not a description: in v0.2 a small model skipped the tool on 17 of 20
# small hand-written problems and got 7 of them wrong, while it was right every time it called the tool.
INTRO = ("Use for every question that asks for the best plan, quantities, assignment, schedule or choice under "
         "limits, even a small one: an optimum worked out by hand is a guess until the solver proves it. Pick the "
         "kind that fits and pass the numbers as stated (decimals are fine).")
OUTRO = "Returns the plan, its value and the status (OPTIMAL = proved best); when no plan exists, the rules that conflict."

# How an agent should use the tools. The MCP server sends it as its instructions, which clients pass to the
# model; frameworks that ignore server instructions can put it in their system prompt. Until v0.3 this was a
# Claude Code skill: the words are unchanged, so benchmark results carry over.
GUIDANCE = """For any request to plan, assign, schedule, cut, ship, produce or choose the best option under limits, call solve, even when the problem looks small enough to do in your head: it takes one call, and hand-worked optima are often wrong.

1. Pick the kind that fits and pass the numbers exactly as stated. For shifts, call get_format first. If a rule is ambiguous, say which reading you used.
2. If the answer is INVALID_INPUT, fix what it lists and call again.
3. Report the status meaning, the plan and its value. If no plan exists, show the conflicting rules and ask which one to relax; never drop a rule yourself."""


def catalogue() -> str:
    """What every agent reads: one line per kind, about 400 tokens in all."""
    return "\n".join([INTRO, *(f"{name}: {k.line}" for name, k in KINDS.items()), OUTRO])


def get_format(kind: str) -> str:
    if kind not in KINDS:
        return f"Unknown kind {kind!r}. Kinds: {', '.join(KIND_NAMES)}."
    return f"{kind}: {KINDS[kind].format}"


def solve(kind: str, data) -> dict:
    """Solve one problem of a known kind; see catalogue() and get_format(kind)."""
    if kind not in KINDS:
        return invalid([f"unknown kind {kind!r}: choose one of {', '.join(KIND_NAMES)}"])
    k = KINDS[kind]
    parsed, error = parse(k.input, data)
    if error:
        return error | {"help": f'call get_format("{kind}") for the full format'}
    return k.run(parsed)
