"""Staff scheduling, the first high-level tool.

The caller describes the situation (who, when, how many are needed, who is
unavailable) and this module writes the optimisation model itself, so a
small LLM never has to get the modelling right. It reuses four patterns:

  assignment   work[p, d, s] is 1 if person p works shift s on day d
  coverage     sum over people of work[., d, s] >= people needed on (d, s)
  fairness     helper variables `most` >= every total and `least` <= every
               total; minimising most - least balances the workload
  priorities   weights make the solver respect, strictly in this order:
               1. no shift assigned beyond what is needed (staffing cost)
               2. fairness between people
               3. personal preferences ("rather not work Friday")
               each weight exceeds the largest possible total of every goal
               below it (lexicographic optimisation with one solve)
"""
from pydantic import BaseModel, Field, model_validator

from .explain import explain_infeasible
from .model import Constraint, Model, Objective, Variable
from .solve import solve

ANY = "*"


class Slot(BaseModel):
    day: str
    shift: str = Field(ANY, description='A shift name, or "*" for every shift of that day.')


class Person(BaseModel):
    name: str
    max_shifts: int | None = Field(None, description="Most shifts over the whole period.")
    min_shifts: int = Field(0, description="Fewest shifts over the whole period.")
    unavailable: list[Slot] = Field(default_factory=list, description="Hard: never assigned here.")
    prefers_off: list[Slot] = Field(default_factory=list, description="Soft: avoided when possible.")


class Need(BaseModel):
    day: str = Field(ANY, description='A day name, or "*" for every day.')
    shift: str
    min_staff: int
    max_staff: int | None = None


class ScheduleRequest(BaseModel):
    days: list[str] = Field(description="Days in calendar order, e.g. Mon..Sun.")
    shifts: list[str] = Field(description="Shifts in the order they happen within a day, e.g. morning, evening, night.")
    staff: list[Person]
    needs: list[Need]
    one_shift_per_day: bool = True
    rest_between_days: bool = Field(
        True, description="Nobody works the last shift of a day and then the first shift of the next day.")
    max_consecutive_days: int | None = Field(None, description="Most days in a row anyone may work.")

    @model_validator(mode="after")
    def _known_names(self):
        days, shifts = set(self.days), set(self.shifts)
        for slot in [s for p in self.staff for s in p.unavailable + p.prefers_off]:
            if slot.day not in days or (slot.shift != ANY and slot.shift not in shifts):
                raise ValueError(f"unknown day or shift in {slot.day} {slot.shift}")
        for need in self.needs:
            if (need.day != ANY and need.day not in days) or need.shift not in shifts:
                raise ValueError(f"unknown day or shift in need {need.day} {need.shift}")
        names = [p.name for p in self.staff]
        if len(names) != len(set(names)):
            raise ValueError("staff names must be unique")
        return self


def x(p: str, d: str, s: str) -> str:
    return f"work[{p}, {d}, {s}]"


def _slots(request: ScheduleRequest, slot: Slot) -> list[tuple[str, str]]:
    shifts = request.shifts if slot.shift == ANY else [slot.shift]
    return [(slot.day, s) for s in shifts]


def build_model(request: ScheduleRequest) -> Model:
    r = request
    people, days, shifts = [p.name for p in r.staff], r.days, r.shifts
    total_slots = len(days) * len(shifts)
    variables = [Variable(name=x(p, d, s), type="boolean") for p in people for d in days for s in shifts]
    cons: list[Constraint] = []

    # coverage: enough (and not too many) people on every shift
    needs: dict[tuple[str, str], Need] = {}
    for n in r.needs:  # a specific day overrides a "*" rule
        for d in (days if n.day == ANY else [n.day]):
            if n.day != ANY or (d, n.shift) not in needs:
                needs[(d, n.shift)] = n
    for (d, s), n in needs.items():
        terms = {x(p, d, s): 1 for p in people}
        cons.append(Constraint(name=f"need: at least {n.min_staff} on {d} {s}", terms=terms, sense=">=", rhs=n.min_staff))
        if n.max_staff is not None:
            cons.append(Constraint(name=f"need: at most {n.max_staff} on {d} {s}", terms=terms, sense="<=", rhs=n.max_staff))

    for person in r.staff:
        p = person.name
        for slot in person.unavailable:
            for d, s in _slots(r, slot):
                cons.append(Constraint(name=f"{p} unavailable {d} {s}", terms={x(p, d, s): 1}, sense="<=", rhs=0))
        all_shifts = {x(p, d, s): 1 for d in days for s in shifts}
        if person.max_shifts is not None:
            cons.append(Constraint(name=f"{p}: at most {person.max_shifts} shifts", terms=all_shifts, sense="<=", rhs=person.max_shifts))
        if person.min_shifts:
            cons.append(Constraint(name=f"{p}: at least {person.min_shifts} shifts", terms=all_shifts, sense=">=", rhs=person.min_shifts))
        if r.one_shift_per_day:
            for d in days:
                cons.append(Constraint(name=f"{p}: one shift per day ({d})", terms={x(p, d, s): 1 for s in shifts}, sense="<=", rhs=1))
        if r.rest_between_days and len(shifts) > 1:
            for d, nxt in zip(days, days[1:]):
                cons.append(Constraint(
                    name=f"{p}: rest after {d} {shifts[-1]}",
                    terms={x(p, d, shifts[-1]): 1, x(p, nxt, shifts[0]): 1}, sense="<=", rhs=1))
        if r.max_consecutive_days is not None and r.max_consecutive_days < len(days):
            k = r.max_consecutive_days
            # worked[p, d] is 1 when p works any shift on day d (bookkeeping)
            for d in days:
                w = f"worked[{p}, {d}]"
                variables.append(Variable(name=w, type="boolean"))
                for s in shifts:
                    cons.append(Constraint(name=f"def {w} >= {s}", terms={w: 1, x(p, d, s): -1}, sense=">=", rhs=0, definition=True))
                cons.append(Constraint(name=f"def {w} <= shifts", terms={w: 1, **{x(p, d, s): -1 for s in shifts}}, sense="<=", rhs=0, definition=True))
            for i in range(len(days) - k):
                window = days[i:i + k + 1]
                cons.append(Constraint(
                    name=f"{p}: at most {k} days in a row ({window[0]}-{window[-1]})",
                    terms={f"worked[{p}, {d}]": 1 for d in window}, sense="<=", rhs=k))

    # fairness helpers: total[p], most >= every total, least <= every total
    variables += [Variable(name=f"total[{p}]", lb=0, ub=total_slots) for p in people]
    variables += [Variable(name="most", lb=0, ub=total_slots), Variable(name="least", lb=0, ub=total_slots)]
    for p in people:
        cons.append(Constraint(name=f"def total[{p}]", terms={f"total[{p}]": 1, **{x(p, d, s): -1 for d in days for s in shifts}},
                               sense="==", rhs=0, definition=True))
        cons.append(Constraint(name=f"def most >= total[{p}]", terms={"most": 1, f"total[{p}]": -1}, sense=">=", rhs=0, definition=True))
        cons.append(Constraint(name=f"def least <= total[{p}]", terms={"least": 1, f"total[{p}]": -1}, sense="<=", rhs=0, definition=True))

    # priorities through weights (see module docstring)
    pref_terms = {x(person.name, d, s): 1 for person in r.staff for slot in person.prefers_off for d, s in _slots(r, slot)}
    w_pref = 1
    w_fair = len(pref_terms) + 1                   # beats every possible preference total
    w_cost = w_fair * (total_slots + 1)            # beats every possible fairness gap
    objective_terms: dict[str, int] = {}
    for p in people:
        for d in days:
            for s in shifts:
                objective_terms[x(p, d, s)] = w_cost + w_pref * pref_terms.get(x(p, d, s), 0)
    objective_terms["most"] = w_fair
    objective_terms["least"] = -w_fair
    return Model(variables=variables, constraints=cons, objective=Objective(sense="minimize", terms=objective_terms))


def schedule_shifts(request: ScheduleRequest, time_limit_s: float = 20.0) -> dict:
    model = build_model(request)
    result = solve(model, time_limit_s=time_limit_s)
    out = {
        "status": result["status"],
        "meaning": result["meaning"],
        "model_size": {"variables": len(model.variables), "constraints": len(model.constraints)},
        "solve_seconds": result["solve_seconds"],
    }
    if result["status"] == "INFEASIBLE":
        out["conflict"] = explain_infeasible(model)
        return out
    if "values" not in result:
        return out

    v = result["values"]
    schedule = {d: {s: [p.name for p in request.staff if v[x(p.name, d, s)]] for s in request.shifts} for d in request.days}
    per_person = {}
    for person in request.staff:
        misses = [f"{d} {s}" for slot in person.prefers_off for d, s in _slots(request, slot) if v[x(person.name, d, s)]]
        per_person[person.name] = {"shifts": v[f"total[{person.name}]"], "preference_misses": misses}
    n_prefs = sum(len(_slots(request, slot)) for p in request.staff for slot in p.prefers_off)
    n_missed = sum(len(p["preference_misses"]) for p in per_person.values())
    out.update({
        "schedule": schedule,
        "per_person": per_person,
        "summary": {
            "total_shifts_assigned": sum(p["shifts"] for p in per_person.values()),
            "fairness_gap": v["most"] - v["least"],
            "preferences_respected": f"{n_prefs - n_missed} of {n_prefs}",
        },
        "verification": result["verification"],
    })
    return out
