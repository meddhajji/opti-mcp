"""The general model format every tool speaks.

An optimisation model has three parts:
  variables    what we get to decide, each with a type and bounds
  constraints  linear rules: sum(coefficient * variable) <sense> rhs
  objective    a linear expression to minimise or maximise (optional:
               without one, any solution that respects the rules will do)

Example, as JSON:
  {
    "variables": [
      {"name": "chairs", "type": "integer", "lb": 0, "ub": 100},
      {"name": "tables", "type": "integer", "lb": 0, "ub": 100}
    ],
    "constraints": [
      {"name": "wood", "terms": {"chairs": 2, "tables": 5}, "sense": "<=", "rhs": 200},
      {"name": "hours", "terms": {"chairs": 1, "tables": 3}, "sense": "<=", "rhs": 90}
    ],
    "objective": {"sense": "maximize", "terms": {"chairs": 30, "tables": 80}}
  }

Variables take whole-number values. Coefficients may be decimals (0.25 pounds
of cashews per box), up to 6 decimal places: the engine scales each rule to
whole numbers exactly, so nothing is rounded.
"""
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Sense = Literal["<=", ">=", "=="]
Number = int | float
MAX_DECIMALS = 6


def check_decimals(where: str, numbers) -> None:
    for x in numbers:
        if -Decimal(str(x)).normalize().as_tuple().exponent > MAX_DECIMALS:
            raise ValueError(f"{where}: {x} has more than {MAX_DECIMALS} decimal places")


class Variable(BaseModel):
    name: str
    type: Literal["integer", "boolean"] = "integer"
    lb: int = Field(0, description="Lower bound, 0 if omitted. Ignored for booleans.")
    ub: int | None = Field(
        None,
        description="Upper bound, required for integer variables: the largest value that makes sense, "
                    "for example what the available resources allow. Ignored for booleans.",
    )

    @model_validator(mode="after")
    def _bounds(self):
        if self.type == "boolean":
            self.lb, self.ub = 0, 1
        elif self.ub is None:
            # Never guess a bound: a silent default once capped quantities at 1 and returned a
            # confident, "verified" answer to the wrong problem (see benchmark README).
            raise ValueError(f"integer variable '{self.name}' needs an upper bound 'ub': use the largest value "
                             "that makes sense, for example what the available resources allow")
        if self.lb > self.ub:
            raise ValueError(f"variable {self.name}: lb {self.lb} > ub {self.ub}")
        return self


class Constraint(BaseModel):
    name: str = Field(description="Human-readable: it is what infeasibility explanations quote.")
    terms: dict[str, Number] = Field(description="variable name -> coefficient")
    sense: Sense
    rhs: Number
    definition: bool = Field(
        False,
        description="True for bookkeeping equations (e.g. total = sum of parts) that are never "
                    "the cause of a conflict; they are left out of infeasibility explanations.",
    )


class Objective(BaseModel):
    sense: Literal["minimize", "maximize"]
    terms: dict[str, Number]
    constant: Number = 0


class Model(BaseModel):
    variables: list[Variable]
    constraints: list[Constraint] = []
    objective: Objective | None = None

    @model_validator(mode="after")
    def _references(self):
        names = [v.name for v in self.variables]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate variable names: {dupes}")
        known = set(names)
        cnames = [c.name for c in self.constraints]
        if len(cnames) != len(set(cnames)):
            dupes = sorted({n for n in cnames if cnames.count(n) > 1})
            raise ValueError(f"duplicate constraint names: {dupes}")
        for c in self.constraints:
            unknown = set(c.terms) - known
            if unknown:
                raise ValueError(f"constraint {c.name} uses unknown variables {sorted(unknown)}")
            if not c.terms:
                raise ValueError(f"constraint {c.name} has no terms")
            check_decimals(f"constraint {c.name}", [*c.terms.values(), c.rhs])
        if self.objective:
            unknown = set(self.objective.terms) - known
            if unknown:
                raise ValueError(f"objective uses unknown variables {sorted(unknown)}")
            check_decimals("objective", [*self.objective.terms.values(), self.objective.constant])
        return self
