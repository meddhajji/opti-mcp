# Optimizer: exact optimisation for coding agents

A Claude Code plugin (and MCP server usable by any MCP-capable agent) that lets an agent
solve optimisation problems with a real solver instead of guessing. Ask "plan the shifts
for my five staff this week" or "how many chairs and tables should the workshop make" and
the answer is computed by Google OR-Tools, proved optimal, and re-checked independently
before the agent shows it.

Runs entirely on your machine. No API key, no LLM calls inside the plugin: the agent
describes the problem, the solver decides, the verifier checks.

## Why

Asked to optimise, coding agents tend to write a quick heuristic (a greedy loop, a random
search) whose answer looks plausible but is not optimal, quietly drop a rule, or invent a
plan when none exists. This plugin gives them:

| Tool | What it does |
|---|---|
| `schedule_shifts` | Optimal staff rota from a plain description: people needed per shift, unavailability, max/min shifts, one shift per day, rest between days, max consecutive days, preferences. Goals in strict order: no unnecessary shift, then fairness, then preferences. |
| `solve_model` | Any integer optimisation model (variables, linear constraints, objective) written as JSON. |
| `verify_solution` | Re-checks any proposed solution with plain arithmetic and lists every violated rule. |
| `explain_infeasibility` | When no solution exists, returns a minimal set of rules that cannot all hold, so the user knows exactly what to relax. |

The `optimize` skill tells the agent when to use which tool, to restate the rules before
solving, and to report the solver's status honestly (OPTIMAL means proved best; FEASIBLE
means valid but the time limit stopped the proof).

## Example

From `examples/small_shop.json`: five staff, a week of morning and evening shifts, a
busier Friday and Saturday evening, Sara off on Wednesday, Omar part-time.

- Status OPTIMAL in 0.03 s, verification passed.
- 23 shifts assigned, exactly what the needs add up to.
- Fairness gap of 2, the smallest possible: Omar is capped at 3, and the other 20
  shifts split 5-5-5-5.
- All 4 preferences respected.

Make Friday impossible (Sara and Omar away, Karim unable to do evenings) and the tool
answers with the four rules that clash, leaving out the irrelevant ones:

```
need: at least 3 on Fri evening
Karim unavailable Fri evening
Sara unavailable Fri evening
Omar unavailable Fri evening
```

## How it works

- `optimizer/model.py`: the JSON model format (integer variables and coefficients, the
  form CP-SAT works on; decimals are scaled first).
- `optimizer/solve.py`: builds and solves the model with CP-SAT, then always runs the
  verifier on the answer.
- `optimizer/verify.py`: independent check, no solver involved.
- `optimizer/explain.py`: puts an on/off switch on every rule, asks the solver which
  switches make the problem impossible, then removes rules one by one while it stays
  impossible (deletion filter). The result is an irreducible infeasible subset: removing
  any single rule from it removes the conflict.
- `optimizer/scheduling.py`: turns a rota description into a model using four patterns
  (yes/no assignment, coverage, min-max fairness, weighted priorities).
- `optimizer/server.py`: the MCP server.
- `scripts/launch.py`: what the plugin starts. On first run it creates a private Python
  environment in the plugin's data folder and installs OR-Tools there (about a minute),
  then reuses it.

## Testing

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # bin/python on macOS and Linux
.venv/Scripts/python -m pytest
```

73 tests, including 60 random small problems where the solver's optimum is compared with
brute-force enumeration of every possible assignment (infeasible cases included), and a
check that every reported conflict is truly minimal.

## Install

Requires Python 3.10 or newer, available as `python`.

```bash
claude --plugin-dir /path/to/optimization-plugin
```

## Status and next steps

Version 0.1: the four tools, the `optimize` skill, tests. Next:

1. A benchmark measuring agents with and without the plugin on optimisation word
   problems with known answers (OptiBench and real cases), including a small local model.
2. More high-level tools: assignment, production planning, delivery routing, cutting stock.
3. Continuous variables (linear programming with HiGHS).
