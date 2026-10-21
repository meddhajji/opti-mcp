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

113 tests, including 60 random small problems where the solver's optimum is compared with
brute-force enumeration of every possible assignment (infeasible cases included), and a
check that every reported conflict is truly minimal. The benchmark scorer is tested on every
verdict, and every reference answer must score as optimal.

## Benchmark (in progress)

The question it answers: does an agent solve optimisation problems better with the
plugin than without it, and can a small cheap model with the plugin match a large one?

**Problem set: 20 problems with proven answers** (`benchmark/problems/`).

- **15 from OptiBench** (Yang et al., ICLR 2025, Apache-2.0, see
  `benchmark/optibench/`): linear problems whose expected answers are all whole numbers,
  sampled at random, 11 word problems and 4 with tables. Every one was re-modelled by hand
  (`benchmark/build_optibench.py`) and the solver's proven optimum matches all 15
  expected answers.
- **3 OptiBench problems were excluded** while checking candidates, each with the reason
  recorded in `selection.json`: one asks two questions at once; one is ambiguous (the
  expected 5,900 holds only if a production line's setup cost is paid even when the line
  is unused, otherwise 4,000 is cheaper); one has an inconsistent label (its own expected
  plan earns 12,000, above the stated maximum of 9,500).
- **5 industrial cases written for this benchmark** (`benchmark/build_industrial.py`):
  operator-to-machine assignment, cutting stock, 4-week production planning, exam
  timetabling, cashier staffing. Each optimum is confirmed by a method that does not use
  the model: exhaustive search, a lower bound or dynamic programming.

**Scorer** (`benchmark/score.py`). The agent ends with
`FINAL ANSWER: {"<answer key>": <number>, ...}` and gets one verdict: `optimal`,
`suboptimal`, `infeasible` (breaks a rule, or fractions where whole numbers are needed),
`inconsistent` (the stated value is not what the stated plan achieves), `impossible`
(claims better than the proven optimum), `incomplete` or `no_answer`. Plans are checked
against the reference model rather than compared to one expected plan, so an alternative
optimal plan counts as optimal, and a broken plan is reported with the rule it breaks.

**Runner** (`benchmark/run.py`). Each problem goes to `claude -p` in restricted mode with
an explicit tool list, in a fresh empty folder, with none of the user's plugins, settings
or connectors, and no web search (the problems are public). Three conditions, same
prompt, which never mentions the plugin:

| Condition | Tools | Represents |
|---|---|---|
| reasoning | none | asking a chatbot |
| code | Bash, Read, Write, Edit | a coding agent writing its own solver |
| plugin | the `optimize` skill and the optimizer MCP tools only | this plugin |

### First results: Claude Haiku 5.5, 20 problems, one run each (21 Oct 2026)

`python benchmark/report.py haiku-full` (raw answers in `benchmark/results/haiku-full.jsonl`)

| Condition | Optimal | Suboptimal | Infeasible | Cost (20 runs) | Median time |
|---|---|---|---|---|---|
| plugin | **19 / 20** | 1 | 0 | $0.035 | 21 s |
| reasoning | **19 / 20** | 0 | 1 | $0.032 | 15 s |
| code | **17 / 20** | 1 | 2 | $0.030 | 14 s |

- **Infeasible answers were fractional plans.** Reasoning answered 1,142.857 batches on
  #404. The coding agent twice wrote a continuous solver for a whole-number problem
  (41.27 and 1,142.86 batches). The plugin condition produced no invalid plan.
- **The one plugin miss, #539, is an ambiguous problem.** It opens with "a researcher
  *must perform* two types of experiments" and then asks for the minimum radiation; the
  label is 0 (no experiments). The plugin and code runs read "must perform" as at least
  one of each and answered 5, and every condition explained both readings. Kept in the
  set, because removing a problem after seeing the results would bias it.
- **Haiku chose the plugin on its own in 17 of 20 runs** and solved the other 3 correctly
  by reasoning. Only `solve_model` was called.
- **What this shows:** on textbook-size problems, the cheapest Claude model is already
  nearly perfect by reasoning alone; the plugin removes invalid fractional plans and
  proves optimality, but does not raise the score. With one run per problem, a gap of
  one or two answers is within noise.

Next: a hard tier of realistic-size problems (rotas over a month, many cutting lengths,
large assignments), where a model cannot hold the whole problem in its head, with three
runs per problem to measure variance; then a larger model as reference.

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
