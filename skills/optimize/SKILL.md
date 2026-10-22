---
name: optimize
description: Solve optimisation, scheduling and allocation problems exactly with the optimizer tools instead of reasoning out or hand-coding an answer. Use whenever the user wants the best or a valid choice under rules, such as staff rotas and shift plans, assigning people or machines to tasks, production quantities under limited resources, picking items under a budget, or when they ask why no plan satisfies their constraints.
---

# Optimize: exact answers from a solver, never from guesswork

The optimizer tools run a proven solver (Google OR-Tools CP-SAT) locally. They return
answers that are mathematically optimal or say plainly that they are not, and every
answer is re-checked by an independent verifier. Your job is to describe the problem
correctly and to report the result honestly; the tools do the deciding.

## 1. Recognise the problem and pick the tool

| The user wants... | Use |
|---|---|
| A staff rota: who works which shift on which day | `schedule_shifts` |
| Anything else with whole-number decisions (how many to make, yes/no choices, assignments, budgets) | `solve_model` |
| To check a plan they (or you) already have | `verify_solution` |
| To know why their rules have no solution | `solve_model` reports it; `explain_infeasibility` for a model you already have |

Do not solve the problem yourself, by hand or with a script you write, and then present
the result as optimal. If no tool fits, say so.

## 2. Restate the rules before solving

List back, in plain words, what you are about to give the tool:
- hard rules (must always hold), e.g. "at least 2 people every evening"
- soft wishes (respected when possible), e.g. "Karim prefers Fridays off"
- the goal, e.g. "fewest shifts, then the fairest split"

Ask the user to confirm whenever a rule is ambiguous (per day or per week? at least or
exactly?). A solver gives the perfect answer to the problem it receives; a misunderstood
rule produces a perfect answer to the wrong problem.

## 3. Report the result

- Quote the `meaning` of the status. OPTIMAL means proved best. FEASIBLE means valid but
  not proved best (the time limit stopped the search): say so, and offer a longer
  `time_limit_s`.
- Show the solution as a table, and state the verification result (`valid` and any
  violations). Never present a solution whose verification failed.
- On INFEASIBLE, show the conflicting rules from `conflict` and ask which one the user
  wants to relax. Do not silently drop or change a rule to make the problem solvable.

## 4. Writing a model for `solve_model`

A model is variables, linear constraints and an optional objective, all with integer
numbers. Name every constraint in plain words ("wood available"), because conflict
explanations quote the names. Reusable patterns:

- **Quantities**: an integer variable per decision, with realistic bounds
  (`chairs` in 0..100). Every integer variable needs an `ub`: the largest value that
  makes sense, such as what the scarcest resource allows. If the result lists
  `at_upper_bound`, raise any bound that is not a real limit and solve again.
- **Yes/no choices**: a boolean per option, e.g. `take[item]`. A budget becomes
  `sum(cost[i] * take[i]) <= budget`.
- **Assignment**: boolean `assign[w, t]`; "each task done once" is
  `sum over w of assign[w, t] == 1`; "each worker at most one task" is
  `sum over t of assign[w, t] <= 1`.
- **Resource limits**: `sum(use[r, p] * qty[p]) <= available[r]` for each resource r.
- **If chosen then pay a fixed cost** (big-M): `qty[p] - M * open[p] <= 0`, with
  `open[p]` boolean and M the largest possible `qty[p]`; add `fixed[p] * open[p]` to a
  minimised cost.
- **Fairness (min-max)**: helper variable `most`; constraints `most - load[w] >= 0` for
  every w; minimise `most`. Mark such bookkeeping constraints `"definition": true` so
  they are never blamed in a conflict.
- **Several goals in order**: give each goal a weight larger than the largest possible
  total of all goals below it, then minimise the weighted sum.
- **Decimals**: scale to integers (prices in cents, weights in grams) and say so when
  reporting.

Example:

```json
{
  "variables": [
    {"name": "chairs", "type": "integer", "lb": 0, "ub": 100},
    {"name": "tables", "type": "integer", "lb": 0, "ub": 100}
  ],
  "constraints": [
    {"name": "wood available (units)", "terms": {"chairs": 2, "tables": 5}, "sense": "<=", "rhs": 200},
    {"name": "workshop hours", "terms": {"chairs": 1, "tables": 3}, "sense": "<=", "rhs": 90}
  ],
  "objective": {"sense": "maximize", "terms": {"chairs": 30, "tables": 80}}
}
```
