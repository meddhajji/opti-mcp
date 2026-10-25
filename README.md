# Optimizer: exact optimisation for coding agents

A Claude Code plugin (and MCP server usable by any MCP-capable agent, or a plain Python
library) that lets an agent solve optimisation problems with a real solver instead of
guessing. Ask "which worker should do which job", "how should we cut these bars" or "plan
the shifts for my five staff this week": the agent passes the numbers, Google OR-Tools
finds the answer and proves it optimal, and an independent check confirms it before the
agent shows it.

No API key, no LLM calls inside: the agent reads the problem, the solver decides, the
verifier checks. One engine, three ways in: a local plugin, a Python import, or a web
service any assistant can reach by URL.

## Why

Asked to optimise, agents tend to answer from intuition or write a quick heuristic (a
greedy loop, a brute force that stops scaling) whose answer looks plausible but is not
optimal, quietly drop a rule, or invent a plan when none exists. Small models also lose
accuracy as their input grows, so the tool itself must be cheap to read.

## Two tools, about 700 tokens

| Tool | What it does |
|---|---|
| `solve(kind, data)` | Solves one problem of a known kind from its plain data. Its description is the catalogue below, one line per kind. |
| `get_format(kind)` | The full format of one kind with an example, fetched only when needed. |

| Kind | The agent passes | It gets back |
|---|---|---|
| `assignment` | workers, jobs, a cost table (null = not allowed) | who does which job, total |
| `selection` | items with cost and value, a budget (or one per resource), exclusive groups, dependencies | which to choose, value, spent |
| `cutting` | bar length, pieces with length and count | how to cut each bar, number of bars, waste |
| `production` | demand per period, regular and overtime capacity and cost, holding cost, storage | what to make each period, stock, total cost |
| `product_mix` | products with profit and resource use, resources available | units of each product, total profit |
| `transport` | supply, demand, cost per route | what to ship on each route, total cost |
| `shifts` | days, shifts, staff with availability and limits, staffing needs | an optimal, fair rota |
| `model` | any integer model: variables, linear rules, objective | values (zeros left out), objective |

Every answer carries the solver status (OPTIMAL means proved best; FEASIBLE means valid
but the time limit stopped the proof) and the result of an independent check. When no
plan exists, it lists the smallest set of rules that conflict, in plain words, such as
`period 2: demand of 300 met on time` and `period 2: regular at most 120`. Decimals are
fine everywhere (0.25 lb per box): they are scaled to whole numbers exactly, never rounded.

The same functions work as a library, which suits agents that write code:

```python
from optimizer import solve
solve("cutting", {"bar_length": 6, "pieces": [{"length": 2.5, "count": 4}, {"length": 1.2, "count": 7}]})
# {"status": "OPTIMAL", "bars": 4, "plan": {"cuts": [{"cut": "1 x 2.5 + 2 x 1.2", "bars": 4, "waste": 1.1}],
#  "extra_pieces": {"1.2": 1}}, "checked": true, ...}
```

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

- `optimizer/kinds.py`: each kind turns plain data into a model with readable rule names,
  solves it, and answers in the problem's own words. The loops and variables live here,
  not in the agent's reply.
- `optimizer/model.py`: the general model format (whole-number variables, linear rules).
- `optimizer/engine.py`: builds and solves a model with CP-SAT, scaling decimals exactly,
  then always runs the verifier on the answer.
- `optimizer/verify.py`: independent check with exact fractions, no solver involved.
- `optimizer/explain.py`: puts an on/off switch on every rule, asks the solver which
  switches make the problem impossible, then removes rules one by one while it stays
  impossible (deletion filter). The result is an irreducible infeasible subset: removing
  any single rule from it removes the conflict.
- `optimizer/scheduling.py`: turns a rota description into a model using four patterns
  (yes/no assignment, coverage, min-max fairness, weighted priorities).
- `optimizer/server.py`: the MCP server, two tools.
- `optimizer/web.py`: the same server over HTTP, plus plain JSON endpoints (below).
- `scripts/launch.py`: what the plugin starts. On first run it creates a private Python
  environment in the plugin's data folder and installs OR-Tools there (about a minute),
  then reuses it.

## As a web service

The same engine behind a URL, for assistants that take a connector address and for any
program that can send JSON:

| Endpoint | What it does |
|---|---|
| `/mcp` | MCP over HTTP: the same two tools as the plugin (`solve`, `get_format`) |
| `POST /solve` | `{"kind": "...", "data": {...}}` in, the same answer as the tool out |
| `GET /formats` | the catalogue of kinds; `GET /formats/<kind>` gives one kind's full format |
| `GET /health` | status and version |

```bash
curl -X POST https://<your-host>/solve -H "content-type: application/json" -d '{"kind": "shifts", "data":
  {"days": ["Mon", "Tue"], "shifts": ["am", "pm"], "staff": [{"name": "Sara"}, {"name": "Omar",
  "unavailable": [{"day": "Tue", "shift": "*"}]}, {"name": "Lina"}], "needs": [{"day": "*", "shift": "am",
  "min_staff": 1}, {"day": "*", "shift": "pm", "min_staff": 1}]}}'
```

Nothing a caller sends is stored; the log keeps only the kind, status, size and time of
each solve. A free public instance has to stay usable for everyone, so each request is at
most 256 KB, at most 4 problems are solved at once, and one client can make 30 requests a
minute (`OPTIMIZER_MAX_BODY_BYTES`, `OPTIMIZER_MAX_CONCURRENT`,
`OPTIMIZER_REQUESTS_PER_MINUTE` change them). For private data, run it yourself:

```bash
python -m optimizer.web                        # http://127.0.0.1:8000
docker build -t optimizer . && docker run -p 7860:7860 optimizer
```

## Testing

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # bin/python on macOS and Linux
.venv/Scripts/python -m pytest
```

221 tests. Every kind is checked independently: brute force over every plan for
assignment, selection, product mix and transport, dynamic programming for production
planning, and for cutting stock a different formulation (each piece given to a bar) plus
the total-length lower bound. Each generated benchmark
instance must reach its proven optimum from the data a perfect reader would pass. The
general model is compared with brute-force enumeration on 60 random problems, every
reported conflict is checked to be truly minimal, and a test fails if the tools an agent
reads grow past 700 tokens. The benchmark scorer is tested on every verdict, every
reference answer must score as optimal, and the agent loops are tested against scripted
fake models (tool calls, answer formats, rate limits, dropped connections, outages), so
the tests never call a real model. The web service is tested through its endpoints and
limits, and over real HTTP with the MCP SDK's own client.

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

### Second model: gpt-oss-20b on Groq's free tier, same 20 problems (22 Oct 2026)

`python benchmark/report.py gpt-oss-20b-full` (raw answers in
`benchmark/results/gpt-oss-20b-full.jsonl`). Run through this repo's own agent loop with
`reasoning_effort` low, on plugin version 0.1 (before the fix below).

| Condition | Optimal | Suboptimal | Infeasible | Impossible | No answer |
|---|---|---|---|---|---|
| code | **14 / 20** | 3 | 2 | 0 | 1 |
| plugin | **11 / 20** | 5 | 1 | 1 | 2 |
| reasoning | **8 / 20** | 6 | 4 | 0 | 2 |

- **A 20B model gains from tools, but here more from writing code than from the plugin.**
  It chose the plugin in only 10 of 20 runs (Haiku: 17), and those 10 runs scored 4
  optimal. Solving was never the problem; writing the model was:
  - Two runs hit the bound bug described below (#21, #75).
  - Three runs had modelling errors. On production planning the first model was
    infeasible, so the agent set the demand of three of the four weeks to zero (which the
    skill forbids) and claimed 4,000 against a proven 25,700. On #400 the ingredient
    shares of the mixes were scaled wrongly. On cutting stock the agent modelled 3 pieces
    instead of the whole order, then sent a tool call that was not valid JSON.
  - #473 was solved to the optimum, 785, but the agent wrote `**FINAL ANSWER:**` in bold,
    which the scorer does not accept. The same happened to one correct reasoning answer
    (#56).
- **Fractional plans again came only from the conditions without the plugin.** Reasoning
  gave fractions twice, code once.
- **What this shows:** for a small model, the hard step is turning words into a correct
  general model. Version 0.2 (below) gives it per-family tools where it only fills in the
  data.

### A bug the benchmark found

In version 0.1, an integer variable sent without `ub` silently got `ub = 1`. On #21
gpt-oss-20b sent three quantities without bounds. The tool answered OPTIMAL with every
quantity at 1, a profit of 515, and the verification passed, because that plan was valid
for the model it received. The proven optimum is 27,510. Haiku's tool calls were not
logged, but its plugin answers show no sign of the bug (19 of 20 optimal). Now:

- An integer variable without `ub` is rejected with status `INVALID_MODEL` and one plain
  sentence per problem ("integer variable 'x1' needs an upper bound 'ub': use the largest
  value that makes sense..."), rather than a guessed bound.
- Integer variables that end at their bound are listed in `at_upper_bound` with a
  warning, because an invented bound (say 1,000) can cap the answer the same way.
- Every input mistake comes back as a short list in plain words rather than raw
  validation output, and tool schemas are flattened (no `$ref`), which small models read
  more reliably.

Replaying the logged #21 call with bounds added gives 27,510, the proven optimum. #75
would still fail: its machine-rental constraints were also wrong (250 instead of 75). The
lesson: "verified" only means the answer satisfies the model the agent wrote, so the
plugin must refuse to fill in anything the agent left out.

### Making it light (v0.2, 22 Oct 2026)

The gpt-oss-20b runs exposed what v0.1 cost a small model. Measured on one yardstick
(about 4 characters per token, `tests/test_server.py` keeps it under 700):

| Tokens | v0.1 | v0.2 |
|---|---|---|
| Read on every request (tool descriptions and skill) | 3,906 | 620 |
| Written for one correct call: assignment, 12 workers | 4,246 | 166 |
| Written for one correct call: cutting stock, 8 piece lengths | 11,136 | 65 |
| Returned by the tool: assignment, 12 workers | about 850 | about 60 |

Groq's free tier caps each request at 8,000 tokens, prompt and reply together, so with
v0.1 a correct model of the largest generated problems could not even be sent. What
changed:

- One `solve` tool whose description is a catalogue, one line per kind. The full format
  of a kind is fetched only when needed (`get_format`), the way skills load their details.
- The kind writes the model. The agent passes the data it has already read (a cost
  table, a list of pieces); the loops and the variables are written by tested code.
- Answers are one line of JSON in the problem's words, with zeros left out.
- Decimals are accepted and scaled exactly. The #400 failure above was a hand-scaling
  mistake (0.25 lb written as 25 on a scale where 1 lb was 1000).
- The skill is 140 tokens instead of 1,000.
- The same functions can be imported as a Python library, for agents that write code.

Version 0.3 (below) adds a usage rule and the general model's one-line format: 694 tokens.

**The risk, stated up front.** The kinds cover exactly the five families the generator
produces, so on generated problems they could look good simply because they match the
test. Two checks against that: the families are the textbook ones (assignment, knapsack,
cutting stock, lot sizing, product mix, plus transport) chosen because they are what
industrial engineers meet most, and the 20 hand-written problems, which were not written
for the kinds, are rerun with v0.2 (results below).

### Generated problems, more models and a fourth condition

`benchmark/generate.py` builds word problems from code, in five families with a size
level from 1 to 5: product mix (2 to 12 products), project selection (5 to 35 projects),
assignment (3 to 12 workers), cutting stock (2 to 8 piece lengths) and production
planning (3 to 12 weeks). Numbers never come from an LLM. Every optimum is proven by the
solver and, wherever computable, confirmed independently: dynamic programming at every
size for project selection, assignment and production planning, brute force or a second
formulation for small product-mix and cutting-stock instances; where no check exists the
instance says so. A noise level tells the same instance either plainly or as a longer
story with irrelevant facts, with identical numbers, so noise effects are paired.
Instance ids (`assignment-L3-N1-2`) regenerate exactly the same problem.

Besides Claude Code, `--backend groq` and `--backend gemini` run open and cheap models
(gpt-oss-20b, Gemma 4 26B, Gemini Flash-Lite) through this repo's own agent loop
(`benchmark/agent_common.py`), with the same conditions, prompt and scorer. Two additions
for these runs:

- `--answer-tool`: the agent ends by calling `submit_answer` with one number per answer
  key, so no answer is ever parsed from prose (two correct gpt-oss answers were lost to a
  bold `**FINAL ANSWER:**`). An agent that stops without calling it is reminded once.
- A fourth condition, `library`: plain code execution where scripts can also
  `import optimizer`, with the same guidance as the skill (about 500 tokens).

### v0.2 on 10 large generated problems (23 Oct 2026)

`benchmark/sets/scale-v1`: one problem per family at size levels 4 and 5 (up to 12
workers, 35 projects, 8 piece lengths, 12 weeks), answers through `submit_answer`. Raw
runs in `benchmark/results/scale-v1-*.jsonl`.

| Model | Condition | Optimal | Median tokens per run | Total tokens |
|---|---|---|---|---|
| gpt-oss-20b | code | 4 / 10 | 7,813 | 92,351 |
| gpt-oss-20b | plugin v0.2 | **9 / 10** | **3,354** | 52,493 |
| gpt-oss-20b | library | **10 / 10** | 4,035 | 53,240 |
| Gemma 4 26B | code | 9 / 10 | 27,102 | 384,730 |
| Gemma 4 26B | plugin v0.2 | **10 / 10** | 11,698 | 139,827 |
| Gemma 4 26B | library | **10 / 10** | **10,467** | 110,787 |

- **For the weaker model, the toolkit more than doubles accuracy with about 43% fewer
  tokens.** Five of code's six misses were numbers it never computed: after its scripts
  crashed (PuLP is not installed, for instance), gpt-oss submitted a lower bound or zero,
  such as 31 and 24 bars against proven optima of 32 and 25, or a cost of 0 against
  124,498. The sixth, 35 projects, was close but not optimal (1,554,000 vs 1,557,000).
- **For Gemma, accuracy is nearly equal; cost is not.** The plugin and the library use
  under half of code's tokens per run (about a third in total) and finish in a median of
  80 to 100 s instead of 337 s.
- **The plugin's one miss came from the format, not the data.** On product mix with 8
  products the agent put an "at least 1 chair" rule in a field that does not exist, worked
  around the error by reserving one chair's resources, then forgot to add that chair's
  profit (5,505 instead of 5,565).
- 38 of the 40 plugin and library runs used the matching kind. The other two were Gemma
  on product mix, which wrote the problem as a general model (one of them only partly
  visible in a truncated log), and both still reached the optimum.

### v0.2 on the 20 hand-written problems (24 Oct 2026)

gpt-oss-20b, with the same text answer format as the v0.1 runs above
(`benchmark/results/gpt-oss-20b-v02.jsonl`):

| Condition | Optimal | Called the tool | Optimal when it did | Median tokens |
|---|---|---|---|---|
| reasoning (v0.1 runs) | 8 / 20 | | | 1,370 |
| code (v0.1 runs) | 14 / 20 | | | 1,572 |
| plugin v0.1 | 11 / 20 | 10 | 4 | 6,568 |
| plugin v0.2 | 13 / 20 | **3** | 3 | 1,687 |
| library | **15 / 20** | 16 | 11 | 5,784 |

- **The library is the most accurate condition on these problems too.**
- **The plugin's weakness moved.** v0.2 fixed what went wrong when the tool was called
  (3 of 3 optimal, against 4 of 10 with v0.1), but gpt-oss called it on only 3 of the 20
  problems. All 7 plugin misses are problems it answered by hand: on textbook-size
  problems, a 20B model often decides it does not need a tool, and is then often wrong.
- Two library runs lost to a dropped connection were redone.

### v0.3: asking for the tool even on small problems (test in progress)

The change: `solve`'s description now opens with a usage rule, echoed in the skill ("use
for every question that asks for the best plan, quantities, assignment, schedule or
choice under limits, even a small one: an optimum worked out by hand is a guess until the
solver proves it"), and the general `model` kind carries its own one-line format, so a
small problem outside the families takes one call instead of two. 694 tokens.

The test, with its success criteria fixed before any run: gpt-oss-20b calls the tool on
at least 15 of 20 problems in each set (from 3 of 20), the plugin is at least as accurate
as the library on the hand-written problems, the large problems do not get worse, and the
tools stay under 700 tokens. v0.2 and v0.3 answer the same problems in the same format,
with v0.2 run from a pinned checkout:

- **20 unseen story problems** (`benchmark/sets/small-story-v1`): size levels 1 and 2,
  told with irrelevant facts, from fresh seeds (101 and 102).
- **10 new OptiBench problems** (`benchmark/sets/optibench-heldout-v1`), drawn at random
  after v0.3 was designed (seed 20261024), with two exclusions recorded in
  `benchmark/optibench/heldout.json`. All 10 labels match the solver's proven optimum,
  and only 3 of the 10 fit a kind, so 7 test the general path.

Results so far:

| | v0.2 optimal | v0.2 called the tool | v0.3 optimal | v0.3 called the tool |
|---|---|---|---|---|
| Gemma 4 26B, 20 story problems | 17 | 20 | 14 | 20 |
| Gemma 4 26B, 10 new OptiBench problems | 10 | 9 | 10 | 10 |
| gpt-oss-20b, 20 story problems | 18 | 6 | running | running |
| gpt-oss-20b, 10 new OptiBench problems | running | running | running | running |

- **Gemma already calls the tool on almost every problem**, so v0.3 has nothing to fix for
  it. Its lower v0.3 score on the story problems is answer format, not numbers: five runs
  (two with v0.2, three with v0.3) wrote the right optimum with the number as its own
  label (`{"91000": 91000}`), and two v0.3 runs ended with no text. Counting right numbers,
  19 against 17.
- **gpt-oss with v0.2 called the tool on 6 of the 20 story problems**, and all 6 were
  optimal; both misses were answered by hand. Its v0.3 runs are still going.
- **A flaw in the story generator.** Gemma's one wrong story answer had every number right:
  the gap was exactly an office rent mentioned as an irrelevant fact ("7,600 dollars per
  month, paid whatever the plan"), which it added to "the total cost of meeting demand".
  That reading is defensible, so future sets must keep money facts out of cost and
  profit questions. Both versions face the same problems, so this comparison stays fair.

### What the harness got wrong, and how it was fixed

Not model results, but each one changed what could be measured:

- **Groq's "try again in 539ms" was read as 539 minutes**, so runs slept for up to 9
  hours. Milliseconds are now read first, a test covers it, and every wait is logged.
- **A 30-minute loss of internet** turned 10 runs into "no answer", and a Gemini request
  sent as the connection dropped hung for over an hour. Connections are now retried,
  requests time out after 10 minutes, and a lasting outage stops the run so that it
  resumes later instead of recording failures. The failed runs were set aside, not
  deleted, and redone.
- **In the plugin condition the MCP client wraps every error in an exception group**, so a
  stop signal was recorded as an ordinary error. Errors are now unwrapped.
- **Gemini's 504 (the server gave up) is now retried**, and cached prompt tokens, which do
  not count against Groq's limits, are recorded separately.

## Install

Requires Python 3.10 or newer, available as `python`.

```bash
claude --plugin-dir /path/to/optimization-plugin
```

As a library:

```bash
pip install git+https://github.com/meddhajji/optimization-plugin
```

## Status and next steps

Version 0.3: eight problem kinds behind two tools (about 700 tokens), the `optimize`
skill, the Python library, 214 tests, and the benchmark above. Next:

1. Finish the v0.3 test (gpt-oss-20b on the new OptiBench problems, the hand-written
   problems and the large ones).
2. If small models still skip the tool on small problems, rethink how it is offered
   rather than reword it again. The library condition, where the agent writes code that
   imports the solver, is the strongest candidate so far.
3. The size sweep (levels 1 to 5, several seeds, noise on and off) as charts of accuracy
   and tokens against problem size, after fixing the money-distractor flaw.
4. Infeasible instances, to measure whether agents report conflicts instead of
   inventing a plan.
5. More kinds where agents fail most: vehicle routing, bin packing, job-shop scheduling.
6. Continuous variables (linear programming with HiGHS).
