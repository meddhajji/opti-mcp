# Optimizer: exact optimisation for cheap and open-source LLMs

[![tests](https://github.com/meddhajji/opti-mcp/actions/workflows/tests.yml/badge.svg)](https://github.com/meddhajji/opti-mcp/actions/workflows/tests.yml)

An MCP server that lets any LLM, especially small and open-source ones, solve
optimisation problems exactly. The model reads the problem and passes the numbers;
Google OR-Tools finds the best plan and proves it optimal; an independent check confirms
it before the model answers. No API key, and no LLM inside.

Use it by URL, self-host it with Docker, or import it in Python.

## Why

Open models are cheap enough to run at scale but not trusted to plan. Asked for the best
production plan or assignment, they answer from intuition, or write solver code that
crashes and then report a number they never computed. This server gives them a tool that
cannot get the maths wrong, and keeps it light (about 700 tokens per request) for small
models with small context windows.

Measured on 10 large generated problems (up to 12 workers, 35 projects, 8 piece lengths,
12 weeks), each with a proven optimum:

| Model | Writing its own code | With this server | Tokens per run (median) |
|---|---|---|---|
| gpt-oss-20b | 4 / 10 optimal | **9 / 10** | 7,813 → **3,354** |
| Gemma 4 26B | 9 / 10 | **10 / 10** | 27,102 → **11,698** |

Five of gpt-oss's six misses with its own code were numbers it never computed. What the
server does not fix yet is measured too: on small textbook problems a 20B model often
skips the tool, and when a problem needs a general model rather than a known kind, it
often writes that model wrongly. The full benchmark, failures included, is in
[docs/benchmark.md](docs/benchmark.md).

## What it solves

Two tools: `solve(kind, data)` and `get_format(kind)`. The model picks a kind and passes
the data it read; the server writes and solves the model.

| Kind | The model passes | It gets back |
|---|---|---|
| `assignment` | workers, jobs, a cost table (null = not allowed) | who does which job, total |
| `selection` | items with cost and value, a budget (or one per resource), exclusive groups, dependencies | which to choose, value, spent |
| `cutting` | bar length, pieces with length and count | how to cut each bar, number of bars, waste |
| `production` | demand per period, regular and overtime capacity and cost, holding cost, storage | what to make each period, stock, total cost |
| `product_mix` | products with profit and resource use, resources available | units of each product, total profit |
| `transport` | supply, demand, cost per route | what to ship on each route, total cost |
| `shifts` | days, shifts, staff with availability and limits, staffing needs | an optimal, fair rota |
| `model` | any integer model: variables, linear rules, objective | values (zeros left out), objective |

Every answer carries the solver status (OPTIMAL means proved best) and the result of an
independent check. When no plan exists, it names the smallest set of rules that conflict,
in plain words ("period 2: demand of 300 met on time", "period 2: regular at most 120").
Decimals are scaled to whole numbers exactly, never rounded.

## Try it

With Docker:

```bash
git clone https://github.com/meddhajji/opti-mcp && cd opti-mcp
docker build -t optimizer . && docker run -p 7860:7860 optimizer
```

Or with Python 3.10 or newer:

```bash
pip install git+https://github.com/meddhajji/opti-mcp
optimizer-web     # MCP over HTTP at http://127.0.0.1:8000/mcp, JSON at /solve
optimizer-mcp     # MCP over stdio, for clients that start the server themselves
```

Then ask it something (port 7860 for Docker, 8000 for `optimizer-web`):

```bash
curl -X POST http://127.0.0.1:7860/solve -H "content-type: application/json" \
  -d '{"kind": "assignment", "data": {"workers": ["Ana", "Ben"], "jobs": ["J1", "J2"], "cost": [[4, 7], [6, 3]]}}'
# {"status": "OPTIMAL", "total": 7, "plan": {"Ana": "J1", "Ben": "J2"}, "checked": true, ...}
```

A public instance is on its way; its address will be added here.

## Connect a model

- **Any MCP client** (agent frameworks, chat apps that take a connector URL): point it at
  `http(s)://<host>/mcp`, or let it start `optimizer-mcp` over stdio. The server sends its
  usage guidance to the client as MCP instructions.
- **Frameworks that ignore server instructions:** add the same guidance to the system
  prompt with `from optimizer import GUIDANCE`.
- **No MCP at all:** call `POST /solve`, or import the same functions:

```python
from optimizer import solve
solve("cutting", {"bar_length": 6, "pieces": [{"length": 2.5, "count": 4}, {"length": 1.2, "count": 7}]})
# {"status": "OPTIMAL", "bars": 4, "plan": {"cuts": [...]}, "checked": true, ...}
```

The web service keeps nothing a caller sends and logs only the kind, status, size and
time of each solve. A public instance allows 256 KB per request, 4 solves at once and 30
requests a minute per client (all configurable). For private data, run your own.

## How it works

- `optimizer/kinds.py`: each kind turns plain data into a model with readable rule names
  and answers in the problem's words.
- `optimizer/engine.py`: solves the model with CP-SAT (OR-Tools), scaling decimals exactly.
- `optimizer/verify.py`: an independent check with exact fractions, no solver involved.
- `optimizer/explain.py`: when no plan exists, finds the smallest set of rules that
  conflict (an irreducible infeasible subset).
- `optimizer/server.py`, `optimizer/web.py`: MCP over stdio, and over HTTP with JSON
  endpoints.
- `benchmark/`: problem generators with proven optima, held-out sets, the agent harness
  and the scorer.

## How it got here

- **v0.1, 19 to 21 Oct 2026.** Four general tools, packaged as a Claude Code plugin.
  Claude Haiku scored 19/20 on textbook problems by reasoning alone: strong models barely
  need it.
- **The gpt-oss-20b baseline.** A 20B model did better writing its own code (14/20) than
  with the tool (11/20), and the benchmark found a bug: an integer variable sent without
  an upper bound was silently capped at 1, which gave a confident, "verified" wrong answer.
- **v0.2, 22 Oct.** Made light (3,906 to 620 tokens per request) and specific: the model
  passes the data of a known kind instead of writing a model. On large problems gpt-oss
  scored 9/10 with it, against 4/10 writing its own code.
- **v0.3, 24 to 25 Oct.** Small models skipped the tool on small problems. A usage rule
  raised tool calls (3 to 10 of 20 on the hand-written set) but not accuracy: the extra
  calls landed on problems that need a general model, which a 20B model often gets wrong.
- **v0.4, 26 Oct.** A standalone MCP server and web service. The Claude Code packaging is
  gone: the models that need this are the cheap ones.

Next: building blocks for the general model (percentage limits, setup costs, covering
rules) and the tool restating each model in plain words, measured on fresh held-out
problems.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

222 tests, run on GitHub at every push: every kind against an independent method (brute
force, dynamic programming, a second formulation), every benchmark answer against its
proven optimum, the server as a client sees it (including real MCP over HTTP), and a
budget that fails if what a model reads grows past 700 tokens. The tests never call a
real model.

## License

MIT. The OptiBench problems in `benchmark/` come from Yang et al. (ICLR 2025, Apache-2.0).
