"""Summarise a benchmark results file: scores per model and condition, then
every answer that was not optimal, with the scorer's reason.

Run: python benchmark/report.py haiku-full
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH.parent))
sys.path.insert(0, str(BENCH))

from score import OUTCOMES  # noqa: E402


def load(name: str) -> list[dict]:
    path = BENCH / "results" / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    rows = load(sys.argv[1])
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["condition"])].append(r)

    print(f"| model | condition | runs | optimal | {' | '.join(o for o in OUTCOMES[1:])} | used plugin | cost | median time |")
    print("|---|---|---|---|" + "---|" * (len(OUTCOMES) - 1) + "---|---|---|")
    for (model, condition), rs in sorted(groups.items()):
        c = Counter(r["outcome"] for r in rs)
        times = sorted(r["seconds"] for r in rs)
        cost = sum(r["cost_usd"] or 0 for r in rs)
        used = sum(r["plugin_used"] for r in rs)
        print(f"| {model} | {condition} | {len(rs)} | **{c['optimal']}** ({c['optimal'] / len(rs):.0%}) | "
              + " | ".join(str(c[o]) for o in OUTCOMES[1:])
              + f" | {used if condition == 'plugin' else '-'} | ${cost:.3f} | {times[len(times) // 2]:.0f}s |")

    print("\nAnswers that were not optimal:")
    for r in sorted(rows, key=lambda r: (r["condition"], r["problem"])):
        if r["outcome"] != "optimal":
            extra = f" (error: {r['error'][:80]})" if r.get("error") else ""
            print(f"- {r['condition']:9} {r['model']:6} {r['problem']:32} {r['outcome']:12} {r['detail']}{extra}")

    by_problem = defaultdict(dict)
    for r in rows:
        by_problem[r["problem"]][(r["model"], r["condition"])] = r["outcome"] == "optimal"
    hard = [p for p, res in by_problem.items() if not all(res.values())]
    print(f"\n{len(hard)} of {len(by_problem)} problems had at least one non-optimal answer")


if __name__ == "__main__":
    main()
