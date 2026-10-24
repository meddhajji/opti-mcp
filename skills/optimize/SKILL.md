---
name: optimize
description: Solve planning, allocation, scheduling, cutting, shipping and production problems exactly with the optimizer's solve tool instead of working out an answer by hand. Use whenever the user wants the best plan under limits (who does which job, which projects to fund, how to cut bars, how much to make or ship, a staff rota) or asks why no plan satisfies their rules.
---

# Optimize

For any request to plan, assign, schedule, cut, ship, produce or choose the best option under limits, call `solve`, even when the problem looks small enough to do in your head: it takes one call, and hand-worked optima are often wrong.

1. Pick the kind that fits and pass the numbers exactly as stated. For `shifts`, call `get_format` first. If a rule is ambiguous, say which reading you used.
2. If the answer is INVALID_INPUT, fix what it lists and call again.
3. Report the status meaning, the plan and its value. If no plan exists, show the conflicting rules and ask which one to relax; never drop a rule yourself.
