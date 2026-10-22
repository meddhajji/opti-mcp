"""Score an agent's final answer to a benchmark problem.

The agent is asked to end with a line
    FINAL ANSWER: {"<answer key>": <number>, ...}
using the problem's answer keys. Each answer gets exactly one verdict:

  optimal       the objective equals the proven optimum and, when the problem asks for
                decisions, they respect every rule and really achieve it (an alternative
                optimal plan counts: decisions are checked, not compared to one plan)
  suboptimal    a valid plan that achieves less than the optimum
  infeasible    the plan breaks a rule, or gives fractions where whole numbers are needed
  inconsistent  the stated objective is not what the stated plan achieves
  impossible    the claimed objective beats the proven optimum (objective-only problems)
  incomplete    an answer key is missing or not a number
  no_answer     no parsable FINAL ANSWER

A plan is checked by fixing the agent's numbers in the reference model and asking the
solver whether the variables the question does not ask about (stock levels, cutting
patterns...) can be completed; the best completion is the plan's true objective.
"""
import json
import re

from optimizer.explain import explain_infeasible
from optimizer.model import Model
from optimizer.engine import solve_model

FINAL = re.compile(r"FINAL ANSWER\s*:?\s*(?:```(?:json)?\s*)?(\{.*?\})", re.IGNORECASE | re.DOTALL)
OUTCOMES = ["optimal", "suboptimal", "infeasible", "inconsistent", "impossible", "incomplete", "no_answer"]


def parse_final_answer(text: str) -> dict | None:
    """The last FINAL ANSWER JSON object in the text, or None."""
    for match in reversed(FINAL.findall(text or "")):
        try:
            data = json.loads(match)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def to_number(value) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[\s$,]|dollars?|DM|MAD", "", value, flags=re.IGNORECASE)
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(b)) + 0.005


def better(a: float, b: float, sense: str) -> bool:
    return (a > b if sense == "maximize" else a < b) and not close(a, b)


def score(problem: dict, answer: dict | None) -> dict:
    if answer is None:
        return {"outcome": "no_answer", "detail": "no parsable FINAL ANSWER"}
    numbers = {k: to_number(answer.get(k)) for k in problem["answer_keys"]}
    missing = [k for k, v in numbers.items() if v is None]
    if missing:
        return {"outcome": "incomplete", "detail": f"missing or not a number: {missing}"}

    optimum, sense = problem["optimum"], problem["objective_sense"]
    claimed = numbers[problem["objective_key"]]
    keymap = problem["reference"]["key_to_variable"]
    decisions = {keymap[k]: v for k, v in numbers.items() if keymap[k] != "objective"}

    if not decisions:  # objective-only question
        if close(claimed, optimum):
            return {"outcome": "optimal", "detail": f"objective {claimed:g} is the proven optimum"}
        if better(claimed, optimum, sense):
            return {"outcome": "impossible", "detail": f"claims {claimed:g}, but the proven optimum is {optimum:g}"}
        return {"outcome": "suboptimal", "detail": f"claims {claimed:g}, the optimum is {optimum:g}"}

    fractional = {v: x for v, x in decisions.items() if not float(x).is_integer()}
    if fractional:
        return {"outcome": "infeasible", "detail": f"whole numbers required, got {fractional}"}

    reference = problem["reference"]["model"]
    fixed = dict(reference, constraints=reference["constraints"] + [
        {"name": f"agent: {v} = {int(x)}", "terms": {v: 1}, "sense": "==", "rhs": int(x)} for v, x in decisions.items()
    ])
    model = Model.model_validate(fixed)
    result = solve_model(model, time_limit_s=20, workers=4)
    if result["status"] == "INFEASIBLE":
        conflict = explain_infeasible(model, time_limit_s=20)
        broken = [c["rule"] for c in conflict.get("conflict", []) if not c["rule"].startswith("agent:")]
        return {"outcome": "infeasible", "detail": f"the plan breaks: {broken or 'variable bounds'}"}
    if result["status"] != "OPTIMAL":
        return {"outcome": "incomplete", "detail": f"could not check the plan ({result['status']})"}

    achieved = result["objective"] / problem["reference"]["objective_divisor"]
    if not close(claimed, achieved):
        return {"outcome": "inconsistent", "detail": f"states {claimed:g}, but this plan achieves {achieved:g}"}
    if close(achieved, optimum):
        return {"outcome": "optimal", "detail": f"valid plan achieving the optimum {optimum:g}"}
    return {"outcome": "suboptimal", "detail": f"valid plan achieving {achieved:g}, the optimum is {optimum:g}"}
