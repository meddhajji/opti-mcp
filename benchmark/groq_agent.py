"""Benchmark agent for models served by Groq (OpenAI-style chat with tools).

Shared behaviour (conditions, tools, logging) lives in agent_common.py. Here only
Groq's specifics: its free-tier limits and error shapes. Differences from Claude
Code, stated plainly: the skill is given up front as a system message (Claude Code
shows its description and loads it on demand), and the loop is this file's.

The API key is read from GROQ_API_KEY, or from .env at the repo root.
"""
import os
import re
import time
from pathlib import Path

import groq

from agent_common import (MAX_TURNS, CallLog, DailyLimitReached, condition_tools, deref,  # noqa: F401
                          env_key, run_python, skill_text)

# Reasoning models (gpt-oss) spend output tokens thinking before they answer; Groq's
# default cap (2,048) cut answers off mid-thought, which Claude Code runs never hit.
# Groq also counts the requested cap against the tokens-per-minute limit, so each
# request asks for whatever fits under that limit after its prompt.
MAX_COMPLETION_TOKENS = 8192
TPM_LIMIT = int(os.environ.get("GROQ_TPM_LIMIT", "8000"))  # free (on_demand) tier for gpt-oss-20b
MIN_COMPLETION_TOKENS = 256
# gpt-oss models think at length by default; under the free 8,000 tokens-per-minute limit
# the plugin condition (whose prompt carries ~4,000 tokens of tool definitions and skill)
# was left too little room to answer. "low" is applied to every condition alike, and is
# what the laptop tracker's assistant uses in production.
REASONING_EFFORT = os.environ.get("GROQ_REASONING_EFFORT", "low")


def api_key() -> str:
    return env_key("GROQ_API_KEY")


def is_daily_limit(error: Exception) -> bool:
    text = str(error)
    return "per day" in text or "(TPD)" in text or "(RPD)" in text


def retry_after_seconds(error: Exception) -> float:
    headers = getattr(getattr(error, "response", None), "headers", {}) or {}
    try:
        return float(headers.get("retry-after", 20))
    except ValueError:
        return 20.0


class RequestTooLarge(RuntimeError):
    """The prompt alone leaves no room for an answer under the per-minute token limit."""


def completion_budget(kwargs: dict) -> int:
    """Largest output cap that fits under the per-minute limit after the prompt (estimated)."""
    import json

    prompt_chars = len(json.dumps(kwargs.get("messages", []))) + len(json.dumps(kwargs.get("tools", [])))
    return max(MIN_COMPLETION_TOKENS, min(MAX_COMPLETION_TOKENS, TPM_LIMIT - int(prompt_chars / 3.2) - 300))


def overshoot(error: Exception) -> int | None:
    """From "Limit 8000, Requested 11506": by how many tokens the request was too large."""
    m = re.search(r"Limit (\d+), Requested (\d+)", str(error))
    return int(m.group(2)) - int(m.group(1)) if m else None


def chat(client, **kwargs):
    """One completion: sizes the output cap, waits out per-minute limits, stops on daily ones."""
    kwargs["max_completion_tokens"] = completion_budget(kwargs)
    for _ in range(8):
        try:
            return client.chat.completions.create(**kwargs)
        except groq.RateLimitError as e:
            if is_daily_limit(e):
                raise DailyLimitReached(str(e)[:300]) from e
            time.sleep(min(retry_after_seconds(e) + 1, 120))
        except groq.APIStatusError as e:
            excess = overshoot(e) if e.status_code == 413 else None
            if excess is None:
                raise
            smaller = kwargs["max_completion_tokens"] - excess - 200
            if smaller < MIN_COMPLETION_TOKENS:
                raise RequestTooLarge(str(e)[:300]) from e
            kwargs["max_completion_tokens"] = smaller
    raise RuntimeError("still rate-limited after 8 retries")


async def run(prompt: str, model: str, condition: str, workdir: Path, client=None) -> dict:
    """Returns {text, tools, turns, tokens, events, calls} in the shape run.py records."""
    client = client or groq.Groq(api_key=api_key(), max_retries=0)
    messages = [{"role": "user", "content": prompt}]
    if condition == "plugin":
        messages.insert(0, {"role": "system", "content": skill_text()})
    events, tokens = [], {"prompt": 0, "completion": 0}

    async with condition_tools(condition, workdir) as (specs, call_tool):
        log = CallLog(call_tool)
        tools = [{"type": "function", "function": s} for s in specs]
        text, turns = "", MAX_TURNS
        for turn in range(1, MAX_TURNS + 1):
            kwargs = {"model": model, "messages": messages}
            if "gpt-oss" in model:
                kwargs["reasoning_effort"] = REASONING_EFFORT
            if tools:
                kwargs |= {"tools": tools, "tool_choice": "auto"}
            try:
                response = chat(client, **kwargs)
            except RequestTooLarge:
                events.append(f"turn {turn}: conversation too long for the per-minute token limit")
                text, turns = "", turn
                break
            except groq.BadRequestError as e:  # e.g. a malformed tool call rejected by Groq
                events.append(f"turn {turn}: bad request: {str(e)[:200]}")
                messages.append({"role": "user", "content": f"Your last reply could not be processed ({str(e)[:300]}). "
                                                            "Please try again."})
                continue
            if response.usage:
                tokens["prompt"] += response.usage.prompt_tokens or 0
                tokens["completion"] += response.usage.completion_tokens or 0
            choice = response.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                events.append(f"turn {turn}: reply cut off by the output cap")
            message = choice.message
            calls = message.tool_calls or []
            messages.append({"role": "assistant", "content": message.content or "",
                             **({"tool_calls": [c.model_dump() for c in calls]} if calls else {})})
            if not calls:
                text, turns = message.content or "", turn
                break
            for call in calls:
                result = await log.execute(call.function.name, call.function.arguments or "{}")
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return {"text": text, "tools": log.names, "turns": turns, "tokens": tokens, "events": events, "calls": log.calls}
