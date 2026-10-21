"""Benchmark agent for Gemini API models (Gemini Flash-Lite, Gemma 4).

Shared behaviour (conditions, tools, logging) lives in agent_common.py. Here only
the Gemini API's specifics. The skill goes in as a system instruction, except for
Gemma models, where it is placed at the top of the user message because Gemma on
this API does not reliably accept system instructions.

The API key is read from GEMINI_API_KEY, or from .env at the repo root.
"""
import re
import time
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from agent_common import MAX_TURNS, CallLog, DailyLimitReached, condition_tools, env_key, skill_text

MAX_RETRIES = 8


def is_daily_limit(error: Exception) -> bool:
    return "PerDay" in str(error)


def retry_delay_seconds(error: Exception) -> float:
    """From "retryDelay': '23s'" or "retry in 23.4s"; 30 s when the error does not say."""
    m = re.search(r"retryDelay'?:\s*'(\d+(?:\.\d+)?)s", str(error)) or re.search(r"retry in (\d+(?:\.\d+)?)s", str(error))
    return float(m.group(1)) if m else 30.0


def generate(client, **kwargs):
    """One generation: waits out per-minute limits and overloads, stops on daily limits."""
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(**kwargs)
        except genai_errors.APIError as e:
            if e.code == 429 and is_daily_limit(e):
                raise DailyLimitReached(str(e)[:300]) from e
            if e.code in (429, 500, 503) and attempt < MAX_RETRIES - 1:
                time.sleep(min(retry_delay_seconds(e) + 1, 120))
                continue
            raise
    raise RuntimeError("unreachable")


def answer_text(content) -> str:
    return "".join(p.text for p in (content.parts or []) if p.text and not getattr(p, "thought", False))


async def run(prompt: str, model: str, condition: str, workdir: Path, client=None) -> dict:
    """Returns {text, tools, turns, tokens, events, calls} in the shape run.py records."""
    client = client or genai.Client(api_key=env_key("GEMINI_API_KEY"))
    system = skill_text() if condition == "plugin" else None
    if system and "gemma" in model:
        prompt, system = f"{system}\n\n---\n\n{prompt}", None
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    events, tokens = [], {"prompt": 0, "completion": 0}

    async with condition_tools(condition, workdir) as (specs, call_tool):
        log = CallLog(call_tool)
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=s["name"], description=s["description"], parameters_json_schema=s["parameters"])
                for s in specs])] if specs else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        text, turns = "", MAX_TURNS
        for turn in range(1, MAX_TURNS + 1):
            response = generate(client, model=model, contents=contents, config=config)
            usage = response.usage_metadata
            if usage:
                tokens["prompt"] += usage.prompt_token_count or 0
                tokens["completion"] += (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
            candidate = response.candidates[0] if response.candidates else None
            if candidate is None or candidate.content is None:
                reason = getattr(candidate, "finish_reason", None) or getattr(response, "prompt_feedback", None)
                events.append(f"turn {turn}: empty reply ({reason})")
                if candidate is not None and candidate.finish_reason == types.FinishReason.MALFORMED_FUNCTION_CALL:
                    contents.append(types.Content(role="user", parts=[types.Part(
                        text="Your last tool call was malformed and could not be run. Please try again.")]))
                    continue
                text, turns = "", turn
                break
            if candidate.finish_reason == types.FinishReason.MAX_TOKENS:
                events.append(f"turn {turn}: reply cut off by the output cap")
            contents.append(candidate.content)  # kept whole: Gemini needs its thought signatures back
            calls = [p.function_call for p in candidate.content.parts or [] if p.function_call]
            if not calls:
                text, turns = answer_text(candidate.content), turn
                break
            results = []
            for call in calls:
                result = await log.execute(call.name, dict(call.args or {}))
                results.append(types.Part.from_function_response(name=call.name, response={"result": result}))
            contents.append(types.Content(role="user", parts=results))
    return {"text": text, "tools": log.names, "turns": turns, "tokens": tokens, "events": events, "calls": log.calls}
