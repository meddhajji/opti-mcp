"""Benchmark agent for Gemini API models (Gemini Flash-Lite, Gemma 4).

Shared behaviour (conditions, tools, logging) lives in agent_common.py. Here only
the Gemini API's specifics. A condition's guidance goes in as a system instruction,
except for Gemma models, where it is placed at the top of the user message because
Gemma on this API does not reliably accept system instructions.

The API key is read from GEMINI_API_KEY, or from .env at the repo root.
"""
import re
import time
from pathlib import Path

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from agent_common import (MAX_TURNS, REMINDER, CallLog, DailyLimitReached, ProviderUnavailable, condition_tools,
                          env_key)

MAX_RETRIES = 8
CONNECTION_RETRIES = 5
# Without a timeout, a request sent as the connection dropped once hung for over an hour.
REQUEST_TIMEOUT_MS = 600_000


def is_daily_limit(error: Exception) -> bool:
    return "PerDay" in str(error)


def retry_delay_seconds(error: Exception) -> float:
    """From "retryDelay': '23s'" or "retry in 23.4s"; 30 s when the error does not say."""
    m = re.search(r"retryDelay'?:\s*'(\d+(?:\.\d+)?)s", str(error)) or re.search(r"retry in (\d+(?:\.\d+)?)s", str(error))
    return float(m.group(1)) if m else 30.0


def generate(client, **kwargs):
    """One generation: waits out per-minute limits and overloads, stops on daily limits and outages."""
    connection_retries = 0
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(**kwargs)
        except httpx.TransportError as e:  # no connection, or no reply within the timeout
            connection_retries += 1
            if connection_retries > CONNECTION_RETRIES:
                raise ProviderUnavailable(f"Gemini API unreachable: {type(e).__name__}") from e
            time.sleep(10 * connection_retries)
        except genai_errors.APIError as e:
            if e.code == 429 and is_daily_limit(e):
                raise DailyLimitReached(str(e)[:300]) from e
            if e.code in (429, 500, 503, 504) and attempt < MAX_RETRIES - 1:  # 504: the server gave up, retry
                time.sleep(min(retry_delay_seconds(e) + 1, 120))
                continue
            raise
    raise RuntimeError("unreachable")


def answer_text(content) -> str:
    return "".join(p.text for p in (content.parts or []) if p.text and not getattr(p, "thought", False))


async def run(prompt: str, model: str, condition: str, workdir: Path, client=None,
              answer_keys: list[str] | None = None) -> dict:
    """Returns {text, tools, turns, tokens, events, calls, answer} in the shape run.py records;
    answer is what submit_answer received (always None without answer_keys)."""
    client = client or genai.Client(api_key=env_key("GEMINI_API_KEY"),
                                    http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
    events, tokens = [], {"prompt": 0, "completion": 0, "cached": 0}

    async with condition_tools(condition, workdir, answer_keys) as (specs, call_tool, system):
        if system and "gemma" in model:
            prompt, system = f"{system}\n\n---\n\n{prompt}", None
        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        log = CallLog(call_tool, answer_keys)
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=s["name"], description=s["description"], parameters_json_schema=s["parameters"])
                for s in specs])] if specs else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        text, turns, reminded = "", MAX_TURNS, False
        for turn in range(1, MAX_TURNS + 1):
            response = generate(client, model=model, contents=contents, config=config)
            usage = response.usage_metadata
            if usage:
                tokens["prompt"] += usage.prompt_token_count or 0
                tokens["completion"] += (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
                tokens["cached"] += usage.cached_content_token_count or 0
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
                if answer_keys and not reminded:
                    reminded = True
                    events.append(f"turn {turn}: stopped without submit_answer, reminded")
                    contents.append(types.Content(role="user", parts=[types.Part(text=REMINDER)]))
                    continue
                text, turns = answer_text(candidate.content), turn
                break
            results = []
            for call in calls:
                result = await log.execute(call.name, dict(call.args or {}))
                results.append(types.Part.from_function_response(name=call.name, response={"result": result}))
            contents.append(types.Content(role="user", parts=results))
            if log.answer is not None:
                text, turns = answer_text(candidate.content), turn
                break
    return {"text": text, "tools": log.names, "turns": turns, "tokens": tokens, "events": events, "calls": log.calls,
            "answer": log.answer}
