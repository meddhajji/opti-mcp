import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))

from groq_agent import MAX_COMPLETION_TOKENS, MIN_COMPLETION_TOKENS, TPM_LIMIT, completion_budget, overshoot  # noqa: E402


def test_small_prompts_get_room_but_stay_under_the_minute_limit():
    budget = completion_budget({"messages": [{"role": "user", "content": "short question"}]})
    assert budget <= MAX_COMPLETION_TOKENS and budget < TPM_LIMIT


def test_large_prompts_shrink_the_cap_to_the_floor():
    huge = {"messages": [{"role": "user", "content": "x" * 40_000}]}
    assert completion_budget(huge) == MIN_COMPLETION_TOKENS


def test_overshoot_is_read_from_groqs_message():
    msg = "Request too large ... (TPM): Limit 8000, Requested 11506, please reduce your message size"
    assert overshoot(Exception(msg)) == 3506
    assert overshoot(Exception("something else")) is None


import groq_agent  # noqa: E402
from groq_agent import try_again_seconds  # noqa: E402


def test_try_again_durations_are_parsed():
    assert try_again_seconds(Exception("Please try again in 17.28s.")) == 17.28
    assert try_again_seconds(Exception("Please try again in 7m30.5s.")) == 450.5
    assert try_again_seconds(Exception("Please try again in 1h2m3s.")) == 3723
    assert try_again_seconds(Exception("Please try again in 539ms.")) == 0.539  # once read as 539 minutes
    assert try_again_seconds(Exception("Please try again in 2m0.5s.")) == 120.5
    assert try_again_seconds(Exception("no hint")) is None


class _RateLimit(Exception):
    pass


def _client(errors, monkeypatch):
    """A client failing with the given rate-limit messages, then succeeding."""
    import groq

    monkeypatch.setattr(groq, "RateLimitError", _RateLimit)
    queue = list(errors)

    def create(**kwargs):
        if queue:
            raise _RateLimit(queue.pop(0))
        return "ok"

    return type("C", (), {"chat": type("Ch", (), {"completions": type("Co", (), {"create": staticmethod(create)})})})


def test_short_daily_waits_are_waited_out(monkeypatch):
    slept = []
    monkeypatch.setattr(groq_agent.time, "sleep", slept.append)
    daily = "Rate limit reached on tokens per day (TPD): Limit 200000, Used 199957. Please try again in 1m30s."
    assert groq_agent.chat(_client([daily, daily], monkeypatch), model="m", messages=[]) == "ok"
    assert slept == [92.0, 92.0]


def test_long_daily_waits_stop_the_run(monkeypatch):
    monkeypatch.setattr(groq_agent.time, "sleep", lambda s: None)
    daily = "tokens per day (TPD): Limit 200000. Please try again in 5h10m0s."
    try:
        groq_agent.chat(_client([daily], monkeypatch), model="m", messages=[])
        raise AssertionError("expected DailyLimitReached")
    except groq_agent.DailyLimitReached:
        pass


class _Dropped(Exception):
    pass


def test_dropped_connections_are_retried(monkeypatch):
    import groq

    monkeypatch.setattr(groq, "APIConnectionError", _Dropped)
    monkeypatch.setattr(groq_agent.time, "sleep", lambda s: None)
    failures = [_Dropped("Connection error."), _Dropped("Connection error.")]

    def create(**kwargs):
        if failures:
            raise failures.pop(0)
        return "ok"

    client = type("C", (), {"chat": type("Ch", (), {"completions": type("Co", (), {"create": staticmethod(create)})})})
    assert groq_agent.chat(client, model="m", messages=[]) == "ok"
