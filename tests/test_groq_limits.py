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
