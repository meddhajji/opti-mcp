import json
from pathlib import Path

import pytest

from optimizer.scheduling import ScheduleRequest, schedule_shifts

SHOP = json.loads((Path(__file__).parent.parent / "examples" / "small_shop.json").read_text())


@pytest.fixture(scope="module")
def result():
    return schedule_shifts(ScheduleRequest.model_validate(SHOP))


def test_shop_schedule_is_optimal_and_verified(result):
    assert result["status"] == "OPTIMAL"
    assert result["verification"]["valid"]


def test_no_shift_beyond_what_is_needed(result):
    # 7 mornings x 1, 5 weekday evenings x 2, Fri and Sat evenings x 3
    assert result["summary"]["total_shifts_assigned"] == 7 + 10 + 6


def test_fairness_gap_is_the_smallest_possible(result):
    # Omar is capped at 3; the other 20 shifts split 5-5-5-5, so the gap cannot be below 2
    assert result["summary"]["fairness_gap"] == 2
    assert result["per_person"]["Omar"]["shifts"] == 3


def test_every_rule_holds_in_the_returned_rota(result):
    schedule, shifts = result["schedule"], SHOP["shifts"]
    days = SHOP["days"]
    assert all("Sara" not in schedule["Wed"][s] for s in shifts)
    assert all("Omar" not in schedule[d][s] for d in ("Mon", "Tue") for s in shifts)
    for d, nxt in zip(days, days[1:]):  # rest between evening and next morning
        assert not set(schedule[d]["evening"]) & set(schedule[nxt]["morning"])
    for name in ("Karim", "Sara", "Youssef", "Imane", "Omar"):  # at most 4 days in a row
        worked = [any(name in schedule[d][s] for s in shifts) for d in days]
        assert not any(all(worked[i:i + 5]) for i in range(len(days) - 4))
    assert result["summary"]["preferences_respected"] == "4 of 4"


def test_impossible_friday_explains_the_exact_conflict():
    req = json.loads(json.dumps(SHOP))
    for p in req["staff"]:
        if p["name"] in ("Sara", "Omar"):
            p["unavailable"] = p.get("unavailable", []) + [{"day": "Fri"}]
        if p["name"] == "Karim":
            p["unavailable"] = [{"day": "Fri", "shift": "evening"}]
    r = schedule_shifts(ScheduleRequest.model_validate(req))
    assert r["status"] == "INFEASIBLE"
    assert {c["rule"] for c in r["conflict"]["conflict"]} == {
        "need: at least 3 on Fri evening",
        "Karim unavailable Fri evening",
        "Sara unavailable Fri evening",
        "Omar unavailable Fri evening",
    }


def test_unknown_day_is_rejected():
    req = json.loads(json.dumps(SHOP))
    req["staff"][0]["unavailable"] = [{"day": "Funday"}]
    with pytest.raises(ValueError, match="unknown day"):
        ScheduleRequest.model_validate(req)
