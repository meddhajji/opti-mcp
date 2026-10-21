import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))

from generate import FAMILIES, build, fitting_patterns  # noqa: E402
from score import score  # noqa: E402


@pytest.mark.parametrize("family", FAMILIES)
def test_same_id_gives_the_same_problem(family):
    a, b = build(family, 2, 1, 7), build(family, 2, 1, 7)
    assert a["question"] == b["question"] and a["optimum"] == b["optimum"]


@pytest.mark.parametrize("family", FAMILIES)
def test_noise_keeps_the_numbers_and_adds_context(family):
    clean, noisy = build(family, 2, 0, 3), build(family, 2, 1, 3)
    assert clean["optimum"] == noisy["optimum"] and clean["reference"] == noisy["reference"]
    assert all(line in noisy["question"] for line in clean["question"].splitlines() if line.strip())
    assert len(noisy["question"]) > len(clean["question"])


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("level", [1, 2])
def test_small_instances_have_an_independent_check(family, level):
    problem = build(family, level, 0, 1)  # build() raises if the check disagrees with the solver
    assert not problem["independent_check"].startswith("none")


@pytest.mark.parametrize("family", FAMILIES)
def test_generated_problems_work_with_the_scorer(family):
    p = build(family, 3, 0, 2)
    assert score(p, p["expected"])["outcome"] == "optimal"
    worse = p["optimum"] - 1 if p["objective_sense"] == "maximize" else p["optimum"] + 1
    assert score(p, {p["objective_key"]: worse})["outcome"] == "suboptimal"


def test_cutting_patterns_are_exactly_those_that_fit():
    patterns = set(fitting_patterns([250, 180], 600))
    brute = {(a, b) for a in range(3) for b in range(4) if 250 * a + 180 * b <= 600}
    assert patterns == brute
