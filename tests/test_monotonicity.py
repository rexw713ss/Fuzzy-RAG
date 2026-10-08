"""Tests for the output-monotonicity report (spec v0.3, section 4.2).

The coarse-grid counts are pinned to the spec's spacing-sensitivity table; the full 0.05 grid
(about a minute) is reproduced by scripts/monotonicity_report.py instead of here.
"""

import numpy as np
import pytest

from main_logic import fuzzy as fz
from main_logic import monotonicity as mo


@pytest.fixture(scope="module")
def grid10():
    return mo.grid_scores(10)


def test_coarsest_grid_matches_the_spec_table():
    """Section 4.2 table, spacing 0.2000: 4,320 pairs, 64 violations, max 0.0067."""
    r = mo.report(mo.grid_scores(5))
    assert (r["pairs"], r["violations"]) == (4320, 64)
    assert r["drop_max"] == pytest.approx(0.0067, abs=5e-5)


def test_tenth_grid_matches_the_spec_table(grid10):
    """Section 4.2 table, spacing 0.1000: 53,240 pairs, 1,584 violations, max 0.0067."""
    r = mo.report(grid10)
    assert (r["pairs"], r["violations"]) == (53240, 1584)
    assert r["drop_max"] == pytest.approx(0.0067, abs=5e-5)


def test_violations_are_equal_per_signal(grid10):
    """The rule base treats the four signals symmetrically, so their counts must match."""
    per = mo.report(grid10)["per_signal"]
    assert len(set(per.values())) == 1 and sum(per.values()) == 1584


@pytest.mark.parametrize("steps", [1, 2, 5, 20])
def test_pair_count_formula(steps):
    """4 signals x steps moves x (steps + 1)^3 positions of the other three."""
    n_pairs, _ = mo.violations(np.zeros((steps + 1,) * 4))
    assert n_pairs == 4 * steps * (steps + 1) ** 3


def test_worst_single_case_from_the_spec():
    """Section 4.2: raising S1 from 0.30 to 0.35 drops the score 0.612179 -> 0.574834."""
    before = fz.escalate([0.30, 0.40, 0.70, 0.75])[0]
    after = fz.escalate([0.35, 0.40, 0.70, 0.75])[0]
    assert before == pytest.approx(0.612179, abs=1e-6)
    assert after == pytest.approx(0.574834, abs=1e-6)


def synthetic():
    """A 2 x 2 x 2 x 2 grid, 0.9 everywhere except: the origin 0.62 and one step up in S3 0.58
    (the one planted drop; every other move from those two points rises to 0.9), and a 1e-12
    bump at (1, 1, 1, 0) so that raising S4 there drops by 1e-12, below the tolerance."""
    s = np.full((2,) * 4, 0.9)
    s[0, 0, 0, 0], s[0, 0, 1, 0] = 0.62, 0.58
    s[1, 1, 1, 0] = 0.9 + 1e-12
    return s


def test_planted_drop_is_found_and_noise_is_not():
    n_pairs, rec = mo.violations(synthetic())
    assert rec["drop"].size == 1
    assert rec["signal"].tolist() == [2]
    assert rec["before"][0] == pytest.approx(0.62) and rec["after"][0] == pytest.approx(0.58)


def test_downgrade_uses_half_open_intervals():
    """0.62 -> 0.58 downgrades at theta = 0.60 and 0.62 (score was >= theta, now below), not at
    0.58 (the lower score is still >= 0.58) or 0.63 (it was already below)."""
    _, rec = mo.violations(synthetic())
    assert [mo.downgrades(rec, t) for t in (0.58, 0.60, 0.62, 0.63)] == [0, 1, 1, 0]


def test_report_counts_straddles_and_the_selected_pair():
    r = mo.report(synthetic(), theta1=0.30, theta2=0.60)
    assert r["violations"] == 1
    assert r["straddle_any_threshold"] == 1
    assert r["per_threshold"][0.6] == 1 and r["per_threshold"][0.58] == 0
    assert r["thresholds_without_flips"] == 61 - 4          # 0.59, 0.60, 0.61, 0.62
    assert r["selected"] == {"theta1": 0.30, "theta2": 0.60, "downgrades": 1}
    assert mo.report(synthetic(), theta1=0.30, theta2=0.70)["selected"]["downgrades"] == 0


def test_monotone_grid_reports_zeros():
    r = mo.report(np.zeros((3,) * 4))
    assert r["violations"] == 0 and r["drop_max"] == 0.0 and r["straddle_any_threshold"] == 0


def test_threshold_values_are_the_61_grid_values():
    t = mo.threshold_values()
    assert len(t) == 61 and t[0] == 0.20 and t[-1] == 0.80


def test_ablations_can_be_measured_the_same_way():
    s = mo.grid_scores(2, defuzz="sugeno")
    assert s.shape == (3,) * 4
    assert s[0, 0, 0, 0] == pytest.approx(fz.SUGENO_Z["Low"])


@pytest.mark.parametrize("bad", [np.zeros((3, 3, 3)), np.zeros((3, 3, 3, 4))])
def test_grid_shape_is_checked(bad):
    with pytest.raises(ValueError, match="4-d grid"):
        mo.violations(bad)


def test_steps_must_be_a_positive_integer():
    with pytest.raises(ValueError, match="steps"):
        mo.grid_scores(2.5)
