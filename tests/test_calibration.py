"""Tests for jitter calibration (spec v0.3, section 7.3).

Includes the section 8.1 acceptance test "w <= 0.25 for every signal after calibration, with any
clipping flagged".
"""

import numpy as np
import pytest

from main_logic import calibration as cb
from main_logic import fuzzy as fz


def cluster(s1, others=(0.5, 0.5, 0.5)):
    """A 3-phrasing cluster where S1 takes the given values and S2-S4 are constant."""
    return np.array([[v, *others] for v in s1])


def test_hand_example_pairs():
    """S1 = 0.42, 0.47, 0.39 -> pairwise differences 0.05, 0.08, 0.03."""
    diffs, used, skipped = cb.pair_differences([cluster([0.42, 0.47, 0.39])])
    assert sorted(diffs[:, 0]) == pytest.approx([0.03, 0.05, 0.08])
    assert (diffs[:, 1:] == 0).all()
    assert (used, skipped) == (1, 0)


def test_small_jitter_stays_at_the_floor():
    out = cb.calibrate([cluster([0.42, 0.47, 0.39])])
    s1 = out["signals"]["S1"]
    assert s1["jitter"] == pytest.approx(0.065)          # p75 of 0.03, 0.05, 0.08
    assert s1["width"] == cb.W_FLOOR and s1["at_floor"] and not s1["capped"]
    assert out["all_at_floor"]


def test_mid_jitter_becomes_the_width():
    # Each cluster gives S1 diffs 0.22, 0.22, 0; the 75th percentile of the pooled six is 0.22.
    out = cb.calibrate([cluster([0.0, 0.22, 0.22]), cluster([0.0, 0.22, 0.22])])
    s1 = out["signals"]["S1"]
    assert s1["jitter"] == pytest.approx(0.22)
    assert s1["width"] == pytest.approx(0.22)
    assert not s1["at_floor"] and not s1["capped"]
    assert not out["all_at_floor"]


def test_large_jitter_is_capped_and_flagged():
    out = cb.calibrate([cluster([0.0, 0.5, 1.0])])
    s1 = out["signals"]["S1"]
    assert s1["jitter"] > cb.W_CAP
    assert s1["width"] == cb.W_CAP and s1["capped"]
    assert out["signals"]["S2"]["width"] == cb.W_FLOOR and not out["signals"]["S2"]["capped"]


def test_acceptance_widths_never_exceed_cap_and_clipping_is_flagged():
    """Section 8.1: for any data, every width <= 0.25, flagged exactly where jitter > 0.25,
    and the widths are accepted by the controller."""
    rng = np.random.default_rng(0)
    for spread in (0.02, 0.1, 0.3, 0.6, 1.0):
        clusters = [np.clip(0.5 + spread * (rng.random((3, 4)) - 0.5), 0, 1) for _ in range(30)]
        out = cb.calibrate(clusters)
        for s in out["signals"].values():
            assert cb.W_FLOOR <= s["width"] <= cb.W_CAP
            assert s["capped"] == (s["jitter"] > cb.W_CAP)
        fz.escalate([0.5] * 4, widths=out["widths"])       # must not raise


def test_differences_are_pooled_over_clusters():
    a = cluster([0.0, 0.1, 0.1])      # S1 diffs 0.1, 0.1, 0.0
    b = cluster([0.0, 0.3, 0.3])      # S1 diffs 0.3, 0.3, 0.0
    diffs, used, _ = cb.pair_differences([a, b])
    assert used == 2 and len(diffs) == 6
    assert cb.calibrate([a, b])["signals"]["S1"]["jitter"] == pytest.approx(
        np.percentile([0.1, 0.1, 0.0, 0.3, 0.3, 0.0], 75))


def test_median_sensitivity_version():
    out = cb.calibrate([cluster([0.0, 0.22, 0.22]), cluster([0.0, 0.24, 0.24])])
    s1 = out["signals"]["S1"]
    assert s1["jitter_median"] == pytest.approx(np.median([0.22, 0.22, 0, 0.24, 0.24, 0]))
    assert out["widths_median"][0] == pytest.approx(min(max(s1["jitter_median"], 0.20), 0.25))


def test_incomplete_clusters_are_skipped_and_counted():
    full = cluster([0.0, 0.22, 0.22])
    two = cluster([0.0, 1.0])                             # would dominate if it were used
    one = cluster([0.5])
    out = cb.calibrate([full, two, one])
    assert (out["n_clusters"], out["n_skipped"]) == (1, 2)
    assert out["signals"]["S1"]["jitter"] == pytest.approx(
        cb.calibrate([full])["signals"]["S1"]["jitter"])


def test_no_complete_cluster_is_an_error():
    with pytest.raises(ValueError, match="no complete clusters"):
        cb.calibrate([cluster([0.1, 0.2])])


@pytest.mark.parametrize("bad", [
    np.array([[0.5, 0.5, 0.5, 1.2]] * 3),          # not scaled
    np.array([[0.5, np.nan, 0.5, 0.5]] * 3),       # not finite
    np.array([[0.5, 0.5, 0.5]] * 3),               # 3 signals
    np.array([[0.5] * 4] * 4),                     # 4 phrasings
])
def test_bad_input_rejected(bad):
    with pytest.raises(ValueError):
        cb.calibrate([bad])
