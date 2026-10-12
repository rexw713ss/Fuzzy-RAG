"""Tests for the stage 1 checks (spec v0.3, section 8.2)."""

import numpy as np
import pytest

from main_logic import checks as ck


def clean_signal(**kw):
    return {"non_finite": 0, "most_common_share": 0.05, "clipped": 0.10, "auc_hard": 0.6, **kw}


def summary(nq=None, hq=None, corr=0.5, floor=False):
    def one(over):
        sig = {s: clean_signal() for s in ck.SIGNALS}
        sig.update(over or {})
        return {"signals": sig, "all_at_floor": floor,
                "correlations": {"S1-S4": {"spearman": corr, "pearson": corr}}}
    return {"nq": one(nq), "hotpotqa": one(hq)}


def test_most_common_share():
    assert ck.most_common_share([1, 1, 1, 2]) == 0.75
    assert ck.most_common_share([0.1, 0.1 + 1e-15, 0.3]) == pytest.approx(2 / 3)
    assert np.isnan(ck.most_common_share([]))


def test_auc():
    assert ck.auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert ck.auc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert ck.auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    assert np.isnan(ck.auc([1, 2], [1, 1]))


def test_correlations_are_spearman_and_pearson():
    x = np.arange(10.0)
    X = np.stack([x, x ** 3, -x, np.sin(x)], axis=1)
    c = ck.correlations(X)
    assert set(c) == {"S1-S2", "S1-S3", "S1-S4", "S2-S3", "S2-S4", "S3-S4"}
    assert c["S1-S2"]["spearman"] == pytest.approx(1.0)
    assert c["S1-S2"]["pearson"] < 1.0
    assert c["S1-S3"]["spearman"] == pytest.approx(-1.0)


def test_clean_pilot_has_no_flags():
    assert ck.flags(summary()) == []


@pytest.mark.parametrize("over, level, check", [
    ({"non_finite": 1}, "stop", "non-finite"),
    ({"most_common_share": 0.81}, "stop", "identical"),
    ({"clipped": 0.81}, "stop", "clipped"),
    ({"clipped": 0.21}, "revisit", "clipped"),
])
def test_single_signal_conditions(over, level, check):
    f = ck.flags(summary(nq={"S2": clean_signal(**over)}))
    assert [(x["level"], x["dataset"], x["check"]) for x in f] == [(level, "nq", check)]


def test_thresholds_are_strict():
    at = clean_signal(most_common_share=0.80, clipped=0.20)
    assert ck.flags(summary(nq={"S1": at})) == []
    assert ck.flags(summary(corr=0.90)) == []


def test_high_correlation_either_sign_stops():
    assert {x["check"] for x in ck.flags(summary(corr=0.91))} == {"correlation"}
    assert {x["check"] for x in ck.flags(summary(corr=-0.95))} == {"correlation"}


def test_all_widths_at_floor_is_reported():
    f = ck.flags(summary(floor=True))
    assert {(x["level"], x["check"]) for x in f} == {("explain", "calibration")}


def test_s3_opposite_directions():
    f = ck.flags(summary(nq={"S3": clean_signal(auc_hard=0.6)},
                         hq={"S3": clean_signal(auc_hard=0.4)}))
    assert [(x["level"], x["check"]) for x in f] == [("explain", "S3 direction")]
    same = summary(nq={"S3": clean_signal(auc_hard=0.6)}, hq={"S3": clean_signal(auc_hard=0.7)})
    assert ck.flags(same) == []
