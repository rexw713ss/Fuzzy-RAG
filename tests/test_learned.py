"""Tests for the query-only learned router (spec v0.3, section 5)."""

import numpy as np

from main_logic import learned as lr


def test_features():
    f = lr.features("Which is older, Big Ben or the Eiffel Tower in 1900?")
    assert len(f) == 12
    assert f[0] == 11                       # words
    assert f[1] == 4                        # Big, Ben, Eiffel, Tower
    assert f[2] == 1.0                      # has a number
    assert f[3 + lr.WH.index("which")] == 1.0 and f[3:11].sum() == 1.0
    assert f[11] == 2                       # "older", "or"
    assert lr.features("Name the capital")[10] == 1.0          # no question word: "other"


def test_fit_learns_a_separable_rule_and_is_deterministic():
    qs = [f"who is person {i}" for i in range(20)] + [f"which is older, A{i} or B{i}" for i in range(20)]
    y = [0] * 20 + [2] * 20
    m = lr.fit(qs, y)
    assert m == lr.fit(qs, y)
    p = lr.probabilities(m, qs)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p.argmax(axis=1) == np.array(y)).all()
    s = lr.scores(m, qs)
    assert ((s >= 0) & (s <= 1)).all() and s[:20].max() < s[20:].min()


def test_constant_feature_does_not_break_standardization():
    m = lr.fit(["who a", "who b"], [0, 1])
    assert np.isfinite(lr.scores(m, ["who c"])).all()
