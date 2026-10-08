"""Tests for the controller baselines B1-B4 (spec v0.3, section 5).

The real action-outcome rows do not exist until stage 3, so the tables here are hand-built:
three or four instances whose best routing can be worked out by hand.
"""

import numpy as np
import pytest

from main_logic import baselines as bl
from main_logic import fuzzy as fz
from main_logic import routing as rt

COST = [1.0, 2.0, 5.0]                  # A0 cheap, A1 mid, A2 dear


def table(f1_rows):
    """One instance per F1 row, cluster c0, variants v0, v1, ...; the same costs everywhere."""
    inst = [("c0", f"v{i}") for i in range(len(f1_rows))]
    return rt.OutcomeTable(inst, f1_rows, [COST] * len(f1_rows))


def sig(tab, rows):
    return {k: r for k, r in zip(tab.instances, rows)}


# Three instances: easy (only A0 needed), medium (A1 needed), hard (only A2 works).
EMH_F1 = [[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]


# --- grids -------------------------------------------------------------------------------------

def test_weight_grid_has_35_vectors_summing_to_one():
    g = bl.weight_grid()
    assert len(g) == 35 and len(set(g)) == 35
    assert all(abs(sum(w) - 1.0) < 1e-12 and min(w) >= 0 for w in g)
    assert (0.25, 0.25, 0.25, 0.25) in g
    for i in range(4):
        assert tuple(1.0 if j == i else 0.0 for j in range(4)) in g


def test_weight_grid_rejects_a_step_that_does_not_divide_one():
    with pytest.raises(ValueError, match="whole number"):
        bl.weight_grid(0.3)


def test_cut_grid_is_every_integer_pair_up_to_nine():
    g = bl.cut_grid()
    assert len(g) == 45
    assert all(0 <= c1 < c2 <= 9 for c1, c2 in g)


# --- B2 crisp labels ---------------------------------------------------------------------------

@pytest.mark.parametrize("x, level", [
    (0.0, 0), (0.3499, 0), (0.35, 1), (0.5, 1), (0.6499, 1), (0.65, 2), (1.0, 2),
])
def test_crisp_labels_are_half_open(x, level):
    assert bl.crisp_levels(x) == level


def test_crisp_labels_match_fuzzy_labels_outside_the_overlap():
    """Section 5: B2 is the w -> 0 limit. Away from the bands the fuzzy label is already crisp."""
    for x in (0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 1.0):
        assert int(np.argmax(fz.memberships(x, fz.W_DEFAULT))) == bl.crisp_levels(x)


def test_ordinal_sum_spans_0_to_8():
    s = np.array([[0.0] * 4, [0.5] * 4, [1.0] * 4, [0.1, 0.5, 0.9, 0.5]])
    assert bl.ordinal_sums(s).tolist() == [0, 4, 8, 4]


def test_b2_picks_the_cheapest_perfect_cut_points():
    """t = 0, 4, 8. Perfect routing is A0, A1, A2; every c1 in 1..4, c2 in 5..8 does it at the
    same cost, so the tie-break takes the lowest pair."""
    tab = table(EMH_F1)
    s = sig(tab, [[0.0] * 4, [0.5] * 4, [1.0] * 4])
    fit = bl.fit_b2(tab, s, budget=10.0)
    assert (fit["c1"], fit["c2"]) == (1, 5)
    assert fit["quality"] == 1.0
    assert fit["n_grid"] == 45
    assert bl.route(fit, tab, s).tolist() == ["A0", "A1", "A2"]


# --- B1 weighted sum ---------------------------------------------------------------------------

def test_weighted_scores_rejects_bad_weights():
    s = np.full((2, 4), 0.5)
    for w in ([0.5, 0.5, 0.5, 0.5], [1.5, -0.5, 0, 0], [1, 0, 0]):
        with pytest.raises(ValueError, match="weights"):
            bl.weighted_scores(s, w)


def test_b1_finds_the_one_informative_signal():
    """Only S3 orders the instances; the others are reversed. Perfect routing needs S3 to dominate."""
    tab = table(EMH_F1)
    s = sig(tab, [[0.9, 0.9, 0.1, 0.9], [0.5, 0.5, 0.6, 0.5], [0.1, 0.1, 0.95, 0.1]])
    fit = bl.fit_b1(tab, s, budget=10.0)
    assert fit["quality"] == 1.0
    assert fit["weights"] == (0.0, 0.0, 1.0, 0.0)
    assert fit["n_weights"] == 35
    assert bl.route(fit, tab, s).tolist() == ["A0", "A1", "A2"]


def test_b1_tie_across_weights_takes_the_first_in_lattice_order():
    """All four signals equal: every weight vector gives the same scores."""
    tab = table(EMH_F1)
    s = sig(tab, [[0.1] * 4, [0.6] * 4, [0.9] * 4])
    fit = bl.fit_b1(tab, s, budget=10.0)
    assert fit["weights"] == bl.weight_grid()[0] == (0.0, 0.0, 0.0, 1.0)


def test_b1_with_one_weight_vector_is_plain_threshold_selection():
    tab = table(EMH_F1)
    rows = [[0.2, 0.3, 0.1, 0.4], [0.5, 0.6, 0.4, 0.5], [0.9, 0.7, 0.8, 0.9]]
    w = (0.25, 0.25, 0.25, 0.25)
    fit = bl.fit_b1(tab, sig(tab, rows), budget=10.0, weights=[w])
    sel = rt.select_thresholds(tab, bl.weighted_scores(np.array(rows), w), 10.0)
    assert (fit["theta1"], fit["theta2"], fit["quality"]) == (sel["theta1"], sel["theta2"],
                                                              sel["quality"])


def test_b1_raises_when_no_weight_meets_the_budget():
    tab = table(EMH_F1)
    s = sig(tab, [[0.9] * 4] * 3)                      # every pair routes all to A2
    with pytest.raises(ValueError, match="cost budget"):
        bl.fit_b1(tab, s, budget=1.5)


# --- B3 / B4 fuzzy -----------------------------------------------------------------------------

def test_b3_is_select_thresholds_on_the_fuzzy_score():
    tab = table(EMH_F1)
    rows = [[0.1] * 4, [0.5] * 4, [0.95] * 4]
    fit = bl.fit_b3(tab, sig(tab, rows), budget=10.0)
    scores = [fz.escalate(r, 0.20)[0] for r in rows]
    sel = rt.select_thresholds(tab, scores, 10.0)
    assert (fit["theta1"], fit["theta2"]) == (sel["theta1"], sel["theta2"])
    assert fit["widths"] == (0.20,) * 4


def test_b4_at_the_floor_equals_b3():
    """Section 5 honest reporting: calibration that leaves every width at 0.20 changes nothing."""
    tab = table(EMH_F1)
    s = sig(tab, [[0.1, 0.2, 0.3, 0.2], [0.5, 0.4, 0.6, 0.5], [0.9, 0.8, 0.95, 0.7]])
    b3, b4 = bl.fit_b3(tab, s, budget=10.0), bl.fit_b4(tab, s, [0.20] * 4, budget=10.0)
    for key in ("theta1", "theta2", "quality", "cost"):
        assert b3[key] == b4[key]


def test_b4_uses_its_own_widths():
    tab = table(EMH_F1)
    s = sig(tab, [[0.1] * 4, [0.5] * 4, [0.95] * 4])
    fit = bl.fit_b4(tab, s, [0.20, 0.25, 0.22, 0.20], budget=10.0)
    assert fit["widths"] == (0.20, 0.25, 0.22, 0.20)


def test_b4_refuses_widths_above_the_cap():
    tab = table(EMH_F1)
    s = sig(tab, [[0.5] * 4] * 3)
    with pytest.raises(ValueError, match="w must be"):
        bl.fit_b4(tab, s, [0.30] * 4, budget=10.0)


# --- shared plumbing ---------------------------------------------------------------------------

def test_signals_are_keyed_by_instance_not_position():
    tab = table(EMH_F1)
    s = sig(tab, [[0.1] * 4, [0.5] * 4, [0.9] * 4])
    shuffled = dict(reversed(list(s.items())))
    assert np.array_equal(bl.signal_matrix(tab, s), bl.signal_matrix(tab, shuffled))


def test_missing_or_unscaled_signals_are_errors():
    tab = table(EMH_F1)
    s = sig(tab, [[0.1] * 4, [0.5] * 4, [0.9] * 4])
    with pytest.raises(ValueError, match="no signals"):
        bl.signal_matrix(tab, dict(list(s.items())[:2]))
    s[tab.instances[0]] = [1.2, 0.5, 0.5, 0.5]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        bl.signal_matrix(tab, s)


def test_route_applies_a_fit_to_another_split():
    """Thresholds fitted on one table are applied unchanged to another (validation -> test)."""
    val = table(EMH_F1)
    fit = bl.fit_b2(val, sig(val, [[0.0] * 4, [0.5] * 4, [1.0] * 4]), budget=10.0)
    test = table([[1, 1, 1], [1, 1, 1]])
    acts = bl.route(fit, test, sig(test, [[1.0] * 4, [0.0] * 4]))
    assert acts.tolist() == ["A2", "A0"]


def test_route_rejects_an_unknown_baseline():
    tab = table(EMH_F1)
    s = sig(tab, [[0.5] * 4] * 3)
    with pytest.raises(ValueError, match="unknown baseline"):
        bl.route({"baseline": "B9", "theta1": 0.3, "theta2": 0.6}, tab, s)
