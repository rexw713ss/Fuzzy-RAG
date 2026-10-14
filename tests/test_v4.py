"""Tests for the v0.4 redesign (REDESIGN_2026-10-15): three signals (S1 dropped) and two actions
(A0 merged into A1; A1 answer now, A2 escalate) with one threshold. Hand-built tables, as in
the v0.3 tests, so every expected number can be checked by hand."""

import numpy as np
import pytest

from main_logic import baselines as bl
from main_logic import calibration as cb
from main_logic import evaluation as ev
from main_logic import fuzzy as fz
from main_logic import learned as lr
from main_logic import monotonicity as mo
from main_logic import routing as rt

V4 = rt.ACTIONS_V4
SIG3 = ("S2", "S3", "S4")


def table(f1_rows, cost=(2.0, 6.0), clusters=None):
    """Two-action table: per-instance F1 pairs (A1, A2); the same costs everywhere."""
    n = len(f1_rows)
    clusters = clusters or [f"c{i // 3}" for i in range(n)]
    variants = [f"v{sum(c == x for x in clusters[:i])}" for i, c in enumerate(clusters)]
    return rt.OutcomeTable(list(zip(clusters, variants)), f1_rows, [list(cost)] * n, actions=V4)


# --- routing -------------------------------------------------------------------------------

def test_assign_action_with_one_threshold():
    acts = rt.assign_action([0.1, 0.5, 0.9], 0.5, actions=V4)
    assert acts.tolist() == ["A1", "A2", "A2"]
    assert rt.assign_action(0.49, 0.5, actions=V4) == "A1"


@pytest.mark.parametrize("cut", [(), (0.3, 0.6), (np.nan,)])
def test_two_actions_need_one_finite_threshold(cut):
    with pytest.raises(ValueError):
        rt.assign_action([0.5], *cut, actions=V4)


def test_theta_grid_runs_from_always_a2_to_always_a1():
    g = rt.theta_grid()
    assert len(g) == 101 and g[0] == (0.0,) and g[-1] == (1.0,)
    assert rt.default_grid(V4) == g and len(rt.default_grid(rt.ACTIONS)) == 2600


def test_outcome_table_from_rows_with_two_actions():
    rows = [{"cluster_id": "c", "variant_id": v, "action": a, "F1": f, "latency_total": c}
            for v in (0, 1) for a, f, c in (("A1", 0.5, 2.0), ("A2", 1.0, 6.0))]
    t = rt.OutcomeTable.from_rows(rows, actions=V4)
    assert t.actions == V4 and t.f1.shape == (2, 2)
    with pytest.raises(ValueError, match="unknown action"):
        rt.OutcomeTable.from_rows(rows + [{**rows[0], "action": "A0"}], actions=V4)


def test_select_threshold_escalates_the_instances_that_need_it_within_budget():
    # Instances 2 and 3 need A2; the score ranks them highest. Budget: half escalated.
    t = table([[1.0, 1.0], [1.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    scores = [0.1, 0.2, 0.7, 0.8]
    sel = rt.select_thresholds(t, scores, budget=4.0)
    assert sel["counts"] == {"A1": 2, "A2": 2} and sel["quality"] == 1.0
    assert sel["thresholds"] == (sel["theta"],) and 0.2 < sel["theta"] <= 0.7
    assert sel["theta"] == pytest.approx(0.21)          # the lowest theta of the tie
    # A tighter budget allows one escalation only: the highest score.
    tight = rt.select_thresholds(t, scores, budget=3.0)
    assert tight["counts"] == {"A1": 3, "A2": 1} and tight["quality"] == 0.75


def test_pareto_frontier_two_actions_is_monotone():
    t = table([[1.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    front = rt.pareto_frontier(t, [0.1, 0.6, 0.9])
    qs = [p["quality"] for p in front]
    assert qs == sorted(qs) and front[-1]["quality"] == 1.0 and "theta" in front[0]


# --- baselines -----------------------------------------------------------------------------

def test_b1_lattice_and_b2_cuts_for_three_signals_two_actions():
    g = bl.weight_grid(dim=3)
    assert len(g) == 15 and all(abs(sum(w) - 1) < 1e-12 for w in g)
    assert bl.cut_grid(3, 2) == [(c,) for c in range(8)]
    assert len(bl.cut_grid()) == 45                      # v0.3 unchanged


def test_baselines_fit_and_route_on_three_signals():
    # Easy instances: low signals, A1 enough. Hard: high signals, only A2 works.
    t = table([[1.0, 1.0], [1.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    sig = dict(zip(t.instances, [(0.1, 0.1, 0.1), (0.2, 0.1, 0.2), (0.9, 0.8, 0.9), (0.8, 0.9, 0.9)]))
    for fit in (bl.fit_b1(t, sig, budget=4.0), bl.fit_b2(t, sig, budget=4.0),
                bl.fit_b3(t, sig, budget=4.0), bl.fit_b4(t, sig, [0.229, 0.2, 0.25], budget=4.0)):
        assert fit["quality"] == 1.0, fit["baseline"]
        assert bl.route(fit, t, sig).tolist() == ["A1", "A1", "A2", "A2"], fit["baseline"]
    assert len(bl.fit_b1(t, sig, budget=4.0)["weights"]) == 3
    assert bl.fit_b4(t, sig, 0.2, budget=4.0)["signals"] == SIG3
    assert "c" in bl.fit_b2(t, sig, budget=4.0)


def test_signal_matrix_checks_the_number_of_signals():
    t = table([[1.0, 1.0]])
    with pytest.raises(ValueError, match="need 3 signals"):
        bl.signal_matrix(t, {t.instances[0]: (0.1, 0.2, 0.3, 0.4)}, 3)


# --- evaluation ----------------------------------------------------------------------------

def test_oracle_two_actions():
    t = table([[0.50, 0.60], [0.60, 0.61], [0.0, 0.0]])
    assert ev.oracle_actions(t).tolist() == [1, 0, 0]
    sh = ev.oracle_shares(t)
    assert sh["a1_suffices"] == pytest.approx(2 / 3) and sh["a2_required"] == pytest.approx(1 / 3)
    assert sh["all_identical"] == pytest.approx(1 / 3)


def test_flip_rates_two_actions():
    # One cluster: v0 -> A1, v1 -> A2, v2 -> A2. Pairs (0,1), (0,2) flip; (1,2) does not.
    # v1, v2 escalated but A2 buys nothing on them (F1 equal): both flips are harmful.
    t = table([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    r = ev.flip_rates(t, ["A1", "A2", "A2"])
    assert r["rfr"] == pytest.approx(2 / 3) and r["hrfr"] == pytest.approx(2 / 3)


# --- calibration, query-only router, monotonicity ------------------------------------------

def test_calibrate_three_signals():
    c = np.array([[0.1, 0.2, 0.3], [0.5, 0.2, 0.3], [0.1, 0.6, 0.3]])
    out = cb.calibrate([c], signals=SIG3)
    assert set(out["signals"]) == set(SIG3) and len(out["widths"]) == 3
    assert out["widths"][2] == pytest.approx(0.20)       # S4 never moves: floor


def test_query_only_scores_two_classes_is_p_a2():
    qs = [f"who is person {i}" for i in range(10)] + [f"which is older, A{i} or B{i}" for i in range(10)]
    model = lr.fit(qs, [0] * 10 + [1] * 10, n_classes=2)
    p = lr.probabilities(model, qs)
    assert np.allclose(lr.scores(model, qs), p[:, 1])
    assert lr.scores(model, qs)[:10].max() < lr.scores(model, qs)[10:].min()


def test_monotonicity_three_signals():
    s = mo.grid_scores(4, signals=SIG3)
    assert s.shape == (5, 5, 5)
    rep = mo.report(s, thresholds=[0.5], signals=SIG3, theta=0.5)
    assert set(rep["per_signal"]) == set(SIG3) and "downgrades" in rep["selected"]
    with pytest.raises(ValueError, match="3-d grid"):
        mo.violations(np.zeros((3, 3)), n_signals=3)
