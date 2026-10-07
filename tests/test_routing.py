"""Tests for the action thresholds and the cost constraint (spec v0.2, section 3.4).

Covers the section 8.1 acceptance test "theta1 < theta2 enforced in the grid search",
the half-open interval convention, and the user-chosen tie-break and Pareto sweep.
The real outcomes.jsonl does not exist until stage 3, so the outcome tables here are
hand-built: small enough that every expected mean is computable by hand.
"""

import numpy as np
import pytest

from main_logic import routing as rt


def make_rows(f1_by_action, cost_by_action, n=4):
    """n instances, each with the same per-action F1 and cost. One cluster, n variants."""
    return [
        {"cluster_id": "c0", "variant_id": f"v{i}", "action": a,
         "F1": f1_by_action[a], "latency_total": cost_by_action[a]}
        for i in range(n) for a in rt.ACTIONS
    ]


def simple_table(n=4):
    """A0 cheap and weak, A1 mid, A2 dear and strong."""
    return rt.OutcomeTable.from_rows(make_rows(
        {"A0": 0.30, "A1": 0.50, "A2": 0.60},
        {"A0": 1.0, "A1": 2.0, "A2": 5.0}, n=n))


# --- thresholds and the score -> action map (section 3.4) --------------------------


def test_theta1_must_be_below_theta2():
    """Section 8.1 acceptance test, at the single-pair level."""
    with pytest.raises(ValueError, match="theta1 < theta2"):
        rt.check_thresholds(0.5, 0.5)
    with pytest.raises(ValueError, match="theta1 < theta2"):
        rt.check_thresholds(0.6, 0.4)
    assert rt.check_thresholds(0.4, 0.6) == (0.4, 0.6)


def test_every_grid_pair_satisfies_theta1_below_theta2():
    """Section 8.1 acceptance test, over the whole grid."""
    grid = rt.threshold_grid()
    assert grid, "grid must not be empty"
    assert all(t1 < t2 for t1, t2 in grid)


def test_grid_has_960_pairs_on_the_exact_lattice():
    """31 x 31 candidates less the one (0.50, 0.50) that fails theta1 < theta2."""
    grid = rt.threshold_grid()
    assert len(grid) == 31 * 31 - 1 == 960
    t1s = sorted({t1 for t1, _ in grid})
    t2s = sorted({t2 for _, t2 in grid})
    assert t1s[0] == 0.20 and t1s[-1] == 0.50
    assert t2s[0] == 0.50 and t2s[-1] == 0.80
    # No float drift: every value equals the literal it should be.
    assert 0.27 in t1s and 0.73 in t2s
    assert (0.50, 0.50) not in grid


def test_intervals_are_half_open():
    """esc < t1 -> A0; t1 <= esc < t2 -> A1; esc >= t2 -> A2."""
    t1, t2 = 0.30, 0.70
    assert rt.assign_action(0.2999, t1, t2) == "A0"
    assert rt.assign_action(0.30, t1, t2) == "A1"      # exactly theta1 escalates
    assert rt.assign_action(0.6999, t1, t2) == "A1"
    assert rt.assign_action(0.70, t1, t2) == "A2"      # exactly theta2 escalates
    assert rt.assign_action(1.0, t1, t2) == "A2"


def test_assign_action_vectorizes_consistently():
    scores = np.array([0.1, 0.35, 0.5, 0.75, 0.9])
    got = rt.assign_action(scores, 0.3, 0.7)
    assert list(got) == ["A0", "A1", "A1", "A2", "A2"]
    assert [rt.assign_action(float(s), 0.3, 0.7) for s in scores] == list(got)


def test_non_finite_scores_rejected():
    with pytest.raises(ValueError, match="finite"):
        rt.assign_action([0.5, np.nan], 0.3, 0.7)


# --- the outcome table (section 7.4) ----------------------------------------------


def test_table_from_rows_orders_instances_and_actions():
    table = simple_table(n=3)
    assert len(table) == 3
    assert table.instances == [("c0", "v0"), ("c0", "v1"), ("c0", "v2")]
    assert table.f1[0].tolist() == [0.30, 0.50, 0.60]
    assert table.cost[0].tolist() == [1.0, 2.0, 5.0]


def test_table_rejects_instance_missing_an_action():
    rows = make_rows({"A0": 0.3, "A1": 0.5, "A2": 0.6},
                     {"A0": 1.0, "A1": 2.0, "A2": 5.0}, n=2)
    rows = [r for r in rows if not (r["variant_id"] == "v1" and r["action"] == "A2")]
    with pytest.raises(ValueError, match="no cached row for A2"):
        rt.OutcomeTable.from_rows(rows)


def test_table_rejects_duplicate_rows():
    rows = make_rows({"A0": 0.3, "A1": 0.5, "A2": 0.6},
                     {"A0": 1.0, "A1": 2.0, "A2": 5.0}, n=1)
    with pytest.raises(ValueError, match="duplicate cached row"):
        rt.OutcomeTable.from_rows(rows + [rows[0]])


def test_cost_can_sum_component_latencies():
    """Section 3.4 secondary metrics / component latencies from section 7.4."""
    rows = [{"cluster_id": "c0", "variant_id": "v0", "action": a, "F1": 0.5,
             "retrieval_latency": 1.0, "generation_latency": 2.0} for a in rt.ACTIONS]
    table = rt.OutcomeTable.from_rows(
        rows, cost=("retrieval_latency", "generation_latency"))
    assert table.cost[0].tolist() == [3.0, 3.0, 3.0]
    assert table.cost_key == "retrieval_latency+generation_latency"


def test_align_rejects_a_missing_score():
    table = simple_table(n=2)
    with pytest.raises(ValueError, match="no escalation score"):
        table.align({("c0", "v0"): 0.5})


def test_align_is_keyed_not_positional():
    table = simple_table(n=2)
    got = table.align({("c0", "v1"): 0.8, ("c0", "v0"): 0.2})
    assert got.tolist() == [0.2, 0.8]


# --- policy value, the shared primitive -------------------------------------------


def test_policy_value_and_always_policies():
    table = simple_table(n=4)
    for action, expected in [("A0", (0.30, 1.0)), ("A1", (0.50, 2.0)),
                             ("A2", (0.60, 5.0))]:
        assert rt.always_policy(table, action) == pytest.approx(expected)
        assert rt.policy_value(table, [action] * 4) == pytest.approx(expected)
    # Two of each: means are the averages.
    mixed = rt.policy_value(table, ["A0", "A0", "A2", "A2"])
    assert mixed == pytest.approx((0.45, 3.0))


def test_policy_value_rejects_wrong_length_and_bad_labels():
    table = simple_table(n=3)
    with pytest.raises(ValueError, match="one action per instance"):
        rt.policy_value(table, ["A0", "A1"])
    with pytest.raises(ValueError, match="unknown action"):
        rt.policy_value(table, ["A0", "A1", "A3"])


def test_action_counts_keeps_zeros():
    """Section 3.4 removed the usage floor, so an unused action must stay visible."""
    counts = rt.action_counts(np.array(["A0", "A0", "A1"]))
    assert counts == {"A0": 2, "A1": 1, "A2": 0}


# --- threshold selection (section 3.4) --------------------------------------------


def test_selection_respects_the_budget():
    """Scores chosen so the default always-A1 budget is actually reachable here."""
    table = simple_table(n=4)
    scores = {k: v for k, v in zip(table.instances, [0.15, 0.25, 0.35, 0.85])}
    out = rt.select_thresholds(table, scores)
    # Budget is the always-A1 mean cost.
    assert out["budget"] == pytest.approx(2.0)
    assert out["cost"] <= out["budget"] + 1e-12
    assert out["n_feasible"] >= 1
    assert out["n_grid"] == 960
    assert out["quality_key"] == "F1" and out["cost_key"] == "latency_total"


def test_selection_maximizes_quality_under_the_budget():
    """A2 is strictly better and affordable for one of four instances."""
    table = rt.OutcomeTable.from_rows(make_rows(
        {"A0": 0.20, "A1": 0.20, "A2": 0.90},
        {"A0": 1.0, "A1": 2.0, "A2": 5.0}, n=4))
    scores = {k: v for k, v in zip(table.instances, [0.10, 0.20, 0.30, 0.95])}
    out = rt.select_thresholds(table, scores)
    # One A2 (cost 5) + three A0 (cost 1) = mean 2.0, exactly the always-A1 budget.
    assert out["counts"]["A2"] == 1
    assert out["cost"] == pytest.approx(2.0)
    assert out["quality"] == pytest.approx((0.20 * 3 + 0.90) / 4)


def test_ties_break_to_the_cheaper_policy():
    """A1 and A2 answer identically, so the tie-break must not pay for A2."""
    table = rt.OutcomeTable.from_rows(make_rows(
        {"A0": 0.10, "A1": 0.80, "A2": 0.80},
        {"A0": 1.0, "A1": 2.0, "A2": 5.0}, n=4))
    scores = {k: v for k, v in zip(table.instances, [0.55, 0.60, 0.65, 0.70])}
    out = rt.select_thresholds(table, scores, budget=5.0)
    assert out["quality"] == pytest.approx(0.80)
    assert out["counts"] == {"A0": 0, "A1": 4, "A2": 0}
    assert out["cost"] == pytest.approx(2.0)


def test_ties_then_break_to_the_lowest_thresholds():
    """All actions identical: every pair ties on quality and cost, so lowest theta wins."""
    table = rt.OutcomeTable.from_rows(make_rows(
        {a: 0.5 for a in rt.ACTIONS}, {a: 1.0 for a in rt.ACTIONS}, n=3))
    scores = {k: 0.5 for k in table.instances}
    out = rt.select_thresholds(table, scores)
    assert (out["theta1"], out["theta2"]) == (0.20, 0.50)


def test_selection_is_deterministic():
    table = simple_table(n=5)
    scores = {k: v for k, v in zip(table.instances, [0.2, 0.4, 0.5, 0.6, 0.8])}
    first = rt.select_thresholds(table, scores, budget=10.0)
    assert rt.select_thresholds(table, scores, budget=10.0) == first


def test_unused_actions_are_reported():
    """Never-selected actions are a section 3.4 finding, not an error."""
    table = simple_table(n=3)
    scores = {k: 0.1 for k in table.instances}        # everything below any theta1
    out = rt.select_thresholds(table, scores, budget=10.0)
    assert out["counts"]["A0"] == 3
    assert out["unused_actions"] == ["A1", "A2"]


def test_infeasible_budget_raises_with_the_cheapest_cost():
    table = simple_table(n=3)
    scores = {k: 0.5 for k in table.instances}
    with pytest.raises(ValueError, match="cheapest reachable policy"):
        rt.select_thresholds(table, scores, budget=0.5)


def test_always_a1_budget_is_generally_unreachable():
    """The budget is a bound, not an attainable point.

    Fuzzy scores span [0.1333, 0.8667] while theta1 >= 0.20 and theta2 <= 0.80, so no
    pair sends every instance to A1 once scores sit at both ends of that span.
    """
    table = simple_table(n=2)
    scores = {k: v for k, v in zip(table.instances, [0.1333, 0.8667])}
    for t1, t2 in rt.threshold_grid():
        assert set(rt.assign_action(table.align(scores), t1, t2)) != {"A1"}


def test_default_always_a1_budget_can_itself_be_infeasible():
    """A finding about section 3.4, not a bug.

    A score at or above 0.80 is forced to A2 (theta2 <= 0.80) and one at or above 0.50 can
    never be A0 (theta1 <= 0.50). When A2 is much dearer than A1, the forced A2 instances
    alone can push the mean past the always-A1 budget, leaving the constrained problem
    empty. The Pareto sweep is therefore the reporting that always has an answer.
    """
    table = simple_table(n=5)
    scores = {k: v for k, v in zip(table.instances, [0.2, 0.4, 0.5, 0.6, 0.8])}
    assert rt.always_policy(table, "A1")[1] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="no threshold pair meets the cost budget"):
        rt.select_thresholds(table, scores)
    # The sweep still yields a frontier over the costs that are reachable.
    frontier = rt.pareto_frontier(table, scores)
    assert frontier and min(p["budget"] for p in frontier) == pytest.approx(2.2)


# --- Pareto frontier (section 3.4) ------------------------------------------------


def test_pareto_quality_is_non_decreasing_in_budget():
    table = simple_table(n=5)
    scores = {k: v for k, v in zip(table.instances, [0.15, 0.35, 0.50, 0.65, 0.85])}
    frontier = rt.pareto_frontier(table, scores)
    assert len(frontier) >= 2
    budgets = [p["budget"] for p in frontier]
    qualities = [p["quality"] for p in frontier]
    assert budgets == sorted(budgets)
    assert all(b == pytest.approx(a) or b > a for a, b in zip(qualities, qualities[1:]))
    assert all(p["cost"] <= p["budget"] + 1e-12 for p in frontier)


def test_pareto_covers_every_achievable_cost_level():
    """Sweep points are the distinct achievable mean costs, with no binning."""
    table = simple_table(n=4)
    scores = {k: v for k, v in zip(table.instances, [0.15, 0.40, 0.60, 0.85])}
    grid = rt.threshold_grid()
    achievable = {rt.policy_value(table, rt.assign_action(table.align(scores), t1, t2))[1]
                  for t1, t2 in grid}
    frontier = rt.pareto_frontier(table, scores)
    assert {p["budget"] for p in frontier} == achievable


def test_pareto_top_matches_unconstrained_selection():
    """At the largest budget the frontier must agree with an unconstrained grid search."""
    table = simple_table(n=4)
    scores = {k: v for k, v in zip(table.instances, [0.15, 0.40, 0.60, 0.85])}
    frontier = rt.pareto_frontier(table, scores)
    top = frontier[-1]
    out = rt.select_thresholds(table, scores, budget=top["budget"])
    assert (out["theta1"], out["theta2"]) == (top["theta1"], top["theta2"])
    assert out["quality"] == pytest.approx(top["quality"])
