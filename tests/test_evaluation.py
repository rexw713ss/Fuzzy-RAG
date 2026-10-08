"""Tests for evaluation (spec v0.3, section 6). Tables are hand-built so every number can be
checked by hand."""

import numpy as np
import pytest

from main_logic import evaluation as ev
from main_logic.routing import OutcomeTable


def table(f1_rows, clusters=None):
    """OutcomeTable from per-instance F1 triples (A0, A1, A2); costs 1/2/5.
    clusters: cluster id per row (default: every 3 rows form a cluster)."""
    n = len(f1_rows)
    clusters = clusters or [f"c{i // 3}" for i in range(n)]
    variants = [f"v{sum(c == x for x in clusters[:i])}" for i, c in enumerate(clusters)]
    return OutcomeTable(list(zip(clusters, variants)), f1_rows, [[1.0, 2.0, 5.0]] * n)


# --- 6.1 oracle --------------------------------------------------------------------------


def test_oracle_picks_cheapest_within_delta():
    t = table([[0.50, 0.51, 0.60],     # only A2 within 0.02 of 0.60
               [0.60, 0.61, 0.60],     # all within 0.02 of 0.61 -> A0
               [0.40, 0.70, 0.71]])    # A1 within 0.02 of 0.71 -> A1
    assert ev.oracle_actions(t).tolist() == [2, 0, 1]


def test_oracle_gap_of_exactly_delta_counts_as_within():
    """0.50 vs best 0.52: exactly 0.02 apart; float rounding must not exclude A0."""
    assert ev.oracle_actions(table([[0.50, 0.52, 0.52]])).tolist() == [0]


def test_oracle_shares():
    t = table([[0.3, 0.3, 0.3],      # identical -> A0
               [0.5, 0.5, 0.5],      # identical -> A0
               [0.2, 0.9, 0.9],      # A1
               [0.1, 0.1, 0.8]])     # A2
    s = ev.oracle_shares(t)
    assert s == pytest.approx({"a0_suffices": 0.5, "a1_required": 0.25,
                               "a2_required": 0.25, "all_identical": 0.5})
    assert s["a0_suffices"] + s["a1_required"] + s["a2_required"] == pytest.approx(1.0)


# --- 6.2 flip rates ----------------------------------------------------------------------


def test_rfr_counts_all_three_pairs():
    t = table([[0.5, 0.5, 0.5]] * 3)
    # actions A0, A0, A1: pairs (0,1) same, (0,2) and (1,2) differ -> 2/3
    assert ev.flip_rates(t, ["A0", "A0", "A1"])["rfr"] == pytest.approx(2 / 3)


def test_same_action_everywhere_has_no_flips():
    t = table([[0.5, 0.6, 0.7]] * 6)
    r = ev.flip_rates(t, ["A1"] * 6)
    assert (r["rfr"], r["hrfr"], r["n_pairs"], r["n_clusters"]) == (0.0, 0.0, 6, 2)


def test_harmful_a_sibling_action_would_have_been_better():
    """Phrasing 0 got A0 (F1 0.5) but A1 would have given it 0.9 -> harmful by (a)."""
    t = table([[0.5, 0.9, 0.9],
               [0.5, 0.9, 0.9],
               [0.5, 0.9, 0.9]])
    _, flipped, harmful, _ = ev.pair_flags(t, ["A0", "A1", "A1"])
    assert flipped.tolist() == [True, True, False]
    assert harmful.tolist() == [True, True, False]


def test_harmful_b_paid_more_for_nothing():
    """Phrasing 1 got A2 but gains only 0.01 over A0 on its own query -> harmful by (b), even
    though A0 would not have helped phrasing 0 (no harm by (a))."""
    t = table([[0.60, 0.60, 0.60],
               [0.60, 0.60, 0.61],
               [0.60, 0.60, 0.61]])
    _, flipped, harmful, _ = ev.pair_flags(t, ["A0", "A2", "A2"])
    assert flipped.tolist() == [True, True, False]
    assert harmful.tolist() == [True, True, False]


def test_a_justified_flip_is_not_harmful():
    """Phrasing 1 needs A2 (gain 0.4); phrasing 0 does fine with A0 and gains nothing from A2.
    Different actions, but each is right for its own query."""
    t = table([[0.80, 0.80, 0.80],
               [0.40, 0.40, 0.80],
               [0.80, 0.80, 0.80]])
    r = ev.flip_rates(t, ["A0", "A2", "A0"])
    assert r["rfr"] == pytest.approx(2 / 3)
    assert r["hrfr"] == 0.0


def test_hrfr_never_exceeds_rfr():
    rng = np.random.default_rng(0)
    for _ in range(50):
        t = table(rng.random((30, 3)).round(2).tolist())
        r = ev.flip_rates(t, rng.integers(0, 3, 30))
        assert r["hrfr"] <= r["rfr"]


def test_incomplete_clusters_are_skipped_and_counted():
    t = table([[0.5, 0.5, 0.5]] * 5, clusters=["a", "a", "a", "b", "b"])
    r = ev.flip_rates(t, ["A0", "A0", "A1", "A0", "A2"])
    assert (r["n_clusters"], r["n_skipped"], r["n_pairs"]) == (1, 1, 3)


def test_no_complete_cluster_is_an_error():
    t = table([[0.5, 0.5, 0.5]] * 2, clusters=["a", "a"])
    with pytest.raises(ValueError, match="no complete clusters"):
        ev.flip_rates(t, ["A0", "A1"])


def test_bad_actions_rejected():
    t = table([[0.5, 0.5, 0.5]] * 3)
    with pytest.raises(ValueError):
        ev.flip_rates(t, ["A0", "A1"])
    with pytest.raises(ValueError):
        ev.flip_rates(t, ["A0", "A1", "A3"])


def test_within_cluster_measures():
    t = table([[0.2, 0.5, 0.9],
               [0.4, 0.5, 0.9],
               [0.6, 0.5, 0.9]])
    w = ev.within_cluster(t, ["A0", "A0", "A0"], [0.3, 0.4, 0.5])
    assert w["score_sd"] == pytest.approx(np.std([0.3, 0.4, 0.5]))
    assert w["f1_sd"] == pytest.approx(np.std([0.2, 0.4, 0.6]))
    assert w["worst_drop"] == pytest.approx(0.4)
    w2 = ev.within_cluster(t, ["A2", "A2", "A2"], [0.3, 0.4, 0.5])
    assert w2["f1_sd"] == pytest.approx(0.0) and w2["worst_drop"] == pytest.approx(0.0)


# --- 6.3 bootstrap -----------------------------------------------------------------------


def test_bootstrap_resamples_whole_clusters():
    """Two clusters, one all 0 and one all 1: any resample mean is 0, 0.5 or 1, never anything
    else, which is only true if clusters are kept whole."""
    values = [0, 0, 0, 1, 1, 1]
    groups = ["a", "a", "a", "b", "b", "b"]
    r = ev.bootstrap_ci(values, groups)
    assert r["mean"] == 0.5
    assert r["lo"] in (0.0, 0.5) and r["hi"] in (0.5, 1.0)


def test_bootstrap_is_reproducible_and_contains_the_mean():
    rng = np.random.default_rng(1)
    values = rng.random(300)
    groups = np.repeat(np.arange(100), 3)
    r = ev.bootstrap_ci(values, groups)
    assert r == ev.bootstrap_ci(values, groups)
    assert r["lo"] <= r["mean"] <= r["hi"]
    assert ev.bootstrap_ci(values, groups, seed=7) != r


def test_bootstrap_of_a_constant_has_zero_width():
    r = ev.bootstrap_ci([0.4] * 9, np.repeat([1, 2, 3], 3))
    assert r == pytest.approx({"mean": 0.4, "lo": 0.4, "hi": 0.4})


def test_paired_bootstrap():
    rng = np.random.default_rng(2)
    base = rng.random(300)
    groups = np.repeat(np.arange(100), 3)
    same = ev.paired_bootstrap(base, base, groups)
    assert same["diff"] == 0.0 and not same["significant"]
    better = ev.paired_bootstrap(base + 0.1, base, groups)
    assert better["diff"] == pytest.approx(0.1)
    assert better["lo"] == pytest.approx(0.1) and better["significant"]
    noise = ev.paired_bootstrap(base + rng.normal(0, 0.2, 300), base, groups)
    assert noise["lo"] < noise["diff"] < noise["hi"]


def test_bootstrap_input_checks():
    with pytest.raises(ValueError):
        ev.bootstrap_ci([], [])
    with pytest.raises(ValueError):
        ev.paired_bootstrap([1, 2], [1], ["a", "a"])
