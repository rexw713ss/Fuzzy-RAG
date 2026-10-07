"""Tests for the A2 merge rule and rewrite fallback (spec v0.3, section 1.2).

Includes the section 8.1 acceptance test "A2 merge produces no duplicate passage IDs".
"""

import numpy as np
import pytest

from main_logic import actions as ac


def test_merge_produces_no_duplicate_ids():
    """Section 8.1 acceptance test, on many random overlapping pools."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        orig_ids = rng.choice(80, size=50, replace=False)
        rew_ids = rng.choice(80, size=50, replace=False)       # heavy overlap with orig
        orig = [(str(d), float(s)) for d, s in zip(orig_ids, rng.random(50))]
        rew = [(str(d), float(s)) for d, s in zip(rew_ids, rng.random(50))]
        merged = ac.merge_pools(orig, rew)
        ids = [d for d, _ in merged]
        assert len(ids) == len(set(ids)) == 30
        assert set(ids) <= {d for d, _ in orig} | {d for d, _ in rew}


def test_shared_passage_keeps_its_higher_score():
    orig = [("a", 0.9), ("b", 0.2), ("c", 0.0)]
    rew = [("b", 0.8), ("d", 0.1)]
    merged = dict(ac.merge_pools(orig, rew))
    # b keeps 0.8, not 0.2. Merged raw scores: a .9, b .8, d .1, c 0 -> min-max over [0, .9]
    assert merged == pytest.approx({"a": 1.0, "b": 0.8 / 0.9, "d": 0.1 / 0.9, "c": 0.0})


def test_normalization_spans_the_whole_pool_before_truncation():
    """Spec order: normalize the merged pool, then keep the top k. With 4 passages and k = 2 the
    kept scores are not stretched back to [0, 1]."""
    orig = [("a", 1.0), ("b", 0.5)]
    rew = [("c", 0.25), ("d", 0.0)]
    assert ac.merge_pools(orig, rew, k=2) == [("a", 1.0), ("b", 0.5)]


def test_normalization_does_not_change_order():
    orig = [("a", 0.7), ("b", 0.6), ("c", 0.55)]
    rew = [("d", 0.65), ("e", 0.5)]
    raw = sorted(orig + rew, key=lambda x: -x[1])
    assert [d for d, _ in ac.merge_pools(orig, rew)] == [d for d, _ in raw]


def test_ties_are_deterministic_whatever_the_input_order():
    orig = [("10", 0.5), ("9", 0.5), ("x", 0.1)]
    rew = [("2", 0.5)]
    first = ac.merge_pools(orig, rew)
    assert ac.merge_pools(list(reversed(orig)), rew) == first
    assert ac.merge_pools(rew, orig) == first
    assert [d for d, _ in first] == ["10", "2", "9", "x"]      # string order, as in fuse()


def test_one_empty_pool_and_both_empty():
    orig = [("a", 0.8), ("b", 0.4)]
    assert ac.merge_pools(orig, []) == [("a", 1.0), ("b", 0.0)]
    assert ac.merge_pools([], []) == []


def test_constant_scores_map_to_one():
    """Same rule as signals._minmax: a pool with no spread maps to 1.0, not a division by zero."""
    assert ac.merge_pools([("a", 0.3)], [("b", 0.3)]) == [("a", 1.0), ("b", 1.0)]


def test_non_finite_scores_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        ac.merge_pools([("a", float("nan"))], [("b", 0.2)])


@pytest.mark.parametrize("rewrite", [None, "", "   ", "Who wrote Hamlet?",
                                     "  who   wrote hamlet?  ", "WHO WROTE HAMLET?"])
def test_unusable_rewrites_fall_back_to_a1(rewrite):
    assert not ac.rewrite_usable("Who wrote Hamlet?", rewrite)


@pytest.mark.parametrize("rewrite", ["Which playwright wrote Hamlet?", "Who wrote Hamlet"])
def test_real_rewrites_are_usable(rewrite):
    """Any change beyond whitespace and case counts, even dropping the question mark."""
    assert ac.rewrite_usable("Who wrote Hamlet?", rewrite)
