"""Tests for the routing signals (spec v0.2, section 2).

The four S2 cases in section 2.3 ("Required unit tests") are implemented verbatim.
Everything else is an anchor on a hand-computable value, or pins down one of the
choices the spec does not make (see CLAUDE.md, "Decisions taken in signals.py").
"""

import numpy as np
import pytest

from main_logic import signals as sg


# --- 2.1 fusion ------------------------------------------------------------

def test_fuse_equal_weight_and_missing_docs():
    # bm25 min-max: a=1.0, b=0.5, c=0.0 ; dense min-max: b=1.0, d=0.0
    # fused: b=0.75, a=0.5, c=0.0, d=0.0 ; the c/d tie breaks on doc id.
    out = sg.fuse({"a": 10, "b": 5, "c": 0}, {"b": 1, "d": 0})
    assert [d for d, _ in out] == ["b", "a", "c", "d"]
    assert out[0][1] == pytest.approx(0.75)
    assert out[1][1] == pytest.approx(0.5)


def test_minmax_constant_scores_map_to_one():
    """Decision 1: an all-equal retriever list normalizes to all 1.0, not 0.0."""
    assert sg._minmax({"a": 3.0, "b": 3.0}) == {"a": 1.0, "b": 1.0}


# --- 2.2 S1 margin uncertainty --------------------------------------------

def test_s1_hand_computed():
    # s1=1.0, s2=0.9, s10=0.0  ->  1 - 0.1/1.0 = 0.9
    scores = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0]
    assert sg.s1_margin(scores) == pytest.approx(0.9, abs=1e-6)


def test_s1_extremes():
    tied = [1.0, 1.0] + [0.5] * 7 + [0.0]          # no separation -> maximal
    runaway = [1.0] + [0.0] * 9                     # rank 1 alone -> minimal
    assert sg.s1_margin(tied) == pytest.approx(1.0, abs=1e-6)
    assert sg.s1_margin(runaway) == pytest.approx(0.0, abs=1e-6)


# --- 2.3 S2 rank disagreement (spec's four required cases) ----------------

def test_s2_identical_lists_is_zero():
    a = list(range(10))
    assert sg.s2_disagreement(a, list(a)) == pytest.approx(0.0, abs=1e-12)


def test_s2_disjoint_lists_is_one():
    assert sg.s2_disagreement(range(10), range(100, 110)) == pytest.approx(1.0)


def test_s2_deep_swap_matters_less_than_shallow_swap():
    base = list(range(10))
    deep = base[:8] + [base[9], base[8]]            # swap ranks 9 and 10
    shallow = [base[1], base[0]] + base[2:]         # swap ranks 1 and 2
    assert sg.s2_disagreement(base, deep) < sg.s2_disagreement(base, shallow)


def test_s2_in_unit_interval_on_random_lists():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = rng.permutation(15)[:10].tolist()
        b = rng.permutation(15)[:10].tolist()
        s2 = sg.s2_disagreement(a, b)
        assert np.isfinite(s2) and 0.0 <= s2 <= 1.0


# --- 2.4 S3 evidence dispersion -------------------------------------------

def test_s3_identical_and_orthogonal_embeddings():
    same = np.ones((10, 4))
    assert sg.s3_dispersion(same) == pytest.approx(0.0, abs=1e-6)
    assert sg.s3_dispersion(np.eye(10)) == pytest.approx(1.0, abs=1e-6)


def test_s3_can_exceed_one():
    """Decision 3: S3 raw lives in [0, 2]; p5/p95 rescaling handles the range.

    Five copies of +e and five of -e: 20 pairs at cos +1, 25 at cos -1,
    mean cos = -1/9, so S3 = 1 + 1/9.
    """
    e = np.zeros((10, 2))
    e[:5, 0], e[5:, 0] = 1.0, -1.0
    assert sg.s3_dispersion(e) == pytest.approx(1.0 + 1 / 9, abs=1e-6)


# --- 2.5 S4 score entropy -------------------------------------------------

def test_s4_uniform_is_one_and_peaked_is_near_zero():
    assert sg.s4_entropy([0.5] * 10) == pytest.approx(1.0, abs=1e-6)
    assert sg.s4_entropy([1.0] + [0.0] * 9) < 0.01   # tau = 0.1 is sharp


# --- short lists ----------------------------------------------------------

@pytest.mark.parametrize(
    "call",
    [
        lambda: sg.s1_margin([1.0] * 9),
        lambda: sg.s2_disagreement(range(9), range(9)),
        lambda: sg.s3_dispersion(np.ones((9, 4))),
        lambda: sg.s4_entropy([1.0] * 9),
    ],
)
def test_signals_reject_short_lists(call):
    """Decision 2: fewer than k=10 items raises rather than silently computing."""
    with pytest.raises(ValueError):
        call()


# --- 2.6 rescaling --------------------------------------------------------

def test_apply_scaler_clips_to_unit_interval():
    scaler = sg.fit_scaler(np.arange(101.0))        # p5 = 5, p95 = 95
    scaled, clipped = sg.apply_scaler([5.0, 50.0, 95.0, -10.0, 200.0], scaler)
    assert scaled.tolist() == pytest.approx([0.0, 0.5, 1.0, 0.0, 1.0], abs=1e-6)
    # Only -10 and 200 fall strictly outside [p5, p95]; 5 and 95 sit exactly on the ends.
    assert clipped == pytest.approx(0.4)


def test_apply_scaler_endpoints_are_symmetric():
    """D1: a value exactly at p5 and one exactly at p95 are both not clipped."""
    scaler = sg.fit_scaler(np.arange(101.0))
    assert sg.apply_scaler([5.0], scaler)[1] == 0.0
    assert sg.apply_scaler([95.0], scaler)[1] == 0.0
    assert sg.apply_scaler([4.999], scaler)[1] == 1.0
    assert sg.apply_scaler([95.001], scaler)[1] == 1.0


# --- end to end -----------------------------------------------------------

def test_compute_signals_returns_all_four_finite():
    rng = np.random.default_rng(1)
    bm25 = {f"d{i}": float(rng.random()) for i in range(50)}
    dense = {f"d{i}": float(rng.random()) for i in range(25, 75)}
    fused = sg.fuse(bm25, dense)
    top = [d for d, _ in fused[:10]]
    bm25_top = sorted(bm25, key=bm25.get, reverse=True)[:10]
    dense_top = sorted(dense, key=dense.get, reverse=True)[:10]
    emb = rng.normal(size=(10, 8))
    out = sg.compute_signals(fused, bm25_top, dense_top, emb)
    assert set(out) == {"S1", "S2", "S3", "S4"}
    assert all(np.isfinite(v) for v in out.values())
    assert len(top) == 10
