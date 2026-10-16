"""Verify resampling estimands against literal duplicated-query calculations."""

from dataclasses import replace

import numpy as np
import pytest

from experiment_ver2.data import Split, Dataset
from experiment_ver2.policies import quota_actions
from experiment_ver2.robustness.statistics import (
    cluster_counts, reranked_counts, interval, _draw_statistics, subset_diagnostics,
)


def fixture():
    n = 6
    split = Split(X=np.zeros((n, 4)), f1=np.array([[0, 1], [0, 1], [0, 1], [1, 0], [0, .5], [.5, 0.]]),
                  cost=np.ones((n, 2)), em=np.zeros((n, 2)),
                  keys=(("d", "a", 0), ("d", "a", 1), ("d", "a", 2), ("d", "b", 0), ("d", "c", 0), ("d", "c", 1)),
                  questions=tuple("abcdef"), actions=("A1", "A2"))
    return Dataset("d", split, split, split, np.full(4, .2), actions=split.actions)


@pytest.mark.parametrize("scores", [[1, .5, .5, 0, -1, .1], [.1] * 6, [-1] * 6])
@pytest.mark.parametrize("quota", [0, .25, .5, 1])
def test_vectorized_rerank_matches_literal_batch(scores, quota):
    questions = tuple("abcdef")
    mult = np.array([[1, 2, 3, 0, 1, 2], [0, 0, 3, 2, 0, 1]])
    observed = reranked_counts(scores, questions, mult, quota)
    for b, weights in enumerate(mult):
        rows = np.repeat(np.arange(6), weights)
        chosen = quota_actions(np.asarray(scores)[rows], tuple(questions[i] for i in rows), quota, base=0, escalation=1)
        expected = np.bincount(rows[chosen == 1], minlength=6)
        np.testing.assert_array_equal(observed[b], expected)


def test_fallback_and_invalid_multiplicities():
    assert not reranked_counts([1], ["x"], np.array([[5]]), fallback=True).any()
    with pytest.raises(ValueError):
        reranked_counts([1], ["x"], np.array([[1.5]]))
    with pytest.raises(ValueError):
        reranked_counts([float("nan")], ["x"], np.array([[1]]))
    with pytest.raises(ValueError, match="unique"):
        reranked_counts([1, 1], ["same", "same"], np.array([[1, 1]]))


def test_clusters_drawn_as_units():
    counts, inv = cluster_counts(["a", "a", "a", "b"], 100, 7)
    assert (counts.sum(axis=1) == 2).all()
    mult = counts[:, inv]
    np.testing.assert_array_equal(mult[:, 0], mult[:, 2])
    assert set(mult.sum(axis=1)) == {2, 4, 6}


def test_three_estimands_against_literal_formulas():
    data = fixture()
    scores = {m: np.array([.1, .2, .3, .4, .5, .6]) for m in ("fuzzy-calibrated", "ridge", "fuzzy-fixed")}
    fits = {m: {"quota": .5, "candidate": "fit"} for m in scores}
    actions = {m: quota_actions(s, data.test.questions, .5, base=0, escalation=1) for m, s in scores.items()}
    observed = _draw_statistics(data.test, actions, scores, fits, 30, 7)
    counts, inv = cluster_counts(data.test.groups, 30, 7)
    for b, multiplicities in enumerate(counts[:, inv]):
        rows = np.repeat(np.arange(6), multiplicities)
        f1 = data.test.f1[rows]
        assigned = actions["fuzzy-calibrated"][rows]
        actual = f1[np.arange(len(rows)), assigned].mean()
        fixed_ref = .5 * f1[:, 0].mean() + .5 * f1[:, 1].mean()
        p = assigned.mean()
        matched_ref = (1 - p) * f1[:, 0].mean() + p * f1[:, 1].mean()
        assert observed["frozen_fixed_fraction"]["random"][b] == pytest.approx(actual - fixed_ref)
        assert observed["frozen_matched_fraction"]["random"][b] == pytest.approx(actual - matched_ref)
        new_actions = quota_actions(scores["fuzzy-calibrated"][rows], tuple(data.test.questions[i] for i in rows), .5, base=0, escalation=1)
        new_p = new_actions.mean()
        new_actual = f1[np.arange(len(rows)), new_actions].mean()
        assert observed["reranked_batch"]["random"][b] == pytest.approx(new_actual - ((1 - new_p) * f1[:, 0].mean() + new_p * f1[:, 1].mean()))
    assert not np.allclose(observed["frozen_fixed_fraction"]["random"], observed["frozen_matched_fraction"]["random"])
    assert (observed["reranked_batch"]["ridge"] == 0).all()


def test_bonferroni_is_wider_without_invented_pvalue():
    result = interval(np.arange(1000) / 1000, .5, 8)
    assert result["bonferroni_percentile"][0] < result["percentile_95"][0]
    assert result["bonferroni_percentile"][1] > result["percentile_95"][1]
    assert result["bonferroni_marginal_confidence"] == .99375


def test_missing_variants_do_not_silently_change_population():
    data = fixture()
    actions = {m: np.zeros(6, dtype=int) for m in ("fuzzy-calibrated", "ridge", "fuzzy-fixed")}
    result = subset_diagnostics(data, actions)["subsets"]
    assert result["original_only"]["queries"] == 3
    assert result["complete_clusters_only"]["clusters"] == 1
    assert result["incomplete_clusters_only"]["clusters"] == 2
    block = result["all"]["methods"]["fuzzy-calibrated"]
    assert block["F1_query_weighted"] == pytest.approx(1.5 / 6)
    assert block["F1_cluster_equal_weighted"] == pytest.approx((0 + 1 + .25) / 3)
    assert block["gain_vs_cluster_equal_fraction_random"] == 0


def test_evidence_json_bytes_are_platform_portable(tmp_path):
    from experiment_ver2.robustness.run import write_json
    path = tmp_path / "evidence.json"
    write_json(path, {"dataset": "真實資料", "scores": [1, 2]})
    assert b"\r" not in path.read_bytes()
    assert path.read_bytes().endswith(b"\n")
