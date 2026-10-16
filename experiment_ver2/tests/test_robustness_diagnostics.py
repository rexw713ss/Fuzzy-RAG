"""Cluster-mass support, tied ranks, shrinkage and frozen-input diagnostics."""

from copy import deepcopy
import json

import numpy as np
import pytest

from experiment_ver2.data import Dataset, Split
from experiment_ver2.robustness import diagnostics as dg
from experiment_ver2.run import fit_candidates, freeze_policy, apply_policy
from experiment_ver2 import models


@pytest.fixture
def fixture():
    def split(name, values):
        X, f1, keys, questions = [], [], [], []
        for cluster, value in enumerate(values):
            for variant in range(3):
                X.append([value] * 4)
                f1.append([.5, .7 if value == 0 else .4])
                keys.append(("toy", f"{name}-{cluster}", variant))
                questions.append(f"{name}-{cluster} wording {variant}")
        return Split(np.array(X), np.array(f1), np.tile([1., 2.], (len(X), 1)),
                     np.zeros((len(X), 2)), tuple(keys), tuple(questions), actions=("A1", "A2"))

    data = Dataset("toy", split("train", [0, 1] * 90), split("val", [0, 1] * 4),
                   split("test", [0, .5, 1] * 3), np.full(4, .2), actions=("A1", "A2"))
    candidates = fit_candidates(data)
    fits, assignments = {}, {}
    for method, variants in candidates.items():
        scores = {}
        test_scores = {}
        for name, model in variants.items():
            q, _ = models.predict(model, data.val.X)
            scores[name] = q[:, 1] - q[:, 0]
            q, _ = models.predict(model, data.test.X)
            test_scores[name] = q[:, 1] - q[:, 0]
        fits[method] = freeze_policy(method, scores, data.val, .25)
        assignments[method] = apply_policy(method, fits[method], test_scores, data.test.questions, data.actions)
    return data, candidates, fits, assignments


def test_cluster_support_sums_180_and_ess_does_not_count_540_paraphrase_rows(fixture):
    report = dg.run_diagnostics(*fixture)
    support = report["rule_support"]["fuzzy-fixed"]
    assert support["n_train_clusters"] == 180
    assert support["sum_rule_support"] == pytest.approx(180)
    assert sum(row["cluster_equivalent_support"] for row in support["rules"]) == pytest.approx(180)
    assert support["rules"][0]["cluster_equivalent_support"] == pytest.approx(90)
    assert support["rules"][0]["cluster_mass_kish_ess"] == pytest.approx(90)
    assert support["rules"][0]["clusters_with_nonzero_membership"] == 90
    assert support["rules"][80]["cluster_mass_kish_ess"] == pytest.approx(90)
    assert support["unseen_rule_count"] == 79
    assert fixture[1]["fuzzy-fixed"]["prior=1.0"]["rule_effective_rows"][0] == pytest.approx(270)


def test_cluster_mass_ess_is_unchanged_when_phrasing_rows_are_duplicated(fixture):
    data = fixture[0]
    split = deepcopy(data.train)
    basis = models.rule_weights(split.X)
    original = dg._cluster_support(split, basis)
    split.X = np.repeat(split.X, 2, axis=0)
    split.keys = tuple(key for key in split.keys for _ in range(2))
    duplicate = dg._cluster_support(split, np.repeat(basis, 2, axis=0))
    for actual, expected in zip(duplicate, original):
        np.testing.assert_allclose(actual, expected)


def test_selected_prior_fraction_matches_shrinkage_and_unseen_rule_fallback(fixture):
    report = dg.run_diagnostics(*fixture)
    support = report["rule_support"]["fuzzy-fixed"]["selected_model_test_support"]
    assert support["status"] == "used"
    prior = support["prior_strength"]
    for row, x in zip(support["per_query"], fixture[0].test.X):
        if x[0] == .5:
            assert row["global_mean_prior_fraction"] == 1
            assert row["membership_mass_on_unseen_rules"] == 1
            assert row["membership_weighted_cluster_support"] == 0
        else:
            assert row["global_mean_prior_fraction"] == pytest.approx(prior / (90 + prior))
            assert row["membership_weighted_cluster_support"] == pytest.approx(90)
            assert row["membership_weighted_cluster_kish_ess"] == pytest.approx(90)
    assert support["all_test"]["n"] == len(fixture[0].test)
    assert support["selected_A2"]["n"] == np.sum(fixture[3]["fuzzy-fixed"] == 1)


@pytest.mark.parametrize("scores,gains,expected", [
    ([0, 0, 1, 1], [2, 2, 4, 4], 1),
    ([0, 0, 1, 1], [4, 4, 2, 2], -1),
    ([0, 0, 1, 1], [0, 1, 0, 1], 0),
    ([1, 1, 1], [1, 2, 3], None),
    ([1, 2, 3], [1, 1, 1], None),
    ([1], [3], None),
])
def test_spearman_with_ties_and_constants(scores, gains, expected):
    actual = dg.spearman(scores, gains)
    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected)


def test_quartiles_keep_ties_together_even_if_bins_are_empty():
    scores = np.array([.1] * 12 + [.9] * 4)
    gains = np.arange(16, dtype=float)
    groups = np.array([f"g{i}" for i in range(16)])
    result = dg._quartiles(scores, gains, groups, predicted_gain=True)
    nonempty = [group for group in result["groups"] if group["n"]]
    assert len(nonempty) == 2
    assert [group["n"] for group in nonempty] == [12, 4]
    for group in nonempty:
        assert group["score"]["min"] == group["score"]["max"]
        assert group["predicted_gain_mean"] == group["score"]["mean"]
    assert sum(group["n"] for group in result["groups"]) == 16


def test_feature_saturation_uses_scaled_endpoints_not_claimed_raw_clipping(fixture):
    report = dg.run_diagnostics(*fixture)
    signal = report["feature_drift"]["signals"]["S1"]
    assert signal["train"]["mean"] == .5
    assert signal["train"]["std"] == .5
    assert signal["train"]["at_zero_percent"] == 50
    assert signal["train"]["at_one_percent"] == 50
    assert signal["test"]["at_either_boundary_percent"] == pytest.approx(200 / 3)
    assert "raw clipping cannot be inferred" in report["feature_drift"]["boundary_interpretation"]


def test_fallback_has_no_selected_fuzzy_score_or_shrinkage(fixture):
    data, candidates, fits, assignments = fixture
    fits["fuzzy-fixed"] = {"candidate": "always-A1"}
    assignments["fuzzy-fixed"] = np.zeros(len(data.test), dtype=int)
    report = dg.run_diagnostics(data, candidates, fits, assignments)
    assert report["rule_support"]["fuzzy-fixed"]["selected_model_test_support"]["status"] == "not_used"
    assert report["gain_ranking"]["fuzzy-fixed"]["status"] == "not_used"
    assert report["rule_support"]["fuzzy-fixed"]["sum_rule_support"] == pytest.approx(180)


def test_diagnostics_are_json_serializable_and_do_not_modify_frozen_inputs(fixture):
    data, candidates, fits, assignments = fixture
    old_candidates = json.dumps(candidates, sort_keys=True, allow_nan=False)
    old_fits = deepcopy(fits)
    old_actions = {name: actions.copy() for name, actions in assignments.items()}
    old_features = {name: split.X.copy() for name, split in data.splits.items()}
    report = dg.run_diagnostics(*fixture)
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert old_candidates == json.dumps(candidates, sort_keys=True, allow_nan=False)
    assert old_fits == fits
    for method in assignments:
        np.testing.assert_array_equal(old_actions[method], assignments[method])
    for name, split in data.splits.items():
        np.testing.assert_array_equal(old_features[name], split.X)


def test_rejects_model_support_fitted_with_paraphrase_row_weights(fixture):
    data, candidates, fits, assignments = fixture
    candidates["fuzzy-fixed"]["prior=1.0"] = models.fit(data.train.X, data.train.f1, data.train.cost)
    with pytest.raises(ValueError, match="one total TRAIN weight per cluster"):
        dg.run_diagnostics(data, candidates, fits, assignments)


@pytest.mark.parametrize("scores,gains", [([], []), ([1], [1, 2]), ([np.nan], [1]), ([1], [np.inf])])
def test_invalid_ranking_inputs_rejected(scores, gains):
    with pytest.raises(ValueError):
        dg.spearman(scores, gains)
