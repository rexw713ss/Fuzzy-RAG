"""Resampled policy mechanics, exact-model parity and test-outcome isolation."""

from copy import deepcopy
import json

import numpy as np
import pytest

from experiment_ver2 import models
from experiment_ver2.data import Dataset, Split
from experiment_ver2.robustness import stability as st
from experiment_ver2.run import fit_candidates, freeze_policy, apply_policy
from main_logic import calibration


def _dataset():
    rng = np.random.default_rng(713)

    def split(name, count):
        features, quality, keys, questions = [], [], [], []
        for cluster in range(count):
            centre = rng.uniform(.1, .9, 4)
            for variant in range(3):
                x = np.clip(centre + rng.normal(0, .1, 4), 0, 1)
                # Deliberately informative small synthetic example; never evidence.
                gain = .3 if centre[0] < .5 else -.15
                features.append(x)
                quality.append([.45, .45 + gain])
                keys.append(("toy", f"{name}-{cluster}", variant))
                questions.append(f"{name} question {cluster} wording {variant}")
        return Split(X=np.array(features), f1=np.array(quality),
                     cost=np.tile([1.0, 2.0], (len(keys), 1)),
                     em=np.zeros((len(keys), 2)), keys=tuple(keys), questions=tuple(questions),
                     actions=("A1", "A2"))

    train, val, test = split("train", 8), split("val", 5), split("test", 6)
    widths = calibration.calibrate([train.X[r] for r in st._cluster_rows(train)])["widths"]
    return Dataset("toy", train, val, test, np.array(widths), actions=("A1", "A2"))


@pytest.mark.parametrize("prior", [1.0, 5.0, 20.0])
def test_cached_basis_fit_exactly_matches_original_model(prior):
    data = _dataset()
    widths = [.2, .22, .24, .25]
    weights = np.arange(len(data.train)) % 4 / 3
    basis = models.rule_weights(data.train.X, widths)
    cached = st._fuzzy_from_basis(data.train, weights, widths, basis, prior)
    original = models.fit(data.train.X, data.train.f1, data.train.cost, widths=widths,
                          prior_strength=prior, sample_weight=weights)
    assert cached == original
    for split in (data.val, data.test):
        actual = np.clip(models.rule_weights(split.X, widths) @ cached["rule_quality"], 0, 1)
        expected, _ = models.predict(original, split.X)
        np.testing.assert_array_equal(actual, expected)


def test_training_draw_weights_equal_whole_cluster_multiplicity():
    rows = [np.array([0]), np.array([1, 2]), np.array([3, 4, 5])]
    draws, weights = st._draw_training(rows, 6, np.random.default_rng(18))
    counts = np.bincount(draws, minlength=len(rows))
    assert len(draws) == len(rows)
    np.testing.assert_allclose(weights.sum(), len(rows))
    for row, count in zip(rows, counts):
        np.testing.assert_allclose(weights[row], count / len(row))
        assert weights[row].sum() == pytest.approx(count)


def test_validation_materializes_whole_cluster_copies_in_sampled_order():
    data = _dataset()
    rows = st._cluster_rows(data.val)
    draws, indices, sampled = st._draw_validation(data.val, rows, np.random.default_rng(82))
    expected = np.concatenate([rows[i] for i in draws])
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(sampled.X, data.val.X[expected])
    np.testing.assert_array_equal(sampled.f1, data.val.f1[expected])
    assert sampled.questions == tuple(data.val.questions[i] for i in expected)
    for chunk in indices.reshape(-1, 3):
        assert len({data.val.keys[i][1] for i in chunk}) == 1


def test_stability_deterministic_json_roundtrip_and_original_policy_equivalence():
    data = _dataset()
    result = st.run_stability(data, repetitions=3, seed=42)
    assert result == st.run_stability(data, repetitions=3, seed=42)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert "not confirmatory" in result["interpretation"]
    assert "scaler-estimation variation is omitted" in result["limitation"]
    candidates = fit_candidates(data)
    for method in st.METHODS:
        val_scores = {}
        test_scores = {}
        for name, model in candidates[method].items():
            val, _ = models.predict(model, data.val.X)
            test, _ = models.predict(model, data.test.X)
            val_scores[name] = val[:, 1] - val[:, 0]
            test_scores[name] = test[:, 1] - test[:, 0]
        fit = freeze_policy(method, val_scores, data.val, .25)
        actions = apply_policy(method, fit, test_scores, data.test.questions, data.actions)
        original = result["methods"][method]["original"]
        assert original["fit"] == fit
        assert original["actions"] == actions.tolist()
        summary = result["methods"][method]
        assert sum(summary["candidate_selection_counts"].values()) == 3
        assert sum(summary["candidate_selection_frequencies"].values()) == pytest.approx(1)
        assert summary["valid_repetitions"] == 3
        for record in summary["repetitions"]:
            assert record["metrics"]["A2_count"] <= int(.25 * len(data.test))
            assert 0 <= record["metrics"]["action_agreement_with_original"] <= 1
    for draw in result["resamples"]:
        assert draw["training_weight"] == pytest.approx(8)
        rows = st._cluster_rows(data.train)
        clusters = [data.train.X[rows[i]] for i in draw["train_cluster_draws"]]
        expected = calibration.calibrate(clusters)["widths"]
        np.testing.assert_array_equal(draw["calibrated_widths"], expected)


def test_changing_test_outcomes_cannot_change_any_model_selection_or_action():
    data = _dataset()
    changed = deepcopy(data)
    changed.test.f1 = 1 - data.test.f1
    changed.test.cost = data.test.cost * 123 + 44
    changed.test.em = 1 - data.test.em
    first = st.run_stability(data, repetitions=3, seed=312)
    second = st.run_stability(changed, repetitions=3, seed=312)
    assert first["resamples"] == second["resamples"]
    for method in st.METHODS:
        a, b = first["methods"][method], second["methods"][method]
        assert a["candidate_selection_counts"] == b["candidate_selection_counts"]
        assert a["test_A2_selection_frequency"] == b["test_A2_selection_frequency"]
        for left, right in zip([a["original"]] + a["repetitions"],
                               [b["original"]] + b["repetitions"]):
            assert left["fit"] == right["fit"]
            assert left["actions"] == right["actions"]
            assert left["candidate_models_sha256"] == right["candidate_models_sha256"]
            assert left["selected_model_sha256"] == right["selected_model_sha256"]
            assert left["metrics"]["cached_cost"] != right["metrics"]["cached_cost"]


def test_no_complete_cluster_draw_is_recorded_not_redrawn(monkeypatch):
    data = _dataset()
    keep = np.r_[np.arange(3), np.arange(3, len(data.train), 3)]
    data.train.X = data.train.X[keep]
    data.train.f1 = data.train.f1[keep]
    data.train.cost = data.train.cost[keep]
    data.train.em = data.train.em[keep]
    data.train.keys = tuple(data.train.keys[i] for i in keep)
    data.train.questions = tuple(data.train.questions[i] for i in keep)

    def incomplete_only(rows, count, rng):
        index = next(i for i, row in enumerate(rows) if len(row) < 3)
        draws = np.full(len(rows), index)
        weights = np.zeros(count)
        weights[rows[index]] = len(rows) / len(rows[index])
        return draws, weights

    monkeypatch.setattr(st, "_draw_training", incomplete_only)
    result = st.run_stability(data, repetitions=2)
    cal = result["methods"]["fuzzy-calibrated"]
    assert cal["failed_repetitions"] == 2
    assert cal["valid_repetitions"] == 0
    assert cal["test_A2_selection_frequency"] is None
    assert cal["distributions"]["F1"] is None
    assert len(result["failures"]) == 2
    assert result["methods"]["fuzzy-fixed"]["valid_repetitions"] == 2


@pytest.mark.parametrize("repetitions", [0, 1, 2.5, True])
def test_invalid_repetition_count_rejected(repetitions):
    with pytest.raises(ValueError, match="repetitions"):
        st.run_stability(_dataset(), repetitions=repetitions)


@pytest.mark.parametrize("seed", [-1, 2.5, True])
def test_invalid_seed_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        st.run_stability(_dataset(), repetitions=2, seed=seed)


def test_three_signal_ablation_rejected_for_prespecified_primary_analysis():
    data = _dataset()
    data.signals = ("S2", "S3", "S4")
    with pytest.raises(ValueError, match="four signals"):
        st.run_stability(data, repetitions=2)
