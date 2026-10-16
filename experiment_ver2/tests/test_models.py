"""Analytic recovery, no-label prediction, sparse support and persistence checks."""

import json

import numpy as np
import pytest

from experiment_ver2 import models
from main_logic import fuzzy as fz


X = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
QUALITY = np.array([[0.2, 0.4, 0.9], [0.8, 0.9, 0.1]])
COST = np.array([[1.0, 2.0, 4.0], [2.0, 3.0, 7.0]])


def test_product_rule_partition_and_original_rule_order():
    rng = np.random.default_rng(713)
    inputs = np.vstack((X, rng.uniform(size=(20, 4)), [0.35] * 4))
    widths = [0.20, 0.21, 0.23, 0.25]
    weights = models.rule_weights(inputs, widths)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert np.all(weights >= 0)
    assert weights.shape == (len(inputs), 81)
    for row, memberships in zip(inputs, weights):
        expected = [np.prod([fz.memberships(row[j], widths[j])[fz.LABELS.index(label)]
                             for j, label in enumerate(labels)])
                    for _, labels, _, _ in fz.RULES]
        np.testing.assert_allclose(memberships, expected, atol=1e-15)
    assert weights[0, 0] == 1
    assert weights[1, 80] == 1


def test_shrinkage_matches_hand_calculation_and_unseen_rule_uses_global():
    model = models.fit(X, QUALITY, COST, prior_strength=2)
    quality, cost = models.predict(model, np.vstack((X, [0.5] * 4)))
    global_quality, global_cost = QUALITY.mean(axis=0), COST.mean(axis=0)
    np.testing.assert_allclose(quality[:2], (QUALITY + 2 * global_quality) / 3)
    np.testing.assert_allclose(cost[:2], (COST + 2 * global_cost) / 3)
    np.testing.assert_allclose(quality[2], global_quality)
    np.testing.assert_allclose(cost[2], global_cost)
    assert model["rule_support"][0] == 1
    assert model["rule_support"][40] == 0
    assert model["rule_effective_rows"][0] == 1


def test_zero_prior_recovers_observed_rule_and_handles_unseen_rule():
    model = models.fit(X, QUALITY, COST, prior_strength=0)
    quality, cost = models.predict(model, np.vstack((X, [0.5] * 4)))
    np.testing.assert_allclose(quality[:2], QUALITY)
    np.testing.assert_allclose(cost[:2], COST)
    np.testing.assert_allclose(quality[2], QUALITY.mean(axis=0))


def test_model_can_learn_that_high_uncertainty_does_not_favor_expensive_action():
    model = models.fit(np.repeat(X, 12, axis=0), np.repeat(QUALITY, 12, axis=0),
                       np.repeat(COST, 12, axis=0), prior_strength=1)
    quality, _ = models.predict(model, X)
    assert quality[0, 2] > quality[0, 1]
    assert quality[1, 2] < quality[1, 1]


def test_fractional_firing_mass_controls_shrinkage_not_kish_row_count():
    # Almost all weight falls in Low; a tiny Med activation is not a full
    # observation, even though that rule has Kish effective row count one.
    train_X = np.array([[0.2502, 0, 0, 0], [1, 1, 1, 1]])
    model = models.fit(train_X, QUALITY, COST, prior_strength=1)
    idx = 27  # Med,Low,Low,Low, following original rule order.
    mass = model["rule_support"][idx]
    assert 0 < mass < 0.01
    assert model["rule_effective_rows"][idx] == pytest.approx(1)
    np.testing.assert_allclose(model["rule_quality"][idx],
                               (mass * QUALITY[0] + QUALITY.mean(axis=0)) / (mass + 1))


@pytest.mark.parametrize("fitter", [models.fit, models.fit_global, models.fit_ridge])
def test_cluster_weights_make_duplicate_paraphrases_mass_preserving(fitter):
    base = fitter(X, QUALITY, COST)
    duplicate = fitter(np.repeat(X, 3, axis=0), np.repeat(QUALITY, 3, axis=0),
                       np.repeat(COST, 3, axis=0), sample_weight=np.full(6, 1 / 3))
    for before, after in zip(models.predict(base, X), models.predict(duplicate, X)):
        np.testing.assert_allclose(before, after, atol=1e-14)


def test_fuzzy_prediction_is_continuous_at_membership_boundaries():
    model = models.fit(X, QUALITY, COST, prior_strength=1)
    for boundary in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75):
        points = np.array([[boundary - 1e-8, 0, 0, 0],
                           [boundary + 1e-8, 0, 0, 0]])
        quality, cost = models.predict(model, points)
        assert np.max(np.abs(quality[0] - quality[1])) < 1e-6
        assert np.max(np.abs(cost[0] - cost[1])) < 1e-6


def test_global_baseline_is_signal_independent_and_uses_weighted_means():
    model = models.fit_global(X, QUALITY, COST, sample_weight=[3, 1])
    quality, cost = models.predict(model, np.array([[0.3] * 4, [0.7] * 4]))
    np.testing.assert_allclose(quality, np.broadcast_to((3 * QUALITY[0] + QUALITY[1]) / 4,
                                                       quality.shape))
    np.testing.assert_allclose(cost, np.broadcast_to((3 * COST[0] + COST[1]) / 4,
                                                    cost.shape))


def test_ridge_recovers_linear_outcomes_and_constant_feature_is_safe():
    rng = np.random.default_rng(23)
    train_X = rng.uniform(size=(80, 4))
    train_X[:, 3] = 0.5
    quality = np.column_stack((0.1 + 0.5 * train_X[:, 0],
                               0.9 - 0.3 * train_X[:, 1],
                               0.3 + 0.2 * train_X[:, 2]))
    cost = 1 + 3 * quality
    model = models.fit_ridge(train_X, quality, cost, alpha=0)
    predicted_quality, predicted_cost = models.predict(model, train_X)
    np.testing.assert_allclose(predicted_quality, quality, atol=1e-12)
    np.testing.assert_allclose(predicted_cost, cost, atol=1e-12)
    assert model["feature_scale"][3] == 1


def test_ridge_regularizes_toward_global_mean_without_penalizing_intercept():
    model = models.fit_ridge(X, QUALITY, COST, alpha=1e12)
    quality, cost = models.predict(model, X)
    np.testing.assert_allclose(quality, np.tile(QUALITY.mean(axis=0), (2, 1)), atol=1e-10)
    np.testing.assert_allclose(cost, np.tile(COST.mean(axis=0), (2, 1)), atol=1e-10)
    constant = models.fit_ridge(np.zeros((2, 4)), QUALITY, COST, alpha=0)
    quality, _ = models.predict(constant, X)
    np.testing.assert_allclose(quality, np.tile(QUALITY.mean(axis=0), (2, 1)))


def test_ridge_clips_extrapolation_to_physical_outcome_ranges():
    train_X = X * 0.01 + 0.49
    model = models.fit_ridge(train_X, QUALITY, COST, alpha=0)
    quality, cost = models.predict(model, X)
    assert np.all((quality >= 0) & (quality <= 1))
    assert np.all(cost >= 0)


@pytest.mark.parametrize("fitter", [models.fit, models.fit_global, models.fit_ridge])
def test_perfect_training_quality_survives_weighted_sum_roundoff(fitter):
    rng = np.random.default_rng(13)
    train_X = rng.random((500, 4))
    model = fitter(train_X, np.ones((500, 3)), np.ones((500, 3)),
                   sample_weight=rng.random(500))
    quality, cost = models.predict(model, train_X[:10])
    np.testing.assert_allclose(quality, 1)
    np.testing.assert_allclose(cost, 1)


def test_constant_boundary_features_survive_weighted_mean_roundoff():
    rng = np.random.default_rng(13)
    model = models.fit_ridge(np.ones((500, 4)), np.ones((500, 3)),
                             np.ones((500, 3)), sample_weight=rng.random(500))
    quality, cost = models.predict(model, X)
    np.testing.assert_allclose(quality, 1)
    np.testing.assert_allclose(cost, 1)


@pytest.mark.parametrize("fitter", [models.fit, models.fit_global, models.fit_ridge])
def test_model_json_roundtrip_and_empty_prediction(fitter):
    model = fitter(X, QUALITY, COST)
    restored = json.loads(json.dumps(model, allow_nan=False))
    for original, reloaded in zip(models.predict(model, X), models.predict(restored, X)):
        np.testing.assert_allclose(original, reloaded)
    assert [a.shape for a in models.predict(restored, np.empty((0, 4)))] == [(0, 3), (0, 3)]


@pytest.mark.parametrize("bad_X", [np.ones((2, 2)), np.ones((2, 5)), [[0, 0, 0, np.nan]],
                                    [[0, 0, 0, np.inf]], [[0, 0, 0, 1.1]],
                                    [[0, 0, 0, -0.1]], np.empty((0, 4))])
def test_rejects_invalid_training_features(bad_X):
    with pytest.raises(ValueError):
        models.fit(bad_X, QUALITY, COST)


@pytest.mark.parametrize("bad_quality", [QUALITY + 1, QUALITY - 1,
                                          np.full((2, 3), np.nan), QUALITY[:, :2],
                                          QUALITY[:1]])
def test_rejects_invalid_quality(bad_quality):
    with pytest.raises(ValueError):
        models.fit(X, bad_quality, COST)


@pytest.mark.parametrize("bad_cost", [-COST, np.full((2, 3), np.inf), COST[:, :2], COST[:1]])
def test_rejects_invalid_cost(bad_cost):
    with pytest.raises(ValueError):
        models.fit(X, QUALITY, bad_cost)


@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [np.nan, 1], [1], [[1], [1]]])
def test_rejects_invalid_sample_weights(weights):
    with pytest.raises(ValueError, match="sample_weight"):
        models.fit(X, QUALITY, COST, sample_weight=weights)


@pytest.mark.parametrize("widths", [0, -0.1, 0.26, np.nan, [0.2, 0.2], [[0.2] * 4]])
def test_rejects_invalid_membership_widths(widths):
    with pytest.raises(ValueError, match="widths"):
        models.rule_weights(X, widths)


@pytest.mark.parametrize("value", [-1, np.inf, np.nan, [1]])
def test_rejects_invalid_regularization(value):
    with pytest.raises(ValueError, match="prior_strength"):
        models.fit(X, QUALITY, COST, prior_strength=value)
    with pytest.raises(ValueError, match="alpha"):
        models.fit_ridge(X, QUALITY, COST, alpha=value)


def test_rejects_corrupted_model_parameters():
    model = models.fit_ridge(X, QUALITY, COST)
    model["feature_scale"][0] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        models.predict(model, X)
    model = models.fit(X, QUALITY, COST)
    model["rule_quality"][0][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        models.predict(model, X)
    model = models.fit(X, QUALITY, COST)
    del model["widths"]
    with pytest.raises(ValueError, match="missing predictive parameter"):
        models.predict(model, X)
    with pytest.raises(ValueError, match="schema_version"):
        models.predict({}, X)


@pytest.mark.parametrize("n_features", [3, 4])
@pytest.mark.parametrize("n_actions", [2, 3])
@pytest.mark.parametrize("fitter", [models.fit, models.fit_global, models.fit_ridge])
def test_supported_dimensions_roundtrip_without_artificial_action_or_feature(
        n_features, n_actions, fitter):
    train_X = X[:, :n_features]
    # Two-action data are the real A1/A2 columns, without a copied A0.
    quality, cost = QUALITY[:, -n_actions:], COST[:, -n_actions:]
    model = fitter(train_X, quality, cost, sample_weight=[1 / 3, 1 / 3])
    assert model["n_features"] == n_features
    assert model["n_actions"] == n_actions
    assert len(model["global_quality"]) == n_actions
    restored = json.loads(json.dumps(model, allow_nan=False))
    predictions = models.predict(model, train_X)
    for original, reloaded in zip(predictions, models.predict(restored, train_X)):
        assert original.shape == (2, n_actions)
        np.testing.assert_allclose(original, reloaded)
    empty = models.predict(restored, np.empty((0, n_features)))
    assert [a.shape for a in empty] == [(0, n_actions), (0, n_actions)]
    if model["kind"] == "fuzzy":
        assert len(model["widths"]) == n_features
        assert np.shape(model["rule_quality"]) == (3**n_features, n_actions)
    if model["kind"] == "ridge":
        assert np.shape(model["coefficients"]) == (n_features, 2 * n_actions)


def test_three_signal_product_partition_matches_original_27_rule_order():
    rng = np.random.default_rng(143)
    train_X = np.vstack((X[:, :3], rng.uniform(size=(20, 3)), [0.35] * 3))
    widths = [0.20, 0.22, 0.25]
    weights = models.rule_weights(train_X, widths)
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    assert np.all(weights >= 0)
    assert weights.shape == (len(train_X), 27)
    assert weights[0, 0] == weights[1, 26] == 1
    indices = models.rule_index(3)
    original_rules = fz.rule_base(3)
    for idx, (_, labels, _, _) in zip(indices, original_rules):
        assert tuple(fz.LABELS[i] for i in idx) == labels
    for row, actual in zip(train_X, weights):
        degrees = np.array([fz.memberships(x, w) for x, w in zip(row, widths)])
        expected = np.prod(degrees[np.arange(3), indices], axis=1)
        np.testing.assert_allclose(actual, expected, atol=1e-15)
    assert models.rule_weights(np.empty((0, 3)), widths).shape == (0, 27)
    np.testing.assert_array_equal(models.rule_index(4), models.RULE_INDEX)


def test_two_action_three_signal_shrinkage_and_rule_trace_reconstruction():
    train_X, quality, cost = X[:, :3], QUALITY[:, 1:], COST[:, 1:]
    model = models.fit(train_X, quality, cost, widths=[0.2, 0.21, 0.24], prior_strength=2)
    test_X = np.vstack((train_X, [0.5] * 3, [0.35, 0.6, 0.9]))
    predicted_quality, predicted_cost = models.predict(model, test_X)
    np.testing.assert_allclose(predicted_quality[:2], (quality + 2 * quality.mean(axis=0)) / 3)
    np.testing.assert_allclose(predicted_cost[:2], (cost + 2 * cost.mean(axis=0)) / 3)
    np.testing.assert_allclose(predicted_quality[2], quality.mean(axis=0))
    assert model["rule_support"][13] == 0  # Med,Med,Med.
    active = models.rule_weights(test_X, model["widths"])
    np.testing.assert_allclose(active @ model["rule_quality"], predicted_quality)
    np.testing.assert_allclose(active @ model["rule_cost"], predicted_cost)
    assert predicted_quality[0, 1] > predicted_quality[0, 0]
    assert predicted_quality[1, 1] < predicted_quality[1, 0]


def test_three_feature_two_action_ridge_recovers_linear_outcomes_on_new_inputs():
    rng = np.random.default_rng(38)
    train_X = rng.uniform(size=(80, 3))
    test_X = rng.uniform(size=(20, 3))

    def targets(features):
        quality = np.column_stack((0.1 + 0.3 * features[:, 0] + 0.2 * features[:, 2],
                                   0.9 - 0.3 * features[:, 1]))
        return quality, 1 + 3 * quality

    model = models.fit_ridge(train_X, *targets(train_X), alpha=0)
    for actual, expected in zip(models.predict(model, test_X), targets(test_X)):
        np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("fitter", [models.fit, models.fit_global, models.fit_ridge])
@pytest.mark.parametrize("n_features", [3, 4])
def test_prediction_rejects_other_supported_but_untrained_feature_count(fitter, n_features):
    model = fitter(X[:, :n_features], QUALITY[:, 1:], COST[:, 1:])
    other_width = 7 - n_features
    with pytest.raises(ValueError, match="X must have shape"):
        models.predict(model, np.zeros((2, other_width)))
    with pytest.raises(ValueError, match="X must have shape"):
        models.predict(model, np.empty((0, other_width)))


@pytest.mark.parametrize("fitter", [models.fit, models.fit_global, models.fit_ridge])
def test_legacy_schema_one_models_still_predict_four_signals_three_actions(fitter):
    current = fitter(X, QUALITY, COST)
    legacy = {key: value for key, value in current.items() if key not in ("n_features", "n_actions")}
    legacy["schema_version"] = 1
    for expected, actual in zip(models.predict(current, X), models.predict(legacy, X)):
        np.testing.assert_allclose(actual, expected)
    with pytest.raises(ValueError, match="X must have shape"):
        models.predict(legacy, X[:, :3])


@pytest.mark.parametrize("key,value", [("n_features", 2), ("n_features", 5),
                                        ("n_features", 3.0), ("n_features", True),
                                        ("n_actions", 1), ("n_actions", 4),
                                        ("n_actions", 2.0), ("n_actions", True)])
def test_rejects_invalid_model_dimension_metadata(key, value):
    model = models.fit_global(X[:, :3], QUALITY[:, 1:], COST[:, 1:])
    model[key] = value
    with pytest.raises(ValueError, match=key):
        models.predict(model, X[:, :3])


@pytest.mark.parametrize("fitter", [models.fit, models.fit_global, models.fit_ridge])
def test_predict_checks_action_count_and_required_dimension_metadata(fitter):
    model = fitter(X[:, :3], QUALITY[:, 1:], COST[:, 1:])
    model["global_quality"].append(0.5)
    with pytest.raises(ValueError, match="global_quality"):
        models.predict(model, X[:, :3])
    model = fitter(X[:, :3], QUALITY[:, 1:], COST[:, 1:])
    del model["n_features"]
    with pytest.raises(ValueError, match="missing predictive parameter n_features"):
        models.predict(model, X[:, :3])


def test_three_signal_widths_and_rule_count_are_exact():
    with pytest.raises(ValueError, match="3 signal widths"):
        models.fit(X[:, :3], QUALITY[:, 1:], COST[:, 1:], widths=[0.2] * 4)
    model = models.fit(X[:, :3], QUALITY[:, 1:], COST[:, 1:])
    model["rule_quality"] = model["rule_quality"][:26]
    with pytest.raises(ValueError, match="exactly 27"):
        models.predict(model, X[:, :3])
    model = models.fit(X[:, :3], QUALITY[:, 1:], COST[:, 1:])
    model["rule_cost"] = [row + [1.0] for row in model["rule_cost"]]
    with pytest.raises(ValueError, match="rule_cost must have shape"):
        models.predict(model, X[:, :3])


@pytest.mark.parametrize("count", [0, 1, 2, 5, 3.0, "3", True])
def test_rule_index_rejects_unsupported_or_noninteger_dimension(count):
    with pytest.raises(ValueError, match="n_features"):
        models.rule_index(count)
