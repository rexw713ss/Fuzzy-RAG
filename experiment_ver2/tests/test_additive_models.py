"""Additive basis, weighted-ridge equivalence and persisted model checks."""

import json

import numpy as np
import pytest

from experiment_ver2 import additive_models as additive
from main_logic import fuzzy as fz


def _sample(n_features=4, n_actions=2, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(80, n_features))
    quality = rng.uniform(.1, .9, (len(X), n_actions))
    cost = rng.uniform(1, 5, (len(X), n_actions))
    return X, quality, cost


@pytest.mark.parametrize("n_features", [3, 4])
def test_membership_basis_has_only_single_signal_terms_in_exact_order(n_features):
    X, _, _ = _sample(n_features)
    widths = np.linspace(.2, .25, n_features)
    basis = additive.additive_basis(X, widths)
    assert basis.shape == (len(X), 3 * n_features)
    np.testing.assert_allclose(basis.reshape(len(X), n_features, 3).sum(axis=2), 1)
    np.testing.assert_allclose(basis.sum(axis=1), n_features)
    for j in range(n_features):
        expected = np.array([fz.memberships(value, widths[j]) for value in X[:, j]])
        np.testing.assert_array_equal(basis[:, 3*j:3*j+3], expected)
    assert additive.additive_basis(np.empty((0, n_features))).shape == (0, 3 * n_features)


def test_changing_one_signal_changes_only_its_three_basis_columns():
    before = additive.additive_basis([[.1, .3, .7, .9]])
    after = additive.additive_basis([[.1, .9, .7, .9]])
    np.testing.assert_array_equal(before[:, :3], after[:, :3])
    np.testing.assert_array_equal(before[:, 6:], after[:, 6:])
    assert not np.array_equal(before[:, 3:6], after[:, 3:6])


def test_matches_independent_weighted_ridge_solve_with_unpenalized_intercept():
    X, quality, cost = _sample()
    weights = np.arange(len(X)) % 5 / 3
    alpha = .7
    model = additive.fit(X, quality, cost, widths=[.2, .21, .24, .25],
                         alpha=alpha, sample_weight=weights)
    basis = additive.additive_basis(X, model["widths"])
    mean = np.average(basis, axis=0, weights=weights)
    scale = np.sqrt(np.average((basis-mean)**2, axis=0, weights=weights))
    scale[scale < 1e-12] = 1
    design = np.column_stack(((basis-mean)/scale, np.ones(len(X))))
    penalty = np.diag([alpha] * basis.shape[1] + [0])
    expected = np.linalg.solve(design.T @ (weights[:, None]*design) + penalty,
                               design.T @ (weights[:, None]*np.column_stack((quality, cost))))
    np.testing.assert_allclose(model["coefficients"], expected[:-1], atol=1e-12)
    np.testing.assert_allclose(model["intercept"], expected[-1], atol=1e-12)


def test_single_signal_nonmonotone_quality_is_recoverable_on_unseen_inputs():
    X, _, _ = _sample(seed=18)
    def targets(features):
        basis = additive.additive_basis(features)
        q = np.column_stack((.2 + .5*basis[:, 1], .7 - .3*basis[:, 6]))
        return q, 1 + 2*q
    model = additive.fit(X, *targets(X), alpha=1e-8)
    test_X, _, _ = _sample(seed=83)
    for actual, expected in zip(additive.predict(model, test_X), targets(test_X)):
        np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_balanced_xor_interaction_is_not_learned_by_additive_model():
    X = np.array([[0, 0, .5, .5], [0, 1, .5, .5], [1, 0, .5, .5], [1, 1, .5, .5]])
    q = np.array([[.2, .8], [.8, .2], [.8, .2], [.2, .8]])
    model = additive.fit(X, q, np.ones_like(q), alpha=.1)
    predicted, _ = additive.predict(model, X)
    np.testing.assert_allclose(predicted, .5, atol=1e-14)


def test_paraphrase_duplication_with_divided_weights_preserves_fit():
    X, q, c = _sample()
    first = additive.fit(X, q, c, sample_weight=np.ones(len(X)))
    repeat = additive.fit(np.repeat(X, 3, axis=0), np.repeat(q, 3, axis=0),
                          np.repeat(c, 3, axis=0), sample_weight=np.full(3*len(X), 1/3))
    for expected, actual in zip(additive.predict(first, X), additive.predict(repeat, X)):
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_all_constant_features_and_perfect_quality_are_finite_at_boundaries():
    rng = np.random.default_rng(33)
    X = np.ones((100, 4))
    q = np.ones((100, 2))
    c = np.tile([2, 3], (100, 1))
    model = additive.fit(X, q, c, sample_weight=rng.uniform(size=100))
    assert model["feature_scale"] == [1.] * 12
    quality, cost = additive.predict(model, np.zeros((2, 4)))
    np.testing.assert_allclose(quality, 1)
    np.testing.assert_allclose(cost, np.tile([2, 3], (2, 1)))


@pytest.mark.parametrize("n_features", [3, 4])
@pytest.mark.parametrize("n_actions", [2, 3])
def test_dimensions_json_roundtrip_and_prediction_shape(n_features, n_actions):
    X, q, c = _sample(n_features, n_actions)
    model = additive.fit(X, q, c)
    assert model["n_features"] == n_features
    assert model["n_basis"] == 3*n_features
    assert model["n_actions"] == n_actions
    assert np.shape(model["coefficients"]) == (3*n_features, 2*n_actions)
    restored = json.loads(json.dumps(model, allow_nan=False))
    for before, after in zip(additive.predict(model, X), additive.predict(restored, X)):
        np.testing.assert_array_equal(before, after)
    assert [v.shape for v in additive.predict(restored, np.empty((0, n_features)))] == [(0, n_actions)]*2
    with pytest.raises(ValueError, match="X must have shape"):
        additive.predict(model, np.zeros((2, 7-n_features)))


def test_prediction_clips_to_physical_ranges_without_modifying_model():
    X, q, c = _sample()
    model = additive.fit(X, q, c)
    model["intercept"] = [-10., 10., -10., 10.]
    model["coefficients"] = np.zeros((12, 4)).tolist()
    original = json.dumps(model, sort_keys=True)
    predicted_quality, predicted_cost = additive.predict(model, X[:1])
    np.testing.assert_array_equal(predicted_quality, [[0, 1]])
    np.testing.assert_array_equal(predicted_cost, [[0, 10]])
    assert original == json.dumps(model, sort_keys=True)


@pytest.mark.parametrize("alpha", [0, -1, np.nan, np.inf, [1], True])
def test_requires_positive_finite_regularization(alpha):
    with pytest.raises(ValueError, match="alpha"):
        additive.fit(*_sample(), alpha=alpha)


@pytest.mark.parametrize("widths", [0, .26, np.nan, [.2, .2, .2], [[.2]*4]])
def test_invalid_widths_rejected(widths):
    with pytest.raises(ValueError, match="widths"):
        additive.fit(*_sample(), widths=widths)


def test_invalid_training_values_and_weights_rejected():
    X, q, c = _sample()
    with pytest.raises(ValueError, match="quality"):
        additive.fit(X, q+1, c)
    with pytest.raises(ValueError, match="cost"):
        additive.fit(X, q, -c)
    with pytest.raises(ValueError, match="sample_weight"):
        additive.fit(X, q, c, sample_weight=np.zeros(len(X)))
    with pytest.raises(ValueError, match="X"):
        additive.fit(X[:, :2], q, c)


def test_serialized_parameter_corruption_rejected():
    X, q, c = _sample()
    model = additive.fit(X, q, c)
    model["feature_scale"][0] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        additive.predict(model, X)
    model = additive.fit(X, q, c)
    model["coefficients"][0][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        additive.predict(model, X)
    model = additive.fit(X, q, c)
    model["n_basis"] = 9
    with pytest.raises(ValueError, match="n_basis"):
        additive.predict(model, X)
    model = additive.fit(X, q, c)
    del model["widths"]
    with pytest.raises(ValueError, match="missing predictive parameter widths"):
        additive.predict(model, X)
    with pytest.raises(ValueError, match="schema_version"):
        additive.predict({}, X)
