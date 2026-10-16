"""Additive fuzzy action-outcome model with TRAIN-standardized memberships.

Each signal contributes its Low/Med/High memberships as three separate basis
columns: there are 9 columns for three signals or 12 for four. There are no
cross-feature products. An unpenalized intercept plus ridge coefficients fit
each action's quality and cost independently. Positive regularization resolves
the collinear partition-of-unity basis. Widths/alpha are supplied by the frozen
experiment protocol; this module performs no hyperparameter selection.
"""

import numpy as np

from experiment_ver2 import models
from main_logic import fuzzy as fz


SCHEMA_VERSION = 1
KIND = "additive_fuzzy"


def _alpha(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("alpha must be a finite strictly positive scalar")
    alpha = models._nonnegative_scalar(value, "alpha")
    if alpha == 0:
        raise ValueError("alpha must be strictly positive for the collinear membership basis")
    return alpha


def additive_basis(X, widths=0.2):
    """Return (n,3*d) single-signal memberships, feature-major L/M/H order.

    X has 3 or 4 scaled features in [0,1]. Each feature's three columns sum to
    one; the entire row sums to d. Empty batches retain their basis dimension.
    """
    X = models._matrix(X, "X", models.SIGNAL_COUNTS, allow_empty=True, unit_interval=True)
    widths = models._widths(widths, X.shape[1])
    if not len(X):
        return np.empty((0, 3 * X.shape[1]))
    return np.array([[fz.memberships(x, width) for x, width in zip(row, widths)]
                     for row in X]).reshape(len(X), 3 * X.shape[1])


def fit(X, quality, cost, widths=0.2, alpha=1.0, sample_weight=None):
    """Fit weighted squared loss + alpha*||coef||² on additive memberships.

    Membership means/scales use TRAIN weights only, exactly as in the raw
    feature ridge comparator. Near-constant columns use scale 1. The intercept
    is the weighted TRAIN output mean and is not penalized. Sample weights
    may be 1/k for k variants of one cluster; their total defines the loss
    scale. Returns a JSON-serializable model with no synthetic action column.
    """
    X, quality, cost, weights = models._training(X, quality, cost, sample_weight)
    widths = models._widths(widths, X.shape[1])
    alpha = _alpha(alpha)
    basis = additive_basis(X, widths)
    n_basis, n_actions = basis.shape[1], quality.shape[1]
    mean = np.clip(np.average(basis, axis=0, weights=weights), 0, 1)
    scale = np.sqrt(np.average(np.square(basis - mean), axis=0, weights=weights))
    scale[scale < 1e-12] = 1.0
    design = (basis - mean) / scale
    outcomes = np.column_stack((quality, cost))
    intercept = np.average(outcomes, axis=0, weights=weights)
    root_weight = np.sqrt(weights[:, None])
    lhs = np.vstack((design * root_weight, np.sqrt(alpha) * np.eye(n_basis)))
    rhs = np.vstack(((outcomes - intercept) * root_weight,
                     np.zeros((n_basis, 2 * n_actions))))
    coefficients = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    model = models._common(KIND, X, quality, cost, weights)
    model.update({"schema_version": SCHEMA_VERSION, "n_basis": n_basis,
                  "basis_order": "feature-major: Low, Med, High for each input signal",
                  "widths": widths.tolist(), "alpha": alpha,
                  "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
                  "intercept": intercept.tolist(), "coefficients": coefficients.tolist()})
    return model


def predict(model, X):
    """Predict per-action (quality,cost), clipped to [0,1] and >=0 respectively.

    Fitting is additive before these physical-range clips. Global clipping can
    produce nonadditive boundary effects; it does not add learned interactions.
    Serialized predictive parameter dimensions and finite ranges are checked.
    """
    if (not isinstance(model, dict) or type(model.get("schema_version")) is not int
            or model["schema_version"] != SCHEMA_VERSION or model.get("kind") != KIND):
        raise ValueError("unsupported additive model kind or schema_version")
    try:
        n_features = models._dimension(model["n_features"], "model n_features", models.SIGNAL_COUNTS)
        n_actions = models._dimension(model["n_actions"], "model n_actions", models.ACTION_COUNTS)
        n_basis = models._dimension(model["n_basis"], "model n_basis", (3 * n_features,))
        X = models._matrix(X, "X", n_features, allow_empty=True, unit_interval=True)
        widths = models._widths(model["widths"], n_features)
        _alpha(model["alpha"])
        models._vector(model["global_quality"], "global_quality", n_actions, unit_interval=True)
        models._vector(model["global_cost"], "global_cost", n_actions, nonnegative=True)
        mean = models._vector(model["feature_mean"], "feature_mean", n_basis, unit_interval=True)
        scale = models._vector(model["feature_scale"], "feature_scale", n_basis)
        if np.any(scale <= 0):
            raise ValueError("model feature_scale must be strictly positive")
        coefficients = models._matrix(model["coefficients"], "coefficients", 2 * n_actions)
        if len(coefficients) != n_basis:
            raise ValueError(f"model coefficients must have {n_basis} basis rows")
        intercept = models._vector(model["intercept"], "intercept", 2 * n_actions)
        raw = ((additive_basis(X, widths) - mean) / scale) @ coefficients + intercept
        if not np.isfinite(raw).all():
            raise ValueError("model produced non-finite predictions")
        return np.clip(raw[:, :n_actions], 0, 1), np.maximum(raw[:, n_actions:], 0)
    except KeyError as error:
        raise ValueError(f"model is missing predictive parameter {error.args[0]}") from error
