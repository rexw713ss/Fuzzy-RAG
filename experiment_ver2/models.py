"""Train-only action-outcome models; NumPy is the sole numerical dependency.

The fuzzy model learns action *quality*, not an ordinal escalation label. Each
of the 81 antecedents is a product of the original Low/Med/High memberships.
The partition sums to one. Its consequent for each action is a weighted mean
shrunk toward the training-wide mean for that action. Unseen rules fall back
exactly to that mean; weakly activated rules receive little evidence. Costs
are learned separately in their input units, without assuming an action order.

All fits return plain dictionaries that can be round-tripped through JSON.
Hyperparameters and widths must be chosen without examining test outcomes.
"""

import itertools

import numpy as np

from main_logic import fuzzy as fz


N_SIGNALS = 4
N_ACTIONS = 3
RULE_INDEX = np.array(list(itertools.product(range(3), repeat=4)), dtype=int)
SCHEMA_VERSION = 1


def _matrix(value, name, columns, *, allow_empty=False, unit_interval=False):
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or a.shape[1] != columns or (not allow_empty and len(a) == 0):
        raise ValueError(f"{name} must have shape (n, {columns}) with "
                         f"{'n >= 0' if allow_empty else 'n > 0'}; got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain only finite values")
    if unit_interval and np.any((a < 0) | (a > 1)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return a


def _widths(widths):
    w = np.asarray(widths, dtype=float)
    if w.ndim != 0 and w.shape != (N_SIGNALS,):
        raise ValueError("widths must be one scalar or four signal widths")
    w = np.broadcast_to(w, (N_SIGNALS,)).copy()
    if not np.all(np.isfinite(w)) or np.any((w <= 0) | (w > fz.W_MAX)):
        raise ValueError(f"widths must be finite and in (0, {fz.W_MAX}]")
    return w


def _nonnegative_scalar(value, name):
    a = np.asarray(value, dtype=float)
    if a.ndim != 0 or not np.isfinite(a) or a < 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return float(a)


def _training(X, quality, cost, sample_weight):
    X = _matrix(X, "X", N_SIGNALS, unit_interval=True)
    quality = _matrix(quality, "quality", N_ACTIONS, unit_interval=True)
    cost = _matrix(cost, "cost", N_ACTIONS)
    if len(quality) != len(X) or len(cost) != len(X):
        raise ValueError("X, quality and cost must have the same number of rows")
    if np.any(cost < 0):
        raise ValueError("cost must be nonnegative")
    w = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if w.shape != (len(X),) or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("sample_weight must be one finite nonnegative value per row")
    if not np.isfinite(w.sum()) or w.sum() <= 0:
        raise ValueError("sample_weight must have a positive finite sum")
    return X, quality, cost, w


def _common(kind, quality, cost, w):
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "n_train": len(quality),
        "training_weight": float(w.sum()),
        # Weighted summation may exceed the mathematical bound by one ulp.
        "global_quality": np.clip(np.average(quality, axis=0, weights=w), 0, 1).tolist(),
        "global_cost": np.average(cost, axis=0, weights=w).tolist(),
    }


def rule_weights(X, widths=0.2):
    """Return (n, 81) product weights in the original rule-base order.

    Each row sums to one and depends on signals only. The four input columns
    are the existing scaled S1..S4, each in [0, 1].
    """
    X = _matrix(X, "X", N_SIGNALS, allow_empty=True, unit_interval=True)
    widths = _widths(widths)
    if len(X) == 0:
        return np.empty((0, len(RULE_INDEX)))
    mu = np.array([[fz.memberships(value, width) for value, width in zip(row, widths)]
                   for row in X])
    weights = np.ones((len(X), len(RULE_INDEX)))
    for j in range(N_SIGNALS):
        weights *= mu[:, j, RULE_INDEX[:, j]]
    # Only remove floating-point drift; the membership functions already form
    # a partition and are never renormalized using labels or held-out data.
    return weights / weights.sum(axis=1, keepdims=True)


def fit(X, quality, cost, widths=0.2, prior_strength=5.0, sample_weight=None):
    """Fit shrunken fuzzy action means and return a JSON-serializable model.

    A rule with evidence mass m has consequent
    (sum_i weight_i * membership_ir * outcome_i + prior_strength * global) /
    (m + prior_strength). The prior is a pseudo-count in sample-weight units.
    Give each of a cluster's k rows weight 1/k to make one cluster one unit.
    With prior_strength=0, observed rules use their raw weighted mean and
    unseen rules still use the global mean. No monotonicity is imposed.
    """
    X, quality, cost, w = _training(X, quality, cost, sample_weight)
    widths = _widths(widths)
    prior_strength = _nonnegative_scalar(prior_strength, "prior_strength")
    model = _common("fuzzy", quality, cost, w)
    memberships = rule_weights(X, widths)
    weighted = memberships * w[:, None]
    support = weighted.sum(axis=0)
    denominator = support + prior_strength

    def consequents(values, global_mean, unit_interval=False):
        means = np.broadcast_to(global_mean, (len(RULE_INDEX), N_ACTIONS)).copy()
        numerator = weighted.T @ values + prior_strength * global_mean
        np.divide(numerator, denominator[:, None], out=means,
                  where=denominator[:, None] > 0)
        if unit_interval:
            means = np.clip(means, 0, 1)
        return means.tolist()

    # Kish effective *row* count is diagnostic, not independent-cluster ESS.
    # Membership mass governs shrinkage: a tiny firing degree must not look
    # like a full observation merely because its Kish count is one.
    square_mass = np.square(weighted).sum(axis=0)
    effective_n = np.divide(np.square(support), square_mass,
                            out=np.zeros_like(support), where=square_mass > 0)
    model.update({
        "widths": widths.tolist(),
        "prior_strength": prior_strength,
        "rule_support": support.tolist(),
        "rule_effective_rows": effective_n.tolist(),
        "rule_quality": consequents(quality, np.array(model["global_quality"]), True),
        "rule_cost": consequents(cost, np.array(model["global_cost"])),
    })
    return model


def fit_global(X, quality, cost, sample_weight=None):
    """Signal-free comparator: training mean outcome for each action."""
    _, quality, cost, w = _training(X, quality, cost, sample_weight)
    return _common("global", quality, cost, w)


def fit_ridge(X, quality, cost, alpha=1.0, sample_weight=None):
    """Linear action-outcome comparator with an unpenalized intercept.

    Fits weighted squared error + alpha * ||coefficients||^2. Features are
    standardized from training rows only. Costs and F1 share the feature
    design, but each action/output has separate coefficients. Predictions
    are clipped to physical ranges (F1 [0,1], cost >=0); alpha is not tuned
    here. alpha=0 uses least squares, including rank-deficient inputs.
    """
    X, quality, cost, w = _training(X, quality, cost, sample_weight)
    alpha = _nonnegative_scalar(alpha, "alpha")
    model = _common("ridge", quality, cost, w)
    mean = np.clip(np.average(X, axis=0, weights=w), 0, 1)
    scale = np.sqrt(np.average(np.square(X - mean), axis=0, weights=w))
    scale[scale < 1e-12] = 1.0
    design = (X - mean) / scale
    outcomes = np.column_stack((quality, cost))
    intercept = np.average(outcomes, axis=0, weights=w)
    lhs = design * np.sqrt(w[:, None])
    rhs = (outcomes - intercept) * np.sqrt(w[:, None])
    if alpha > 0:
        lhs = np.vstack((lhs, np.sqrt(alpha) * np.eye(N_SIGNALS)))
        rhs = np.vstack((rhs, np.zeros((N_SIGNALS, 2 * N_ACTIONS))))
    coefficients = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    model.update({"alpha": alpha, "feature_mean": mean.tolist(),
                  "feature_scale": scale.tolist(), "intercept": intercept.tolist(),
                  "coefficients": coefficients.tolist()})
    return model


def _vector(value, name, length, *, nonnegative=False, unit_interval=False):
    a = np.asarray(value, dtype=float)
    if a.shape != (length,) or not np.all(np.isfinite(a)):
        raise ValueError(f"model {name} must contain {length} finite values")
    if (nonnegative or unit_interval) and np.any(a < 0):
        raise ValueError(f"model {name} must be nonnegative")
    if unit_interval and np.any(a > 1):
        raise ValueError(f"model {name} must lie in [0, 1]")
    return a


def predict(model, X):
    """Predict per-action (quality, cost), each shape (n,3), from signals only.

    Reading a JSON-restored model validates its predictive parameters rather
    than silently accepting malformed or non-finite coefficients.
    """
    X = _matrix(X, "X", N_SIGNALS, allow_empty=True, unit_interval=True)
    if not isinstance(model, dict) or model.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported or missing model schema_version")
    kind = model.get("kind")
    try:
        global_quality = _vector(model["global_quality"], "global_quality", N_ACTIONS,
                                 unit_interval=True)
        global_cost = _vector(model["global_cost"], "global_cost", N_ACTIONS,
                              nonnegative=True)
        if kind == "global":
            return (np.broadcast_to(global_quality, (len(X), N_ACTIONS)).copy(),
                    np.broadcast_to(global_cost, (len(X), N_ACTIONS)).copy())
        if kind == "fuzzy":
            weights = rule_weights(X, model["widths"])
            quality = _matrix(model["rule_quality"], "rule_quality", N_ACTIONS,
                              unit_interval=True)
            cost = _matrix(model["rule_cost"], "rule_cost", N_ACTIONS)
            if len(quality) != len(RULE_INDEX) or len(cost) != len(RULE_INDEX):
                raise ValueError("fuzzy model must contain exactly 81 rule consequents")
            if np.any(cost < 0):
                raise ValueError("model rule_cost must be nonnegative")
            return np.clip(weights @ quality, 0, 1), np.maximum(weights @ cost, 0)
        if kind == "ridge":
            mean = _vector(model["feature_mean"], "feature_mean", N_SIGNALS,
                           unit_interval=True)
            scale = _vector(model["feature_scale"], "feature_scale", N_SIGNALS)
            if np.any(scale <= 0):
                raise ValueError("model feature_scale must be strictly positive")
            coefficients = _matrix(model["coefficients"], "coefficients", 2 * N_ACTIONS)
            if len(coefficients) != N_SIGNALS:
                raise ValueError("model coefficients must have four feature rows")
            intercept = _vector(model["intercept"], "intercept", 2 * N_ACTIONS)
            raw = ((X - mean) / scale) @ coefficients + intercept
            if not np.all(np.isfinite(raw)):
                raise ValueError("model produced non-finite predictions")
            return np.clip(raw[:, :N_ACTIONS], 0, 1), np.maximum(raw[:, N_ACTIONS:], 0)
    except KeyError as error:
        raise ValueError(f"model is missing predictive parameter {error.args[0]}") from error
    raise ValueError(f"unknown model kind {kind!r}")
