"""Train-only action-outcome models; NumPy is the sole numerical dependency.

The fuzzy model learns action *quality*, not an ordinal escalation label. Each
of the 27 or 81 antecedents is a product of the original Low/Med/High memberships.
The partition sums to one. Its consequent for each action is a weighted mean
shrunk toward the training-wide mean for that action. Unseen rules fall back
exactly to that mean; weakly activated rules receive little evidence. Costs
are learned separately in their input units, without assuming an action order.

Fits accept three/four signals and two/three action columns in their supplied
order; no artificial action or feature is added. All fits return plain
dictionaries that can be round-tripped through JSON.
Hyperparameters and widths must be chosen without examining test outcomes.
"""

import itertools

import numpy as np

from main_logic import fuzzy as fz


N_SIGNALS = 4
N_ACTIONS = 3
SIGNAL_COUNTS = (3, 4)
ACTION_COUNTS = (2, 3)
SCHEMA_VERSION = 2


def _dimension(value, name, allowed):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}")
    return int(value)


def rule_index(n_features=N_SIGNALS):
    """Antecedent label indices, first feature slowest and last fastest.

    Row i corresponds to rule id i+1, matching main_logic.fuzzy.rule_base(n).
    Three features produce 27 rows; four features produce 81 rows.
    """
    n_features = _dimension(n_features, "n_features", SIGNAL_COUNTS)
    return np.array(list(itertools.product(range(3), repeat=n_features)), dtype=int)


RULE_INDEX = rule_index()  # Legacy four-signal public constant.


def _matrix(value, name, columns, *, allow_empty=False, unit_interval=False):
    a = np.asarray(value, dtype=float)
    counts = (columns,) if isinstance(columns, int) else columns
    if a.ndim != 2 or a.shape[1] not in counts or (not allow_empty and len(a) == 0):
        description = " or ".join(str(c) for c in counts)
        raise ValueError(f"{name} must have shape (n, {description}) with "
                         f"{'n >= 0' if allow_empty else 'n > 0'}; got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain only finite values")
    if unit_interval and np.any((a < 0) | (a > 1)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return a


def _widths(widths, n_features=N_SIGNALS):
    w = np.asarray(widths, dtype=float)
    if w.ndim != 0 and w.shape != (n_features,):
        raise ValueError(f"widths must be one scalar or {n_features} signal widths")
    w = np.broadcast_to(w, (n_features,)).copy()
    if not np.all(np.isfinite(w)) or np.any((w <= 0) | (w > fz.W_MAX)):
        raise ValueError(f"widths must be finite and in (0, {fz.W_MAX}]")
    return w


def _nonnegative_scalar(value, name):
    a = np.asarray(value, dtype=float)
    if a.ndim != 0 or not np.isfinite(a) or a < 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return float(a)


def _training(X, quality, cost, sample_weight):
    X = _matrix(X, "X", SIGNAL_COUNTS, unit_interval=True)
    quality = _matrix(quality, "quality", ACTION_COUNTS, unit_interval=True)
    cost = _matrix(cost, "cost", quality.shape[1])
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


def _common(kind, X, quality, cost, w):
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "n_features": X.shape[1],
        "n_actions": quality.shape[1],
        "n_train": len(quality),
        "training_weight": float(w.sum()),
        # Weighted summation may exceed the mathematical bound by one ulp.
        "global_quality": np.clip(np.average(quality, axis=0, weights=w), 0, 1).tolist(),
        "global_cost": np.average(cost, axis=0, weights=w).tolist(),
    }


def rule_weights(X, widths=0.2):
    """Return (n, 3**n_features) weights in the original rule-base order.

    Each row sums to one and depends on signals only. The three/four input
    columns must each be scaled to [0,1]. Per-feature widths must match the
    input column order and count; a scalar applies to every feature.
    """
    X = _matrix(X, "X", SIGNAL_COUNTS, allow_empty=True, unit_interval=True)
    n_features = X.shape[1]
    widths = _widths(widths, n_features)
    indices = rule_index(n_features)
    if len(X) == 0:
        return np.empty((0, len(indices)))
    mu = np.array([[fz.memberships(value, width) for value, width in zip(row, widths)]
                   for row in X])
    weights = np.ones((len(X), len(indices)))
    for j in range(n_features):
        weights *= mu[:, j, indices[:, j]]
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
    widths = _widths(widths, X.shape[1])
    prior_strength = _nonnegative_scalar(prior_strength, "prior_strength")
    model = _common("fuzzy", X, quality, cost, w)
    memberships = rule_weights(X, widths)
    weighted = memberships * w[:, None]
    support = weighted.sum(axis=0)
    denominator = support + prior_strength

    def consequents(values, global_mean, unit_interval=False):
        means = np.broadcast_to(global_mean, (memberships.shape[1], quality.shape[1])).copy()
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
    X, quality, cost, w = _training(X, quality, cost, sample_weight)
    return _common("global", X, quality, cost, w)


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
    model = _common("ridge", X, quality, cost, w)
    n_features, n_actions = X.shape[1], quality.shape[1]
    mean = np.clip(np.average(X, axis=0, weights=w), 0, 1)
    scale = np.sqrt(np.average(np.square(X - mean), axis=0, weights=w))
    scale[scale < 1e-12] = 1.0
    design = (X - mean) / scale
    outcomes = np.column_stack((quality, cost))
    intercept = np.average(outcomes, axis=0, weights=w)
    lhs = design * np.sqrt(w[:, None])
    rhs = (outcomes - intercept) * np.sqrt(w[:, None])
    if alpha > 0:
        lhs = np.vstack((lhs, np.sqrt(alpha) * np.eye(n_features)))
        rhs = np.vstack((rhs, np.zeros((n_features, 2 * n_actions))))
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
    """Predict (quality, cost), each (n,n_actions), from training-width signals.

    Reading a JSON-restored model validates its predictive parameters rather
    than silently accepting malformed or non-finite coefficients. Legacy
    schema 1 models use exactly four features and three actions.
    """
    if not isinstance(model, dict) or model.get("schema_version") not in (1, SCHEMA_VERSION):
        raise ValueError("unsupported or missing model schema_version")
    kind = model.get("kind")
    try:
        if model["schema_version"] == 1:
            n_features, n_actions = N_SIGNALS, N_ACTIONS
        else:
            n_features = _dimension(model["n_features"], "model n_features", SIGNAL_COUNTS)
            n_actions = _dimension(model["n_actions"], "model n_actions", ACTION_COUNTS)
        X = _matrix(X, "X", n_features, allow_empty=True, unit_interval=True)
        global_quality = _vector(model["global_quality"], "global_quality", n_actions,
                                 unit_interval=True)
        global_cost = _vector(model["global_cost"], "global_cost", n_actions,
                              nonnegative=True)
        if kind == "global":
            return (np.broadcast_to(global_quality, (len(X), n_actions)).copy(),
                    np.broadcast_to(global_cost, (len(X), n_actions)).copy())
        if kind == "fuzzy":
            weights = rule_weights(X, model["widths"])
            quality = _matrix(model["rule_quality"], "rule_quality", n_actions,
                              unit_interval=True)
            cost = _matrix(model["rule_cost"], "rule_cost", n_actions)
            if len(quality) != 3**n_features or len(cost) != 3**n_features:
                raise ValueError(f"fuzzy model must contain exactly {3**n_features} rule consequents")
            if np.any(cost < 0):
                raise ValueError("model rule_cost must be nonnegative")
            return np.clip(weights @ quality, 0, 1), np.maximum(weights @ cost, 0)
        if kind == "ridge":
            mean = _vector(model["feature_mean"], "feature_mean", n_features,
                           unit_interval=True)
            scale = _vector(model["feature_scale"], "feature_scale", n_features)
            if np.any(scale <= 0):
                raise ValueError("model feature_scale must be strictly positive")
            coefficients = _matrix(model["coefficients"], "coefficients", 2 * n_actions)
            if len(coefficients) != n_features:
                raise ValueError(f"model coefficients must have {n_features} feature rows")
            intercept = _vector(model["intercept"], "intercept", 2 * n_actions)
            raw = ((X - mean) / scale) @ coefficients + intercept
            if not np.all(np.isfinite(raw)):
                raise ValueError("model produced non-finite predictions")
            return np.clip(raw[:, :n_actions], 0, 1), np.maximum(raw[:, n_actions:], 0)
    except KeyError as error:
        raise ValueError(f"model is missing predictive parameter {error.args[0]}") from error
    raise ValueError(f"unknown model kind {kind!r}")
