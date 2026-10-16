"""TRAIN/VAL cluster-resampling sensitivity on one unchanged test batch.

The resampling distribution diagnoses dependence on the observed training and
validation clusters. Its percentiles are NOT confidence intervals: test rows,
archived scalers, model families and the primary quota stay fixed. Test outcomes
are read only after all resampled policies have been selected and applied.
"""

from collections import Counter
from dataclasses import replace
import hashlib
import json

import numpy as np

from experiment_ver2 import models
from experiment_ver2.run import PRIORS, RIDGE_ALPHAS, PRIMARY_BUDGET, apply_policy, freeze_policy
from main_logic import calibration


METHODS = ("global", "ridge", "fuzzy-fixed", "fuzzy-calibrated")
PERCENTILES = (2.5, 50.0, 97.5)


def _cluster_rows(split):
    groups = split.groups
    return [np.flatnonzero(groups == group) for group in np.unique(groups)]


def _draw_training(rows, n_instances, rng):
    """One draw per original cluster, represented by multiplicity / cluster size."""
    draws = rng.integers(len(rows), size=len(rows))
    counts = np.bincount(draws, minlength=len(rows))
    weights = np.zeros(n_instances)
    for block, count in zip(rows, counts):
        weights[block] = count / len(block)
    return draws, weights


def _draw_validation(split, rows, rng):
    """Materialize sampled cluster copies; top-q is applied to this entire batch.

    Duplicate copies retain their question text, features and outcomes. The
    existing quota policy uses question hash then row position for exact ties;
    copies may receive different actions at a boundary, as repeated arrivals.
    """
    draws = rng.integers(len(rows), size=len(rows))
    indices = np.concatenate([rows[i] for i in draws])
    sampled = replace(split, X=split.X[indices], f1=split.f1[indices],
                      cost=split.cost[indices], em=split.em[indices],
                      keys=tuple(split.keys[i] for i in indices),
                      questions=tuple(split.questions[i] for i in indices),
                      secondary={k: v[indices] for k, v in split.secondary.items()})
    return draws, indices, sampled


def _fuzzy_from_basis(train, weights, widths, basis, prior):
    """The exact models.fit sufficient statistics, reusing a label-free basis.

    Equivalence to models.fit is regression-tested, including zero-weight rows.
    The cache eliminates repeated membership evaluation for each prior/draw.
    """
    X, quality, cost, weights = models._training(train.X, train.f1, train.cost, weights)
    model = models._common("fuzzy", X, quality, cost, weights)
    weighted = basis * weights[:, None]
    support = weighted.sum(axis=0)
    denominator = support + prior

    def consequent(values, global_mean, unit_interval=False):
        mean = np.broadcast_to(global_mean, (basis.shape[1], values.shape[1])).copy()
        numerator = weighted.T @ values + prior * global_mean
        np.divide(numerator, denominator[:, None], out=mean, where=denominator[:, None] > 0)
        return (np.clip(mean, 0, 1) if unit_interval else mean).tolist()

    square_mass = np.square(weighted).sum(axis=0)
    effective = np.divide(np.square(support), square_mass, out=np.zeros_like(support),
                          where=square_mass > 0)
    model.update({"widths": list(map(float, widths)), "prior_strength": float(prior),
                  "rule_support": support.tolist(), "rule_effective_rows": effective.tolist(),
                  "rule_quality": consequent(quality, np.asarray(model["global_quality"]), True),
                  "rule_cost": consequent(cost, np.asarray(model["global_cost"]))})
    return model


def _fit_candidates(train, weights, widths, bases):
    fitted = {"global": {"global": models.fit_global(train.X, train.f1, train.cost,
                                                      sample_weight=weights)},
              "ridge": {}, "fuzzy-fixed": {}}
    for alpha in RIDGE_ALPHAS:
        fitted["ridge"][f"alpha={alpha}"] = models.fit_ridge(
            train.X, train.f1, train.cost, alpha=alpha, sample_weight=weights)
    variants = [("fuzzy-fixed", np.full(4, .2))]
    if widths is not None:
        fitted["fuzzy-calibrated"] = {}
        variants.append(("fuzzy-calibrated", widths))
    for method, w in variants:
        for prior in PRIORS:
            fitted[method][f"prior={prior}"] = _fuzzy_from_basis(
                train, weights, w, bases[method]["train"], prior)
    return fitted


def _scores(fitted, split, split_name, bases, indices=None):
    base, escalation = (split.actions.index(a) for a in ("A1", "A2"))
    scores = {}
    for method, variants in fitted.items():
        scores[method] = {}
        for name, model in variants.items():
            if model["kind"] == "fuzzy":
                basis = bases[method][split_name]
                if indices is not None:
                    basis = basis[indices]
                quality = np.clip(basis @ np.asarray(model["rule_quality"]), 0, 1)
            else:
                quality, _ = models.predict(model, split.X)
            scores[method][name] = quality[:, escalation] - quality[:, base]
    return scores


def _fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _select_and_apply(dataset, train_weights, widths, bases, validation, val_indices=None):
    """No test outcome access: fit TRAIN, select VAL, predict TEST features."""
    fitted = _fit_candidates(dataset.train, train_weights, widths, bases)
    val_scores = _scores(fitted, validation, "val", bases, val_indices)
    selected = {method: freeze_policy(method, variants, validation, PRIMARY_BUDGET)
                for method, variants in val_scores.items()}
    test_scores = _scores(fitted, dataset.test, "test", bases)
    output = {}
    for method, fit in selected.items():
        actions = apply_policy(method, fit, test_scores[method], dataset.test.questions, dataset.actions)
        chosen = fitted[method].get(fit["candidate"])
        output[method] = {"fit": fit, "actions": actions.tolist(),
                          "candidate_models_sha256": _fingerprint(fitted[method]),
                          "selected_model_sha256": _fingerprint(chosen) if chosen is not None else None}
    return output


def _metrics(split, actions, original_actions):
    base, escalation = (split.actions.index(a) for a in ("A1", "A2"))
    actions = np.asarray(actions)
    selected = split.f1[np.arange(len(split)), actions]
    fraction = float(np.mean(actions == escalation))
    random = (1 - fraction) * split.f1[:, base] + fraction * split.f1[:, escalation]
    return {"F1": float(selected.mean()),
            "gain_vs_always_A1": float(np.mean(selected - split.f1[:, base])),
            "gain_vs_random_same_A2_count": float(np.mean(selected - random)),
            "A2_fraction": fraction, "A2_count": int(np.sum(actions == escalation)),
            "cached_cost": float(split.cost[np.arange(len(split)), actions].mean()),
            "action_agreement_with_original": float(np.mean(actions == np.asarray(original_actions)))}


def _distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()), "max": float(values.max()),
            "percentiles": {str(p): float(v) for p, v in zip(PERCENTILES, np.percentile(values, PERCENTILES))}}


def run_stability(dataset, repetitions=100, seed=20261016):
    """Return reproducible TRAIN+VAL cluster-bootstrap stability diagnostics.

    The primary four-signal family, hyperparameter grids and q=.25 are fixed.
    Each TRAIN cluster copy contributes one unit of sample weight. VAL copies
    are materialized and scored as an instance-weighted repeated batch. Seeds
    for TRAIN and VAL use separate NumPy SeedSequence child streams.

    A draw with no complete TRAIN cluster has no calibrated-width fit; it is
    recorded as a failure for that method, never redrawn or silently replaced.
    All decisions are fixed before any TEST F1/cost summary is computed.
    """
    if isinstance(repetitions, (bool, np.bool_)) or not isinstance(repetitions, (int, np.integer)) or repetitions < 2:
        raise ValueError("repetitions must be an integer >= 2")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if tuple(dataset.signals) != ("S1", "S2", "S3", "S4"):
        raise ValueError("stability analysis requires the prespecified four signals S1..S4")
    for split in dataset.splits.values():
        if not len(split) or tuple(split.actions) != tuple(dataset.actions) or tuple(split.signals) != tuple(dataset.signals):
            raise ValueError("all nonempty splits must have the same actions and four signals")
        if split.X.shape != (len(split), 4):
            raise ValueError("split features must have shape (n,4)")
    if "A1" not in dataset.actions or "A2" not in dataset.actions:
        raise ValueError("the experiment must contain A1 and A2")
    train_rows, val_rows = _cluster_rows(dataset.train), _cluster_rows(dataset.val)
    if any(len(block) > 3 for block in train_rows + val_rows):
        raise ValueError("clusters must have at most three variants")
    if not any(len(block) == 3 for block in train_rows):
        raise ValueError("at least one complete TRAIN cluster is required for width calibration")

    fixed = {name: models.rule_weights(split.X, .2) for name, split in dataset.splits.items()}

    def bases_for(widths):
        bases = {"fuzzy-fixed": fixed}
        if widths is not None:
            bases["fuzzy-calibrated"] = (
                fixed if np.array_equal(widths, np.full(4, .2)) else
                {name: models.rule_weights(split.X, widths) for name, split in dataset.splits.items()})
        return bases

    original_weights = np.zeros(len(dataset.train))
    for block in train_rows:
        original_weights[block] = 1 / len(block)
    original = _select_and_apply(dataset, original_weights, dataset.widths,
                                 bases_for(dataset.widths), dataset.val)
    sequences = np.random.SeedSequence(int(seed)).spawn(2)
    train_rng, val_rng = [np.random.default_rng(s) for s in sequences]
    records = {method: [] for method in METHODS}
    resamples = []
    failures = []
    for repetition in range(int(repetitions)):
        train_draws, weights = _draw_training(train_rows, len(dataset.train), train_rng)
        val_draws, val_indices, validation = _draw_validation(dataset.val, val_rows, val_rng)
        clusters = [dataset.train.X[train_rows[i]] for i in train_draws]
        complete = sum(len(block) == 3 for block in clusters)
        width_result = calibration.calibrate(clusters, signals=dataset.signals) if complete else None
        widths = np.asarray(width_result["widths"]) if width_result is not None else None
        policies = _select_and_apply(dataset, weights, widths, bases_for(widths), validation, val_indices)
        if widths is None:
            failures.append({"repetition": repetition, "method": "fuzzy-calibrated",
                             "reason": "sampled TRAIN contains no complete cluster for jitter calibration"})
        resamples.append({"repetition": repetition, "train_cluster_draws": train_draws.tolist(),
                          "val_cluster_draws": val_draws.tolist(),
                          "training_weight": float(weights.sum()), "validation_instances": len(validation),
                          "calibrated_widths": widths.tolist() if widths is not None else None,
                          "calibration_complete_cluster_copies": complete})
        for method, policy in policies.items():
            records[method].append({"repetition": repetition, **policy})

    # Evaluation starts here, after every model, selection and action is frozen.
    methods = {}
    escalation = dataset.actions.index("A2")
    for method in METHODS:
        reference = original[method]
        reference["metrics"] = _metrics(dataset.test, reference["actions"], reference["actions"])
        selected_counts = Counter()
        for record in records[method]:
            record["metrics"] = _metrics(dataset.test, record["actions"], reference["actions"])
            selected_counts[record["fit"]["candidate"]] += 1
        n_valid = len(records[method])
        summaries = {metric: _distribution([r["metrics"][metric] for r in records[method]])
                     for metric in reference["metrics"]}
        action_matrix = np.array([r["actions"] for r in records[method]])
        methods[method] = {"original": reference, "valid_repetitions": n_valid,
                           "failed_repetitions": int(repetitions) - n_valid,
                           "candidate_selection_counts": dict(sorted(selected_counts.items())),
                           "candidate_selection_frequencies": {k: v / n_valid for k, v in sorted(selected_counts.items())},
                           "distributions": summaries, "repetitions": records[method],
                           "test_A2_selection_frequency": (np.mean(action_matrix == escalation, axis=0).tolist()
                                                            if n_valid else None)}
    return {"dataset": dataset.name, "seed": int(seed), "repetitions": int(repetitions),
            "quota": PRIMARY_BUDGET, "signals": list(dataset.signals), "actions": list(dataset.actions),
            "methods": methods, "resamples": resamples, "failures": failures,
            "test_keys": [list(k) for k in dataset.test.keys],
            "train_cluster_order": [str(dataset.train.groups[rows[0]]) for rows in train_rows],
            "val_cluster_order": [str(dataset.val.groups[rows[0]]) for rows in val_rows],
            "hyperparameters": {"priors": list(PRIORS), "ridge_alphas": list(RIDGE_ALPHAS)},
            "interpretation": "stability diagnostics only; percentile ranges are not confirmatory confidence intervals",
            "training_resampling": "whole clusters; each row weight = sampled cluster multiplicity / number of its variants",
            "validation_resampling": "independent whole-cluster draws materialized as repeated rows; quota applied to repeated batch",
            "preprocessing": "scaled features and archived scaler stay fixed; TRAIN jitter widths recomputed per draw",
            "limitation": "raw training signals unavailable: scaler-estimation variation is omitted; the same previously inspected test batch is reused",
            "test_outcomes_usage": "all candidate fitting, validation selection and test assignments finish before test F1/cost evaluation"}
