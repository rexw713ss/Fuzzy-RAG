"""Descriptive diagnostics of already-frozen primary policies; no fitting.

Cluster support averages memberships within each original question before
computing Kish effective counts. This avoids presenting correlated paraphrase
rows as independent evidence. Gain ranks and feature shifts are descriptions,
not tests, causal effects, hyperparameter searches or new policy decisions.
"""

import numpy as np

from experiment_ver2 import models
from main_logic import fuzzy as fz


def _summary(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return {"n": len(values), "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "p25": float(np.percentile(values, 25)),
            "median": float(np.median(values)), "p75": float(np.percentile(values, 75)),
            "max": float(values.max())}


def _average_ranks(values):
    order = np.argsort(values, kind="stable")
    ranked = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranked[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranked


def spearman(scores, gains):
    """Average-rank Spearman rho; undefined constant vectors return None.

    No p value is produced: the rows include related paraphrases.
    """
    scores, gains = np.asarray(scores, dtype=float), np.asarray(gains, dtype=float)
    if scores.ndim != 1 or scores.shape != gains.shape or not len(scores):
        raise ValueError("scores and gains must be nonempty one-dimensional arrays of equal length")
    if not np.isfinite(scores).all() or not np.isfinite(gains).all():
        raise ValueError("scores and gains must be finite")
    if len(np.unique(scores)) < 2 or len(np.unique(gains)) < 2:
        return None
    left, right = _average_ranks(scores), _average_ranks(gains)
    left -= left.mean()
    right -= right.mean()
    return float(np.clip(np.dot(left, right) / np.sqrt(np.dot(left, left) * np.dot(right, right)), -1, 1))


def _quartiles(scores, gains, groups, predicted_gain):
    # Outcome-blind quantiles; all equal scores go to the same bin. Empty bins
    # are retained rather than breaking ties with outcome information.
    cuts = np.percentile(scores, [25, 50, 75])
    bins = np.searchsorted(cuts, scores, side="right")
    result = []
    for index in range(4):
        chosen = bins == index
        score_summary, gain_summary = _summary(scores[chosen]), _summary(gains[chosen])
        result.append({"quartile_low_to_high": index + 1, "n": int(chosen.sum()),
                       "n_clusters": len(np.unique(groups[chosen])),
                       "score": score_summary, "actual_gain": gain_summary,
                       "predicted_gain_mean": score_summary["mean"] if predicted_gain and score_summary else None,
                       "actual_gain_mean": gain_summary["mean"] if gain_summary else None,
                       "benefit_fraction_gain_gt_zero": float(np.mean(gains[chosen] > 0)) if chosen.any() else None,
                       "harm_fraction_gain_lt_zero": float(np.mean(gains[chosen] < 0)) if chosen.any() else None})
    return {"score_cut_values": cuts.tolist(), "groups": result,
            "definition": "nominal score quartiles within this split; ties kept together and equal-to-cut scores enter the higher bin; bins may be empty"}


def _feature_drift(dataset):
    output = {}
    for j, signal in enumerate(dataset.signals):
        training = dataset.train.X[:, j]
        mean, sd = float(training.mean()), float(training.std())
        by_split = {}
        for name, split in dataset.splits.items():
            values = split.X[:, j]
            shift = float(values.mean() - mean)
            by_split[name] = {
                **_summary(values), "n_clusters": len(np.unique(split.groups)),
                "at_zero_percent": float(np.mean(values == 0) * 100),
                "at_one_percent": float(np.mean(values == 1) * 100),
                "at_either_boundary_percent": float(np.mean((values == 0) | (values == 1)) * 100),
                "mean_shift_from_train": shift,
                "mean_shift_in_train_sd": shift / sd if sd > 0 else None,
            }
        output[signal] = by_split
    return {"signals": output, "weighting": "instance-weighted descriptive summaries; standard deviations use ddof=0",
            "boundary_interpretation": "exact scaled 0/1 saturation, including possible natural endpoints; raw clipping cannot be inferred for archived TRAIN/VAL",
            "inference": "no drift hypothesis tests; archived preprocessing is frozen"}


def _cluster_support(train, basis):
    _, inverse, sizes = np.unique(train.groups, return_inverse=True, return_counts=True)
    cluster_mass = np.zeros((len(sizes), basis.shape[1]))
    np.add.at(cluster_mass, inverse, basis / sizes[inverse, None])
    support = cluster_mass.sum(axis=0)
    squared = np.square(cluster_mass).sum(axis=0)
    ess = np.divide(np.square(support), squared, out=np.zeros_like(support), where=squared > 0)
    active = np.sum(cluster_mass > 0, axis=0)
    return support, ess, active, len(sizes)


def _support_diagnostic(dataset, method, variants, fit, actions):
    selected_name = fit["candidate"]
    model = variants.get(selected_name)
    # Every candidate in one fuzzy family shares widths; fallback still has a
    # meaningful feature-space support description, but no selected prior.
    geometry = next(iter(variants.values()))
    widths = geometry["widths"]
    if any(candidate["kind"] != "fuzzy" or candidate["widths"] != widths for candidate in variants.values()):
        raise ValueError(f"{method} candidates must share fuzzy membership widths")
    training_basis = models.rule_weights(dataset.train.X, widths)
    support, ess, active, n_clusters = _cluster_support(dataset.train, training_basis)
    for candidate in variants.values():
        if not np.allclose(candidate["rule_support"], support, rtol=1e-10, atol=1e-12):
            raise ValueError("rule support is inconsistent with one total TRAIN weight per cluster")
    indices = models.rule_index(dataset.train.X.shape[1])
    result = {
        "selected_candidate": selected_name, "widths": widths, "n_train_clusters": n_clusters,
        "sum_rule_support": float(support.sum()),
        "support_conservation_error": float(abs(support.sum() - n_clusters)),
        "unseen_rule_count": int(np.sum(support == 0)),
        "rules_below_one_cluster_equivalent_mass": int(np.sum(support < 1)),
        "rule_support_distribution": _summary(support),
        "cluster_mass_kish_ess_distribution": _summary(ess),
        "rules": [{"rule_id": i + 1, "antecedents": [fz.LABELS[j] for j in indices[i]],
                   "cluster_equivalent_support": float(support[i]),
                   "cluster_mass_kish_ess": float(ess[i]), "clusters_with_nonzero_membership": int(active[i])}
                  for i in range(len(support))],
        "support_definition": "sum over TRAIN clusters of the mean rule membership within each cluster; masses across all rules sum to number of TRAIN clusters",
        "ess_definition": "(sum cluster masses)^2 / sum squared cluster masses; uses original-question clusters, not paraphrase rows; a diagnostic effective count, not a confidence bound",
    }
    if model is None:
        result["selected_model_test_support"] = {
            "status": "not_used", "reason": "validation selected always-A1; no fuzzy model is used by this policy"}
        return result
    prior = float(model["prior_strength"])
    # With a zero prior, an entirely unseen rule still uses the global mean.
    fraction = np.divide(prior, support + prior, out=np.ones_like(support), where=support + prior > 0)
    basis = models.rule_weights(dataset.test.X, widths)
    prior_fraction = basis @ fraction
    weighted_support, weighted_ess = basis @ support, basis @ ess
    unseen_mass = basis @ (support == 0).astype(float)
    chosen = actions == dataset.actions.index("A2")

    def summaries(mask):
        return {"n": int(np.sum(mask)),
                "global_mean_prior_fraction": _summary(prior_fraction[mask]),
                "membership_weighted_cluster_support": _summary(weighted_support[mask]),
                "membership_weighted_cluster_kish_ess": _summary(weighted_ess[mask]),
                "membership_mass_on_unseen_rules": _summary(unseen_mass[mask])}

    result["selected_model_test_support"] = {
        "status": "used", "prior_strength": prior,
        "all_test": summaries(np.ones(len(dataset.test), dtype=bool)), "selected_A2": summaries(chosen),
        "per_query": [{"key": list(key), "action": dataset.actions[int(actions[i])],
                       "global_mean_prior_fraction": float(prior_fraction[i]),
                       "membership_weighted_cluster_support": float(weighted_support[i]),
                       "membership_weighted_cluster_kish_ess": float(weighted_ess[i]),
                       "membership_mass_on_unseen_rules": float(unseen_mass[i])}
                      for i, key in enumerate(dataset.test.keys)],
        "shrinkage_definition": "sum_r membership(x,r) * prior_strength/(TRAIN_support_r+prior_strength); unseen zero-prior rules have coefficient 1",
        "interpretation": "deterministic weight on the training global action means; not a Bayesian posterior probability or uncertainty interval; weighted rule ESS is not a query-level independent sample count",
    }
    return result


def run_diagnostics(dataset, candidates, fits, assignments):
    """Describe already-selected primary policies (fits keyed by method).

    ``candidates`` is run.fit_candidates(dataset); ``fits`` is one primary
    budget's method dictionary; ``assignments`` holds the corresponding TEST
    action indices. This function never modifies inputs, fits or chooses models.
    """
    if set(fits) != set(assignments):
        raise ValueError("fits and assignments must name the same methods")
    for split in dataset.splits.values():
        if not len(split) or split.X.shape != (len(split), len(dataset.signals)):
            raise ValueError("each split needs nonempty, aligned feature rows")
        if not np.isfinite(split.X).all() or np.any((split.X < 0) | (split.X > 1)):
            raise ValueError("scaled signals must be finite and lie in [0,1]")
    base, escalation = (dataset.actions.index(label) for label in ("A1", "A2"))
    ranking, support = {}, {}
    for method, fit in fits.items():
        actions = np.asarray(assignments[method])
        if actions.shape != (len(dataset.test),) or actions.dtype.kind not in "iu" or not np.isin(actions, [base, escalation]).all():
            raise ValueError("assignments must contain one A1/A2 index per TEST row")
        selected = fit["candidate"]
        variants = candidates.get(method)
        if method.startswith("fuzzy-"):
            if not variants:
                raise ValueError(f"missing fuzzy candidates for {method}")
            support[method] = _support_diagnostic(dataset, method, variants, fit, actions)
        if selected == "always-A1":
            if np.any(actions != base):
                raise ValueError("always-A1 fit disagrees with TEST assignments")
            ranking[method] = {"selected_candidate": selected, "status": "not_used",
                               "reason": "constant fallback policy has no selected score or gain ranking"}
            continue
        is_uncertainty = method in ("uncertainty-B3", "uncertainty-B4")
        if not is_uncertainty and (not variants or selected not in variants):
            raise ValueError(f"missing selected model {method}/{selected}")
        by_split = {}
        for name, split in dataset.splits.items():
            if is_uncertainty:
                widths = .2 if method == "uncertainty-B3" else dataset.widths
                scores = np.asarray([fz.escalate(x, widths, signals=dataset.signals)[0] for x in split.X])
            else:
                quality, _ = models.predict(variants[selected], split.X)
                scores = quality[:, escalation] - quality[:, base]
            gains = split.f1[:, escalation] - split.f1[:, base]
            rho = spearman(scores, gains)
            by_split[name] = {"n_instances": len(split), "n_clusters": len(np.unique(split.groups)),
                              "score_unique_values": len(np.unique(scores)), "gain_unique_values": len(np.unique(gains)),
                              "spearman_rho": rho,
                              "undefined_reason": "at least one vector is constant" if rho is None else None,
                              "quartiles": _quartiles(scores, gains, split.groups, not is_uncertainty)}
        ranking[method] = {"selected_candidate": selected, "status": "used",
                           "score_kind": "uncertainty" if is_uncertainty else "predicted_A2_minus_A1_F1",
                           "splits": by_split}
    return {"dataset": dataset.name, "signals": list(dataset.signals), "actions": list(dataset.actions),
            "rule_support": support, "feature_drift": _feature_drift(dataset), "gain_ranking": ranking,
            "interpretation": "descriptive diagnostics of frozen policies; no test-driven tuning, causal attribution or confirmatory inference",
            "ranking_limitation": "instance-weighted Spearman with average tied ranks, no p values; TRAIN is resubstitution, VAL was used for model selection, TEST is previously inspected; cluster dependence is not an independent-row significance test"}
