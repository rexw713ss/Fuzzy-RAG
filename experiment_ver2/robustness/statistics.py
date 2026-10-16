"""Cluster bootstrap sensitivity for frozen batch policies, not new confirmation.

Three estimands are deliberately distinct: original frozen assignments against
the original random-mixture fraction; frozen assignments against each draw's
realized fraction; and a fresh top-quota allocation of each resampled batch.
No version retrains or selects a model on test labels.
"""

import hashlib
import math

import numpy as np


METHOD = "fuzzy-calibrated"
REFERENCES = ("random", "always-A1", "ridge", "fuzzy-fixed")


def cluster_counts(groups, repetitions, seed):
    if not isinstance(repetitions, int) or repetitions < 2:
        raise ValueError("repetitions must be an integer >= 2")
    ids, inverse = np.unique(groups, return_inverse=True)
    if not len(ids):
        raise ValueError("groups must not be empty")
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(ids), size=(repetitions, len(ids)))
    counts = np.zeros((repetitions, len(ids)), dtype=np.int64)
    np.add.at(counts, (np.arange(repetitions)[:, None], draws), 1)
    return counts, inverse


def reranked_counts(scores, questions, multiplicities, quota=.25, *, fallback=False):
    """Number of A2 copies for every original row in each resampled batch.

    Equivalent to materializing repeated clusters and applying quota_actions.
    Ties between copies of the exact same row are interchangeable in outcomes.
    Only strictly positive utility scores are eligible.
    """
    scores, mult = np.asarray(scores, float), np.asarray(multiplicities)
    if scores.shape != (len(questions),) or not np.isfinite(scores).all():
        raise ValueError("finite scores required for each question")
    if len(set(questions)) != len(questions):
        raise ValueError("original question strings must be unique for compressed tie-breaking; repeated copies of a row are allowed via multiplicities")
    if mult.ndim != 2 or mult.shape[1] != len(scores) or mult.dtype.kind not in "iu" or np.any(mult < 0):
        raise ValueError("nonnegative integer multiplicities required")
    if not np.isfinite(quota) or not 0 <= quota <= 1:
        raise ValueError("quota must lie in [0,1]")
    result = np.zeros_like(mult)
    if fallback:
        return result
    ties = [hashlib.sha256(str(q).encode()).hexdigest() for q in questions]
    order = np.asarray(sorted(range(len(scores)), key=lambda i: (-scores[i], ties[i], i)))
    ordered = mult[:, order]
    previous = np.cumsum(ordered, axis=1) - ordered
    limit = np.floor(quota * mult.sum(axis=1) + 1e-12).astype(int)
    chosen = np.clip(limit[:, None] - previous, 0, ordered)
    chosen[:, scores[order] <= 0] = 0
    result[:, order] = chosen
    return result


def interval(samples, point, family_size=8):
    if not isinstance(family_size, int) or family_size < 1:
        raise ValueError("family_size must be positive integer")
    samples = np.asarray(samples, float)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("finite one-dimensional samples required")
    lo, hi = np.quantile(samples, [.025, .975])
    adjusted = np.quantile(samples, [.025 / family_size, 1 - .025 / family_size])
    return {"difference": float(point), "percentile_95": [float(lo), float(hi)],
            "bonferroni_percentile": adjusted.tolist(), "bonferroni_marginal_confidence": 1 - .05 / family_size,
            "bootstrap_sd": float(np.std(samples, ddof=1)),
            "scope": "approximate exploratory percentile intervals; multiplicity correction cannot repair reused test data"}


def _draw_statistics(split, assignments, chosen_scores, fits, n_boot, seed):
    counts, inv = cluster_counts(split.groups, n_boot, seed)
    mult = counts[:, inv]
    totals = mult.sum(axis=1)
    base, escalation = (split.actions.index(a) for a in ("A1", "A2"))
    baseline = mult @ split.f1[:, base] / totals
    mean_gain = mult @ (split.f1[:, escalation] - split.f1[:, base]) / totals
    fixed, fractions, reranked, reranked_fractions = {}, {}, {}, {}
    for method in (METHOD, "ridge", "fuzzy-fixed"):
        actions = np.asarray(assignments[method])
        fixed[method] = mult @ split.f1[np.arange(len(split)), actions] / totals
        fractions[method] = mult @ (actions == escalation).astype(float) / totals
        selected = reranked_counts(chosen_scores[method], split.questions, mult,
                                  quota=fits[method]["quota"], fallback=fits[method]["candidate"] == "always-A1")
        reranked[method] = baseline + selected @ (split.f1[:, escalation] - split.f1[:, base]) / totals
        reranked_fractions[method] = selected.sum(axis=1) / totals
    original_p = float(np.mean(assignments[METHOD] == escalation))
    output = {}
    for label, values, p in (("frozen_fixed_fraction", fixed, original_p),
                              ("frozen_matched_fraction", fixed, fractions[METHOD]),
                              ("reranked_batch", reranked, reranked_fractions[METHOD])):
        references = {"random": baseline + p * mean_gain, "always-A1": baseline,
                      "ridge": values["ridge"], "fuzzy-fixed": values["fuzzy-fixed"]}
        output[label] = {ref: values[METHOD] - v for ref, v in references.items()}
    return output


def bootstrap_sensitivity(dataset, assignments, chosen_scores, fits, n_boot=10000, seed=20261016, family_size=8):
    split = dataset.test
    base, escalation = (split.actions.index(a) for a in ("A1", "A2"))
    values = {m: split.f1[np.arange(len(split)), a] for m, a in assignments.items()}
    p = float(np.mean(assignments[METHOD] == escalation))
    references = {"random": split.f1[:, base] + p * (split.f1[:, escalation] - split.f1[:, base]),
                  "always-A1": split.f1[:, base], "ridge": values["ridge"], "fuzzy-fixed": values["fuzzy-fixed"]}
    points = {ref: float(np.mean(values[METHOD] - v)) for ref, v in references.items()}
    draws = _draw_statistics(split, assignments, chosen_scores, fits, n_boot, seed)
    result = {"n_boot": n_boot, "seed": seed, "independent_sampling_unit": "question cluster",
              "family_size": family_size,
              "family_definition": "four F1 comparisons times two datasets, separately within each bootstrap estimand; does not cover all exploratory analyses",
              "fits": "TRAIN/VAL fits and hyperparameters frozen in all three estimands",
              "random_reference": "expected uniform allocation; allocation randomness integrated, not simulated",
              "estimands": {mode: {ref: interval(x, points[ref], family_size) for ref, x in comparisons.items()}
                            for mode, comparisons in draws.items()}}
    result["monte_carlo_seed_sensitivity"] = []
    for check_seed in (42, 314159):
        check = _draw_statistics(split, assignments, chosen_scores, fits, 2000, check_seed)
        result["monte_carlo_seed_sensitivity"].append({"seed": check_seed, "n_boot": 2000,
            "frozen_matched_fraction": {ref: np.quantile(x, [.025, .975]).tolist()
                                        for ref, x in check["frozen_matched_fraction"].items()}})
    return result


def subset_diagnostics(dataset, assignments, cluster_metadata=None):
    """Preserve frozen full-batch assignments when changing the scoring population."""
    split = dataset.test
    base, escalation = (split.actions.index(a) for a in ("A1", "A2"))
    ids, inv, sizes = np.unique(split.groups, return_inverse=True, return_counts=True)
    original = np.array([k[2] == 0 for k in split.keys])
    subsets = {"all": np.ones(len(split), bool), "original_only": original,
               "paraphrase_only": ~original, "complete_clusters_only": sizes[inv] == 3,
               "incomplete_clusters_only": sizes[inv] != 3}
    if cluster_metadata:
        types = np.array([cluster_metadata[k[1]].get("type", "unspecified") for k in split.keys])
        for label in sorted(set(types)):
            subsets[f"question_type={label}"] = types == label
    results = {}
    row = np.arange(len(split))
    for label, mask in subsets.items():
        if not mask.any():
            continue
        groups = split.groups[mask]
        _, sub_inv, sub_size = np.unique(groups, return_inverse=True, return_counts=True)
        cluster_weights = 1 / sub_size[sub_inv]
        block = {"queries": int(mask.sum()), "clusters": len(sub_size), "methods": {}}
        for method in (METHOD, "ridge", "fuzzy-fixed"):
            actions = assignments[method]
            quality = split.f1[row, actions][mask]
            p = np.mean(actions[mask] == escalation)
            gain = split.f1[mask, escalation] - split.f1[mask, base]
            diff = quality - split.f1[mask, base] - p * gain
            block["methods"][method] = {"F1_query_weighted": float(quality.mean()),
                "F1_cluster_equal_weighted": float(np.average(quality, weights=cluster_weights)),
                "A2_fraction_in_subset": float(p),
                "gain_vs_subset_matched_random_query_weighted": float(diff.mean()),
                "gain_vs_subset_matched_random_cluster_equal_weighted": float(np.average(diff, weights=cluster_weights))}
        # Calculate a separate truly cluster-equal allocation reference: random
        # probability equals the cluster-equal weighted observed A2 fraction.
        for method, entry in block["methods"].items():
            p_cluster = np.average(assignments[method][mask] == escalation, weights=cluster_weights)
            actual = split.f1[row, assignments[method]][mask]
            diff = actual - split.f1[mask, base] - p_cluster * (split.f1[mask, escalation] - split.f1[mask, base])
            entry["cluster_equal_A2_fraction"] = float(p_cluster)
            entry["gain_vs_cluster_equal_fraction_random"] = float(np.average(diff, weights=cluster_weights))
        results[label] = block
    # Original-only scores compare the same 150 source questions, even when
    # screening produced unequal numbers of variants. No incomplete data dropped.
    return {"assignment_scope": "frozen full-batch actions; subset usage may exceed q=.25; no subset-specific retuning",
            "intervals": "descriptive only; subsets are neither independent nor prespecified confirmatory endpoints",
            "subsets": results}


def influence_and_planning(dataset, assignments):
    split = dataset.test
    base, escalation = (split.actions.index(a) for a in ("A1", "A2"))
    ids, inv, sizes = np.unique(split.groups, return_inverse=True, return_counts=True)
    quality = split.f1[np.arange(len(split)), assignments[METHOD]]
    chosen = (assignments[METHOD] == escalation).astype(float)
    all_gain = split.f1[:, escalation] - split.f1[:, base]
    sums = lambda a: np.bincount(inv, weights=a, minlength=len(ids))
    residual = sums(quality - split.f1[:, base])
    gains, counts = sums(all_gain), sums(chosen)
    remaining = len(split) - sizes
    loo = ((residual.sum() - residual) / remaining -
           (counts.sum() - counts) / remaining * (gains.sum() - gains) / remaining)
    influential = np.argsort(np.abs(loo - (residual.sum() / len(split) - chosen.mean() * all_gain.mean())))[-5:][::-1]
    ridge_diff = quality - split.f1[np.arange(len(split)), assignments["ridge"]]
    cluster_difference = sums(ridge_diff) / sizes
    sd = float(cluster_difference.std(ddof=1))
    # z_.975 + z_.80, two-sided unadjusted approximate normal planning.
    future = {str(delta): max(2, math.ceil(((1.959963984540054 + .8416212335729143) * sd / delta) ** 2)) if sd > 0 else None
              for delta in (.01, .02, .03)}
    return {"leave_one_cluster_out_matched_random": {
                "minimum": float(loo.min()), "maximum": float(loo.max()),
                "top_five_influential": [{"group": str(ids[i]), "difference_without_group": float(loo[i])} for i in influential]},
            "future_sample_size_sensitivity": {
                "comparison": "fuzzy-calibrated versus ridge; equal cluster means; fixed fitted policies",
                "observed_cluster_difference_sd": sd, "target_F1_difference_to_independent_clusters": future,
                "power": .8, "two_sided_alpha": .05,
                "limitations": "rough normal approximation using noisy exploratory variance; not observed power or a guarantee; does not cover model-selection, generation, multiple endpoints or dataset shift; plan new independent data conservatively"}}
