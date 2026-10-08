"""Evaluation (spec v0.3, section 6): oracle action, routing flip rates, within-cluster robustness,
bootstrap confidence intervals and paired comparisons.

Everything reads the cached action-outcome rows held by routing.OutcomeTable (section 7.4), so no
metric costs a model call.

Choices the spec leaves open (user decisions, Oct 8, 2026):
  1. The oracle's "cheapest" action follows the fixed cost order A0 < A1 < A2 (section 1), not
     measured latency, which is noisy.
  2. HRFR's denominator is all pairs, so HRFR <= RFR.
  3. Flip rates use complete clusters only (3 phrasings, 3 pairs); incomplete ones are skipped
     and counted, the same rule as calibration.
  4. Confidence intervals: 95%, percentile method (2.5th and 97.5th of the resamples), seed 42.
  5. "Paired cluster-level tests" (6.3) are a paired bootstrap: both systems are scored on the
     same resampled clusters; the difference counts as significant if its 95% CI excludes 0.
  6. Worst-case drop within a cluster = best phrasing's F1 minus worst phrasing's F1.

Comparisons against delta use a 1e-12 tolerance, so a gap of exactly 0.02 counts as 0.02 despite
float rounding (0.52 - 0.02 is not exactly 0.50 in binary).
"""

import itertools

import numpy as np

from main_logic.calibration import CLUSTER_SIZE
from main_logic.routing import ACTIONS

DELTA = 0.02        # oracle tolerance (6.1), also used by HRFR (6.2)
N_BOOT = 1000       # bootstrap resamples over clusters (6.3)
CI_LEVEL = 0.95
SEED = 42
TOL = 1e-12


def _indices(actions, n):
    """Action labels ('A0'...) or indices (0..2) -> int array of length n."""
    a = np.asarray(actions)
    if a.shape != (n,):
        raise ValueError(f"need one action per instance ({n}), got {a.shape}")
    if a.dtype.kind in "US":
        unknown = set(a.tolist()) - set(ACTIONS)
        if unknown:
            raise ValueError(f"unknown action label(s): {sorted(unknown)}")
        return np.array([ACTIONS.index(x) for x in a])
    idx = a.astype(int)
    if not np.all((idx >= 0) & (idx < len(ACTIONS))):
        raise ValueError("action indices must be 0, 1 or 2")
    return idx


# --- 6.1 oracle --------------------------------------------------------------------------


def oracle_actions(table, delta=DELTA):
    """Per instance, the cheapest action with F1 >= best F1 - delta (cost order A0 < A1 < A2)."""
    best = table.f1.max(axis=1, keepdims=True)
    ok = table.f1 >= best - delta - TOL
    return ok.argmax(axis=1)            # first True = cheapest qualifying action


def oracle_shares(table, delta=DELTA):
    """Section 6.1 reporting: shares of instances where A0 suffices, A1 is required, A2 is
    required (these three sum to 1), and where all three actions give identical F1 (a subset of
    'A0 suffices')."""
    a = oracle_actions(table, delta)
    identical = np.all(np.abs(table.f1 - table.f1[:, :1]) <= TOL, axis=1)
    return {"a0_suffices": float(np.mean(a == 0)), "a1_required": float(np.mean(a == 1)),
            "a2_required": float(np.mean(a == 2)), "all_identical": float(np.mean(identical))}


# --- 6.2 robustness ----------------------------------------------------------------------


def complete_clusters(table):
    """({cluster_id: row indices in variant order}, n_skipped). Only clusters with exactly
    CLUSTER_SIZE phrasings are kept."""
    rows = {}
    for i, (cluster, _variant) in enumerate(table.instances):
        rows.setdefault(cluster, []).append(i)
    complete = {c: r for c, r in rows.items() if len(r) == CLUSTER_SIZE}
    return complete, len(rows) - len(complete)


def pair_flags(table, actions, delta=DELTA):
    """Every within-cluster pair of a complete cluster, with whether it flipped and whether the
    flip was harmful. Returns (cluster ids, flipped, harmful), one entry per pair, and n_skipped.

    A pair (i, j) with a_i != a_j is harmful if either
      (a) one query would have scored better with its sibling's action:
          F1_i(a_j) > F1_i(a_i) + delta  or  F1_j(a_i) > F1_j(a_j) + delta;
      (b) the more expensive action bought nothing: on the query that received it, it beats the
          cheaper action by less than delta.
    """
    a = _indices(actions, len(table))
    f1 = table.f1
    clusters, n_skipped = complete_clusters(table)
    ids, flipped, harmful = [], [], []
    for cluster, rows in clusters.items():
        for i, j in itertools.combinations(rows, 2):
            ai, aj = a[i], a[j]
            flip = ai != aj
            harm = False
            if flip:
                better_swap = (f1[i, aj] > f1[i, ai] + delta + TOL
                               or f1[j, ai] > f1[j, aj] + delta + TOL)
                k, hi, lo = (i, ai, aj) if ai > aj else (j, aj, ai)
                paid_for_nothing = f1[k, hi] - f1[k, lo] < delta - TOL
                harm = bool(better_swap or paid_for_nothing)
            ids.append(cluster)
            flipped.append(bool(flip))
            harmful.append(harm)
    return ids, np.array(flipped, dtype=bool), np.array(harmful, dtype=bool), n_skipped


def flip_rates(table, actions, delta=DELTA):
    """RFR and HRFR over all within-cluster pairs of complete clusters (section 6.2)."""
    ids, flipped, harmful, n_skipped = pair_flags(table, actions, delta)
    if not ids:
        raise ValueError(f"no complete clusters ({n_skipped} incomplete skipped)")
    return {"rfr": float(flipped.mean()), "hrfr": float(harmful.mean()), "n_pairs": len(ids),
            "n_clusters": len(set(ids)), "n_skipped": n_skipped}


def within_cluster(table, actions, scores):
    """Section 6.2 'also report', averaged over complete clusters: sd of the escalation score,
    sd of answer F1 under the routed actions, and the worst-case drop (best - worst F1).

    scores: {(cluster_id, variant_id): score} or an array in table.instances order.
    Standard deviations are population sd (ddof = 0) over a cluster's 3 phrasings.
    """
    a = _indices(actions, len(table))
    s = table.align(scores) if isinstance(scores, dict) else np.asarray(scores, dtype=float)
    if s.shape != (len(table),):
        raise ValueError(f"need one score per instance ({len(table)}), got {s.shape}")
    f1 = table.f1[np.arange(len(table)), a]
    clusters, n_skipped = complete_clusters(table)
    if not clusters:
        raise ValueError(f"no complete clusters ({n_skipped} incomplete skipped)")
    sd_s = [s[r].std() for r in clusters.values()]
    sd_f = [f1[r].std() for r in clusters.values()]
    drop = [f1[r].max() - f1[r].min() for r in clusters.values()]
    return {"score_sd": float(np.mean(sd_s)), "f1_sd": float(np.mean(sd_f)),
            "worst_drop": float(np.mean(drop)), "worst_drop_max": float(np.max(drop)),
            "n_clusters": len(clusters), "n_skipped": n_skipped}


# --- 6.3 statistics ----------------------------------------------------------------------


def _cluster_sums(values, groups):
    """Per-cluster sums of each value array and unit counts, in a fixed cluster order."""
    groups = np.asarray(groups)
    _, inv = np.unique(groups, return_inverse=True)
    n_clusters = inv.max() + 1
    counts = np.bincount(inv, minlength=n_clusters).astype(float)
    sums = [np.bincount(inv, weights=np.asarray(v, dtype=float), minlength=n_clusters)
            for v in values]
    return sums, counts


def _resamples(n_clusters, n_boot, seed):
    return np.random.default_rng(seed).integers(0, n_clusters, size=(n_boot, n_clusters))


def _interval(boot, ci):
    tail = (1 - ci) / 2 * 100
    lo, hi = np.percentile(boot, [tail, 100 - tail])
    return float(lo), float(hi)


def bootstrap_ci(values, groups, n_boot=N_BOOT, ci=CI_LEVEL, seed=SEED):
    """Mean of `values` with a cluster bootstrap CI (section 6.3).

    values: one number per unit (an instance's F1 or cost, or a pair's flip flag); groups: the
    cluster of each unit. Whole clusters are resampled with replacement, so a cluster's
    phrasings always stay together. Returns {"mean", "lo", "hi"}.
    """
    values = np.asarray(values, dtype=float)
    if values.shape != np.shape(groups) or values.ndim != 1 or len(values) == 0:
        raise ValueError("values and groups must be non-empty 1-D arrays of the same length")
    (sums,), counts = _cluster_sums([values], groups)
    idx = _resamples(len(counts), n_boot, seed)
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = _interval(boot, ci)
    return {"mean": float(values.mean()), "lo": lo, "hi": hi}


def paired_bootstrap(values_a, values_b, groups, n_boot=N_BOOT, ci=CI_LEVEL, seed=SEED):
    """Paired cluster-level comparison of two systems on the same units (section 6.3).

    Both are scored on the same resampled clusters; returns the mean difference a - b, its CI,
    and whether the CI excludes 0.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape or a.shape != np.shape(groups) or a.ndim != 1 or len(a) == 0:
        raise ValueError("values_a, values_b and groups must be non-empty 1-D arrays "
                         "of the same length")
    (sa, sb), counts = _cluster_sums([a, b], groups)
    idx = _resamples(len(counts), n_boot, seed)
    n = counts[idx].sum(axis=1)
    boot = (sa[idx].sum(axis=1) - sb[idx].sum(axis=1)) / n
    lo, hi = _interval(boot, ci)
    return {"diff": float(a.mean() - b.mean()), "lo": lo, "hi": hi,
            "significant": bool(lo > 0 or hi < 0)}
