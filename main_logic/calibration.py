"""Jitter calibration of the band widths (spec v0.3, section 7.3; pipeline step 7).

Jitter is how much a signal moves when only the wording of a question changes. For each signal,
take the absolute difference between every pair of phrasings in each training cluster, pool the
differences over clusters, and take the 75th percentile:

    w_s = clip(jitter_s, 0.20, 0.25)

A width clipped at 0.25 is flagged: the signal is potentially unsuitable as a stable routing
feature (a pre-registered failure condition, section 3.1). The median version is reported as a
sensitivity check. Uses scaled signals only (section 2.6) — no labels, answers or test data.

Only complete clusters (exactly 3 phrasings) are used; incomplete ones are skipped and counted
(user decision, Oct 7), matching section 6.2, which compares 3 pairs per cluster.
"""

import itertools

import numpy as np

from main_logic import fuzzy as fz

W_FLOOR = fz.W_DEFAULT      # 0.20: calibration can only widen the default band
W_CAP = fz.W_MAX            # 0.25: hard cap (section 3.1)
CLUSTER_SIZE = 3
QUANTILE = 75


def pair_differences(clusters):
    """Pooled |S_a - S_b| over all phrasing pairs of every complete cluster.

    clusters: iterable of arrays (n_phrasings, 4) of scaled signals, one per training cluster.
    Returns (diffs of shape (n_pairs, 4), n_used, n_skipped).
    """
    diffs, used, skipped = [], 0, 0
    for c in clusters:
        c = np.asarray(c, dtype=float)
        if c.ndim != 2 or c.shape[1] != 4:
            raise ValueError(f"each cluster must be (n_phrasings, 4), got {c.shape}")
        if not np.all((c >= 0.0) & (c <= 1.0)):
            raise ValueError("signals must be scaled to [0, 1] (and finite) before calibration")
        if len(c) > CLUSTER_SIZE:
            raise ValueError(f"a cluster has {len(c)} phrasings; at most {CLUSTER_SIZE} expected")
        if len(c) < CLUSTER_SIZE:
            skipped += 1
            continue
        used += 1
        diffs.extend(np.abs(c[i] - c[j]) for i, j in itertools.combinations(range(len(c)), 2))
    return np.array(diffs).reshape(-1, 4), used, skipped


def calibrate(clusters):
    """Per-signal widths from jitter (section 7.3), with clipping flagged (section 8.1).

    Returns a dict:
      widths          final w per signal (S1..S4), ready for fuzzy.escalate
      widths_median   the median-based sensitivity version
      signals         per signal: jitter (p75), jitter_median, width, at_floor, capped
      all_at_floor    True if every width stayed 0.20: calibration did not change the
                      controller, so no contribution from it may be claimed (section 5)
      n_clusters, n_skipped
    """
    diffs, used, skipped = pair_differences(clusters)
    if used == 0:
        raise ValueError(f"no complete clusters ({skipped} incomplete skipped)")

    p75 = np.percentile(diffs, QUANTILE, axis=0)
    med = np.median(diffs, axis=0)
    widths = np.clip(p75, W_FLOOR, W_CAP)
    widths_median = np.clip(med, W_FLOOR, W_CAP)

    signals = {
        name: {"jitter": float(p75[i]), "jitter_median": float(med[i]),
               "width": float(widths[i]), "at_floor": bool(p75[i] <= W_FLOOR),
               "capped": bool(p75[i] > W_CAP)}
        for i, name in enumerate(fz.SIGNALS)
    }
    return {
        "widths": widths.tolist(),
        "widths_median": widths_median.tolist(),
        "signals": signals,
        "all_at_floor": all(s["at_floor"] for s in signals.values()),
        "n_clusters": used,
        "n_skipped": skipped,
    }
