"""Stage 1 checks on the pilot's signals (spec v0.3, section 8.2 stop conditions, section 2.6).

Stage 1 exit (section 8): signals vary meaningfully and separate easy from hard queries.
Section 8.2 halts the pipeline if any of these occur:
  - a signal contains NaN or Inf;
  - more than 80% of a signal's values are identical;
  - more than 80% of a signal's values clip to 0 or 1 (section 2.6 adds: revisit above 20%);
  - two signals correlate above 0.90 with no redundancy analysis;
  - S3 behaves in opposite directions on HotpotQA and NQ without explanation;
  - all calibrated widths equal the fixed width (report; do not claim calibration helped).

flags() turns a per-dataset summary into a list of findings, each with a level:
  "stop"     a section 8.2 halt condition;
  "revisit"  section 2.6's 20% clipping limit;
  "explain"  S3's direction differs between datasets, or all widths at the floor: report it.
Correlation is Spearman's (rank) correlation, the measure used in the smoke tests; Pearson's is
reported alongside.
"""

import numpy as np

SIGNALS = ("S1", "S2", "S3", "S4")
MAX_IDENTICAL = 0.80
MAX_CLIPPED = 0.80          # section 8.2
REVISIT_CLIPPED = 0.20      # section 2.6
MAX_CORRELATION = 0.90


def most_common_share(values):
    """Share of values equal to the most common value (rounded to 12 decimals)."""
    v = np.round(np.asarray(values, dtype=float), 12)
    if v.size == 0:
        return float("nan")
    _, counts = np.unique(v, return_counts=True)
    return float(counts.max() / v.size)


def ranks(x):
    """Average ranks (1-based), ties sharing their mean rank."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def correlations(X):
    """{"S1-S2": {"spearman", "pearson"}, ...} for the columns of X (instances, 4)."""
    X = np.asarray(X, dtype=float)
    p = np.corrcoef(X.T)
    s = np.corrcoef(np.array([ranks(X[:, j]) for j in range(X.shape[1])]))
    return {f"{a}-{b}": {"spearman": float(s[i, j]), "pearson": float(p[i, j])}
            for i, a in enumerate(SIGNALS) for j, b in enumerate(SIGNALS) if i < j}


def auc(score, positive):
    """P(score of a random positive > score of a random negative), ties counted half.
    0.5 = no separation; above 0.5 = higher on the positives. nan if a class is empty."""
    positive = np.asarray(positive, dtype=bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks(score)[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def flags(summary):
    """Findings for {dataset: {"signals": {S: {"non_finite", "most_common_share", "clipped",
    "auc_hard"}}, "correlations": correlations(...), "all_at_floor": bool}}.

    Returns a list of {"level", "dataset", "check", "detail"}.
    """
    out = []

    def add(level, dataset, check, detail):
        out.append({"level": level, "dataset": dataset, "check": check, "detail": detail})

    for ds, d in summary.items():
        for s in SIGNALS:
            x = d["signals"][s]
            if x["non_finite"]:
                add("stop", ds, "non-finite", f"{s}: {x['non_finite']} NaN/Inf values")
            if x["most_common_share"] > MAX_IDENTICAL:
                add("stop", ds, "identical", f"{s}: {x['most_common_share']:.0%} identical")
            if x["clipped"] > MAX_CLIPPED:
                add("stop", ds, "clipped", f"{s}: {x['clipped']:.0%} clipped to 0 or 1")
            elif x["clipped"] > REVISIT_CLIPPED:
                add("revisit", ds, "clipped", f"{s}: {x['clipped']:.0%} clipped to 0 or 1")
        for pair, c in d["correlations"].items():
            if abs(c["spearman"]) > MAX_CORRELATION:
                add("stop", ds, "correlation", f"{pair}: Spearman {c['spearman']:.2f}")
        if d.get("all_at_floor"):
            add("explain", ds, "calibration", "all four widths at the fixed width 0.20")

    a = {ds: d["signals"]["S3"]["auc_hard"] for ds, d in summary.items()}
    if len(a) == 2 and all(np.isfinite(v) for v in a.values()):
        (d1, v1), (d2, v2) = a.items()
        if (v1 - 0.5) * (v2 - 0.5) < 0:
            add("explain", "both", "S3 direction",
                f"S3 AUC for hard: {d1} {v1:.2f}, {d2} {v2:.2f} (opposite sides of 0.5)")
    return out
