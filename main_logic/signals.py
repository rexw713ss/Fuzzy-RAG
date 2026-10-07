"""Routing signals S1-S4 and their rescaling (spec v0.2, section 2).

Every raw signal is oriented so that a higher value means "escalate more".
Raw values are rescaled to [0, 1] with percentiles fitted on training clusters
only (fit_scaler / apply_scaler).
"""

import numpy as np

EPS = 1e-9
K = 10           # signal window
POOL = 50        # fused pool size
RBO_P = 0.9
TAU = 0.1        # S4 softmax temperature


def _minmax(scores):
    """Min-max normalize a dict {doc_id: score}. A constant list maps to all 1.0."""
    v = np.array(list(scores.values()), dtype=float)
    lo, hi = v.min(), v.max()
    if hi - lo < EPS:
        return {d: 1.0 for d in scores}
    return {d: (s - lo) / (hi - lo) for d, s in scores.items()}


def fuse(bm25, dense, pool=POOL):
    """Equal-weight min-max fusion (section 2.1). Not RRF.

    bm25, dense: {doc_id: raw score} for each retriever's own top-50.
    A document missing from one retriever's list scores 0 for that retriever.
    Returns [(doc_id, fused_score), ...] sorted descending, truncated to `pool`.
    """
    nb, nd = _minmax(bm25), _minmax(dense)
    fused = {d: 0.5 * nb.get(d, 0.0) + 0.5 * nd.get(d, 0.0) for d in set(nb) | set(nd)}
    # Sort by score, then doc_id so ties are deterministic.
    ranked = sorted(fused.items(), key=lambda x: (-x[1], str(x[0])))
    return ranked[:pool]


def s1_margin(scores, k=K):
    """S1 = 1 - (s1 - s2) / (s1 - s_k + eps)  (section 2.2).

    scores: fused scores sorted descending, at least k of them.
    """
    s = np.asarray(scores, dtype=float)
    if len(s) < k:
        raise ValueError(f"need at least {k} scores, got {len(s)}")
    return float(1.0 - (s[0] - s[1]) / (s[0] - s[k - 1] + EPS))


def rbo_ext(a, b, p=RBO_P, k=K):
    """Extrapolated rank-biased overlap of two rankings at depth k (section 2.3).

    RBO_EXT = (1-p) * sum_{d=1..k} p^(d-1) * A_d  +  p^k * A_k,
    where A_d = |a[:d] & b[:d]| / d. Identical lists give exactly 1.
    """
    a, b = list(a)[:k], list(b)[:k]
    if len(a) < k or len(b) < k:
        raise ValueError(f"both rankings need at least {k} items")
    total, seen_a, seen_b, overlap = 0.0, set(), set(), 0
    for d in range(1, k + 1):
        x, y = a[d - 1], b[d - 1]
        # Incremental overlap: count each new match once.
        if x == y:
            overlap += 1
        else:
            overlap += (x in seen_b) + (y in seen_a)
        seen_a.add(x)
        seen_b.add(y)
        total += p ** (d - 1) * overlap / d
    a_k = overlap / k
    return (1 - p) * total + p ** k * a_k


def s2_disagreement(bm25_top, dense_top, p=RBO_P, k=K):
    """S2 = 1 - RBO_EXT between the two retrievers' own top-k lists."""
    return float(1.0 - rbo_ext(bm25_top, dense_top, p, k))


def s3_dispersion(emb, k=K):
    """S3 = mean pairwise cosine distance among the top-k passage embeddings (section 2.4).

    emb: array (k, dim), rows in rank order.
    """
    e = np.asarray(emb, dtype=float)[:k]
    if len(e) < k:
        raise ValueError(f"need at least {k} embeddings, got {len(e)}")
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + EPS)
    cos = e @ e.T
    iu = np.triu_indices(k, 1)   # the k(k-1)/2 distinct pairs
    return float(1.0 - cos[iu].mean())


def s4_entropy(scores, k=K, tau=TAU):
    """S4 = normalized entropy of softmax(s / tau) over the top-k fused scores (section 2.5)."""
    s = np.asarray(scores, dtype=float)[:k]
    if len(s) < k:
        raise ValueError(f"need at least {k} scores, got {len(s)}")
    z = (s - s.max()) / tau      # shift for numerical stability
    p = np.exp(z) / np.exp(z).sum()
    h = -(p * np.log(p + EPS)).sum()
    return float(h / np.log(k))


def compute_signals(fused, bm25_top, dense_top, emb, k=K):
    """All four raw signals for one query instance.

    fused: output of fuse(); bm25_top, dense_top: each retriever's own top-k doc ids;
    emb: embeddings of the top-k fused passages, in fused rank order.
    """
    scores = [s for _, s in fused]
    return {
        "S1": s1_margin(scores, k),
        "S2": s2_disagreement(bm25_top, dense_top, k=k),
        "S3": s3_dispersion(emb, k),
        "S4": s4_entropy(scores, k),
    }


def fit_scaler(values):
    """Fit p5/p95 on training-split raw values for one signal and one dataset (section 2.6)."""
    v = np.asarray(values, dtype=float)
    return {"p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}


def apply_scaler(values, scaler):
    """Rescale to [0, 1] and clip. Returns (scaled, fraction clipped to 0 or 1).

    A value counts as clipped only if it falls strictly outside [0, 1] before clipping, so a
    value exactly at p5 and one exactly at p95 are treated alike (spec v0.3 O-5 / review D1).
    """
    v = np.asarray(values, dtype=float)
    raw = (v - scaler["p5"]) / (scaler["p95"] - scaler["p5"] + EPS)
    scaled = np.clip(raw, 0.0, 1.0)
    clipped = float(np.mean((raw < 0.0) | (raw > 1.0)))
    return scaled, clipped
