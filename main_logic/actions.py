"""The three actions (spec v0.3, section 1). So far: A2's merge rule and its fallback test.

A2 rewrites the question and runs a second hybrid retrieval, giving a second fused top-50 pool.
merge_pools combines the two pools before the cross-encoder (section 1.2); rewrite_usable decides
when A2 must fall back to A1 instead.
"""

import numpy as np

from main_logic import signals as sg

A2_RERANK = 30      # passages the cross-encoder re-scores after the A2 merge (section 1)


def merge_pools(original, rewrite, k=A2_RERANK):
    """A2 merge rule (section 1.2). Returns the merged top-k as [(doc_id, score), ...].

    original, rewrite: fused pools as returned by signals.fuse, [(doc_id, score), ...].
    1. Union of both pools.
    2. A passage in both keeps max(score_original, score_rewrite), so ids are unique.
    3. Min-max normalize over the whole merged pool (signals' rule: a constant pool maps to 1.0).
    4. Sort descending, ties broken by doc id as in signals.fuse, and keep the top k.

    Step 3 is a straight-line rescaling, so it cannot change which passages make the top k or
    their order; the cross-encoder re-scores them anyway. It is kept because the spec asks for it.
    """
    best = {}
    for doc, score in list(original) + list(rewrite):
        score = float(score)
        if not np.isfinite(score):
            raise ValueError(f"non-finite score {score} for passage {doc}")
        if doc not in best or score > best[doc]:
            best[doc] = score
    if not best:
        return []
    scaled = sg._minmax(best)
    ranked = sorted(scaled.items(), key=lambda x: (-x[1], str(x[0])))
    return ranked[:k]


def rewrite_usable(original, rewrite):
    """False when A2 must fall back to A1 (section 1.2).

    Unusable: the rewrite call failed (rewrite is None), the rewrite is empty, or it is the
    original question once whitespace runs and letter case are ignored (spec v0.3: a rewrite that
    only changes capitalization or spacing is not a rewrite). A fallback still counts as an A2
    instance, with a flag in the results.
    """
    if rewrite is None:
        return False
    norm = " ".join(rewrite.split()).lower()
    return bool(norm) and norm != " ".join(original.split()).lower()
