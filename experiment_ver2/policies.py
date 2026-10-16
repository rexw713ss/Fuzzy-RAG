"""Outcome-blind, batch-budget A1/A2 rescue policies.

The batch may use all query features, but NEVER its F1, EM, or measured costs.
The quota is a maximum count, not an online latency guarantee.
"""

import hashlib
import math

import numpy as np


def quota_actions(scores, questions, fraction, *, positive_only=True, base=1, escalation=2):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(questions) or not len(scores):
        raise ValueError("need one score and question per nonempty batch")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must lie in [0, 1]")
    # Hash ties without gold labels, cluster siblings, or cached latency.
    ties = [hashlib.sha256(str(q).encode("utf-8")).hexdigest() for q in questions]
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], ties[i], i))
    eligible = [i for i in order if not positive_only or scores[i] > 0]
    actions = np.full(len(scores), base, dtype=int)
    actions[eligible[:math.floor(fraction * len(scores) + 1e-12)]] = escalation
    return actions


def choose_on_validation(candidates, f1, cost, *, base=1, escalation=2):
    """Select by validation F1, then fewer A2 calls, then measured validation cost.

    Every candidate is (stable_name, action_indices). The always-A1 fallback must
    be supplied explicitly. No requirement forces a method to use two actions.
    """
    f1, cost = np.asarray(f1, float), np.asarray(cost, float)
    if f1.ndim != 2 or f1.shape[1] not in (2, 3) or f1.shape != cost.shape or not len(f1):
        raise ValueError("validation matrices must be nonempty (n, 2) or (n, 3)")
    if base == escalation or not all(0 <= a < f1.shape[1] for a in (base, escalation)):
        raise ValueError("base and escalation must be distinct valid action indices")
    if not np.isfinite(f1).all() or not np.isfinite(cost).all():
        raise ValueError("validation outcomes must be finite")
    if np.any((f1 < 0) | (f1 > 1)) or np.any(cost < 0):
        raise ValueError("validation F1 must lie in [0,1] and cost must be nonnegative")
    choices = []
    for name, actions in candidates:
        a = np.asarray(actions)
        if a.shape != (len(f1),) or a.dtype.kind not in "iu" or not np.isin(a, [base, escalation]).all():
            raise ValueError("candidate must assign A1 or A2 once per validation query")
        row = np.arange(len(f1))
        choices.append({"candidate": name, "F1": float(f1[row, a].mean()),
                        "A2_count": int((a == escalation).sum()), "cost": float(cost[row, a].mean())})
    if not choices:
        raise ValueError("no candidate policies")
    return min(choices, key=lambda p: (-p["F1"], p["A2_count"], p["cost"], p["candidate"]))
