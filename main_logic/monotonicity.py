"""Output monotonicity of the fuzzy controller (spec v0.3, section 4.2).

The rule base is monotone by construction, but the defuzzified score need not be: raising one
signal can lower the score a little, and if that small drop crosses a threshold the action is
downgraded while a signal said "escalate more". This module measures it on a regular grid and
returns the five reporting items of section 4.2.

Protocol (section 4.2, C5):
  - grid: `steps` equal steps on [0, 1] per signal (20 steps = 0.05 spacing, 21^4 = 194,481
    points); w = 0.20; the 1,001-point output axis;
  - pair: two grid points one step apart in exactly one signal (4 x steps x (steps+1)^3 pairs);
  - violation: the score drops by more than 1e-9 when that signal increases. The tolerance
    excludes floating-point noise on flat plateaus (drops of ~1e-17).

Item 5 (does a violation change the action?): with half-open intervals (section 3.4) a drop
from `before` to `after` downgrades the action at threshold theta exactly when
after < theta <= before.
"""

import numpy as np

from main_logic import fuzzy as fz
from main_logic import routing as rt

STEPS = 20          # 0.05 spacing (section 4.2)
TOL = 1e-9          # violation tolerance (section 4.2)


def grid_scores(steps=STEPS, widths=fz.W_DEFAULT, **operators):
    """Escalation score at every point of the grid: an array of shape (steps+1,) * 4.

    operators: passed to fuzzy.escalate (tnorm, aggregation, defuzz), so an ablation can be
    measured the same way. The full 0.05 grid takes about a minute.
    """
    if int(steps) != steps or steps < 1:
        raise ValueError(f"steps must be a positive integer, got {steps}")
    axis = np.linspace(0.0, 1.0, int(steps) + 1)
    m = axis.size
    scores = np.empty((m,) * 4)
    for idx in np.ndindex(scores.shape):
        scores[idx] = fz.escalate(axis[list(idx)], widths, **operators)[0]
    return scores


def violations(scores, tol=TOL):
    """Every adjacent pair whose score drops by more than tol when one signal rises one step.

    Returns (n_pairs, records): records is a dict of arrays, one entry per violation:
    signal (0..3), before (score at the lower input), after (score one step higher), drop.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 4 or len(set(scores.shape)) != 1:
        raise ValueError(f"need a 4-d grid with equal sides, got shape {scores.shape}")
    n_pairs = 0
    sig, before, after = [], [], []
    for k in range(4):
        lo = np.moveaxis(scores, k, 0)[:-1]
        hi = np.moveaxis(scores, k, 0)[1:]
        n_pairs += lo.size
        bad = (lo - hi) > tol
        sig.append(np.full(int(bad.sum()), k))
        before.append(lo[bad])
        after.append(hi[bad])
    before, after = np.concatenate(before), np.concatenate(after)
    return n_pairs, {"signal": np.concatenate(sig), "before": before, "after": after,
                     "drop": before - after}


def downgrades(records, theta):
    """How many violations downgrade the action at threshold theta: after < theta <= before."""
    return int(np.sum((records["after"] < theta) & (theta <= records["before"])))


def threshold_values(grid=None):
    """The distinct threshold values of the section 3.4 grid (61 by default: 0.20 .. 0.80)."""
    grid = rt.threshold_grid() if grid is None else grid
    return sorted({t for pair in grid for t in pair})


def report(scores, tol=TOL, thresholds=None, theta1=None, theta2=None):
    """The five reporting items of section 4.2.

    1 pairs examined; 2 violations (count and rate); 3 per signal; 4 drop max / mean / p95;
    5 threshold effect: how many violations straddle at least one grid threshold, the count
    at each threshold value, and, once theta1 and theta2 are selected, how many violations
    downgrade the action there (an A2 -> A1 or A1 -> A0 change while a signal increased).
    """
    n_pairs, rec = violations(scores, tol)
    n = rec["drop"].size
    thresholds = threshold_values() if thresholds is None else list(thresholds)
    per_threshold = {round(t, 10): downgrades(rec, t) for t in thresholds}
    straddle = np.zeros(n, dtype=bool)
    for t in thresholds:
        straddle |= (rec["after"] < t) & (t <= rec["before"])

    out = {
        "pairs": n_pairs,
        "violations": n,
        "rate": n / n_pairs,
        "per_signal": {s: int(np.sum(rec["signal"] == i)) for i, s in enumerate(fz.SIGNALS)},
        "drop_max": float(rec["drop"].max()) if n else 0.0,
        "drop_mean": float(rec["drop"].mean()) if n else 0.0,
        "drop_p95": float(np.percentile(rec["drop"], 95)) if n else 0.0,
        "straddle_any_threshold": int(straddle.sum()),
        "per_threshold": per_threshold,
        "thresholds_without_flips": sum(1 for v in per_threshold.values() if v == 0),
    }
    if theta1 is not None or theta2 is not None:
        t1, t2 = rt.check_thresholds(theta1, theta2)
        hit = (((rec["after"] < t1) & (t1 <= rec["before"]))
               | ((rec["after"] < t2) & (t2 <= rec["before"])))
        out["selected"] = {"theta1": t1, "theta2": t2, "downgrades": int(hit.sum())}
    return out
