"""Type-1 Mamdani escalation controller (spec v0.2, sections 3-4).

Four scaled signals in [0, 1] -> one continuous escalation score.
Main configuration only: min AND, clipped consequents, max aggregation, centroid.
The ablations (product t-norm, probabilistic sum, Sugeno-0) are not here yet.
Mapping the score to an action (theta1, theta2) is section 3.4 and lives elsewhere.
"""

import itertools

import numpy as np

LABELS = ("Low", "Med", "High")
LEVEL = {"Low": 0, "Med": 1, "High": 2}
SIGNALS = ("S1", "S2", "S3", "S4")

CENTRES = (0.35, 0.65)   # crossover centres (section 3.1)
W_DEFAULT = 0.20         # band width before jitter calibration
W_MAX = 0.25             # hard cap on band width (section 3.1)
N_OUT = 1001             # output-axis resolution for the centroid (not in the spec)

# Output membership functions as trapezoids (a, b, c, d); a triangle has b == c (section 3.2).
OUT_MF = {
    "Low": (0.0, 0.0, 0.0, 0.40),
    "Med": (0.25, 0.50, 0.50, 0.75),
    "High": (0.60, 1.0, 1.0, 1.0),
}


def trapezoid(x, a, b, c, d):
    """Trapezoid membership. a == b gives a left shoulder, c == d a right shoulder."""
    x = np.asarray(x, dtype=float)
    rise = np.ones_like(x) if b == a else (x - a) / (b - a)
    fall = np.ones_like(x) if d == c else (d - x) / (d - c)
    return np.clip(np.minimum(rise, fall), 0.0, 1.0)


def input_mf(w=W_DEFAULT):
    """Low / Med / High trapezoid parameters for band width w (section 3.1).

    Refuses w > 0.25 (spec v0.3, C7): the cap keeps the Med core at least 0.05 wide.
    Clipping a measured jitter to the cap and flagging the signal is the calibration
    step's job (section 7.3), so a wider w arriving here is a bug upstream. It raises
    instead of clipping, so the flag the spec requires can never be skipped silently.
    """
    lo, hi = CENTRES
    # The tolerance lets a w of exactly 0.25 through despite float rounding.
    if not (w > 0.0 and w <= W_MAX + 1e-9):
        raise ValueError(f"w must be in (0, {W_MAX}], got {w}")
    # Med core width is (hi - lo) - w. Only bites if CENTRES are moved closer together.
    if not (hi - lo) - w > 1e-9:
        raise ValueError(f"w = {w} leaves no Med core between centres {CENTRES}")
    h = w / 2
    return {
        "Low": (0.0, 0.0, lo - h, lo + h),
        "Med": (lo - h, lo + h, hi - h, hi + h),
        "High": (hi - h, hi + h, 1.0, 1.0),
    }


def memberships(x, w=W_DEFAULT):
    """Degrees (Low, Med, High) of one scaled signal value. They always sum to 1."""
    mf = input_mf(w)
    return np.array([float(trapezoid(x, *mf[lab])) for lab in LABELS])


def consequent(t):
    """Ordinal-sum cut: t <= 3 -> Low, t = 4 -> Med, t >= 5 -> High (section 4)."""
    return "Low" if t <= 3 else ("Med" if t == 4 else "High")


def rule_base():
    """All 81 rules, in the row order of rule_base_81.csv (S1 slowest, S4 fastest).

    Returns [(rule_id, (label_S1, ..., label_S4), t, consequent), ...].
    """
    rules = []
    for i, labs in enumerate(itertools.product(LABELS, repeat=4), start=1):
        t = sum(LEVEL[lab] for lab in labs)
        rules.append((i, labs, t, consequent(t)))
    return rules


RULES = rule_base()
_RULE_IDX = np.array([[LABELS.index(lab) for lab in labs] for _, labs, _, _ in RULES])
_RULE_OUT = np.array([LABELS.index(c) for _, _, _, c in RULES])


def escalate(s, widths=W_DEFAULT, n_out=N_OUT):
    """Escalation score for one query instance, plus a decision trace.

    s: four scaled signals (S1..S4), each in [0, 1].
    widths: one band width for all signals, or one per signal (after calibration).
    Returns (score, trace). trace holds the membership degrees, every rule with
    non-zero strength, and the clip level of each output label.
    """
    s = np.asarray(s, dtype=float)
    if s.shape != (4,):
        raise ValueError(f"need 4 signals, got shape {s.shape}")
    if not np.all((s >= 0.0) & (s <= 1.0)):
        raise ValueError(f"signals must lie in [0, 1], got {s.tolist()}")
    widths = np.broadcast_to(np.asarray(widths, dtype=float), (4,))

    mu = np.stack([memberships(x, w) for x, w in zip(s, widths)])   # (4 signals, 3 labels)

    # Rule strength = min over the four antecedent degrees (AND = minimum).
    strength = mu[np.arange(4), _RULE_IDX].min(axis=1)              # (81,)

    # Each consequent label is clipped at the strongest rule pointing to it (aggregation = max).
    clip = np.array([strength[_RULE_OUT == j].max() for j in range(3)])

    y = np.linspace(0.0, 1.0, n_out)
    out = np.stack([trapezoid(y, *OUT_MF[lab]) for lab in LABELS])  # (3, n_out)
    agg = np.minimum(out, clip[:, None]).max(axis=0)
    # Some rule always fires (degrees sum to 1 per signal), so agg.sum() > 0.
    score = float((agg * y).sum() / agg.sum())

    fired = [
        {"rule_id": RULES[r][0], "labels": RULES[r][1], "consequent": RULES[r][3],
         "strength": float(strength[r])}
        for r in np.flatnonzero(strength > 0)
    ]
    trace = {
        "memberships": {sig: dict(zip(LABELS, mu[i].tolist())) for i, sig in enumerate(SIGNALS)},
        "fired": fired,
        "clip": dict(zip(LABELS, clip.tolist())),
    }
    return score, trace
