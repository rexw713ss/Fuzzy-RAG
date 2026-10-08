"""Type-1 Mamdani escalation controller (spec v0.2, sections 3-4).

Four scaled signals in [0, 1] -> one continuous escalation score.
Main configuration: min AND, clipped consequents, max aggregation, centroid.
Ablations (spec v0.3, section 3.3), each changing exactly one stage:
  - product t-norm: rule strength = product of the four degrees instead of the minimum;
  - probabilistic sum: clipped consequents combined pointwise with a + b - ab over every
    fired rule (adopted default, to confirm, section 9.3), instead of max;
  - Sugeno-0: each consequent replaced by the constant at its output-set centroid, score =
    strength-weighted average over the fired rules, instead of the centroid.
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

TNORMS = ("min", "product")
AGGREGATIONS = ("max", "probsum")
DEFUZZIFIERS = ("centroid", "sugeno")
# The main configuration and the three single-stage ablations of section 3.3.
ABLATIONS = {
    "main": {},
    "product": {"tnorm": "product"},
    "probsum": {"aggregation": "probsum"},
    "sugeno": {"defuzz": "sugeno"},
}

# Output membership functions as trapezoids (a, b, c, d); a triangle has b == c (section 3.2).
OUT_MF = {
    "Low": (0.0, 0.0, 0.0, 0.40),
    "Med": (0.25, 0.50, 0.50, 0.75),
    "High": (0.60, 1.0, 1.0, 1.0),
}
# Sugeno-0 constants: the analytic centroids of the output sets above (section 3.3), so the
# Sugeno score spans the same range as the main controller. Low = 0.4 / 3, High = 1 - 0.4 / 3.
SUGENO_Z = {"Low": 0.4 / 3, "Med": 0.5, "High": 1.0 - 0.4 / 3}


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
_RULE_Z = np.array([SUGENO_Z[c] for _, _, _, c in RULES])


def escalate(s, widths=W_DEFAULT, n_out=N_OUT, tnorm="min", aggregation="max",
             defuzz="centroid"):
    """Escalation score for one query instance, plus a decision trace.

    s: four scaled signals (S1..S4), each in [0, 1].
    widths: one band width for all signals, or one per signal (after calibration).
    tnorm, aggregation, defuzz: the defaults are the main configuration; see ABLATIONS.
    Returns (score, trace). trace holds the membership degrees, every rule with
    non-zero strength, the clip level of each output label (its strongest rule) and the
    operators used.
    """
    if tnorm not in TNORMS:
        raise ValueError(f"tnorm must be one of {TNORMS}, got {tnorm!r}")
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
    if defuzz not in DEFUZZIFIERS:
        raise ValueError(f"defuzz must be one of {DEFUZZIFIERS}, got {defuzz!r}")
    if defuzz == "sugeno" and aggregation != "max":
        raise ValueError("Sugeno-0 has no output sets to aggregate; leave aggregation='max'")
    s = np.asarray(s, dtype=float)
    if s.shape != (4,):
        raise ValueError(f"need 4 signals, got shape {s.shape}")
    if not np.all((s >= 0.0) & (s <= 1.0)):
        raise ValueError(f"signals must lie in [0, 1], got {s.tolist()}")
    widths = np.broadcast_to(np.asarray(widths, dtype=float), (4,))

    mu = np.stack([memberships(x, w) for x, w in zip(s, widths)])   # (4 signals, 3 labels)

    # Rule strength: AND over the four antecedent degrees (minimum, or product as ablation).
    degrees = mu[np.arange(4), _RULE_IDX]                           # (81 rules, 4 signals)
    strength = degrees.min(axis=1) if tnorm == "min" else degrees.prod(axis=1)   # (81,)

    # Each consequent label's strongest rule. Under max aggregation this is the clip level.
    clip = np.array([strength[_RULE_OUT == j].max() for j in range(3)])

    # Some rule always fires (degrees sum to 1 per signal), so every denominator is > 0.
    if defuzz == "sugeno":
        score = float((strength * _RULE_Z).sum() / strength.sum())
    else:
        y = np.linspace(0.0, 1.0, n_out)
        out = np.stack([trapezoid(y, *OUT_MF[lab]) for lab in LABELS])  # (3, n_out)
        if aggregation == "max":
            agg = np.minimum(out, clip[:, None]).max(axis=0)
        else:
            # Every fired rule's clipped consequent, combined pointwise: 1 - prod(1 - a_r).
            fired = np.flatnonzero(strength > 0)
            clipped = np.minimum(out[_RULE_OUT[fired]], strength[fired][:, None])
            agg = 1.0 - np.prod(1.0 - clipped, axis=0)
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
        "operators": {"tnorm": tnorm, "aggregation": aggregation, "defuzz": defuzz},
    }
    return score, trace
