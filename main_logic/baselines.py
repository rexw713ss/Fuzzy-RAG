"""Controller baselines B1-B4 (spec v0.3, section 5).

All four read the same scaled signals S1..S4 and are scored on the same cached action-outcome
rows (routing.OutcomeTable), so comparing them costs no model calls. Each one turns the four
signals into one number per query instance and splits that number into A0 / A1 / A2 with two
cut points, chosen on the validation split with the section 3.4 objective, tie-break and budget
(routing.select_thresholds).

    B1  hard threshold on a weighted sum w . S          weights and theta1, theta2 tuned
    B2  crisp ordinal sum t (labels at 0.35 / 0.65)     integer cut points c1, c2 tuned
    B3  fuzzy controller, every width 0.20              theta1, theta2 tuned
    B4  fuzzy controller, jitter-calibrated widths      theta1, theta2 tuned (the full method)

Choices the spec leaves open:
  1. B1's weights (user decision, Oct 8, 2026): every weight vector on a 0.25 lattice that
     sums to 1 (35 vectors, including equal weights and each signal alone), each crossed with
     the theta pairs of section 3.4. The weighted sum lies in [0, 1], so the same theta
     grid applies.
  2. B1 tie-break across weight vectors: the section 3.4 key (higher quality, lower cost,
     lower theta1, lower theta2), then the earlier weight vector in lattice order.
  3. B2 labels are half-open as written in section 5: S < 0.35 Low, 0.35 <= S < 0.65 Med,
     S >= 0.65 High (adopted default, to confirm, section 9.3).

v0.4 (REDESIGN_2026-10-15): the same four baselines on any number of signals n (v0.4: S2, S3,
S4) and any action set (v0.4: A1, A2 with one threshold). B1's lattice has n dimensions (15
vectors for n = 3); B2's sum t runs 0..2n and its cuts are integers 0..2n+1 (one cut for two
actions, a pair for three). The signal names travel with the fit ("signals").
"""

import itertools

import numpy as np

from main_logic import fuzzy as fz
from main_logic import routing as rt

BASELINES = ("B1", "B2", "B3", "B4")
WEIGHT_STEP = 0.25                     # B1 weight lattice (user decision)
T_MAX = 8                              # four signals x level 2 (section 5)


def weight_grid(step=WEIGHT_STEP, dim=4):
    """Every non-negative dim-vector on a `step` lattice that sums to 1, in lexicographic order.

    Built from integers so no weight carries float error (0.25 * 3 is exact, but a step such
    as 0.1 would not be). 35 vectors for dim = 4, 15 for dim = 3.
    """
    n = round(1.0 / step)
    if n < 1 or abs(n * step - 1.0) > 1e-9:
        raise ValueError(f"1 / step must be a whole number, got step={step}")
    return [tuple(c / n for c in combo)
            for combo in itertools.product(range(n + 1), repeat=dim) if sum(combo) == n]


def cut_grid(n_signals=4, n_actions=3):
    """B2's cut points on t in 0..2n (section 5).

    Three actions: every integer pair 0 <= c1 < c2 <= 2n+1 (45 pairs for n = 4); c1 = 0 means A0
    is never chosen and c2 = 2n+1 means A2 is never chosen. Two actions (v0.4): every single cut
    c in 0..2n+1, as 1-tuples (8 for n = 3); c = 0 is always A2, c = 2n+1 always A1.
    """
    top = 2 * n_signals + 1
    if n_actions == 3:
        return [(c1, c2) for c1 in range(top + 1) for c2 in range(c1 + 1, top + 1)]
    if n_actions == 2:
        return [(c,) for c in range(top + 1)]
    raise ValueError(f"no B2 cut grid for {n_actions} actions")


def signal_matrix(table, signals, n=4):
    """Scaled signals as an (instances, n) array in the table's instance order.

    signals: {(cluster_id, variant_id): (n values)}, keyed like routing.OutcomeTable.align
    so a mismatch is an error rather than a silent misalignment. Values must lie in [0, 1]
    (after section 2.6 rescaling).
    """
    missing = [k for k in table.instances if k not in signals]
    if missing:
        raise ValueError(f"no signals for {len(missing)} instance(s), first: {missing[0]}")
    s = np.array([np.asarray(signals[k], dtype=float) for k in table.instances])
    if s.shape != (len(table), n):
        raise ValueError(f"need {n} signals per instance, got shape {s.shape}")
    if not np.all((s >= 0.0) & (s <= 1.0)):
        raise ValueError("signals must be scaled to [0, 1] (section 2.6)")
    return s


def weighted_scores(s, weights):
    """B1 score: w . S for each row of s."""
    s = np.asarray(s, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.shape != (s.shape[-1],) or np.any(w < 0) or abs(w.sum() - 1.0) > 1e-9:
        raise ValueError(f"weights must be {s.shape[-1]} non-negative numbers summing to 1, "
                         f"got {weights}")
    return s @ w


def crisp_levels(s):
    """B2 labels: 0 = Low (S < 0.35), 1 = Med (0.35 <= S < 0.65), 2 = High (S >= 0.65)."""
    lo, hi = fz.CENTRES
    s = np.asarray(s, dtype=float)
    return (s >= lo).astype(int) + (s >= hi).astype(int)


def ordinal_sums(s):
    """B2 score: t = sum of the crisp levels, an integer in 0..2n."""
    return crisp_levels(s).sum(axis=-1)


def fuzzy_scores(s, widths, signals=fz.SIGNALS, **operators):
    """B3 / B4 score: the Mamdani escalation score of section 3, one per row of s."""
    return np.array([fz.escalate(row, widths, signals=signals, **operators)[0]
                     for row in np.asarray(s, dtype=float)])


def _names(n):
    """The signal names for n signals: v0.3's S1..S4, or v0.4's S2, S3, S4."""
    if n == len(fz.SIGNALS):
        return fz.SIGNALS
    if n == len(fz.SIGNALS_V4):
        return fz.SIGNALS_V4
    raise ValueError(f"no signal names for {n} signals")


def _scores(fit, s):
    """Recompute a fitted baseline's scores on any signals (e.g. the test split)."""
    name = fit["baseline"]
    if name == "B1":
        return weighted_scores(s, fit["weights"])
    if name == "B2":
        return ordinal_sums(s)
    if name in ("B3", "B4"):
        return fuzzy_scores(s, fit["widths"], signals=tuple(fit["signals"]))
    raise ValueError(f"unknown baseline {name!r}")


def _n_signals(signals):
    first = next(iter(signals.values()))
    return len(first)


def fit_b1(table, signals, budget=None, weights=None, grid=None):
    """B1: choose the weight vector and the thresholds together on validation."""
    n = _n_signals(signals)
    s = signal_matrix(table, signals, n)
    weights = weight_grid(dim=n) if weights is None else list(weights)
    best, best_key = None, None
    n_feasible_weights = 0
    for i, w in enumerate(weights):
        try:
            sel = rt.select_thresholds(table, weighted_scores(s, w), budget, grid)
        except ValueError as e:
            if "cost budget" in str(e):          # no threshold cut is feasible for these weights
                continue
            raise
        n_feasible_weights += 1
        key = (-sel["quality"], sel["cost"], sel["thresholds"], i)
        if best_key is None or key < best_key:
            best, best_key = {**sel, "weights": tuple(w)}, key
    if best is None:
        raise ValueError("no weight vector and threshold pair meets the cost budget")
    return {"baseline": "B1", **best, "n_weights": len(weights),
            "n_feasible_weights": n_feasible_weights}


def fit_b2(table, signals, budget=None):
    """B2: choose integer cut points on the ordinal sum t (section 5)."""
    n = _n_signals(signals)
    t = ordinal_sums(signal_matrix(table, signals, n))
    sel = rt.select_thresholds(table, t, budget, cut_grid(n, len(table.actions)))
    cuts = {"c1": int(sel["theta1"]), "c2": int(sel["theta2"])} if "theta1" in sel else \
        {"c": int(sel["theta"])}
    return {"baseline": "B2", **sel, **cuts}


def fit_b3(table, signals, budget=None, grid=None):
    """B3: fuzzy controller with every width fixed at 0.20."""
    n = _n_signals(signals)
    s = signal_matrix(table, signals, n)
    names = _names(n)
    sel = rt.select_thresholds(table, fuzzy_scores(s, fz.W_DEFAULT, names), budget, grid)
    return {"baseline": "B3", **sel, "widths": (fz.W_DEFAULT,) * n, "signals": names}


def fit_b4(table, signals, widths, budget=None, grid=None):
    """B4: fuzzy controller with the jitter-calibrated widths (calibration.calibrate()['widths'])."""
    n = _n_signals(signals)
    w = tuple(float(x) for x in np.broadcast_to(np.asarray(widths, dtype=float), (n,)))
    s = signal_matrix(table, signals, n)
    names = _names(n)
    sel = rt.select_thresholds(table, fuzzy_scores(s, w, names), budget, grid)
    return {"baseline": "B4", **sel, "widths": w, "signals": names}


def route(fit, table, signals):
    """Apply a fitted baseline to another split: its actions in that table's instance order."""
    s = signal_matrix(table, signals, _n_signals(signals))
    return rt.assign_action(_scores(fit, s), *fit["thresholds"], actions=table.actions)
