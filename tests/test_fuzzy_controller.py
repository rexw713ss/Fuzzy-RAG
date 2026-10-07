"""Tests for the Mamdani escalation controller (spec v0.2, sections 3-4).

Covers the section 8.1 acceptance items that belong to the controller (81 rules,
31/19/31 split, all-Low = minimum, all-High = maximum), the rule CSV, and a
hand-computed example. Ablations and the monotonicity report are not covered yet.
"""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from main_logic import fuzzy as fz

RULE_CSV = Path(__file__).resolve().parents[2] / "Fuzzy_RAG_personal" / "etc" / "rule_base_81.csv"

# Centroids of the output triangles: Low (0, 0, 0.4) -> 0.4/3, High (0.6, 1, 1) -> 1 - 0.4/3.
ESC_MIN = 0.4 / 3
ESC_MAX = 1 - 0.4 / 3


# --- 3.1 input membership functions ---------------------------------------

@pytest.mark.parametrize("w", [0.05, 0.20, 0.25])
def test_memberships_sum_to_one(w):
    for x in np.linspace(0.0, 1.0, 201):
        assert fz.memberships(x, w).sum() == pytest.approx(1.0, abs=1e-12)


def test_flat_cores_and_crossovers():
    # w = 0.20: Low core [0, 0.25], Med core [0.45, 0.55], High core [0.75, 1].
    assert fz.memberships(0.10).tolist() == [1.0, 0.0, 0.0]
    assert fz.memberships(0.50).tolist() == [0.0, 1.0, 0.0]
    assert fz.memberships(0.90).tolist() == [0.0, 0.0, 1.0]
    assert fz.memberships(0.35).tolist() == pytest.approx([0.5, 0.5, 0.0])
    assert fz.memberships(0.65).tolist() == pytest.approx([0.0, 0.5, 0.5])


def test_at_most_two_labels_active():
    for x in np.linspace(0.0, 1.0, 201):
        assert (fz.memberships(x) > 0).sum() <= 2


@pytest.mark.parametrize("w", [0.0, -0.1, 0.26, 0.28, 0.30, np.nan])
def test_invalid_width_rejected(w):
    """w must lie in (0, 0.25] (spec v0.3 C7). 0.26 and 0.28 are still valid trapezoids,
    but they break the pre-registered cap, so fuzzy.py refuses them rather than clipping."""
    with pytest.raises(ValueError):
        fz.input_mf(w)


def test_width_cap_itself_is_accepted():
    """Calibration clips to exactly 0.25, so the cap must pass; its Med core is 0.05 wide."""
    a, b, c, d = fz.input_mf(fz.W_MAX)["Med"]
    assert c - b == pytest.approx(0.05)


def test_one_bad_per_signal_width_is_refused():
    """After calibration each signal has its own width; one over the cap must stop escalate."""
    with pytest.raises(ValueError, match=r"\(0, 0.25\]"):
        fz.escalate([0.5] * 4, widths=[0.20, 0.20, 0.27, 0.20])


# --- 4 rule base ----------------------------------------------------------

def test_81_rules_with_31_19_31_split():
    assert len(fz.RULES) == 81
    assert Counter(c for *_, c in fz.RULES) == {"Low": 31, "Med": 19, "High": 31}


def test_rule_base_matches_csv():
    if not RULE_CSV.exists():
        pytest.skip(f"rule CSV not found at {RULE_CSV}")
    with open(RULE_CSV, encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 81
    for (rid, labs, t, cons), row in zip(fz.RULES, rows):
        assert rid == int(row["rule_id"])
        assert labs == (row["S1_margin"], row["S2_disagreement"],
                        row["S3_dispersion"], row["S4_entropy"])
        assert t == float(row["level_sum_t"])
        assert cons == row["escalation"]


def test_rule_base_is_monotone():
    """Raising any one input's label never lowers a rule's consequent (section 4.2)."""
    out = {labs: fz.LEVEL[c] for _, labs, _, c in fz.RULES}
    for labs, level in out.items():
        for i in range(4):
            k = fz.LABELS.index(labs[i])
            if k < 2:
                up = labs[:i] + (fz.LABELS[k + 1],) + labs[i + 1:]
                assert out[up] >= level


# --- 3.2-3.3 inference and defuzzification ---------------------------------

def test_all_low_is_minimum_and_all_high_is_maximum():
    lo, _ = fz.escalate([0.0] * 4)
    hi, _ = fz.escalate([1.0] * 4)
    assert lo == pytest.approx(ESC_MIN, abs=1e-3)
    assert hi == pytest.approx(ESC_MAX, abs=1e-3)
    rng = np.random.default_rng(0)
    for s in rng.random((500, 4)):
        esc, _ = fz.escalate(s)
        assert lo - 1e-12 <= esc <= hi + 1e-12


def test_hand_computed_example():
    """S = (0.30, 0.50, 0.50, 0.70), w = 0.20.

    S1: Low 0.75 / Med 0.25 ; S2, S3: Med 1 ; S4: Med 0.25 / High 0.75.
    Fired: LMMM (t=3, Low) 0.25, LMMH (t=4, Med) 0.75, MMMM (t=4, Med) 0.25,
    MMMH (t=5, High) 0.25. Clips Low 0.25 / Med 0.75 / High 0.25 are symmetric
    about 0.5, so the centroid is exactly 0.5.
    """
    esc, trace = fz.escalate([0.30, 0.50, 0.50, 0.70])
    assert esc == pytest.approx(0.5, abs=1e-9)
    assert trace["clip"] == pytest.approx({"Low": 0.25, "Med": 0.75, "High": 0.25})
    fired = {r["labels"]: r["strength"] for r in trace["fired"]}
    assert fired == pytest.approx({
        ("Low", "Med", "Med", "Med"): 0.25,
        ("Low", "Med", "Med", "High"): 0.75,
        ("Med", "Med", "Med", "Med"): 0.25,
        ("Med", "Med", "Med", "High"): 0.25,
    })


def test_at_most_16_rules_fire():
    rng = np.random.default_rng(1)
    for s in rng.random((500, 4)):
        _, trace = fz.escalate(s)
        assert 1 <= len(trace["fired"]) <= 16


def test_output_resolution_barely_matters():
    """1,001 vs 10,001 output points: chosen resolution, not in the spec."""
    rng = np.random.default_rng(2)
    for s in rng.random((200, 4)):
        a, _ = fz.escalate(s, n_out=1001)
        b, _ = fz.escalate(s, n_out=10001)
        assert a == pytest.approx(b, abs=1e-3)


def test_per_signal_widths():
    s = [0.30, 0.50, 0.50, 0.70]
    same, _ = fz.escalate(s, widths=0.20)
    listed, _ = fz.escalate(s, widths=[0.20] * 4)
    other, _ = fz.escalate(s, widths=[0.25, 0.20, 0.20, 0.20])
    assert same == listed
    assert other != same


@pytest.mark.parametrize("s", [[0.5] * 3, [0.5, 0.5, 0.5, 1.2], [0.5, -0.1, 0.5, 0.5],
                               [0.5, np.nan, 0.5, 0.5]])
def test_bad_signals_rejected(s):
    with pytest.raises(ValueError):
        fz.escalate(s)
