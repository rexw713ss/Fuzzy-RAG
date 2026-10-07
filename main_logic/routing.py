"""Action thresholds and the cost constraint (spec v0.2, section 3.4).

Two thresholds on the defuzzified escalation score pick one action:

    esc < theta1          -> A0
    theta1 <= esc < theta2 -> A1
    esc >= theta2         -> A2

Thresholds are chosen by grid search on the VALIDATION split only, maximizing mean
quality subject to a cost budget (default: the mean cost of the always-A1 policy).

Everything here is a selection over the cached action-outcome rows of section 7.4, so a
new threshold pair, rule base or ablation costs no model calls. Costs are read from the
cached row rather than inferred from the action label, which is what makes a section 1.2
A2 fallback correct: the row carries A1's measured latency but still counts as A2.

Choices the spec leaves open (user decisions, Oct 6, 2026):
  - Tie-break among pairs with equal mean quality: lower mean cost, then lower theta1,
    then lower theta2. Mirrors the section 6.1 oracle and is deterministic (section 7.5).
  - Pareto sweep points: the distinct mean costs the grid actually achieves, so the
    frontier is exact with no arbitrary binning.
  - The cost field is pluggable (default end-to-end latency, per section 3.4) because
    laptop latency is thermally noisy; the secondary counts can be swapped in.
"""

import itertools

import numpy as np

ACTIONS = ("A0", "A1", "A2")          # strictly ordered by added cost (section 1)
_A = {a: i for i, a in enumerate(ACTIONS)}

T1_RANGE = (0.20, 0.50)               # theta1 grid bounds (section 3.4)
T2_RANGE = (0.50, 0.80)               # theta2 grid bounds
STEP = 0.01
QUALITY = "F1"                        # section 6.1 selects on F1
COST = "latency_total"                # section 3.4 primary cost metric


def check_thresholds(theta1, theta2):
    """theta1 < theta2, both finite (section 3.4, acceptance test section 8.1)."""
    t1, t2 = float(theta1), float(theta2)
    if not (np.isfinite(t1) and np.isfinite(t2)):
        raise ValueError(f"thresholds must be finite, got ({theta1}, {theta2})")
    if not t1 < t2:
        raise ValueError(f"need theta1 < theta2, got theta1={t1}, theta2={t2}")
    return t1, t2


def assign_action(score, theta1, theta2):
    """Action label(s) for one score or an array of scores (section 3.4).

    Intervals are half-open: a score exactly at theta1 is A1, exactly at theta2 is A2.
    """
    t1, t2 = check_thresholds(theta1, theta2)
    s = np.asarray(score, dtype=float)
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    idx = (s >= t1).astype(int) + (s >= t2).astype(int)
    labels = np.asarray(ACTIONS)[idx]
    return labels if labels.ndim else str(labels)


def threshold_grid(step=STEP, t1_range=T1_RANGE, t2_range=T2_RANGE):
    """All (theta1, theta2) pairs on the step lattice with theta1 < theta2 (section 3.4).

    Built as round(lo + i*step, 10) rather than np.arange, which accumulates error and
    has an unreliable endpoint. With the defaults: 31 x 31 candidates, minus the single
    (0.50, 0.50) that fails theta1 < theta2, so 960 pairs.
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")

    def axis(lo, hi):
        n = round((hi - lo) / step)
        return [round(lo + i * step, 10) for i in range(n + 1)]

    pairs = [(t1, t2) for t1, t2 in itertools.product(axis(*t1_range), axis(*t2_range))
             if t1 < t2]
    if not pairs:
        raise ValueError(f"no pair satisfies theta1 < theta2 for {t1_range}, {t2_range}")
    return pairs


class OutcomeTable:
    """Quality and cost of every action on every query instance (section 7.4).

    Built from the cached outcome rows. One instance is a (cluster_id, variant_id) pair;
    all three actions must be present for each, otherwise a router could not be scored
    on it. `instances` is the canonical order of the f1 / cost rows.
    """

    def __init__(self, instances, f1, cost, quality_key=QUALITY, cost_key=COST):
        self.instances = list(instances)
        self.f1 = np.asarray(f1, dtype=float)
        self.cost = np.asarray(cost, dtype=float)
        self.quality_key = quality_key
        self.cost_key = cost_key
        n = len(self.instances)
        if self.f1.shape != (n, 3) or self.cost.shape != (n, 3):
            raise ValueError(f"f1 and cost must both be ({n}, 3), got "
                             f"{self.f1.shape} and {self.cost.shape}")
        if not np.all(np.isfinite(self.f1)) or not np.all(np.isfinite(self.cost)):
            raise ValueError("outcome table contains NaN or Inf")

    def __len__(self):
        return len(self.instances)

    @classmethod
    def from_rows(cls, rows, quality=QUALITY, cost=COST):
        """Build from cached rows (section 7.4).

        rows: dicts with cluster_id, variant_id, action, the quality field, and the cost
        field(s). `cost` is one field name or a sequence of field names to sum, so
        component latencies (retrieval + rerank + rewrite + generation) also work.
        """
        cost_keys = [cost] if isinstance(cost, str) else list(cost)
        if not cost_keys:
            raise ValueError("cost needs at least one field name")

        seen = {}
        for r in rows:
            key = (r["cluster_id"], r["variant_id"])
            action = r["action"]
            if action not in _A:
                raise ValueError(f"unknown action {action!r} for instance {key}")
            if (key, action) in seen:
                raise ValueError(f"duplicate cached row for {key}, action {action}")
            seen[(key, action)] = r

        instances = sorted({k for k, _ in seen}, key=lambda k: (str(k[0]), str(k[1])))
        f1 = np.empty((len(instances), 3), dtype=float)
        cst = np.empty((len(instances), 3), dtype=float)
        for i, key in enumerate(instances):
            for action, j in _A.items():
                r = seen.get((key, action))
                if r is None:
                    raise ValueError(f"instance {key} has no cached row for {action}; "
                                     "the outcome matrix must cover all three actions")
                f1[i, j] = float(r[quality])
                cst[i, j] = float(sum(r[k] for k in cost_keys))
        name = cost if isinstance(cost, str) else "+".join(cost_keys)
        return cls(instances, f1, cst, quality_key=quality, cost_key=name)

    def align(self, scores):
        """Escalation scores as an array in `instances` order.

        scores: {(cluster_id, variant_id): score}. Keyed rather than positional so a
        mismatch is an error instead of a silent misalignment.
        """
        missing = [k for k in self.instances if k not in scores]
        if missing:
            raise ValueError(f"no escalation score for {len(missing)} instance(s), "
                             f"first: {missing[0]}")
        return np.array([float(scores[k]) for k in self.instances], dtype=float)


def policy_value(table, actions):
    """(mean quality, mean cost) of one instance -> action assignment.

    The shared primitive: the section 5 baselines, the always-A* reference points and the
    Pareto sweep are all different ways of producing `actions` for this function.
    """
    a = np.asarray(actions)
    if a.shape != (len(table),):
        raise ValueError(f"need one action per instance ({len(table)}), got {a.shape}")
    if a.dtype.kind in "US":
        unknown = set(np.unique(a)) - set(ACTIONS)
        if unknown:
            raise ValueError(f"unknown action label(s): {sorted(unknown)}")
        idx = np.array([_A[x] for x in a])
    else:
        idx = a.astype(int)
        if not np.all((idx >= 0) & (idx < 3)):
            raise ValueError("action indices must be 0, 1 or 2")
    rows = np.arange(len(table))
    return float(table.f1[rows, idx].mean()), float(table.cost[rows, idx].mean())


def always_policy(table, action):
    """(mean quality, mean cost) of sending every instance to one fixed action."""
    if action not in _A:
        raise ValueError(f"unknown action {action!r}")
    j = _A[action]
    return float(table.f1[:, j].mean()), float(table.cost[:, j].mean())


def action_counts(actions):
    """{action: count} over an assignment, including actions never selected.

    Section 3.4 removed the 5% usage floor: an unused action is a finding to report, not
    something to force, so zero counts are kept rather than dropped.
    """
    a = np.asarray(actions)
    return {x: int((a == x).sum()) for x in ACTIONS}


def _grid_values(table, scores, grid):
    """(mean quality, mean cost) for every pair in the grid, in grid order."""
    return [policy_value(table, assign_action(scores, t1, t2)) for t1, t2 in grid]


def select_thresholds(table, scores, budget=None, grid=None):
    """Grid-search theta1, theta2 on the validation split (section 3.4).

    Maximizes mean quality subject to mean cost <= budget. The budget defaults to the
    always-A1 mean cost; note that the always-A1 policy itself is generally NOT reachable
    by any threshold pair (it would need every score inside [theta1, theta2)), so the
    budget is a bound, not an attainable point, and feasibility is checked rather than
    assumed.

    Ties on mean quality are broken by lower mean cost, then lower theta1, then lower
    theta2 (user decision; keeps the result deterministic for section 7.5).

    Returns a dict with the chosen pair, its quality and cost, the budget, how many pairs
    were feasible, the action counts, and any action never selected.
    """
    s = table.align(scores) if isinstance(scores, dict) else np.asarray(scores, float)
    if s.shape != (len(table),):
        raise ValueError(f"need one score per instance ({len(table)}), got {s.shape}")
    grid = threshold_grid() if grid is None else list(grid)
    for t1, t2 in grid:
        check_thresholds(t1, t2)

    if budget is None:
        budget = always_policy(table, "A1")[1]
    budget = float(budget)

    values = _grid_values(table, s, grid)
    feasible = [(q, c, t1, t2) for (q, c), (t1, t2) in zip(values, grid) if c <= budget]
    if not feasible:
        cheapest = min(c for _, c in values)
        raise ValueError(
            f"no threshold pair meets the cost budget {budget:.6g}; the cheapest "
            f"reachable policy costs {cheapest:.6g}. The always-A1 budget is a bound, "
            "not necessarily attainable by thresholds."
        )

    quality, cost, t1, t2 = min(feasible, key=lambda v: (-v[0], v[1], v[2], v[3]))
    actions = assign_action(s, t1, t2)
    counts = action_counts(actions)
    return {
        "theta1": t1,
        "theta2": t2,
        "quality": quality,
        "cost": cost,
        "budget": budget,
        "quality_key": table.quality_key,
        "cost_key": table.cost_key,
        "n_feasible": len(feasible),
        "n_grid": len(grid),
        "counts": counts,
        "unused_actions": [a for a in ACTIONS if counts[a] == 0],
    }


def pareto_frontier(table, scores, grid=None):
    """Best achievable quality at each cost level (section 3.4 Pareto reporting).

    Sweep points are the distinct mean costs the grid actually achieves (user decision),
    so the staircase is exact. Quality is non-decreasing in budget by construction, since
    a larger budget admits every pair a smaller one did.

    Returns [{budget, quality, cost, theta1, theta2}, ...] ordered by budget.
    """
    s = table.align(scores) if isinstance(scores, dict) else np.asarray(scores, float)
    grid = threshold_grid() if grid is None else list(grid)
    values = _grid_values(table, s, grid)
    points = [(q, c, t1, t2) for (q, c), (t1, t2) in zip(values, grid)]

    frontier = []
    for budget in sorted({c for _, c, _, _ in points}):
        under = [p for p in points if p[1] <= budget]
        quality, cost, t1, t2 = min(under, key=lambda v: (-v[0], v[1], v[2], v[3]))
        frontier.append({"budget": budget, "quality": quality, "cost": cost,
                         "theta1": t1, "theta2": t2})
    return frontier
