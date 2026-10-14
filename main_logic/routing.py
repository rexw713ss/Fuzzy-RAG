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
  - theta2 runs up to 1.00, not the spec's 0.80 (open decision O-3, user decision Oct 11,
    2026). On the stage 3 pilot A1 cost about the same latency as A0 while A2 cost ~3 s more,
    so any score >= 0.80 forced to A2 broke the always-A1 budget and no pair was feasible.
    Up to 1.00, every router can switch A2 off, as B2's cut points already could.
  - theta1 runs down to 0.00, not 0.20, for the same reason: measured latency does not
    strictly follow A0 < A1 (A1's five passages can be shorter, so its generation is faster),
    so forcing low scores to A0 could itself exceed the always-A1 budget (NQ pilot: by
    0.004 s per query). From 0.00, every router can switch A0 off, as B2 can, so the
    always-A1 policy itself is reachable and the budget always has a feasible pair.

v0.4 (REDESIGN_2026-10-15, user decision Oct 15): A0 merged into A1, so a router chooses between
two actions, A1 (answer now) and A2 (escalate), with one threshold theta: A1 if score < theta,
else A2. Everything here takes the action set from the OutcomeTable (`table.actions`), so the
same code serves both: v0.3 tables hold (A0, A1, A2) and threshold pairs; v0.4 tables hold
(A1, A2) and 1-tuples (theta,). A "cut" is the tuple of thresholds, one fewer than the actions.
"""

import itertools

import numpy as np

ACTIONS = ("A0", "A1", "A2")          # strictly ordered by added cost (section 1)
ACTIONS_V4 = ("A1", "A2")             # v0.4: A0 merged into A1 (REDESIGN section 3)
_A = {a: i for i, a in enumerate(ACTIONS)}

T1_RANGE = (0.00, 0.50)               # theta1 grid bounds; spec 0.20, widened (O-3, Oct 11)
T2_RANGE = (0.50, 1.00)               # theta2 grid bounds; spec 0.80, widened (O-3, user Oct 11)
STEP = 0.01
QUALITY = "F1"                        # section 6.1 selects on F1
COST = "latency_total"                # section 3.4 primary cost metric
THETA_RANGE = (0.00, 1.00)            # v0.4 single threshold: 0 = always A2, 1 = always A1


def check_thresholds(theta1, theta2):
    """theta1 < theta2, both finite (section 3.4, acceptance test section 8.1)."""
    t1, t2 = float(theta1), float(theta2)
    if not (np.isfinite(t1) and np.isfinite(t2)):
        raise ValueError(f"thresholds must be finite, got ({theta1}, {theta2})")
    if not t1 < t2:
        raise ValueError(f"need theta1 < theta2, got theta1={t1}, theta2={t2}")
    return t1, t2


def check_cut(thresholds, n_actions):
    """A cut for n_actions actions: n_actions - 1 finite, strictly increasing thresholds."""
    t = tuple(float(x) for x in thresholds)
    if len(t) != n_actions - 1:
        raise ValueError(f"{n_actions} actions need {n_actions - 1} threshold(s), got {len(t)}")
    if n_actions == 3:
        return check_thresholds(*t)
    if not all(np.isfinite(t)):
        raise ValueError(f"thresholds must be finite, got {t}")
    if any(a >= b for a, b in zip(t, t[1:])):
        raise ValueError(f"thresholds must be strictly increasing, got {t}")
    return t


def assign_action(score, *thresholds, actions=ACTIONS):
    """Action label(s) for one score or an array of scores (section 3.4).

    v0.3: assign_action(score, theta1, theta2) over (A0, A1, A2).
    v0.4: assign_action(score, theta, actions=ACTIONS_V4) over (A1, A2).
    Intervals are half-open: a score exactly at a threshold takes the higher action.
    """
    t = check_cut(thresholds, len(actions))
    s = np.asarray(score, dtype=float)
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    idx = sum((s >= x).astype(int) for x in t)
    labels = np.asarray(actions)[idx]
    return labels if labels.ndim else str(labels)


def _axis(lo, hi, step):
    """lo, lo + step, ..., hi as round(lo + i*step, 10): np.arange accumulates error and has an
    unreliable endpoint."""
    n = round((hi - lo) / step)
    return [round(lo + i * step, 10) for i in range(n + 1)]


def threshold_grid(step=STEP, t1_range=T1_RANGE, t2_range=T2_RANGE):
    """All (theta1, theta2) pairs on the step lattice with theta1 < theta2 (section 3.4).

    With the defaults (theta1 0.00-0.50, theta2 0.50-1.00): 51 x 51 candidates minus the single
    (0.50, 0.50) that fails theta1 < theta2, so 2,600 pairs.
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    pairs = [(t1, t2) for t1, t2 in itertools.product(_axis(*t1_range, step),
                                                      _axis(*t2_range, step)) if t1 < t2]
    if not pairs:
        raise ValueError(f"no pair satisfies theta1 < theta2 for {t1_range}, {t2_range}")
    return pairs


def theta_grid(step=STEP, theta_range=THETA_RANGE):
    """v0.4: every single threshold on the step lattice, as 1-tuples (101 by default)."""
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    return [(t,) for t in _axis(*theta_range, step)]


def default_grid(actions):
    """The threshold grid for an action set: pairs for 3 actions, single thetas for 2."""
    if len(actions) == 3:
        return threshold_grid()
    if len(actions) == 2:
        return theta_grid()
    raise ValueError(f"no default grid for {len(actions)} actions")


class OutcomeTable:
    """Quality and cost of every action on every query instance (section 7.4).

    Built from the cached outcome rows. One instance is a (cluster_id, variant_id) pair;
    every action of `actions` must be present for each, otherwise a router could not be scored
    on it. `instances` is the canonical order of the f1 / cost rows; columns follow `actions`
    (v0.3: A0, A1, A2; v0.4: A1, A2), ordered by added cost.
    """

    def __init__(self, instances, f1, cost, quality_key=QUALITY, cost_key=COST, actions=ACTIONS):
        self.instances = list(instances)
        self.actions = tuple(actions)
        self.f1 = np.asarray(f1, dtype=float)
        self.cost = np.asarray(cost, dtype=float)
        self.quality_key = quality_key
        self.cost_key = cost_key
        n, k = len(self.instances), len(self.actions)
        if self.f1.shape != (n, k) or self.cost.shape != (n, k):
            raise ValueError(f"f1 and cost must both be ({n}, {k}), got "
                             f"{self.f1.shape} and {self.cost.shape}")
        if not np.all(np.isfinite(self.f1)) or not np.all(np.isfinite(self.cost)):
            raise ValueError("outcome table contains NaN or Inf")

    def __len__(self):
        return len(self.instances)

    def index(self, action):
        """Column of an action label."""
        if action not in self.actions:
            raise ValueError(f"unknown action {action!r}; this table has {self.actions}")
        return self.actions.index(action)

    @classmethod
    def from_rows(cls, rows, quality=QUALITY, cost=COST, actions=ACTIONS):
        """Build from cached rows (section 7.4).

        rows: dicts with cluster_id, variant_id, action, the quality field, and the cost
        field(s). `cost` is one field name or a sequence of field names to sum, so
        component latencies (retrieval + rerank + rewrite + generation) also work.
        """
        cost_keys = [cost] if isinstance(cost, str) else list(cost)
        if not cost_keys:
            raise ValueError("cost needs at least one field name")
        actions = tuple(actions)

        seen = {}
        for r in rows:
            key = (r["cluster_id"], r["variant_id"])
            action = r["action"]
            if action not in actions:
                raise ValueError(f"unknown action {action!r} for instance {key}")
            if (key, action) in seen:
                raise ValueError(f"duplicate cached row for {key}, action {action}")
            seen[(key, action)] = r

        instances = sorted({k for k, _ in seen}, key=lambda k: (str(k[0]), str(k[1])))
        f1 = np.empty((len(instances), len(actions)), dtype=float)
        cst = np.empty((len(instances), len(actions)), dtype=float)
        for i, key in enumerate(instances):
            for j, action in enumerate(actions):
                r = seen.get((key, action))
                if r is None:
                    raise ValueError(f"instance {key} has no cached row for {action}; "
                                     "the outcome matrix must cover every action")
                f1[i, j] = float(r[quality])
                cst[i, j] = float(sum(r[k] for k in cost_keys))
        name = cost if isinstance(cost, str) else "+".join(cost_keys)
        return cls(instances, f1, cst, quality_key=quality, cost_key=name, actions=actions)

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


def action_indices(actions, labels, n):
    """Action labels or indices -> int array of length n, checked against `labels`."""
    labels = tuple(labels)
    a = np.asarray(actions)
    if a.shape != (n,):
        raise ValueError(f"need one action per instance ({n}), got {a.shape}")
    if a.dtype.kind in "US":
        unknown = set(np.unique(a).tolist()) - set(labels)
        if unknown:
            raise ValueError(f"unknown action label(s): {sorted(unknown)}")
        return np.array([labels.index(x) for x in a.tolist()], dtype=int)
    idx = a.astype(int)
    if not np.all((idx >= 0) & (idx < len(labels))):
        raise ValueError(f"action indices must be in 0..{len(labels) - 1}")
    return idx


def policy_value(table, actions):
    """(mean quality, mean cost) of one instance -> action assignment.

    The shared primitive: the section 5 baselines, the always-A* reference points and the
    Pareto sweep are all different ways of producing `actions` for this function.
    """
    idx = action_indices(actions, table.actions, len(table))
    rows = np.arange(len(table))
    return float(table.f1[rows, idx].mean()), float(table.cost[rows, idx].mean())


def always_policy(table, action):
    """(mean quality, mean cost) of sending every instance to one fixed action."""
    j = table.index(action)
    return float(table.f1[:, j].mean()), float(table.cost[:, j].mean())


def action_counts(actions, labels=ACTIONS):
    """{action: count} over an assignment, including actions never selected.

    Section 3.4 removed the 5% usage floor: an unused action is a finding to report, not
    something to force, so zero counts are kept rather than dropped.
    """
    a = np.asarray(actions)
    return {x: int((a == x).sum()) for x in labels}


def _grid_values(table, scores, grid):
    """(mean quality, mean cost) for every cut in the grid, in grid order."""
    return [policy_value(table, assign_action(scores, *cut, actions=table.actions))
            for cut in grid]


def _cut_fields(cut):
    """Result fields naming a cut: theta1 / theta2 for v0.3 pairs, theta for v0.4."""
    if len(cut) == 2:
        return {"theta1": cut[0], "theta2": cut[1]}
    if len(cut) == 1:
        return {"theta": cut[0]}
    return {}


def select_thresholds(table, scores, budget=None, grid=None):
    """Grid-search the thresholds on the validation split (section 3.4).

    Maximizes mean quality subject to mean cost <= budget. The budget defaults to the
    always-A1 mean cost (v0.3). v0.4 passes its own budget: with only A1 and A2, the always-A1
    budget can only be met by never escalating (REDESIGN section 3). Feasibility is checked
    rather than assumed.

    Ties on mean quality are broken by lower mean cost, then lower thresholds in order
    (user decision; keeps the result deterministic for section 7.5).

    Returns a dict with the chosen cut ("thresholds", plus theta1 / theta2 or theta), its
    quality and cost, the budget, how many cuts were feasible, the action counts, and any action
    never selected.
    """
    s = table.align(scores) if isinstance(scores, dict) else np.asarray(scores, float)
    if s.shape != (len(table),):
        raise ValueError(f"need one score per instance ({len(table)}), got {s.shape}")
    grid = default_grid(table.actions) if grid is None else [tuple(c) for c in grid]
    for cut in grid:
        check_cut(cut, len(table.actions))

    if budget is None:
        budget = always_policy(table, "A1")[1]
    budget = float(budget)

    values = _grid_values(table, s, grid)
    feasible = [(q, c, cut) for (q, c), cut in zip(values, grid) if c <= budget]
    if not feasible:
        cheapest = min(c for _, c in values)
        raise ValueError(
            f"no threshold pair meets the cost budget {budget:.6g}; the cheapest "
            f"reachable policy costs {cheapest:.6g}. The always-A1 budget is a bound, "
            "not necessarily attainable by thresholds."
        )

    quality, cost, cut = min(feasible, key=lambda v: (-v[0], v[1], v[2]))
    counts = action_counts(assign_action(s, *cut, actions=table.actions), table.actions)
    return {
        "thresholds": cut,
        **_cut_fields(cut),
        "quality": quality,
        "cost": cost,
        "budget": budget,
        "quality_key": table.quality_key,
        "cost_key": table.cost_key,
        "n_feasible": len(feasible),
        "n_grid": len(grid),
        "counts": counts,
        "unused_actions": [a for a in table.actions if counts[a] == 0],
    }


def pareto_frontier(table, scores, grid=None):
    """Best achievable quality at each cost level (section 3.4 Pareto reporting).

    Sweep points are the distinct mean costs the grid actually achieves (user decision),
    so the staircase is exact. Quality is non-decreasing in budget by construction, since
    a larger budget admits every cut a smaller one did.

    Returns [{budget, quality, cost, thresholds, theta1 / theta2 or theta}, ...] by budget.
    """
    s = table.align(scores) if isinstance(scores, dict) else np.asarray(scores, float)
    grid = default_grid(table.actions) if grid is None else [tuple(c) for c in grid]
    values = _grid_values(table, s, grid)
    points = [(q, c, cut) for (q, c), cut in zip(values, grid)]

    frontier = []
    for budget in sorted({c for _, c, _ in points}):
        under = [p for p in points if p[1] <= budget]
        quality, cost, cut = min(under, key=lambda v: (-v[0], v[1], v[2]))
        frontier.append({"budget": budget, "quality": quality, "cost": cost,
                         "thresholds": cut, **_cut_fields(cut)})
    return frontier
