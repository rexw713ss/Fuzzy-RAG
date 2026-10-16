"""Paired, cluster-resampled evaluation of frozen policies."""

import numpy as np

from main_logic import evaluation as ev
from main_logic.routing import OutcomeTable


def arrays(split, actions):
    a = np.asarray(actions)
    if a.shape != (len(split),) or a.dtype.kind not in "iu" or not np.isin(a, range(len(split.actions))).all():
        raise ValueError("one valid action index required per row")
    idx = np.arange(len(split))
    return {"F1": split.f1[idx, a], "EM": split.em[idx, a], "cost": split.cost[idx, a]}


def summarize(split, actions, n_boot=1000, seed=42):
    values = arrays(split, actions)
    a = np.asarray(actions, dtype=int)
    result = {k: ev.bootstrap_ci(v, split.groups, n_boot=n_boot, seed=seed)
              for k, v in values.items()}
    table = OutcomeTable([(g, key[2]) for g, key in zip(split.groups, split.keys)],
                         split.f1, split.cost, actions=split.actions)
    ids, flips, harm, skipped = ev.pair_flags(table, a)
    result.update({"counts": {label: int((a == i).sum()) for i, label in enumerate(split.actions)},
                   "A2_fraction": float(np.mean(a == split.actions.index("A2"))),
                   "robustness_incomplete_clusters_skipped": skipped,
                   "robustness_pairs": len(ids)})
    for name, v in (("RFR", flips), ("HRFR_legacy_action_order", harm)):
        result[name] = ev.bootstrap_ci(v.astype(float), ids, n_boot=n_boot, seed=seed) if ids else None
    # HRFR uses the legacy definition (A0<A1<A2), not actual measured cost order.
    result["secondary"] = {k: float(v[np.arange(len(split)), a].mean())
                           for k, v in split.secondary.items()}
    complete, _ = ev.complete_clusters(table)
    quality = values["F1"]
    cluster_ids = list(complete)
    result["within_cluster"] = {}
    for name, function in (("F1_SD", np.std), ("F1_range", np.ptp), ("minimum_F1", np.min)):
        per_cluster = np.array([function(quality[rows]) for rows in complete.values()])
        result["within_cluster"][name] = (
            ev.bootstrap_ci(per_cluster, cluster_ids, n_boot=n_boot, seed=seed) if cluster_ids else None)
    return result


def paired(split, actions, other_actions=None, random_fraction=None, n_boot=1000, seed=42):
    selected = arrays(split, actions)
    if (other_actions is None) == (random_fraction is None):
        raise ValueError("supply exactly one comparison")
    if random_fraction is not None:
        p = float(random_fraction)
        if not 0 <= p <= 1:
            raise ValueError("random_fraction must lie in [0, 1]")
        # Expected uniform random allocation with the SAME number of A2 calls.
        # The probability comes only from the frozen policy's action count.
        # No test F1/cost is used to optimize mixture weights.
        base, escalation = (split.actions.index(label) for label in ("A1", "A2"))
        reference = {k: (1 - p) * v[:, base] + p * v[:, escalation]
                     for k, v in (("F1", split.f1), ("EM", split.em), ("cost", split.cost))}
    else:
        reference = arrays(split, other_actions)
    result = {k: ev.paired_bootstrap(v, reference[k], split.groups, n_boot=n_boot, seed=seed)
              for k, v in selected.items()}
    if random_fraction is not None:
        result["reference"] = {"A2_fraction": p, "kind": "expected_uniform_random_allocation",
                               "means": {k: float(v.mean()) for k, v in reference.items()},
                               "CI_scope": "cluster sampling only; random allocation integrated analytically"}
    return result
