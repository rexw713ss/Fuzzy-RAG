"""v0.4 step 9: fit the two-action routers on validation and evaluate them on the fresh held-out
clusters (primary) and on the v0.3 test split (secondary, already seen in the first run)
(REDESIGN_2026-10-15 sections 3, 5 and 7; rules fixed in RUNLOG_2026-10-15_v04 section 0).

Run from the repo root, in the main environment, after scripts.escalate has written
V4/<dataset>/outcomes.jsonl and V4/fresh/<dataset>/outcomes.jsonl for both datasets:
    .venv\\Scripts\\python.exe -m scripts.evaluate_v4 [--full DIR] [--v4 DIR]

Everything is a selection over cached outcome rows; no model is called.

Per dataset:
  - actions A1 (answer now) and A2 (escalate), one threshold;
  - signals S2, S3, S4, scaled with the v0.3 training scalers (fresh raw signals are scaled the
    same way); B4 widths are the v0.3 jitter widths of S2-S4 (calibration is per signal, so
    dropping S1 leaves the others' widths unchanged);
  - budget: mean latency <= midpoint of always-A1 and always-A2 on VALIDATION;
  - fitted on validation: B1, B2, B3, B4, B4's three operator ablations, B4-4sig (the v0.3
    controller on S1-S4, to show what dropping S1 does) and the query-only router (trained on
    the train split's oracle labels, two classes);
  - signal check on train + val: AUC of each signal and router score for "A2 beats A1";
  - reported on each test set: F1 / EM / cost with 95% cluster-bootstrap CIs, action counts,
    RFR / HRFR, within-cluster sd and worst drop; always-A1, always-A2, the oracle; paired
    cluster bootstrap of B4 against every router and against the best random mix of always-A1
    and always-A2 at B4's cost; the paper criteria;
  - Pareto: each router's validation frontier, every point measured on the fresh set;
  - monotonicity at B4's threshold (3-signal controller) and rule activations on the fresh set.
Pooled: both datasets' test instances, each routed by its own dataset's routers.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from main_logic import baselines as bl
from main_logic import checks as ck
from main_logic import evaluation as ev
from main_logic import fuzzy as fz
from main_logic import learned as lr
from main_logic import monotonicity as mono
from main_logic import routing as rt
from main_logic import signals as sg
from scripts import datasets as ds
from scripts.retrieve_clusters import read_jsonl

ACTIONS = rt.ACTIONS_V4
SIG3 = fz.SIGNALS_V4
ABLATIONS = {f"B4-{k}": v for k, v in fz.ABLATIONS.items() if k != "main"}
SECONDARY = ("retrievals", "ce_passes", "llm_calls", "input_tokens", "output_tokens")
FIXED = ("always-A1", "always-A2")


def key_of(r):
    return (r["cluster_id"], r["variant_id"])


def table_for(rows, keys):
    keys = set(keys)
    return rt.OutcomeTable.from_rows([r for r in rows if key_of(r) in keys], actions=ACTIONS)


class Router:
    def __init__(self, name, score, fit):
        self.name, self.score, self.fit = name, score, fit

    def actions(self, table):
        return rt.assign_action(self.score(table), *self.fit["thresholds"], actions=ACTIONS)


def fit_routers(val, train, sig3, sig4, widths3, widths4, questions, budget):
    """Every router, fitted on validation at the fixed budget."""
    m3 = lambda t: bl.signal_matrix(t, sig3, 3)
    m4 = lambda t: bl.signal_matrix(t, sig4, 4)
    routers, errors = {}, {}

    def add(name, score, grid=None, fit=None):
        try:
            f = fit() if fit else rt.select_thresholds(val, score(val), budget, grid)
            routers[name] = Router(name, score, f)
        except ValueError as e:                      # infeasible budget
            errors[name] = str(e)

    def b1():
        return bl.fit_b1(val, sig3, budget)
    add("B1", lambda t: bl.weighted_scores(m3(t), routers["B1"].fit["weights"]), fit=b1)
    add("B2", lambda t: bl.ordinal_sums(m3(t)), grid=bl.cut_grid(3, 2))
    add("B3", lambda t: bl.fuzzy_scores(m3(t), fz.W_DEFAULT, SIG3))
    add("B4", lambda t: bl.fuzzy_scores(m3(t), widths3, SIG3))
    for name, ops in ABLATIONS.items():
        add(name, lambda t, ops=ops: bl.fuzzy_scores(m3(t), widths3, SIG3, **ops))
    add("B4-4sig", lambda t: bl.fuzzy_scores(m4(t), widths4, fz.SIGNALS))
    model = lr.fit([questions[k] for k in train.instances], ev.oracle_actions(train), n_classes=2)
    add("query-only", lambda t: lr.scores(model, [questions[k] for k in t.instances]))
    return routers, errors


def mixture(table, cost):
    """Per-instance expected F1 of the random mix of always-A1 and always-A2 whose mean cost is
    `cost` (clipped to the two fixed policies), and the mixing weight on A2."""
    c1, c2 = table.cost[:, 0].mean(), table.cost[:, 1].mean()
    lam = float(np.clip((cost - c1) / (c2 - c1), 0.0, 1.0)) if c2 != c1 else 0.0
    return (1 - lam) * table.f1[:, 0] + lam * table.f1[:, 1], lam


def measure(table, actions, extra, scores=None):
    """Test-side metrics of one assignment of actions."""
    a = rt.action_indices(actions, ACTIONS, len(table))
    rows = np.arange(len(table))
    groups = [k[0] for k in table.instances]
    f1, cost = table.f1[rows, a], table.cost[rows, a]
    labels = [ACTIONS[j] for j in a]
    out = {"F1": ev.bootstrap_ci(f1, groups), "cost": ev.bootstrap_ci(cost, groups),
           "EM": float(np.mean([extra[k][x]["EM"] for k, x in zip(table.instances, labels)])),
           "counts": rt.action_counts(np.array(labels), ACTIONS),
           "secondary": {s: float(np.mean([extra[k][x][s] for k, x in zip(table.instances, labels)]))
                         for s in SECONDARY}}
    ids, flipped, harmful, skipped = ev.pair_flags(table, a)
    if ids:
        out["RFR"] = ev.bootstrap_ci(flipped.astype(float), ids)
        out["HRFR"] = ev.bootstrap_ci(harmful.astype(float), ids)
        out["pairs"], out["clusters_skipped"] = len(ids), skipped
    if scores is not None:
        out["within_cluster"] = ev.within_cluster(table, a, scores)
    out["_f1"], out["_cost"], out["_harm"], out["_pair_ids"] = f1, cost, harmful, ids
    return out


def strip(m):
    return {k: v for k, v in m.items() if not k.startswith("_")}


def paired(test, ref, other):
    groups = [k[0] for k in test.instances]
    res = {"F1": ev.paired_bootstrap(ref["_f1"], other["_f1"], groups),
           "cost": ev.paired_bootstrap(ref["_cost"], other["_cost"], groups)}
    if len(ref["_pair_ids"]):
        res["HRFR"] = ev.paired_bootstrap(ref["_harm"].astype(float), other["_harm"].astype(float),
                                          ref["_pair_ids"])
    return res


def evaluate_set(test, routers, extra):
    """Every router and reference policy on one test table."""
    results, actions = {}, {}
    for a in ACTIONS:
        results[f"always-{a}"] = measure(test, [a] * len(test), extra)
        actions[f"always-{a}"] = {k: a for k in test.instances}
    orc = [ACTIONS[i] for i in ev.oracle_actions(test)]
    results["oracle"] = measure(test, orc, extra)
    actions["oracle"] = dict(zip(test.instances, orc))
    for name, r in routers.items():
        acts = r.actions(test)
        results[name] = measure(test, acts, extra, r.score(test))
        actions[name] = dict(zip(test.instances, acts.tolist()))

    comparisons, criteria = {}, None
    if "B4" in results:
        b4 = results["B4"]
        comparisons = {n: paired(test, b4, results[n]) for n in results if n != "B4"}
        mix_f1, lam = mixture(test, b4["cost"]["mean"])
        groups = [k[0] for k in test.instances]
        comparisons["mixture"] = {"lambda_A2": lam,
                                  "F1": ev.paired_bootstrap(b4["_f1"], mix_f1, groups)}
        base = [n for n in ("B1", "B2", "B3") if n in results]
        strongest = max(base, key=lambda n: results[n]["F1"]["mean"]) if base else None
        hrfr = comparisons[strongest].get("HRFR") if strongest else None
        criteria = {
            "actions_used": sum(v > 0 for v in b4["counts"].values()),
            "uses_both_actions": all(v > 0 for v in b4["counts"].values()),
            "strongest_baseline": strongest,
            "HRFR_B4_minus_strongest": hrfr,
            "lower_HRFR_than_strongest": (bool(hrfr["significant"] and hrfr["diff"] < 0)
                                          if hrfr else None),
            "F1_within_1_point": (abs(b4["F1"]["mean"] - results[strongest]["F1"]["mean"]) <= 0.01
                                  if strongest else None),
            "cost_not_higher": (b4["cost"]["mean"] <= results[strongest]["cost"]["mean"]
                                if strongest else None),
            "mixture_F1_at_B4_cost": float(mix_f1.mean()),
            "beats_fixed_and_mixtures": bool(b4["F1"]["mean"] >= mix_f1.mean()),
        }
    return results, actions, comparisons, criteria


def auc_table(table, scores_by_name):
    """AUC of each score for 'A2 beats A1 by more than delta' (train + val signal check)."""
    y = table.f1[:, 1] > table.f1[:, 0] + ev.DELTA + ev.TOL
    out = {"positives": int(y.sum()), "instances": len(y)}
    for name, s in scores_by_name.items():
        out[name] = ck.auc(np.asarray(s, dtype=float), y) if 0 < y.sum() < len(y) else None
    return out


def scale_fresh(raw_rows, scalers):
    """Fresh raw signals scaled with the v0.3 training scalers; fraction clipped per signal."""
    out, clipped = {}, {}
    for s in fz.SIGNALS:
        vals = np.array([r[s] for r in raw_rows], dtype=float)
        scaled, clipped[s] = sg.apply_scaler(vals, scalers[s])
        for r, v in zip(raw_rows, scaled):
            out.setdefault(key_of(r), {})[s] = float(v)
    return out, clipped


def evaluate_dataset(d, full, v4):
    splits = json.loads((full / "splits.json").read_text())[d]
    split_of = {c: s for s, cs in splits.items() for c in cs}
    rows = read_jsonl(v4 / d / "outcomes.jsonl")
    keys = {key_of(r) for r in rows}
    by_split = {s: [k for k in keys if split_of[k[0]] == s] for s in ("train", "val", "test")}
    train, val, old_test = (table_for(rows, by_split[s]) for s in ("train", "val", "test"))

    scaled = {key_of(r): r for r in read_jsonl(full / "signals_scaled.jsonl") if r["dataset"] == d}
    jitter = json.loads((full / "jitter.json").read_text())[d]
    widths4 = jitter["widths"]
    widths3 = [jitter["signals"][s]["width"] for s in SIG3]
    questions = {key_of(q): q["question"] for q in read_jsonl(full / d / "queries.jsonl")}
    extra = {}
    for r in rows:
        extra.setdefault(key_of(r), {})[r["action"]] = r

    # Fresh held-out clusters (primary test set): clusters, retrieval and raw signals in
    # V4/fresh_src/<d> (scripts.build_clusters --offset, scripts.retrieve_clusters), outcome
    # rows in V4/fresh/<d> (scripts.escalate).
    fresh_dir, fresh_src = v4 / "fresh" / d, v4 / "fresh_src" / d
    fresh = None
    if (fresh_dir / "outcomes.jsonl").exists():
        frows = read_jsonl(fresh_dir / "outcomes.jsonl")
        fresh = table_for(frows, {key_of(r) for r in frows})
        scalers = json.loads((full / "scalers.json").read_text())[d]
        fscaled, fclipped = scale_fresh(read_jsonl(fresh_src / "signals.jsonl"), scalers)
        scaled.update({k: {**v, "dataset": d} for k, v in fscaled.items()})
        questions.update({key_of(q): q["question"] for q in read_jsonl(fresh_src / "queries.jsonl")})
        for r in frows:
            extra.setdefault(key_of(r), {})[r["action"]] = r

    sig3 = {k: tuple(v[s] for s in SIG3) for k, v in scaled.items()}
    sig4 = {k: tuple(v[s] for s in fz.SIGNALS) for k, v in scaled.items()}

    # Budget, fixed on validation before any test set is looked at.
    a1_cost, a2_cost = (rt.always_policy(val, a)[1] for a in ACTIONS)
    budget = (a1_cost + a2_cost) / 2
    routers, errors = fit_routers(val, train, sig3, sig4, widths3, widths4, questions, budget)

    # Signal check (REDESIGN step 5). On train + val: the signals and the routers whose ranking
    # has no fitted parameter (B2, B3, B4, ablations). B1's weights are chosen on val and the
    # query-only model is trained on train, so every score is also checked on val alone.
    tv = table_for(rows, by_split["train"] + by_split["val"])
    m = bl.signal_matrix(tv, sig3, 3)
    fixed_rank = [n for n in routers if n not in ("B1", "query-only")]
    signal_check = auc_table(tv, {**{s: m[:, j] for j, s in enumerate(SIG3)},
                                  **{n: routers[n].score(tv) for n in fixed_rank}})
    mv = bl.signal_matrix(val, sig3, 3)
    signal_check["val_only"] = auc_table(val, {**{s: mv[:, j] for j, s in enumerate(SIG3)},
                                               **{n: r.score(val) for n, r in routers.items()}})
    rho = np.corrcoef(np.array([ck.ranks(m[:, j]) for j in range(m.shape[1])]))
    signal_check["spearman"] = {f"{a}-{b}": float(rho[i, j]) for i, a in enumerate(SIG3)
                                for j, b in enumerate(SIG3) if i < j}

    sets = {}
    if fresh is not None:
        sets["fresh"] = fresh
    if len(old_test):                           # absent until A2 has run on the old test split
        sets["old_test"] = old_test
    report = {
        "dataset": d,
        "instances": {"train": len(train), "val": len(val), **{n: len(t) for n, t in sets.items()}},
        "clusters": {n: len({k[0] for k in t.instances}) for n, t in
                     (("train", train), ("val", val), *sets.items())},
        "budget": {"rule": "midpoint of always-A1 and always-A2 mean latency on validation",
                   "always_A1_val": a1_cost, "always_A2_val": a2_cost, "budget": budget},
        "widths_B4": dict(zip(SIG3, widths3)),
        "fits": {n: r.fit for n, r in routers.items()},
        "fit_errors": errors,
        "signal_check_train_val": signal_check,
        "oracle_shares": {n: ev.oracle_shares(t) for n, t in (("train_val", tv), *sets.items())},
    }
    if fresh is not None:
        report["fresh_signal_clipping"] = fclipped
    all_actions = {}
    for name, t in sets.items():
        results, actions, comparisons, criteria = evaluate_set(t, routers, extra)
        report[name] = {"results": {n: strip(x) for n, x in results.items()},
                        "B4_vs": comparisons, "criteria": criteria}
        all_actions[name] = actions

    # Pareto (validation frontier, measured on the primary test set).
    target = sets.get("fresh", sets.get("old_test"))
    pareto = {}
    for name, r in routers.items():
        grid = bl.cut_grid(3, 2) if name == "B2" else None
        s_t = r.score(target)
        pareto[name] = []
        for p in rt.pareto_frontier(val, r.score(val), grid):
            q, c = rt.policy_value(target, rt.assign_action(s_t, *p["thresholds"], actions=ACTIONS))
            pareto[name].append({"val_budget": p["budget"], "thresholds": p["thresholds"],
                                 "val_F1": p["quality"], "test_F1": q, "test_cost": c})
    report["pareto"] = {"measured_on": "fresh" if "fresh" in sets else "old_test", "routers": pareto}

    if "B4" in routers:
        f = routers["B4"].fit
        report["monotonicity_at_B4_theta"] = mono.report(
            mono.grid_scores(widths=widths3, signals=SIG3), thresholds=[f["theta"]],
            signals=SIG3, theta=f["theta"])["selected"]
        fired = {}
        for row in bl.signal_matrix(target, sig3, 3):
            for rule in fz.escalate(row, widths3, signals=SIG3)[1]["fired"]:
                fired[rule["rule_id"]] = fired.get(rule["rule_id"], 0) + 1
        report["rule_activations"] = {"rules_fired": len(fired), "counts": dict(sorted(fired.items()))}
    return report, rows + (frows if fresh is not None else []), all_actions, extra


def plot_pareto(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_name = report["pareto"]["measured_on"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, pts in report["pareto"]["routers"].items():
        if name.startswith("B4-") and name != "B4-4sig":
            continue
        pts = sorted(pts, key=lambda p: p["test_cost"])
        ax.plot([p["test_cost"] for p in pts], [p["test_F1"] for p in pts], marker=".",
                label=name, linewidth=1.0)
    res = report[set_name]["results"]
    for name in FIXED + ("oracle",):
        m = res[name]
        ax.scatter(m["cost"]["mean"], m["F1"]["mean"], marker="*", s=110, color="black", zorder=3)
        ax.annotate(name, (m["cost"]["mean"], m["F1"]["mean"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.axvline(report["budget"]["budget"], color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel(f"mean latency per query (s), {set_name}")
    ax.set_ylabel(f"mean F1, {set_name}")
    ax.set_title(f"{report['dataset']} (v0.4): thresholds chosen on validation at each budget")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_set(title, res, criteria):
    print(f"   -- {title}")
    for n, m in res.items():
        print(f"   {n:12s} F1 {m['F1']['mean']:.3f} [{m['F1']['lo']:.3f},{m['F1']['hi']:.3f}]"
              f"  cost {m['cost']['mean']:6.2f}s"
              + (f"  RFR {m['RFR']['mean']:.3f} HRFR {m['HRFR']['mean']:.3f}" if "RFR" in m else "")
              + f"  {m['counts']}")
    if criteria:
        print("   criteria:", json.dumps(criteria, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=None, help="v0.3 run folder (default data/full)")
    ap.add_argument("--v4", default=None, help="v0.4 folder (default data/v4)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    full = Path(args.full or ds.DATA / "full")
    v4 = Path(args.v4 or ds.DATA / "v4")
    out = v4 / "results"
    out.mkdir(exist_ok=True)

    pooled_rows, pooled_actions, pooled_extra, summary = [], {}, {}, {}
    for d in ds.NAMES:
        report, rows, actions, extra = evaluate_dataset(d, full, v4)
        (out / f"{d}.json").write_text(json.dumps(report, indent=2, default=float))
        plot_pareto(report, out / f"pareto_{d}.png")
        pooled_rows += rows
        pooled_extra.update(extra)
        for set_name, acts in actions.items():
            for n, a in acts.items():
                pooled_actions.setdefault(set_name, {}).setdefault(n, {}).update(a)
        summary[d] = report

    pooled = {}
    for set_name, acts in pooled_actions.items():
        keys = set(acts["always-A1"])
        t = table_for(pooled_rows, keys)
        res = {n: measure(t, [a[k] for k in t.instances], pooled_extra)
               for n, a in acts.items() if set(a) == keys}
        comp = {n: paired(t, res["B4"], res[n]) for n in res if n != "B4"} if "B4" in res else {}
        pooled[set_name] = {"results": {n: strip(m) for n, m in res.items()}, "B4_vs": comp}
    (out / "pooled.json").write_text(json.dumps(pooled, indent=2, default=float))

    for d, r in summary.items():
        print(f"\n== {d}: instances {r['instances']}; budget {r['budget']['budget']:.2f}s "
              f"(A1 {r['budget']['always_A1_val']:.2f}, A2 {r['budget']['always_A2_val']:.2f})")
        if r["fit_errors"]:
            print("   fit errors:", r["fit_errors"])
        sc = r["signal_check_train_val"]
        print("   AUC 'A2 beats A1' (train+val, positives %d/%d): " % (sc["positives"], sc["instances"])
              + ", ".join(f"{k} {v:.3f}" for k, v in sc.items()
                          if isinstance(v, float)))
        for set_name in ("fresh", "old_test"):
            if set_name in r:
                print_set(set_name, r[set_name]["results"], r[set_name]["criteria"])
    for set_name, p in pooled.items():
        print(f"\n== pooled {set_name}")
        for n, m in p["results"].items():
            print(f"   {n:12s} F1 {m['F1']['mean']:.3f} [{m['F1']['lo']:.3f},{m['F1']['hi']:.3f}]"
                  f"  cost {m['cost']['mean']:6.2f}s"
                  + (f"  RFR {m['RFR']['mean']:.3f} HRFR {m['HRFR']['mean']:.3f}" if "RFR" in m else ""))


if __name__ == "__main__":
    main()
