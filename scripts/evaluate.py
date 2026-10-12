"""Pipeline step 9: fit the routers on validation, evaluate on test (spec v0.3, sections 3.4, 5, 6).

Run from the repo root, in the main environment, after scripts.stage1_checks (steps 2, 6, 7) and
scripts.outcomes (step 8) for both datasets:
    .venv\\Scripts\\python.exe -m scripts.evaluate [--root DIR]

Everything is a selection over the cached outcome rows; no model is called.

Per dataset:
  - fitted on VALIDATION (section 3.4 objective, always-A1 budget, tie-break): B1, B2, B3, B4
    (section 5), the three operator ablations of B4 (section 3.3: product t-norm, probabilistic
    sum, Sugeno-0), and the query-only learned router (trained on TRAIN oracle labels);
  - reported on TEST: mean F1 and EM, mean cost, secondary counts, action counts, RFR and HRFR
    (section 6.2), within-cluster sd and worst-case drop, each with a 95% cluster-bootstrap CI;
    always-A0/A1/A2 and the oracle as reference points; paired cluster bootstrap of B4 against
    every other router (F1, cost, harmful flips); the best randomized mixture of the fixed
    policies at B4's cost; the paper's success criteria;
  - the stage 3 question on ALL instances: how often A1 or A2 is required (oracle shares);
  - Pareto: for each router, the validation frontier (best F1 at every budget the grid reaches),
    each point's thresholds then measured on test -> results/pareto_<dataset>.png;
  - section 4.2 item 5 (monotonicity deviations that change B4's action at its thresholds) and
    section 9.4 rule activation counts on test.
The pooled result concatenates both datasets' test instances, each routed by its own dataset's
fitted routers.

Choices the spec leaves open (DECISIONS.md section 14): routers are fitted per dataset (scalers
and widths are per dataset); "strongest matched baseline" = the B1-B3 router with the highest
test F1; "F1 within 1 point" = |F1(B4) - F1(baseline)| <= 0.01.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from main_logic import baselines as bl
from main_logic import evaluation as ev
from main_logic import fuzzy as fz
from main_logic import learned as lr
from main_logic import monotonicity as mono
from main_logic import routing as rt
from main_logic.routing import ACTIONS
from scripts import datasets as ds
from scripts.retrieve_clusters import read_jsonl

ABLATIONS = {f"B4-{k}": v for k, v in fz.ABLATIONS.items() if k != "main"}
SECONDARY = ("retrievals", "ce_passes", "llm_calls", "input_tokens", "output_tokens")
FIXED = ("always-A0", "always-A1", "always-A2")


def table_for(rows, keys):
    keys = set(keys)
    return rt.OutcomeTable.from_rows([r for r in rows if (r["cluster_id"], r["variant_id"]) in keys])


def mixture_f1(points, cost):
    """Best F1 a random mixture of the fixed policies reaches at mean cost <= cost, or None if
    even the cheapest fixed policy costs more. points: [(cost, f1), ...]."""
    best = max((f for c, f in points if c <= cost), default=None)
    for (c1, f1), (c2, f2) in ((p, q) for p in points for q in points):
        if c1 <= cost < c2:
            lam = (cost - c1) / (c2 - c1)
            v = (1 - lam) * f1 + lam * f2
            best = v if best is None else max(best, v)
    return best


class Router:
    """A fitted router: a score function on a table plus its validation-selected thresholds."""

    def __init__(self, name, score, fit):
        self.name, self.score, self.fit = name, score, fit

    def actions(self, table):
        return rt.assign_action(self.score(table), self.fit["theta1"], self.fit["theta2"])


def fit_routers(val, train, signals, widths, questions):
    """Every router of section 5 (+ ablations, query-only), fitted on validation."""
    sm = lambda t: bl.signal_matrix(t, signals)
    routers, errors = {}, {}

    def add(name, score, grid=None, fitter=None):
        try:
            fit = fitter() if fitter else rt.select_thresholds(val, score(val), grid=grid)
            routers[name] = Router(name, score, fit)
        except ValueError as e:                      # infeasible budget (section 9.2 O-3)
            errors[name] = str(e)

    try:
        b1 = bl.fit_b1(val, signals)
        routers["B1"] = Router("B1", lambda t: bl.weighted_scores(sm(t), b1["weights"]), b1)
    except ValueError as e:
        errors["B1"] = str(e)
    add("B2", lambda t: bl.ordinal_sums(sm(t)), grid=bl.cut_grid())
    add("B3", lambda t: bl.fuzzy_scores(sm(t), fz.W_DEFAULT))
    add("B4", lambda t: bl.fuzzy_scores(sm(t), widths))
    for name, ops in ABLATIONS.items():
        add(name, lambda t, ops=ops: np.array([fz.escalate(r, widths, **ops)[0] for r in sm(t)]))
    model = lr.fit([questions[k] for k in train.instances], ev.oracle_actions(train))
    add("query-only", lambda t: lr.scores(model, [questions[k] for k in t.instances]))
    return routers, errors, model


def measure(table, actions, extra, scores=None):
    """Test-side metrics of one assignment of actions."""
    a = np.array([ACTIONS.index(x) if isinstance(x, str) else int(x) for x in actions])
    rows = np.arange(len(table))
    groups = [k[0] for k in table.instances]
    f1, cost = table.f1[rows, a], table.cost[rows, a]
    out = {"F1": ev.bootstrap_ci(f1, groups), "cost": ev.bootstrap_ci(cost, groups),
           "EM": float(np.mean([extra[k][ACTIONS[j]]["EM"] for k, j in zip(table.instances, a)])),
           "counts": rt.action_counts(np.array([ACTIONS[j] for j in a]))}
    out["secondary"] = {s: float(np.mean([extra[k][ACTIONS[j]][s] for k, j in zip(table.instances, a)]))
                        for s in SECONDARY}
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


def evaluate_dataset(d, root):
    rows = read_jsonl(root / d / "outcomes.jsonl")
    splits = json.loads((root / "splits.json").read_text())[d]
    split_of = {c: s for s, cs in splits.items() for c in cs}
    keys = {(r["cluster_id"], r["variant_id"]) for r in rows}
    by_split = {s: [k for k in keys if split_of[k[0]] == s] for s in ("train", "val", "test")}
    train, val, test = (table_for(rows, by_split[s]) for s in ("train", "val", "test"))
    full = table_for(rows, keys)

    signals = {(r["cluster_id"], r["variant_id"]): tuple(r[s] for s in fz.SIGNALS)
               for r in read_jsonl(root / "signals_scaled.jsonl") if r["dataset"] == d}
    widths = json.loads((root / "jitter.json").read_text())[d]["widths"]
    questions = {(q["cluster_id"], q["variant_id"]): q["question"]
                 for q in read_jsonl(root / d / "queries.jsonl")}
    extra = {}
    for r in rows:
        extra.setdefault((r["cluster_id"], r["variant_id"]), {})[r["action"]] = r

    routers, errors, _ = fit_routers(val, train, signals, widths, questions)

    # Test.
    results, test_actions = {}, {}
    for a in ACTIONS:
        results[f"always-{a}"] = measure(test, [a] * len(test), extra)
    results["oracle"] = measure(test, ev.oracle_actions(test), extra)
    for name, r in routers.items():
        acts = r.actions(test)
        results[name] = measure(test, acts, extra, r.score(test))
        test_actions[name] = dict(zip(test.instances, acts))
    test_actions.update({f"always-{a}": {k: a for k in test.instances} for a in ACTIONS})
    test_actions["oracle"] = dict(zip(test.instances, [ACTIONS[i] for i in ev.oracle_actions(test)]))

    comparisons = {n: paired(test, results["B4"], results[n]) for n in results
                   if "B4" in results and n != "B4"}

    fixed = [(results[f]["cost"]["mean"], results[f]["F1"]["mean"]) for f in FIXED]
    criteria = None
    if "B4" in results:
        b4 = results["B4"]
        base = [n for n in ("B1", "B2", "B3") if n in results]
        strongest = max(base, key=lambda n: results[n]["F1"]["mean"]) if base else None
        mix = mixture_f1(fixed, b4["cost"]["mean"])
        criteria = {
            "strongest_baseline": strongest,
            "lower_HRFR_than_strongest": (comparisons[strongest]["HRFR"] if strongest else None),
            "F1_within_1_point": (abs(b4["F1"]["mean"] - results[strongest]["F1"]["mean"]) <= 0.01
                                  if strongest else None),
            "cost_not_higher": (b4["cost"]["mean"] <= results[strongest]["cost"]["mean"]
                                if strongest else None),
            "actions_used": sum(v > 0 for v in b4["counts"].values()),
            "best_fixed_or_mixture_F1_at_B4_cost": mix,
            "beats_fixed_and_mixtures": mix is None or b4["F1"]["mean"] >= mix,
        }

    # Pareto: frontier on validation, each point measured on test.
    pareto = {}
    for name, r in routers.items():
        grid = bl.cut_grid() if name == "B2" else None
        front = rt.pareto_frontier(val, r.score(val), grid)
        s_test = r.score(test)
        pts = []
        for p in front:
            q, c = rt.policy_value(test, rt.assign_action(s_test, p["theta1"], p["theta2"]))
            pts.append({"val_budget": p["budget"], "theta1": p["theta1"], "theta2": p["theta2"],
                        "test_F1": q, "test_cost": c})
        pareto[name] = pts

    # Section 4.2 item 5 and section 9.4 rule activations, for B4.
    extras = {}
    if "B4" in routers:
        f = routers["B4"].fit
        extras["monotonicity_at_B4_thresholds"] = mono.report(
            mono.grid_scores(widths=widths), theta1=f["theta1"], theta2=f["theta2"])["selected"]
        fired = {}
        for row in bl.signal_matrix(test, signals):
            for rule in fz.escalate(row, widths)[1]["fired"]:
                fired[rule["rule_id"]] = fired.get(rule["rule_id"], 0) + 1
        extras["rule_activations_test"] = {"rules_fired": len(fired),
                                           "counts": dict(sorted(fired.items()))}

    report = {
        "dataset": d,
        "instances": {s: len(t) for s, t in (("train", train), ("val", val), ("test", test))},
        "widths_B4": widths,
        "stage3_oracle_shares_all": ev.oracle_shares(full),
        "always_all_instances": {a: {"F1": float(full.f1[:, i].mean()),
                                     "cost": float(full.cost[:, i].mean())}
                                 for i, a in enumerate(ACTIONS)},
        "fits": {n: {k: v for k, v in r.fit.items()} for n, r in routers.items()},
        "fit_errors": errors,
        "test": {n: strip(m) for n, m in results.items()},
        "B4_vs": comparisons,
        "criteria": criteria,
        "pareto": pareto,
        **extras,
    }
    return report, rows, test_actions, extra


def plot_pareto(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, pts in report["pareto"].items():
        if name.startswith("B4-"):
            continue
        ax.plot([p["test_cost"] for p in pts], [p["test_F1"] for p in pts], marker=".",
                label=name, linewidth=1.2)
    for name in FIXED + ("oracle",):
        m = report["test"][name]
        ax.scatter(m["cost"]["mean"], m["F1"]["mean"], marker="*", s=90, zorder=3)
        ax.annotate(name, (m["cost"]["mean"], m["F1"]["mean"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("mean latency per query (s), test")
    ax.set_ylabel("mean F1, test")
    ax.set_title(f"{report['dataset']}: thresholds chosen on validation at each budget")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(args.root or ds.DATA / "stage1")
    out = root / "results"
    out.mkdir(exist_ok=True)

    all_rows, all_actions, all_extra, summary = [], {}, {}, {}
    for d in ds.NAMES:
        report, rows, test_actions, extra = evaluate_dataset(d, root)
        (out / f"{d}.json").write_text(json.dumps(report, indent=2, default=float))
        plot_pareto(report, out / f"pareto_{d}.png")
        all_rows += rows
        all_extra.update(extra)
        for n, acts in test_actions.items():
            all_actions.setdefault(n, {}).update(acts)
        summary[d] = report

    # Pooled test: both datasets' test instances, each routed by its own dataset's routers.
    test_keys = set(all_actions["always-A0"])
    pooled_table = table_for(all_rows, test_keys)
    pooled = {n: strip(measure(pooled_table, [a[k] for k in pooled_table.instances], all_extra))
              for n, a in all_actions.items() if set(a) == test_keys}
    (out / "pooled.json").write_text(json.dumps(pooled, indent=2, default=float))

    for d, r in summary.items():
        print(f"\n== {d}: instances {r['instances']}; stage 3 oracle shares (all): "
              f"{ {k: round(v, 3) for k, v in r['stage3_oracle_shares_all'].items()} }")
        if r["fit_errors"]:
            print("   fit errors:", r["fit_errors"])
        for n, m in r["test"].items():
            print(f"   {n:15s} F1 {m['F1']['mean']:.3f} [{m['F1']['lo']:.3f},{m['F1']['hi']:.3f}]"
                  f"  cost {m['cost']['mean']:.2f}s"
                  + (f"  RFR {m['RFR']['mean']:.3f}  HRFR {m['HRFR']['mean']:.3f}" if "RFR" in m else "")
                  + f"  {m['counts']}")
        print("   criteria:", json.dumps(r["criteria"], default=float))
    print("\n== pooled test")
    for n, m in pooled.items():
        print(f"   {n:15s} F1 {m['F1']['mean']:.3f}  cost {m['cost']['mean']:.2f}s"
              + (f"  RFR {m['RFR']['mean']:.3f}  HRFR {m['HRFR']['mean']:.3f}" if "RFR" in m else ""))


if __name__ == "__main__":
    main()
