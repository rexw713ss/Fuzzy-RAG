"""A fixed exploratory follow-up: additive fuzzy basis versus existing routers."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiment_ver2 import additive_models
from experiment_ver2.archive import load_archive
from experiment_ver2.metrics import summarize
from experiment_ver2.run import RIDGE_ALPHAS, _cluster_weights, freeze_policy, apply_policy, provenance
from experiment_ver2.robustness.run import freeze, verify_reference, write_json
from experiment_ver2.robustness.statistics import cluster_counts, interval


def select_and_predict(dataset):
    """TRAIN fit and VAL selection. TEST labels cannot influence any decision."""
    candidates, fits, scores, assignments = {}, {}, {}, {}
    base, escalation = (dataset.actions.index(a) for a in ("A1", "A2"))
    weights = _cluster_weights(dataset.train)
    for method, widths in (("additive-fixed", .2), ("additive-calibrated", dataset.widths)):
        variants, validation_scores, test_scores = {}, {}, {}
        for alpha in RIDGE_ALPHAS:
            key = f"alpha={alpha}"
            model = additive_models.fit(dataset.train.X, dataset.train.f1, dataset.train.cost,
                                        widths=widths, alpha=alpha, sample_weight=weights)
            variants[key] = model
            for split, output in ((dataset.val, validation_scores), (dataset.test, test_scores)):
                quality, _ = additive_models.predict(model, split.X)
                output[key] = quality[:, escalation] - quality[:, base]
        fit = freeze_policy(method, validation_scores, dataset.val, .25)
        candidates[method], fits[method] = variants, fit
        scores[method] = test_scores.get(fit["candidate"], np.zeros(len(dataset.test)))
        assignments[method] = apply_policy(method, fit, test_scores, dataset.test.questions, dataset.actions)
    return candidates, fits, scores, assignments


def compare(split, actions, reference_actions, n_boot, seed):
    counts, inv = cluster_counts(split.groups, n_boot, seed)
    mult, totals = counts[:, inv], counts[:, inv].sum(axis=1)
    base, escalation = (split.actions.index(a) for a in ("A1", "A2"))
    quality = split.f1[np.arange(len(split)), actions]
    selected = mult @ quality / totals
    p = mult @ (actions == escalation).astype(float) / totals
    refs = {m: split.f1[np.arange(len(split)), a] for m, a in reference_actions.items()}
    random_point = split.f1[:, base] + np.mean(actions == escalation) * (split.f1[:, escalation] - split.f1[:, base])
    random_draw = mult @ split.f1[:, base] / totals + p * (mult @ (split.f1[:, escalation] - split.f1[:, base]) / totals)
    output = {m: interval(selected - mult @ v / totals, np.mean(quality - v), family_size=12) for m, v in refs.items()}
    output["same_count_random"] = interval(selected - random_draw, np.mean(quality - random_point), family_size=12)
    return {"comparisons": output, "family_size": 12,
            "family": "2 additive families times 2 datasets times 3 F1 comparisons",
            "estimand": "frozen assignments; same-count random fraction recomputed inside each cluster resample",
            "limitations": "conditional on fitted policies; no reranking or training variation; exploratory reused test"}


def subgroups(split, actions):
    _, inv, sizes = np.unique(split.groups, return_inverse=True, return_counts=True)
    selected = split.f1[np.arange(len(split)), actions]
    output = {}
    for name, mask in (("original_only", np.array([k[2] == 0 for k in split.keys])),
                       ("complete_clusters_only", sizes[inv] == 3)):
        output[name] = {"queries": int(mask.sum()), "clusters": len(set(split.groups[mask])),
                        "F1": float(selected[mask].mean()) if mask.any() else None,
                        "A2_count": int(np.sum(actions[mask] == split.actions.index("A2")))}
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args(argv)
    if args.bootstrap < 2000 or args.seed < 0:
        parser.error("bootstrap >= 2000 and seed >= 0 required")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be new or empty")
    reference = Path("experiment_ver2/evidence/v04_fresh_4sig")
    for protected in (args.archive, reference):
        if args.out.resolve() == protected.resolve() or protected.resolve() in args.out.resolve().parents:
            parser.error("output must be outside original caches/reference evidence")
    args.out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parent.parent
    protocol = repo / "experiment_ver2/ADDITIVE_EXPERIMENT_PROTOCOL.md"
    manifest = provenance()
    manifest.update({"seed": args.seed, "bootstrap_resamples": args.bootstrap,
                     "test_status": "exploratory follow-up using already-published cached outcomes; no new generation",
                     "comparison_family_size": 12, "budgets": [.25],
                     "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest()})
    manifest["source_sha256"].update({p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (repo / "experiment_ver2/robustness").glob("*.py")})
    summaries = []
    for name in ("nq", "hotpotqa"):
        print(f"{name}: fitting additive families, selection on VAL only", flush=True)
        data = load_archive(args.archive, name, version="v0.4", test_set="fresh")
        _, baseline_fits, baseline_actions, _ = freeze(data)
        verification = verify_reference(data, baseline_fits, baseline_actions, reference)
        candidates, fits, scores, assignments = select_and_predict(data)
        # Test outcomes are first used here, after every new policy is frozen.
        report = {"dataset": name, "data": data.metadata, "baseline_verification": verification,
                  "quota": .25, "methods": {}, "primary_new_family": "additive-calibrated"}
        references = {m: baseline_actions[m] for m in ("ridge", "fuzzy-calibrated")}
        descriptive = {**references, "always-A1": np.full(len(data.test), data.actions.index("A1"), dtype=int)}
        report["descriptive_references"] = {
            method: {"F1": float(data.test.f1[np.arange(len(data.test)), actions].mean()),
                     "EM": float(data.test.em[np.arange(len(data.test)), actions].mean()),
                     "cached_cost": float(data.test.cost[np.arange(len(data.test)), actions].mean()),
                     "A2_count": int(np.sum(actions == data.actions.index("A2")))}
            for method, actions in descriptive.items()}
        decisions = []
        for method, actions in assignments.items():
            print(f"{name}/{method}: selected {fits[method]['candidate']}; evaluating", flush=True)
            result = summarize(data.test, actions, n_boot=args.bootstrap, seed=args.seed)
            result.update({"fit": fits[method], "inference": compare(data.test, actions, references, args.bootstrap, args.seed),
                           "frozen_assignment_subgroups": subgroups(data.test, actions)})
            result["descriptive_gain_vs_always_A1"] = result["F1"]["mean"] - report["descriptive_references"]["always-A1"]["F1"]
            report["methods"][method] = result
            for i, key in enumerate(data.test.keys):
                decisions.append({"dataset": name, "cluster_id": key[1], "variant_id": key[2],
                                  "method": method, "candidate": fits[method]["candidate"], "quota": .25,
                                  "action": data.actions[actions[i]],
                                  "predicted_gain": float(scores[method][i]) if fits[method]["candidate"] != "always-A1" else None})
        write_json(args.out / f"{name}_report.json", report)
        write_json(args.out / f"{name}_models.json", candidates)
        (args.out / f"{name}_decisions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in decisions), encoding="utf-8", newline="\n")
        summaries.append(report)
    write_json(args.out / "manifest.json", manifest)
    lines = ["# Additive fuzzy exploratory follow-up", "", "Real v0.4 cached outcomes; no new retrieval/generation. Primary quota .25. All tests previously inspected.", "",
             "| Dataset | Family | VAL choice | Test F1 | A2 count | Difference vs ridge (95% cluster interval) |", "|---|---|---|---:|---:|---|"]
    for report in summaries:
        for method, r in report["methods"].items():
            c = r["inference"]["comparisons"]["ridge"]
            lo, hi = c["percentile_95"]
            lines.append(f"| {report['dataset']} | {method} | {r['fit']['candidate']} | {r['F1']['mean']:.6f} | {r['counts']['A2']} | {c['difference']:+.6f} [{lo:+.6f}, {hi:+.6f}] |")
    lines += ["", "The report JSON includes all 12 comparisons, conservative multiplicity sensitivity intervals, EM, cached cost, routing flips, and subset scores. No new test-based tuning or selection follows this run."]
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    write_json(args.out / "SHA256.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out.iterdir()) if p.is_file()})
    print(f"Completed: {args.out / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
