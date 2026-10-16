"""Run the frozen ver2 protocol on cached outcomes, without new model calls.

python -m experiment_ver2.run --root path/to/full --out experiment_ver2/runs/replay
python -m experiment_ver2.run --demo --out experiment_ver2/runs/synthetic
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from experiment_ver2 import models
from experiment_ver2.data import discover_datasets, load_dataset
from experiment_ver2.metrics import paired, summarize
from experiment_ver2.policies import choose_on_validation, quota_actions
from main_logic import fuzzy as fz

BUDGETS = (0.0, 0.1, 0.25, 0.5, 1.0)
PRIMARY_BUDGET = 0.25
PRIORS = (1.0, 5.0, 20.0)
RIDGE_ALPHAS = (0.1, 1.0, 10.0)


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _cluster_weights(split):
    _, inv, counts = np.unique(split.groups, return_inverse=True, return_counts=True)
    return 1.0 / counts[inv]


def fit_candidates(dataset):
    """TRAIN outcomes only. Validation and test cannot reach this function."""
    train = dataset.train
    kw = {"sample_weight": _cluster_weights(train)}
    candidates = {"global": {"global": models.fit_global(train.X, train.f1, train.cost, **kw)},
                  "ridge": {}, "fuzzy-fixed": {}, "fuzzy-calibrated": {}}
    for alpha in RIDGE_ALPHAS:
        candidates["ridge"][f"alpha={alpha}"] = models.fit_ridge(
            train.X, train.f1, train.cost, alpha=alpha, **kw)
    for method, widths in (("fuzzy-fixed", .2), ("fuzzy-calibrated", dataset.widths)):
        for prior in PRIORS:
            candidates[method][f"prior={prior}"] = models.fit(
                train.X, train.f1, train.cost, widths=widths, prior_strength=prior, **kw)
    return candidates


def score_candidates(candidates, X, widths):
    """Prediction takes features only; no outcome or sibling-answer access."""
    scores = {}
    for method, params in candidates.items():
        scores[method] = {}
        for name, model in params.items():
            quality, _cost = models.predict(model, X)
            scores[method][name] = quality[:, 2] - quality[:, 1]
    for name, width in (("uncertainty-B3", .2), ("uncertainty-B4", widths)):
        scores[name] = {"original-score": np.array([fz.escalate(row, width)[0] for row in X])}
    return scores


def freeze_policy(method, variants, validation, fraction):
    choices = [("always-A1", np.ones(len(validation), dtype=int))]
    for name, scores in variants.items():
        choices.append((name, quota_actions(scores, validation.questions, fraction,
                                            positive_only=not method.startswith("uncertainty-"))))
    fit = choose_on_validation(choices, validation.f1, validation.cost)
    fit["all_candidates"] = [choose_on_validation([c], validation.f1, validation.cost) for c in choices]
    fit["quota"] = fraction
    return fit


def apply_policy(method, fit, variants, questions):
    if fit["candidate"] == "always-A1":
        return np.ones(len(questions), dtype=int)
    return quota_actions(variants[fit["candidate"]], questions, fit["quota"],
                         positive_only=not method.startswith("uncertainty-"))


def evaluate_dataset(dataset, n_boot=1000):
    start = time.perf_counter()
    candidates = fit_candidates(dataset)
    training_seconds = time.perf_counter() - start
    val_scores = score_candidates(candidates, dataset.val.X, dataset.widths)
    # Freeze every setting before test outcomes are evaluated.
    fits = {str(q): {m: freeze_policy(m, variants, dataset.val, q)
                     for m, variants in val_scores.items()} for q in BUDGETS}
    strongest = {}
    for q, by_method in fits.items():
        baselines = {m: f for m, f in by_method.items() if m not in ("fuzzy-fixed", "fuzzy-calibrated")}
        strongest[q] = min(baselines, key=lambda m: (-baselines[m]["F1"],
                            baselines[m]["A2_count"], baselines[m]["cost"], m))
    start = time.perf_counter()
    test_scores = score_candidates(candidates, dataset.test.X, dataset.widths)
    scoring_seconds = time.perf_counter() - start
    test = dataset.test
    baseline_a1 = np.ones(len(test), dtype=int)
    report = {"dataset": dataset.name, "data": dataset.metadata,
              "training_seconds": training_seconds,
              "diagnostic_all_candidate_scoring_seconds": scoring_seconds,
              "latency_scope": "cached component sum; not new online end-to-end measurement",
              "fixed": {f"always-A{i}": summarize(test, np.full(len(test), i), n_boot)
                        for i in range(3)}, "budgets": {}}
    decisions = []
    for q, by_method in fits.items():
        assignments = {m: apply_policy(m, fit, test_scores[m], test.questions)
                       for m, fit in by_method.items()}
        block = {"quota": float(q), "strongest_baseline_selected_on_validation": strongest[q],
                 "methods": {}}
        for method, actions in assignments.items():
            result = summarize(test, actions, n_boot)
            result["fit"] = by_method[method]
            result["vs_always_A1"] = paired(test, actions, other_actions=baseline_a1, n_boot=n_boot)
            result["vs_random_same_A2_count"] = paired(test, actions,
                                        random_fraction=result["A2_fraction"], n_boot=n_boot)
            result["vs_validation_selected_baseline"] = paired(test, actions,
                                            other_actions=assignments[strongest[q]], n_boot=n_boot)
            # These are unadjusted, exploratory paired intervals; not a paper success flag.
            result["comparison_status"] = "exploratory; no multiplicity-adjusted confirmatory claim"
            block["methods"][method] = result
            candidate = by_method[method]["candidate"]
            selected_scores = test_scores[method].get(candidate)
            active = None
            if float(q) == PRIMARY_BUDGET and method.startswith("fuzzy-") and candidate != "always-A1":
                model = candidates[method][candidate]
                active = models.rule_weights(test.X, model["widths"])
            for i, (key, action) in enumerate(zip(test.keys, actions)):
                decision = {"dataset": key[0], "cluster_id": key[1], "variant_id": key[2],
                            "quota": float(q), "method": method, "action": f"A{int(action)}",
                            "candidate": candidate,
                            "score": float(selected_scores[i]) if selected_scores is not None else None,
                            "score_kind": "uncertainty" if method.startswith("uncertainty-") else "predicted_A2_minus_A1_F1"}
                if active is not None:
                    decision["active_rules"] = [{"rule_id": int(r + 1), "weight": float(active[i, r]),
                        "training_support": model["rule_support"][r],
                        "A2_minus_A1_consequent": model["rule_quality"][r][2] - model["rule_quality"][r][1]}
                        for r in np.flatnonzero(active[i] > 0)]
                decisions.append(decision)
        report["budgets"][q] = block
    train = dataset.train
    gains = train.f1[:, 2] - train.f1[:, 1]
    report["train_action_headroom"] = {
        "mean_A2_minus_A1": float(gains.mean()),
        "share_gain_above_0.02": float(np.mean(gains > .02)),
        "share_harm_below_minus_0.02": float(np.mean(gains < -.02)),
        "oracle_positive_gain_mean": float(np.maximum(gains, 0).mean()),
        "interpretation": "training diagnostic only; oracle gain is not an attainable policy result"}
    report["primary_budget"] = PRIMARY_BUDGET
    return report, candidates, decisions


def provenance():
    repo = Path(__file__).resolve().parent.parent
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    files = sorted((repo / "experiment_ver2").glob("*.py"))
    files += [repo / "main_logic" / name for name in ("signals.py", "calibration.py", "fuzzy.py",
                                                    "evaluation.py", "routing.py")]
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "working_tree_status": git("status", "--porcelain"),
            "source_sha256": {str(p.relative_to(repo)).replace("\\", "/"):
                              hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "python": sys.version, "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "seed": 42, "budgets": BUDGETS, "primary_budget": PRIMARY_BUDGET,
            "priors": PRIORS, "ridge_alphas": RIDGE_ALPHAS,
            "decision_mode": "offline batch quota; no online guarantee",
            "test_status": "previously inspected first-experiment test; exploratory replay"}


def write_summary(out, reports, synthetic):
    text = ["# Ver2 execution report", "",
            "SYNTHETIC MECHANISM CHECK — not NQ/HotpotQA results." if synthetic else
            "Exploratory replay of existing cached outcomes. No new LLM calls were made.", "",
            "Primary analysis: A1/A2 rescue at a maximum A2 fraction of 0.25.",
            "The quota is an operational call limit, not a latency-matched comparison.", "",
            "| Dataset | Method | F1 | A2 fraction | Cached cost | F1 difference vs same-count random (95% CI) |",
            "|---|---|---:|---:|---:|---|"]
    for report in reports:
        for method, m in report["budgets"][str(PRIMARY_BUDGET)]["methods"].items():
            diff = m["vs_random_same_A2_count"]["F1"]
            text.append(f"| {report['dataset']} | {method} | {m['F1']['mean']:.4f} | "
                        f"{m['A2_fraction']:.3f} | {m['cost']['mean']:.4f} | "
                        f"{diff['diff']:+.4f} [{diff['lo']:+.4f}, {diff['hi']:+.4f}] |")
    text += ["", "All intervals are exploratory and unadjusted for multiple comparisons.",
             "Zero A2 usage can reflect the quota, validation fallback, or no positive predicted test gains.",
             "Latency values replay cached component sums and exclude newly measured online serving overhead.",
             "Random references integrate allocation randomness analytically; CIs reflect cluster sampling only.",
             "Bootstrap intervals condition on frozen batch assignments and fitted models; they do not rerank resampled batches.",
             "See report JSON for all budgets, validation fits, EM, RFR/HRFR, and paired comparisons."]
    (out / "REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--root", type=Path, help="original full directory, read only")
    inputs.add_argument("--demo", action="store_true", help="synthetic plumbing check, never benchmark evidence")
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--datasets", nargs="+", help="defaults to datasets in splits.json")
    parser.add_argument("--check-only", action="store_true", help="validate and fingerprint data without fitting")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("--out must be new or empty; existing runs are never overwritten")
    if args.root and (args.out.resolve() == args.root.resolve() or args.root.resolve() in args.out.resolve().parents):
        parser.error("write ver2 outputs outside the original cache root")
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        if args.demo:
            from experiment_ver2.demo import create_demo
            root = args.out / "synthetic_input"
            create_demo(root)
        else:
            root = args.root
        names = args.datasets or discover_datasets(root)
        datasets = [load_dataset(root, name) for name in names]
        manifest = provenance()
        manifest.update({"synthetic": args.demo, "bootstrap_resamples": args.bootstrap,
                         "input_root": str(root.resolve()), "datasets": names})
        if args.demo:
            manifest["test_status"] = "synthetic mechanism check, no empirical dataset evidence"
        _json(args.out / "manifest.json", manifest)
        _json(args.out / "data_audit.json", {d.name: d.metadata for d in datasets})
        if args.check_only:
            print(f"Validated {len(datasets)} dataset(s); audit: {args.out / 'data_audit.json'}")
            return 0
        reports = []
        for dataset in datasets:
            print(f"Evaluating {dataset.name}: {len(dataset.train)}/{len(dataset.val)}/{len(dataset.test)} train/val/test", flush=True)
            report, fitted, decisions = evaluate_dataset(dataset, n_boot=args.bootstrap)
            report["synthetic"] = args.demo
            _json(args.out / f"{dataset.name}_report.json", report)
            _json(args.out / f"{dataset.name}_models.json", fitted)
            (args.out / f"{dataset.name}_decisions.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in decisions), encoding="utf-8")
            reports.append(report)
        write_summary(args.out, reports, args.demo)
        print(f"Finished: {args.out / 'REPORT.md'}")
        return 0
    except (ValueError, FileNotFoundError, KeyError) as exc:
        _json(args.out / "failure.json", {"error": str(exc), "benchmark_completed": False})
        print(f"Cannot evaluate: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
