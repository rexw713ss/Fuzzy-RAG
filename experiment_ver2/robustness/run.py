"""Reproduce the exploratory robustness audit without generation or API calls."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiment_ver2.archive import load_archive
from experiment_ver2.run import fit_candidates, score_candidates, freeze_policy, apply_policy, provenance
from experiment_ver2.robustness.integrity import audit_archive, blind_review_sample, read_rows
from experiment_ver2.robustness.stability import run_stability
from experiment_ver2.robustness.statistics import bootstrap_sensitivity, subset_diagnostics, influence_and_planning


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def freeze(dataset):
    candidates = fit_candidates(dataset)
    val_scores = score_candidates(candidates, dataset.val.X, dataset.widths, dataset.actions, dataset.signals)
    fits = {method: freeze_policy(method, variants, dataset.val, .25) for method, variants in val_scores.items()}
    test_scores = score_candidates(candidates, dataset.test.X, dataset.widths, dataset.actions, dataset.signals)
    assignments = {method: apply_policy(method, fit, test_scores[method], dataset.test.questions, dataset.actions)
                   for method, fit in fits.items()}
    selected = {method: test_scores[method].get(fit["candidate"], np.zeros(len(dataset.test))) for method, fit in fits.items()}
    return candidates, fits, assignments, selected


def verify_reference(dataset, fits, assignments, reference):
    """Fail if the frozen primary policies differ from the previous evidence."""
    report = json.loads((reference / f"{dataset.name}_report.json").read_text(encoding="utf-8"))
    decisions = read_rows(reference / f"{dataset.name}_decisions.jsonl")
    index = {(r["method"], r["cluster_id"], r["variant_id"]): r["action"] for r in decisions if r["quota"] == .25}
    for method, fit in fits.items():
        saved = report["budgets"]["0.25"]["methods"][method]
        if saved["fit"]["candidate"] != fit["candidate"]:
            raise ValueError(f"primary candidate changed: {dataset.name}/{method}")
        expected = [index[method, k[1], k[2]] for k in dataset.test.keys]
        actual = [dataset.actions[a] for a in assignments[method]]
        if expected != actual:
            raise ValueError(f"primary decisions changed: {dataset.name}/{method}")
        observed = dataset.test.f1[np.arange(len(dataset.test)), assignments[method]].mean()
        if abs(observed - saved["F1"]["mean"]) > 1e-12:
            raise ValueError(f"primary F1 changed: {dataset.name}/{method}")
    return {"all_six_families_primary_candidates_actions_F1_match": True,
            "reference_directory": reference.as_posix()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--refits", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--reference", type=Path, default=Path("experiment_ver2/evidence/v04_fresh_4sig"))
    args = parser.parse_args(argv)
    if args.bootstrap < 2000 or args.refits < 2 or args.seed < 0:
        parser.error("bootstrap >= 2000, refits >= 2, seed >= 0 required")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be new or empty")
    for protected in (args.archive, args.reference):
        if args.out.resolve() == protected.resolve() or protected.resolve() in args.out.resolve().parents:
            parser.error("outputs must be outside original caches and reference evidence")
    args.out.mkdir(parents=True, exist_ok=True)
    from experiment_ver2.robustness.diagnostics import run_diagnostics
    repo = Path(__file__).resolve().parents[2]
    manifest = provenance()
    manifest.update({"seed": args.seed, "bootstrap_resamples": args.bootstrap, "train_val_refits": args.refits,
                     "status": "exploratory diagnostics on previously published test; not a new confirmation",
                     "primary_quota": .25, "comparison_family_size": 8,
                     "reference_sha256": {p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in sorted(args.reference.resolve().glob("*.json*"))}})
    manifest["source_sha256"].update({p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted((repo / "experiment_ver2/robustness").glob("*.py"))})
    audits = {}
    for name in ("nq", "hotpotqa"):
        print(f"{name}: validating archive and freezing policies", flush=True)
        data = load_archive(args.archive, name, version="v0.4", test_set="fresh")
        candidates, fits, assignments, selected = freeze(data)
        check = verify_reference(data, fits, assignments, args.reference)
        audits[name] = audit_archive(args.archive, name)
        metadata = {r["cluster_id"]: r for r in read_rows(args.archive / "v0.4/fresh_src" / name / "clusters.jsonl")}
        print(f"{name}: {args.bootstrap} test cluster resamples, three estimands", flush=True)
        report = {"dataset": name, "unchanged_primary_reference": check, "data_provenance": data.metadata,
                  "inference": bootstrap_sensitivity(data, assignments, selected, fits, args.bootstrap, args.seed),
                  "subgroups": subset_diagnostics(data, assignments, metadata),
                  "influence_and_planning": influence_and_planning(data, assignments),
                  "model_diagnostics": run_diagnostics(data, candidates, fits, assignments)}
        write_json(args.out / f"{name}_robustness.json", report)
        print(f"{name}: {args.refits} independent TRAIN/VAL resampling refits", flush=True)
        stability = run_stability(data, repetitions=args.refits, seed=args.seed)
        for method, block in stability["methods"].items():
            if block["original"]["actions"] != assignments[method].tolist():
                raise ValueError(f"stability reference mismatch: {name}/{method}")
        write_json(args.out / f"{name}_stability.json", stability)
    write_json(args.out / "integrity.json", audits)
    tasks, key = blind_review_sample(args.archive, seed=args.seed)
    (args.out / "blind_paraphrase_review.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in tasks), encoding="utf-8", newline="\n")
    write_json(args.out / "blind_review_key.json", key)
    write_json(args.out / "manifest.json", manifest)
    write_json(args.out / "SHA256.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out.iterdir()) if p.is_file()})
    print(f"Saved robustness evidence: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
