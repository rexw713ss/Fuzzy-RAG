"""Read-only reproduction of committed v0.4 reports from cached outcomes.

Run from the repository root:
    python -m experiment_ver2.reproduction --out experiment_ver2/validation/reproduction.json

This verifies numerical reproducibility, not statistical validity or fresh-test
independence. It calls evaluate_dataset directly and never calls the original
script's main(), which would write into the cache tree.
"""

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

from main_logic import answers
from scripts import evaluate_v4 as evaluation


TOLERANCE = 1e-10


def _jsonable(value):
    return json.loads(json.dumps(value, default=float, allow_nan=False))


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(roots):
    files = {p.resolve() for root in roots for p in root.rglob("*")
             if p.is_file() and p.suffix in (".json", ".jsonl")}
    return {str(p): _digest(p) for p in sorted(files)}


def compare(saved, reproduced, tolerance=TOLERANCE):
    """Compare every JSON leaf, retaining all mismatches and numeric counts."""
    result = {"numeric_values_compared": 0, "integer_values_compared": 0,
              "nonnumeric_values_compared": 0, "max_absolute_error": 0.0,
              "mismatches": []}

    def mismatch(path, expected, actual, reason):
        result["mismatches"].append({"path": path, "reason": reason,
                                     "saved": expected, "reproduced": actual})

    def visit(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                child = f"{path}.{key}"
                if key not in a or key not in b:
                    mismatch(child, a.get(key), b.get(key), "missing key")
                else:
                    visit(a[key], b[key], child)
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                mismatch(path, len(a), len(b), "different list length")
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{path}[{index}]")
            return
        # bool is deliberately not treated as an integer.
        if type(a) in (int, float) and type(b) in (int, float):
            result["numeric_values_compared"] += 1
            if type(a) is int and type(b) is int:
                result["integer_values_compared"] += 1
                if a != b:
                    mismatch(path, a, b, "integer/count mismatch")
                error = abs(a - b)
            else:
                error = abs(float(a) - float(b))
                if not (math.isfinite(a) and math.isfinite(b)) or error > tolerance:
                    mismatch(path, a, b, "numeric mismatch")
            result["max_absolute_error"] = max(result["max_absolute_error"], error)
            return
        result["nonnumeric_values_compared"] += 1
        if type(a) is not type(b) or a != b:
            mismatch(path, a, b, "type or nonnumeric value mismatch")

    visit(saved, reproduced, "$")
    result["mismatch_count"] = len(result["mismatches"])
    result["matched"] = not result["mismatches"]
    return result


def _read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def audit_integrity(full, v4):
    """Check cached scoring, cost accounting, coverage, and literal holdout overlap.

    Original-question normalization is casefolding followed by retaining only
    Unicode alphanumeric characters. No semantic duplicate detection or human
    paraphrase-equivalence audit is implied by zero literal overlap.
    """
    result = {"scope": "v0.4 old and fresh action outcomes; original cluster records",
              "absolute_numeric_tolerance": TOLERANCE,
              "original_question_normalization": "str.casefold then retain Unicode str.isalnum characters",
              "limitations": ["No semantic duplicate detection",
                              "No human paraphrase-equivalence audit",
                              "Recomputed EM/F1 uses repository's existing SQuAD-compatible scorer",
                              "Latency sums check accounting, not hardware timing accuracy"],
              "datasets": {}}
    for dataset in ("nq", "hotpotqa"):
        old_source = full / dataset
        fresh_source = v4 / "fresh_src" / dataset
        sources = {"old": old_source, "fresh": fresh_source}
        clusters = {name: _read_rows(src / "clusters.jsonl") for name, src in sources.items()}
        item = {"clusters": {}, "old_fresh_overlap": {}, "outcomes": {}}
        originals = {}
        for name, records in clusters.items():
            ids = Counter(row["cluster_id"] for row in records)
            originals[name] = [next(p["text"] for p in row["phrasings"] if p["variant_id"] == 0)
                               for row in records]
            item["clusters"][name] = {
                "cluster_count": len(records), "unique_cluster_ids": len(ids),
                "duplicate_cluster_id_count": sum(count - 1 for count in ids.values()),
                "complete_clusters": sum(len(row["phrasings"]) == 3 for row in records),
                "incomplete_clusters": sum(len(row["phrasings"]) != 3 for row in records),
                "complete_flag_mismatches": [row["cluster_id"] for row in records
                                             if bool(row["complete"]) != (len(row["phrasings"]) == 3)]}
        for field in ("cluster_id", "source_index"):
            overlap = set(row[field] for row in clusters["old"]) & set(row[field] for row in clusters["fresh"])
            item["old_fresh_overlap"][field] = {"count": len(overlap), "values": sorted(overlap, key=str)}
        normalize = lambda s: "".join(c for c in s.casefold() if c.isalnum())
        overlap = set(map(normalize, originals["old"])) & set(map(normalize, originals["fresh"]))
        item["old_fresh_overlap"]["normalized_original_question"] = {
            "count": len(overlap), "values": sorted(overlap)}
        for name, src in sources.items():
            query_rows = _read_rows(src / "queries.jsonl")
            queries = {(row["cluster_id"], row["variant_id"]): row for row in query_rows}
            outcome_path = (v4 / dataset if name == "old" else v4 / "fresh" / dataset) / "outcomes.jsonl"
            outcomes = _read_rows(outcome_path)
            counts = Counter((row["cluster_id"], row["variant_id"], row["action"]) for row in outcomes)
            expected = {(cluster, variant, action) for cluster, variant in queries for action in ("A1", "A2")}
            seen = set(counts)
            em_errors, f1_errors, latency_errors = [], [], []
            max_em, max_f1, max_latency = 0.0, 0.0, 0.0
            missing_queries = []
            for row in outcomes:
                key = (row["cluster_id"], row["variant_id"], row["action"])
                if key[:2] not in queries:
                    missing_queries.append(key)
                    continue
                gold = queries[key[:2]]["answer"]
                em = answers.exact_match(row["answer"], gold)
                f1 = answers.f1_score(row["answer"], gold)
                em_error, f1_error = abs(em - row["EM"]), abs(f1 - row["F1"])
                max_em, max_f1 = max(max_em, em_error), max(max_f1, f1_error)
                if not math.isfinite(row["EM"]) or em_error > TOLERANCE:
                    em_errors.append({"key": key, "saved": row["EM"], "recomputed": em})
                if not math.isfinite(row["F1"]) or f1_error > TOLERANCE:
                    f1_errors.append({"key": key, "saved": row["F1"], "recomputed": f1})
                components = sum(row[k] for k in ("retrieval_latency", "rerank_latency",
                                                  "followup_latency", "generation_latency"))
                residual = abs(row["latency_total"] - components)
                max_latency = max(max_latency, residual)
                if not math.isfinite(residual) or residual > TOLERANCE:
                    latency_errors.append({"key": key, "saved": row["latency_total"],
                                           "component_sum": components})
            audit = {"outcome_file": str(outcome_path), "query_rows": len(query_rows),
                     "duplicate_query_keys": len(query_rows) - len(queries),
                     "outcome_rows": len(outcomes), "unique_action_keys": len(counts),
                     "duplicate_action_keys": sum(count - 1 for count in counts.values()),
                     "action_counts": dict(sorted(Counter(row["action"] for row in outcomes).items())),
                     "expected_action_keys": len(expected),
                     "missing_action_keys": sorted(expected - seen, key=str),
                     "extra_action_keys": sorted(seen - expected, key=str),
                     "outcomes_without_query": missing_queries,
                     "EM_mismatches": em_errors, "F1_mismatches": f1_errors,
                     "max_EM_absolute_error": max_em, "max_F1_absolute_error": max_f1,
                     "latency_component_fields": ["retrieval_latency", "rerank_latency",
                                                   "followup_latency", "generation_latency"],
                     "max_latency_sum_absolute_error": max_latency,
                     "latency_sum_mismatches": latency_errors,
                     "A2_fallback_count": sum(row["action"] == "A2" and row["fallback"] for row in outcomes)}
            audit["passed"] = not any(audit[key] for key in (
                "duplicate_query_keys", "duplicate_action_keys", "missing_action_keys", "extra_action_keys",
                "outcomes_without_query", "EM_mismatches", "F1_mismatches", "latency_sum_mismatches"))
            item["outcomes"][name] = audit
        item["passed"] = (all(audit["passed"] for audit in item["outcomes"].values())
                          and all(v["count"] == 0 for v in item["old_fresh_overlap"].values())
                          and all(not v["duplicate_cluster_id_count"] and not v["complete_flag_mismatches"]
                                  for v in item["clusters"].values()))
        result["datasets"][dataset] = item
    result["total_outcome_rows"] = sum(v["outcome_rows"] for d in result["datasets"].values()
                                      for v in d["outcomes"].values())
    result["all_checks_passed"] = all(d["passed"] for d in result["datasets"].values())
    return result


def reproduce(full, v4):
    """Match evaluate_v4.main's report assembly without writing original files."""
    reports, pooled_rows, pooled_actions, pooled_extra = {}, [], {}, {}
    for dataset in ("nq", "hotpotqa"):
        print(f"Reproducing {dataset} from cached outcomes", flush=True)
        report, rows, actions, extra = evaluation.evaluate_dataset(dataset, full, v4)
        reports[dataset] = report
        pooled_rows.extend(rows)
        pooled_extra.update(extra)
        for set_name, methods in actions.items():
            for name, assignments in methods.items():
                target = pooled_actions.setdefault(set_name, {}).setdefault(name, {})
                overlap = set(target) & set(assignments)
                if overlap:
                    raise ValueError(f"cross-dataset instance key collision: {next(iter(overlap))}")
                target.update(assignments)
    pooled = {}
    for set_name, methods in pooled_actions.items():
        keys = set(methods["always-A1"])
        table = evaluation.table_for(pooled_rows, keys)
        results = {name: evaluation.measure(table, [assignments[k] for k in table.instances],
                                            pooled_extra)
                   for name, assignments in methods.items() if set(assignments) == keys}
        comparisons = {name: evaluation.paired(table, results["B4"], result)
                       for name, result in results.items() if name != "B4"} if "B4" in results else {}
        pooled[set_name] = {"results": {name: evaluation.strip(result)
                                       for name, result in results.items()},
                            "B4_vs": comparisons}
    reports["pooled"] = pooled
    return _jsonable(reports)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, default=Path("runs/v0.3"))
    parser.add_argument("--v4", type=Path, default=Path("runs/v0.4"))
    parser.add_argument("--out", type=Path,
                        default=Path("experiment_ver2/validation/reproduction.json"))
    args = parser.parse_args(argv)
    full, v4, out = args.full.resolve(), args.v4.resolve(), args.out.resolve()
    if any(out == root or root in out.parents for root in (full, v4)):
        parser.error("--out must be outside both input cache trees")
    if out.exists():
        parser.error("--out already exists; use a new filename to preserve prior validation")
    before = _snapshot((full, v4))
    integrity = audit_integrity(full, v4)
    reports = reproduce(full, v4)
    comparisons = {}
    for name, actual in reports.items():
        path = v4 / "reported" / f"{name}.json"
        expected = json.loads(path.read_text(encoding="utf-8"))
        comparisons[name] = {"saved_report": str(path), "saved_sha256": _digest(path),
                             "reproduced_canonical_sha256": hashlib.sha256(
                                 json.dumps(actual, sort_keys=True, separators=(",", ":"),
                                            allow_nan=False).encode("utf-8")).hexdigest(),
                             **compare(expected, actual)}
    after = _snapshot((full, v4))
    changed = [path for path in sorted(set(before) | set(after)) if before.get(path) != after.get(path)]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    summary = {"kind": "cached_v0.4_numerical_reproduction",
               "interpretation": "reproducibility audit only; no new model outputs or success claim",
               "absolute_numeric_tolerance": TOLERANCE,
               "all_reports_matched": all(r["matched"] for r in comparisons.values()),
               "input_files_checked": len(before), "input_files_changed": changed,
               "input_sha256": before, "reports": comparisons, "dataset_integrity": integrity,
               "python": sys.version, "platform": platform.platform(), "commit": commit,
               "source_sha256": {str(Path(__file__).resolve()): _digest(Path(__file__)),
                                  str(Path(answers.__file__).resolve()): _digest(Path(answers.__file__)),
                                  str(Path(evaluation.__file__).resolve()): _digest(Path(evaluation.__file__))}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"all_reports_matched": summary["all_reports_matched"],
                      "all_integrity_checks_passed": integrity["all_checks_passed"],
                      "total_outcome_rows": integrity["total_outcome_rows"],
                      "input_files_changed": changed,
                      "reports": {name: {k: r[k] for k in ("numeric_values_compared", "mismatch_count",
                                                           "max_absolute_error")}
                                  for name, r in comparisons.items()},
                      "output": str(out)}, indent=2), flush=True)
    return 0 if summary["all_reports_matched"] and integrity["all_checks_passed"] and not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
