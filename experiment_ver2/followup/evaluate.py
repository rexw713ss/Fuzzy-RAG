"""Read-only comparison of NEW action-pilot outcomes, never historical substitutes."""

import argparse
from pathlib import Path

import numpy as np

from experiment_ver2.followup.actions import ACTIONS, score_outcome
from experiment_ver2.followup.collect import validate_queries
from experiment_ver2.followup.io import read_json, read_jsonl, write_json, digest, new_directory, seal
from experiment_ver2.robustness.statistics import cluster_counts, interval
from experiment_ver2.followup.validation import validate_outcome, validate_collection


def evaluate(queries, outcomes, n_boot=10000, seed=20261020):
    queries = validate_queries(queries)
    qindex = {(q["dataset"], q["cluster_id"], q["variant_id"]): q for q in queries}
    index = {}
    for row in outcomes:
        key = (row["dataset"], row["cluster_id"], row["variant_id"])
        pair = key, row["action"]
        if key not in qindex or row["action"] not in ACTIONS or pair in index:
            raise ValueError("unknown/duplicate pilot outcome key")
        if row.get("newly_executed") is not True or "stage_seconds" not in row or "trace" not in row:
            raise ValueError("pilot requires new runtime traces; archived outcomes are not substitutes")
        validate_outcome(row, qindex[key]["question"], qindex[key]["answer"])
        rescored = score_outcome(row, qindex[key]["answer"])
        if any(not np.isfinite(row.get(m, np.nan)) or abs(row[m] - rescored[m]) > 1e-12 for m in ("F1", "EM")):
            raise ValueError("cached scores disagree with generated answer/gold")
        for field in ("latency_total", "component_seconds", "retrievals", "ce_passes", "llm_calls", "input_tokens", "output_tokens", "a1_top_retained", "n_new"):
            if not isinstance(row.get(field), (int, float)) or not np.isfinite(row[field]) or row[field] < 0:
                raise ValueError(f"invalid pilot counter {field}")
        if row["latency_total"] < row["component_seconds"] - 1e-8:
            raise ValueError("component cost exceeds wall-clock total")
        if abs(sum(row["stage_seconds"].values()) - row["component_seconds"]) > 1e-8:
            raise ValueError("stage/component cost mismatch")
        index[pair] = row
    if set(index) != {(key, action) for key in qindex for action in ACTIONS}:
        raise ValueError("all three actions must be newly executed for every planned query; no outcome-based exclusions")
    results = {}
    for dataset in sorted({key[0] for key in qindex}):
        keys = sorted(key for key in qindex if key[0] == dataset)
        groups = np.array([key[1] for key in keys])
        counts, inv = cluster_counts(groups, n_boot, seed)
        mult, totals = counts[:, inv], counts[:, inv].sum(axis=1)
        values = {m: np.asarray([[index[key, a][m] for a in ACTIONS] for key in keys], float)
                  for m in ("F1", "EM", "latency_total", "ce_passes", "llm_calls", "input_tokens", "output_tokens", "a1_top_retained", "n_new")}
        methods = {a: {m: float(v[:, j].mean()) for m, v in values.items()} for j, a in enumerate(ACTIONS)}
        comparisons = {}
        for ref in ("A1", "A2"):
            j = ACTIONS.index(ref)
            delta = values["F1"][:, 2] - values["F1"][:, j]
            comparisons[ref] = interval(mult @ delta / totals, delta.mean(), family_size=4)
        results[dataset] = {"n_queries": len(keys), "n_clusters": len(set(groups)), "methods": methods,
                            "A2_union_F1_comparisons": comparisons}
    return {"status": "new_action_pilot_evaluated", "datasets": results, "n_boot": n_boot, "seed": seed,
            "inference": "exploratory TRAIN pilot; paired cluster percentile intervals, family=2 datasets x2 F1 comparisons",
            "routing_claim": "none; this compares actions, and cannot establish a new router's benefit",
            "latency_claim": "new warmed local wall-clock measurements only; no comparison with historical cached timings",
            "confirmatory_complete": False}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    status = read_json(a.run / "status.json")
    if status.get("status") != "action_collection_complete" or status["outcomes_sha256"] != digest(a.run / "outcomes.jsonl"):
        raise ValueError("collection is incomplete or its outcomes fingerprint changed")
    identity = read_json(a.run / "run_identity.json")
    if identity["inputs"]["queries_file_sha256"] != digest(a.queries):
        raise ValueError("evaluation query file does not match the locked collection")
    queries = read_jsonl(a.queries)
    result = evaluate(queries, validate_collection(a.run, queries))
    out = new_directory(a.out, (a.queries, a.run))
    write_json(out / "report.json", result)
    write_json(out / "inputs.json", {"queries_sha256": digest(a.queries), "run_identity_sha256": digest(a.run / "run_identity.json"),
               "journal_sha256": digest(a.run / "journal.jsonl"), "outcomes_sha256": digest(a.run / "outcomes.jsonl")})
    seal(out)
    print(f"New action pilot report: {out}")


if __name__ == "__main__":
    main()
