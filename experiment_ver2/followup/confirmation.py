"""Evaluate locked original-A2 routers on a genuinely new, reviewed question set.

CLI gates require the full fixed sample size and complete human review. Unit
fixtures only exercise mathematics; they are never exported as benchmark data.
"""

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np

from experiment_ver2 import models
from experiment_ver2.archive import normalize_question
from experiment_ver2.data import Split
from experiment_ver2.followup.actions import score_outcome
from experiment_ver2.followup.collect import validate_queries
from experiment_ver2.followup.io import digest, read_json, read_jsonl, write_json, write_jsonl, new_directory, seal
from experiment_ver2.metrics import summarize
from experiment_ver2.policies import quota_actions
from experiment_ver2.robustness.statistics import reranked_counts, interval
from experiment_ver2.followup.review import task_digest, verified_audit
from experiment_ver2.followup.validation import validate_outcome, validate_collection


SAMPLE_SIZES = {"nq": 1800, "hotpotqa": 3500}
PRIMARY_METHODS = ("fuzzy-calibrated", "ridge")


def check_sample(queries, originals, exposure, sample_manifest):
    validate_queries(queries)
    source = {r["cluster_id"]: r for r in originals}
    if len(source) != len(originals) or len(originals) != sample_manifest["n"]:
        raise ValueError("sample original IDs/count mismatch")
    if {q["cluster_id"] for q in queries} != set(source):
        raise ValueError("every frozen original must be retained; no outcome-based exclusions")
    previous_ids = set(exposure["originals"])
    previous_texts = set(exposure["normalized_texts"])
    originals_seen, text_seen = set(), set()
    for q in queries:
        cid = q["cluster_id"]
        normalized = normalize_question(q["question"])
        if cid in previous_ids or normalized in previous_texts or normalized in text_seen:
            raise ValueError("query overlap with previous exposure or another retained variant")
        text_seen.add(normalized)
        if q["answer"] != source[cid]["answer"] or q["dataset"] != source[cid]["dataset"]:
            raise ValueError("new query gold/dataset differs from frozen original")
        if q["variant_id"] == 0:
            if q["question"] != source[cid]["question"]:
                raise ValueError("original question text changed")
            originals_seen.add(cid)
    if originals_seen != set(source):
        raise ValueError("each cluster must retain its original variant")


def review_subset(queries, review_report, review_key, tasks):
    if review_report.get("status") != "complete_recorded_human_review":
        raise ValueError("new-set independent human review is not complete")
    required = {(q["dataset"], q["cluster_id"], q["variant_id"]) for q in queries if q["variant_id"] != 0}
    if review_report.get("tasks_canonical_sha256") != task_digest(tasks):
        raise ValueError("human-review tasks/text fingerprint mismatch")
    task_index = {r["review_id"]: r for r in tasks}
    qindex = {(q["dataset"], q["cluster_id"], q["variant_id"]): q for q in queries}
    key_rows = review_key["items"]
    mapping, votes = {}, review_report["decisions"]
    seen_votes = set()
    for r in key_rows:
        key = (r["dataset"], r["cluster_id"], r["paraphrase_variant_id"])
        rid = r["review_id"]
        if key in mapping or rid in seen_votes or rid not in votes or rid not in task_index or key not in required:
            raise ValueError("review key/votes are duplicate or incomplete")
        if type(votes[rid].get("valid")) is not bool:
            raise ValueError("review decisions must be explicit booleans")
        task = task_index[rid]
        original = qindex.get((key[0], key[1], 0))
        side = r.get("original_side")
        if original is None or side not in ("left", "right"):
            raise ValueError("review key must identify the original side of each pair")
        other = "right" if side == "left" else "left"
        if task[side + "_question"] != original["question"] or task[other + "_question"] != qindex[key]["question"]:
            raise ValueError("review task text does not match this original/paraphrase pair")
        for label in ("left", "right"):
            expected = hashlib.sha256(task[label + "_question"].encode("utf-8")).hexdigest()
            if r.get(label + "_question_sha256") != expected:
                raise ValueError("review key question-text fingerprint mismatch")
        mapping[key] = rid
        seen_votes.add(rid)
    if set(mapping) != required or seen_votes != set(votes) or seen_votes != set(task_index):
        raise ValueError("human review must cover every retained new paraphrase, not the old 60-pair sample")
    return np.asarray([q["variant_id"] == 0 or votes[mapping[(q["dataset"], q["cluster_id"], q["variant_id"])]]["valid"]
                       for q in queries], bool)


def frozen_actions(lock, raw_features, questions):
    if lock["actions"] != ["A1", "A2"] or lock["signals"] != ["S1", "S2", "S3", "S4"] or lock["quota"] != .25:
        raise ValueError("only the fixed original-A2 four-signal protocol is supported")
    raw = np.asarray(raw_features, float)
    if raw.shape != (len(questions), 4) or not np.isfinite(raw).all():
        raise ValueError("four finite raw pre-action features required per query")
    p5 = np.array([lock["scalers"][s]["p5"] for s in lock["signals"]])
    p95 = np.array([lock["scalers"][s]["p95"] for s in lock["signals"]])
    if not np.isfinite(p5).all() or not np.isfinite(p95).all() or np.any(p95 < p5):
        raise ValueError("invalid frozen scalers")
    X = np.clip((raw - p5) / (p95 - p5 + 1e-9), 0, 1)
    assignments, scores, fallbacks = {}, {}, {}
    for method in PRIMARY_METHODS:
        policy = lock["policies"][method]
        if policy["fit"]["quota"] != .25:
            raise ValueError("policy quota disagrees with locked experiment")
        fallback = policy["fit"]["candidate"] == "always-A1"
        if fallback:
            scores[method] = np.zeros(len(X))
            assignments[method] = np.zeros(len(X), dtype=int)
        else:
            quality, _ = models.predict(policy["model"], X)
            scores[method] = quality[:, 1] - quality[:, 0]
            assignments[method] = quota_actions(scores[method], questions, .25, base=0, escalation=1)
        fallbacks[method] = fallback
    return X, assignments, scores, fallbacks


def _reranked_comparison(split, scores, fallbacks, n_boot, seed):
    if type(n_boot) is not int or n_boot < 2:
        raise ValueError("at least two bootstrap replicates required")
    ids, inv = np.unique(split.groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    samples = []
    delta = split.f1[:, 1] - split.f1[:, 0]
    # Chunk resamples so 1,300 clusters x 3 variants does not require multiple
    # gigabytes of temporary dense multiplicity matrices.
    for start in range(0, n_boot, 128):
        count = min(128, n_boot - start)
        draws = rng.integers(len(ids), size=(count, len(ids)))
        cluster_mult = np.zeros((count, len(ids)), dtype=int)
        np.add.at(cluster_mult, (np.arange(count)[:, None], draws), 1)
        mult = cluster_mult[:, inv]
        totals = mult.sum(axis=1)
        allocations = [reranked_counts(scores[m], split.questions, mult, .25, fallback=fallbacks[m]) for m in PRIMARY_METHODS]
        samples.extend(((allocations[0] - allocations[1]) @ delta / totals).tolist())
    return np.asarray(samples)


def evaluate_locked(queries, outcomes, lock, valid_subset, n_boot=10000, seed=20261021):
    validate_queries(queries)
    if np.asarray(valid_subset).shape != (len(queries),) or np.asarray(valid_subset).dtype.kind != "b":
        raise ValueError("review subset mask must align with queries")
    if {q["dataset"] for q in queries} != {lock["dataset"]}:
        raise ValueError("one dataset matching the frozen policy required")
    index = {}
    qindex = {(q["dataset"], q["cluster_id"], q["variant_id"]): q for q in queries}
    for row in outcomes:
        if row["action"] == "A2-union":
            continue  # Explicitly not this confirmation's action definition.
        key = (row["dataset"], row["cluster_id"], row["variant_id"])
        pair = key, row["action"]
        if key not in qindex or row["action"] not in ("A1", "A2") or pair in index:
            raise ValueError("duplicate or unexpected confirmation outcome")
        validate_outcome(row, qindex[key]["question"], qindex[key]["answer"])
        if row.get("newly_executed") is not True or "trace" not in row or "signals" not in row:
            raise ValueError("confirmation requires new generated outcomes and pre-action signals")
        rescored = score_outcome(row, qindex[key]["answer"])
        if any(not np.isfinite(row.get(m, np.nan)) or abs(row[m] - rescored[m]) > 1e-12 for m in ("F1", "EM")):
            raise ValueError("answer rescoring disagrees")
        if not np.isfinite(row.get("latency_total", np.nan)) or row["latency_total"] <= 0:
            raise ValueError("positive measured wall-clock cost required")
        index[pair] = row
    if set(index) != {(k, a) for k in qindex for a in ("A1", "A2")}:
        raise ValueError("new original-A1/A2 matrix incomplete")
    keys = tuple(qindex)
    questions = tuple(qindex[k]["question"] for k in keys)
    raw = [[index[k, "A1"]["signals"][s] for s in lock["signals"]] for k in keys]
    full_keys, full_questions, full_raw = keys, questions, raw
    valid_subset = np.asarray(valid_subset, bool)
    if not all(valid_subset[i] for i, k in enumerate(keys) if k[2] == 0):
        raise ValueError("primary population must retain every original")
    keys = tuple(k for i, k in enumerate(keys) if valid_subset[i])
    questions = tuple(q for i, q in enumerate(questions) if valid_subset[i])
    raw = np.asarray(raw)[valid_subset]
    before = time.perf_counter()
    X, assignments, scores, fallbacks = frozen_actions(lock, raw, questions)
    router_seconds = time.perf_counter() - before
    # All actions are frozen before outcomes enter metric calculations.
    values = {m: np.asarray([[index[k, a][m] for a in ("A1", "A2")] for k in keys]) for m in ("F1", "EM", "latency_total")}
    split = Split(X, values["F1"], values["latency_total"], values["EM"], keys, questions, actions=("A1", "A2"))
    row_idx = np.arange(len(keys))
    chosen = {m: split.f1[row_idx, a] for m, a in assignments.items()}
    point = float(np.mean(chosen[PRIMARY_METHODS[0]] - chosen[PRIMARY_METHODS[1]]))
    inference = interval(_reranked_comparison(split, scores, fallbacks, n_boot, seed), point, family_size=2)
    inference["estimand"] = "frozen fitted models; cluster-resampled batches reallocated at the fixed q=.25 quota"
    inference["scope"] = "prospective conditional-on-training, approximate percentile cluster bootstrap; validity depends on frozen protocol and unseen source clusters; not an exact finite-sample test or a training-variability interval"
    inference["family"] = "two datasets, one fixed fuzzy-versus-ridge F1 primary comparison per dataset"
    inference["decision"] = "positive primary interval" if inference["bonferroni_percentile"][0] > 0 else "no demonstrated primary superiority"
    masks = {"original_only": np.array([k[2] == 0 for k in keys])}
    results = {m: summarize(split, a, n_boot=n_boot, seed=seed) for m, a in assignments.items()}
    subsets = {label: {"n_queries": int(mask.sum()), "frozen_full_batch_F1": {m: float(v[mask].mean()) if mask.any() else None for m, v in chosen.items()}}
               for label, mask in masks.items()}
    _, all_assignments, _, _ = frozen_actions(lock, full_raw, full_questions)
    all_f1 = np.asarray([[index[k, a]["F1"] for a in ("A1", "A2")] for k in full_keys])
    subsets["full_pipeline_including_invalid_paraphrases"] = {
        "n_queries": len(full_keys), "assignment_scope": "separate full-batch quota allocation; descriptive semantic-error sensitivity only",
        "F1": {m: float(all_f1[np.arange(len(full_keys)), a].mean()) for m, a in all_assignments.items()}}
    decisions = [{"dataset": k[0], "cluster_id": k[1], "variant_id": k[2], "method": m,
                  "action": split.actions[a[i]], "score": None if fallbacks[m] else float(scores[m][i])}
                 for m, a in assignments.items() for i, k in enumerate(keys)]
    return {"methods": results, "primary_inference": inference, "descriptive_subsets": subsets,
            "primary_population": "all frozen originals plus independently human-approved retained paraphrases, fixed before action collection",
            "primary_queries": len(keys), "invalid_paraphrases_retained_in_sensitivity": int((~valid_subset).sum()),
            "secondary_interval_scope": "unadjusted descriptive frozen-assignment intervals for EM, cost and robustness; not additional superiority claims",
            "router_scoring_and_batch_allocation_seconds": router_seconds, "n_boot": n_boot, "seed": seed,
            "quality_cost_limit": "action timings measured separately; policy end-to-end online deployment not measured",
            "status": "locked_policy_evaluation"}, decisions


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("queries", "run", "protocol", "confirmation-bundle", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args(argv)
    from experiment_ver2.followup.freeze import load_binding
    binding, queries, lock, valid = load_binding(a.confirmation_bundle, a.queries, a.protocol)
    dataset = lock["dataset"]
    status, identity = read_json(a.run / "status.json"), read_json(a.run / "run_identity.json")
    if status["status"] != "action_collection_complete" or status["outcomes_sha256"] != digest(a.run / "outcomes.jsonl"):
        raise ValueError("new action collection incomplete or fingerprint changed")
    if identity["inputs"]["queries_file_sha256"] != digest(a.queries) or identity["inputs"]["protocol_sha256"] != digest(a.protocol):
        raise ValueError("generation query/protocol differs from prospective evaluation")
    if identity["inputs"].get("confirmation_binding_sha256") != digest(a.confirmation_bundle / "binding.json"):
        raise ValueError("sample/policy/human review were not bound before this action collection")
    result, decisions = evaluate_locked(queries, validate_collection(a.run, queries), lock, valid)
    result.update({"dataset": dataset, "sample_clusters": SAMPLE_SIZES[dataset], "human_review_recorded_complete": True,
                   "limits": "does not certify semantic deduplication, independent human identities, undisclosed exposure, or original raw scaler provenance"})
    out = new_directory(a.out, (a.queries, a.run, a.protocol, a.confirmation_bundle))
    write_json(out / "report.json", result)
    write_jsonl(out / "decisions.jsonl", decisions)
    write_json(out / "inputs.json", {str(p): digest(p) for p in (a.queries, a.protocol, a.confirmation_bundle / "binding.json", a.run / "outcomes.jsonl", a.run / "run_identity.json")})
    seal(out)
    print(f"Locked-policy new-data report: {out}")


if __name__ == "__main__":
    main()
