"""Bind the unseen sample, policy and completed blind review BEFORE collection."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from experiment_ver2.followup.io import digest, read_json, read_jsonl, write_json, new_directory, seal
from experiment_ver2.followup.review import verified_audit


def validate_inputs(paths):
    from experiment_ver2.followup.confirmation import SAMPLE_SIZES, check_sample, review_subset
    p = {k: Path(v) for k, v in paths.items()}
    queries, originals = read_jsonl(p["queries"]), read_jsonl(p["originals"])
    sample = read_json(p["sample_manifest"])
    dataset = sample.get("dataset")
    if dataset not in SAMPLE_SIZES or sample.get("seed") != 20261018 or sample.get("n") != SAMPLE_SIZES[dataset]:
        raise ValueError("sample must use the fixed dataset, size and seed")
    if (sample.get("sample_sha256") != digest(p["originals"])
            or sample.get("exposure_sha256") != digest(p["exposure"])
            or sample.get("protocol_sha256") != digest(p["protocol"])):
        raise ValueError("sample/source-exposure/protocol fingerprint mismatch")
    exposure = read_json(p["exposure"])
    if exposure.get("dataset") != dataset:
        raise ValueError("exposure dataset mismatch")
    check_sample(queries, originals, exposure, sample)
    policy = read_json(p["policy"])
    manifest = read_json(p["policy_manifest"])
    if (policy["dataset"] != dataset or manifest["policy_lock_sha256"] != digest(p["policy"])
            or manifest["protocol_sha256"] != digest(p["protocol"])):
        raise ValueError("frozen policy/dataset/protocol mismatch")
    generation = read_json(p["generation_status"])
    if generation.get("status") != "paraphrase_generation_complete":
        raise ValueError("prospective paraphrase generation incomplete")
    for label, name in (("queries", "queries"), ("tasks", "review_tasks"), ("key", "review_key")):
        if generation.get(label + "_sha256") != digest(p[name]):
            raise ValueError("generated queries/review task/key fingerprint mismatch")
    generated_identity = read_json(p["generation_identity"])
    if (generated_identity["inputs"]["sample_manifest_sha256"] != digest(p["sample_manifest"])
            or generated_identity["inputs"]["protocol_sha256"] != digest(p["protocol"])):
        raise ValueError("paraphrases were not generated from this frozen sample/protocol")
    tasks = read_jsonl(p["review_tasks"])
    review = verified_audit(p["human_report"], tasks)
    mask = review_subset(queries, review, read_json(p["review_key"]), tasks)
    return queries, policy, mask


def load_binding(directory, queries_path, protocol_path):
    directory = Path(directory)
    binding = read_json(directory / "binding.json")
    for name, value in binding["inputs"].items():
        if digest(value["path"]) != value["sha256"]:
            raise ValueError(f"pre-collection binding changed: {name}")
    if (binding["inputs"]["queries"]["sha256"] != digest(queries_path)
            or binding["inputs"]["protocol"]["sha256"] != digest(protocol_path)):
        raise ValueError("collection query/protocol differs from the pre-collection binding")
    queries, policy, mask = validate_inputs({k: v["path"] for k, v in binding["inputs"].items()})
    if mask.tolist() != binding["primary_population_mask"]:
        raise ValueError("primary population changed after freeze")
    return binding, queries, policy, mask


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("sample", "paraphrases", "lock", "protocol", "human-report", "out"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    paths = {"queries": a.paraphrases / "queries.jsonl", "originals": a.sample / "originals.jsonl",
        "sample_manifest": a.sample / "manifest.json", "exposure": a.sample / "exposure_inventory.json",
        "policy": a.lock, "policy_manifest": a.lock.parent / "pilot_manifest.json", "protocol": a.protocol,
        "generation_status": a.paraphrases / "status.json", "generation_identity": a.paraphrases / "run_identity.json",
        "review_tasks": a.paraphrases / "blind_review_tasks.jsonl", "review_key": a.paraphrases / "blind_review_key.json",
        "human_report": a.human_report, "ratings_A": a.human_report.parent / "rater_A_raw.jsonl",
        "ratings_B": a.human_report.parent / "rater_B_raw.jsonl"}
    adjudications = a.human_report.parent / "adjudications_raw.jsonl"
    if adjudications.exists():
        paths["adjudications"] = adjudications
    queries, policy, mask = validate_inputs(paths)
    out = new_directory(a.out, paths.values())
    write_json(out / "binding.json", {"schema": 1, "status": "ready_for_prospective_collection",
        "created_utc": datetime.now(timezone.utc).isoformat(), "dataset": policy["dataset"],
        "inputs": {k: {"path": str(v.resolve()), "sha256": digest(v)} for k, v in paths.items()},
        "primary_population_mask": mask.tolist(), "primary_queries": int(mask.sum()),
        "all_queries": len(queries), "action_outcomes_observed": False,
        "limit": "local content binding, not independent preregistration or proof of human identity"})
    seal(out)
    print(out)


if __name__ == "__main__":
    main()
