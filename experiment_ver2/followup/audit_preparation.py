"""Read-only audit of real prepared inputs and currently available runtime assets."""
import argparse
from pathlib import Path

from experiment_ver2.followup import runtime, paraphrases, review
from experiment_ver2.followup.io import digest, read_json, read_jsonl, write_json


def audit(root):
    root = Path(root)
    folder = root / "experiment_ver2/followup"
    splits = read_json(root / "runs/v0.3/splits.json")
    sealed = read_json(folder / "prepared/SHA256.json")
    for name, checksum in sealed.items():
        if digest(folder / "prepared" / name) != checksum:
            raise ValueError("prepared input seal mismatch: " + name)
    datasets = {}
    for dataset in ("nq", "hotpotqa"):
        prepared = folder / "prepared" / dataset
        queries = read_jsonl(prepared / "pilot_queries.jsonl")
        manifest = read_json(prepared / "pilot_manifest.json")
        lock = read_json(prepared / "policy_lock.json")
        inventory = read_json(prepared / "exposure_inventory.json")
        previous = read_json(folder / "evidence/prior_preparation" / dataset / "policy_lock.json")
        if lock != previous:
            raise ValueError("policy changed when revising prospective protocol")
        if (len(queries) != 102 or len({q["cluster_id"] for q in queries}) != 34
                or {q["cluster_id"] for q in queries} - set(splits[dataset]["train"])):
            raise ValueError("pilot must contain 34 complete TRAIN clusters, 102 queries")
        if (manifest["protocol_sha256"] != digest(folder / "PROTOCOL.zh-TW.md")
                or manifest["queries_sha256"] != digest(prepared / "pilot_queries.jsonl")
                or manifest["policy_lock_sha256"] != digest(prepared / "policy_lock.json")
                or manifest["exposure_sha256"] != digest(prepared / "exposure_inventory.json")):
            raise ValueError("prepared protocol/query/policy/exposure fingerprints changed")
        for name, checksum in manifest["source_sha256"].items():
            if digest(root / "runs" / name) != checksum:
                raise ValueError("original archive changed: " + name)
        datasets[dataset] = {"pilot_queries": len(queries), "pilot_clusters": 34,
            "pilot_train_only": True, "frozen_policy_unchanged": True,
            "exposed_originals": inventory["n_exposed_originals"],
            "runtime_assets": runtime.preflight({**read_json(folder / "runtime.example.json"), "dataset": dataset})}
    tasks = read_jsonl(root / "experiment_ver2/robustness/evidence/blind_paraphrase_review.jsonl")
    blank_a, blank_b = (read_jsonl(folder / "human_review" / filename) for filename in ("rater_A.jsonl", "rater_B.jsonl"))
    result = review.audit(tasks, blank_a, blank_b)
    return {"status": "preparation_verified_actual_research_incomplete", "datasets": datasets,
        "human_review": {"status": result["status"], "n_tasks": result["n_tasks"],
                         "pending": len(result["pending"]), "resolved_pairs": len(result["decisions"])},
        "paraphrase_assets": paraphrases.preflight(read_json(folder / "runtime.example.json")),
        "source_sha256": {p.relative_to(root).as_posix(): digest(p) for p in sorted(folder.glob("*.py"))},
        "protocol_sha256": digest(folder / "PROTOCOL.zh-TW.md"),
        "limits": "uses example runtime paths because actual assets are unavailable; no new model outcomes or human completion inferred from preparation/tests"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("refuse to overwrite earlier audit evidence")
    write_json(args.out, audit(args.root.resolve()))
    print(args.out)


if __name__ == "__main__":
    main()
