"""Prepare a TRAIN-only action pilot and prospective policy lock without TEST fit."""

import argparse
from pathlib import Path

import numpy as np

from experiment_ver2.archive import load_archive
from experiment_ver2.run import fit_candidates, score_candidates, freeze_policy
from experiment_ver2.followup.io import digest, read_json, read_jsonl, write_json, write_jsonl, new_directory, seal
from experiment_ver2.followup.sampling import exposure_inventory


def prepare(archive, output, protocol, per_dataset=34, seed=42):
    if type(per_dataset) is not int or per_dataset < 1:
        raise ValueError("pilot size must be a positive integer")
    archive = Path(archive)
    splits = read_json(archive / "v0.3/splits.json")
    out = new_directory(output, (archive, protocol))
    for dataset in ("nq", "hotpotqa"):
        folder = out / dataset
        folder.mkdir()
        clusters = read_jsonl(archive / "v0.3" / dataset / "clusters.jsonl")
        train = set(splits[dataset]["train"])
        complete = sorted(c["cluster_id"] for c in clusters if c["cluster_id"] in train and c["complete"] is True)
        if len(complete) < per_dataset:
            raise ValueError("insufficient complete TRAIN clusters for fixed pilot size")
        selected = {complete[i] for i in np.random.default_rng(seed).permutation(len(complete))[:per_dataset]}
        rows = [r for r in read_jsonl(archive / "v0.3" / dataset / "queries.jsonl") if r["cluster_id"] in selected]
        rows = [{"dataset": dataset, **r} for r in rows]
        write_jsonl(folder / "pilot_queries.jsonl", rows)
        data = load_archive(archive, dataset, version="v0.4", test_set="fresh")
        candidates = fit_candidates(data)
        val_scores = score_candidates(candidates, data.val.X, data.widths, data.actions, data.signals)
        # This lock preserves the original primary method. The exposed fresh
        # test is loaded for archive validation but its outcomes never fit/select.
        primary = ("fuzzy-calibrated", "ridge")
        fits = {m: freeze_policy(m, val_scores[m], data.val, .25) for m in primary}
        locked = {m: {"fit": fits[m], "model": candidates[m].get(fits[m]["candidate"])} for m in primary}
        write_json(folder / "policy_lock.json", {"dataset": dataset, "actions": data.actions, "signals": data.signals,
                   "widths": data.widths.tolist(), "policies": locked, "quota": .25,
                   "scalers": read_json(archive / "v0.3/scalers.json")[dataset],
                   "scope": "applies only to original A1/A2 action definitions; new A2-union requires separate TRAIN/VAL outcomes and a new policy lock",
                   "preprocessing_limit": "archived TRAIN scaler; old raw signals unavailable for independent source refit"})
        write_json(folder / "exposure_inventory.json", exposure_inventory(archive, dataset))
        write_json(folder / "pilot_manifest.json", {"dataset": dataset, "seed": seed,
                   "clusters": sorted(selected), "n_clusters": len(selected), "n_queries": len(rows),
                   "split": "train", "protocol_sha256": digest(protocol),
                   "queries_sha256": digest(folder / "pilot_queries.jsonl"),
                   "policy_lock_sha256": digest(folder / "policy_lock.json"),
                   "exposure_sha256": digest(folder / "exposure_inventory.json"),
                   "source_sha256": {str(p.relative_to(archive)): digest(p) for p in
                       sorted(archive.rglob("*")) if p.is_file() and p.suffix in (".json", ".jsonl")},
                   "outcomes_status": "not_collected", "research_complete": False})
    seal(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    a = ap.parse_args()
    print(prepare(a.archive, a.out, a.protocol))


if __name__ == "__main__":
    main()
