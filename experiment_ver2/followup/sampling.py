"""Freeze a new original-question sample without consulting any action outcomes.

Source-order compatibility is verified against EVERY archived original. All
available cluster/query/outcome IDs and normalized phrasings are excluded.
Semantic equivalence and undisclosed prior exposure remain external audit items.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiment_ver2.archive import normalize_question
from experiment_ver2.followup.io import read_json, read_jsonl, write_json, write_jsonl, digest, new_directory, seal


def exposure_inventory(archive, dataset):
    archive = Path(archive)
    required = [archive / prefix / dataset / name for prefix in ("v0.3", "v0.4/fresh_src")
                for name in ("clusters.jsonl", "queries.jsonl")]
    for path in required:
        if not path.is_file():
            raise ValueError(f"missing known exposure archive: {path}")
    originals, texts, identifiers, files = {}, set(), set(), {}
    for path in sorted(archive.rglob("*.jsonl")):
        if dataset not in path.parts:
            continue
        files[path.relative_to(archive).as_posix()] = digest(path)
        for row in read_jsonl(path):
            cid = row.get("cluster_id")
            if cid is None:
                continue
            if not isinstance(cid, str) or not cid.startswith(dataset + "-"):
                raise ValueError(f"unexpected cluster namespace in {path}: {cid}")
            identifiers.add(cid)
            if row.get("question"):
                texts.add(normalize_question(row["question"]))
            if "phrasings" in row:
                if row.get("dataset", dataset) != dataset or type(row.get("source_index")) is not int:
                    raise ValueError("invalid dataset or source_index in archived cluster")
                phrasings = row["phrasings"]
                variants = [p["variant_id"] for p in phrasings]
                if len(variants) != len(set(variants)) or variants.count(0) != 1:
                    raise ValueError("invalid archived original/variant IDs")
                for p in phrasings:
                    texts.add(normalize_question(p["text"]))
                original = next(p["text"] for p in phrasings if p["variant_id"] == 0)
                identity = {"source_index": row["source_index"], "question": normalize_question(original),
                            "source_id": row.get("_id")}
                if cid in originals and originals[cid] != identity:
                    raise ValueError(f"contradictory exposure records: {cid}")
                originals[cid] = identity
    if identifiers - set(originals):
        raise ValueError("some exposed IDs lack original-question metadata; cannot certify exclusion coverage")
    if len({r["source_index"] for r in originals.values()}) != len(originals):
        raise ValueError("multiple archived cluster IDs share one source_index")
    # The original v0.4 pilot selected only old TRAIN clusters. Report that
    # evidence explicitly; do not manufacture missing pilot query files.
    return {"dataset": dataset, "originals": originals, "normalized_texts": sorted(texts),
            "files_sha256": files, "n_exposed_originals": len(originals),
            "scope": "all available archived JSONL records; separate undisclosed exposure requires extra archive records",
            "pilot_scope": "scripts/escalate.py choose_instances selects pilot from complete v0.3 TRAIN clusters; original pilot instances were not archived"}


def load_pool(path, dataset):
    path = Path(path)
    rows = read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("source pool must be a nonempty JSON array or JSONL")
    result, seen = [], set()
    for index, row in enumerate(rows):
        q = row.get("question")
        answers = row.get("answer")
        if isinstance(answers, str):
            answers = [answers]
        if not isinstance(q, str) or not normalize_question(q) or not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
            raise ValueError(f"invalid source question/answers at index {index}")
        if any(k in row for k in ("F1", "EM", "action", "latency_total", "outcomes")):
            raise ValueError("source pool must not contain action outcomes")
        source_id = row.get("_id")
        if dataset == "hotpotqa" and (not isinstance(source_id, str) or not source_id or source_id in seen):
            raise ValueError("HotpotQA requires unique nonempty _id")
        seen.add(source_id)
        result.append({"dataset": dataset, "cluster_id": f"{dataset}-{index:05d}", "source_index": index,
                       "question": q, "answer": answers,
                       **{k: row[k] for k in ("_id", "type", "level") if k in row}})
    return result


def sample_pool(pool, exposure, n, seed):
    if type(n) is not int or n < 1 or type(seed) is not int or seed < 0:
        raise ValueError("positive integer n and nonnegative integer seed required")
    for record in exposure["originals"].values():
        i = record["source_index"]
        if i < 0 or i >= len(pool) or normalize_question(pool[i]["question"]) != record["question"] or pool[i].get("_id") != record["source_id"]:
            raise ValueError("source pool order/content differs from archived source; source_index exclusion is unsafe")
    excluded_texts = set(exposure["normalized_texts"])
    excluded_ids = set(exposure["originals"])
    seen_texts, eligible, excluded = set(), [], defaultdict(int)
    for row in pool:
        text = normalize_question(row["question"])
        if row["cluster_id"] in excluded_ids:
            excluded["previous_original_id"] += 1
        elif text in excluded_texts:
            excluded["previous_normalized_text"] += 1
        elif text in seen_texts:
            excluded["duplicate_in_pool"] += 1
        else:
            eligible.append(row)
        seen_texts.add(text)
    if len(eligible) < n:
        raise ValueError(f"need {n} unused unique originals, only {len(eligible)} eligible")
    order = np.random.default_rng(seed).permutation(len(eligible))[:n]
    return [eligible[i] for i in order], {"pool_size": len(pool), "eligible_originals": len(eligible),
             "excluded": dict(excluded), "n": n, "seed": seed, "source_order_verified_against_all_exposed_originals": True}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=Path("runs"))
    ap.add_argument("--dataset", choices=("nq", "hotpotqa"), required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--source-url", required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    exposure = exposure_inventory(a.archive, a.dataset)
    sample, summary = sample_pool(load_pool(a.pool, a.dataset), exposure, a.n, a.seed)
    protocol_hash = digest(a.protocol)
    out = new_directory(a.out, (a.archive, a.pool, a.protocol))
    write_jsonl(out / "originals.jsonl", sample)
    write_json(out / "exposure_inventory.json", exposure)
    write_json(out / "manifest.json", {**summary, "dataset": a.dataset, "source_url": a.source_url,
               "source_sha256": digest(a.pool), "protocol_sha256": protocol_hash,
               "sample_sha256": digest(out / "originals.jsonl"),
               "exposure_sha256": digest(out / "exposure_inventory.json"),
               "semantic_deduplication": "pending independent semantic review",
               "outcomes_generated": False, "confirmatory_complete": False})
    seal(out)
    print(f"Frozen {len(sample)} originals; outcomes and semantic review remain pending: {out}")


if __name__ == "__main__":
    main()
