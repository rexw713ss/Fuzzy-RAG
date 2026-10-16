"""Additional read-only checks of query identity and paraphrase construction."""

from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from experiment_ver2.archive import normalize_question
from experiment_ver2.data import _split_mapping


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def duplicate_questions(records):
    """Records have split, cluster_id, variant_id, question. Report all collisions."""
    by_text = defaultdict(list)
    for row in records:
        by_text[normalize_question(row["question"])].append(
            {k: row[k] for k in ("split", "cluster_id", "variant_id")})
    within, cross_cluster, cross_split = [], [], []
    for text, rows in by_text.items():
        if len(rows) < 2:
            continue
        item = {"normalized_question": text, "rows": rows}
        if len({(r["split"], r["cluster_id"]) for r in rows}) == 1:
            within.append(item)
        else:
            cross_cluster.append(item)
        if len({r["split"] for r in rows}) > 1:
            cross_split.append(item)
    return {"within_cluster": within, "cross_cluster": cross_cluster, "cross_split": cross_split}


def _answers(value):
    if not isinstance(value, list) or not value or any(not isinstance(v, str) or not v.strip() for v in value):
        return None
    normalized = [normalize_question(v) for v in value]
    return sorted(set(normalized)) if all(normalized) else None


def _valid_cluster_id(value):
    return type(value) is int or (type(value) is str and bool(value.strip()))


def _valid_phrasing(row):
    return (isinstance(row, dict) and type(row.get("variant_id")) is int
            and row["variant_id"] in (0, 1, 2) and isinstance(row.get("text"), str)
            and bool(normalize_question(row["text"])))


def audit_records(queries, clusters):
    """Check retained phrasings against the query table and screening log."""
    issues, siblings, near_duplicates = [], [], []
    indexed = {}
    for row in queries:
        if (not isinstance(row, dict) or not _valid_cluster_id(row.get("cluster_id"))
                or type(row.get("variant_id")) is not int or row["variant_id"] not in (0, 1, 2)
                or not isinstance(row.get("question"), str) or not normalize_question(row["question"])):
            issues.append({"kind": "invalid_query_record"})
            continue
        key = row["cluster_id"], row["variant_id"]
        if key in indexed:
            issues.append({"cluster_id": key[0], "variant_id": key[1], "kind": "duplicate_query_key"})
        indexed[key] = row
    expected = set()
    seen_clusters, complete_count = set(), 0
    for c in clusters:
        if not isinstance(c, dict) or not _valid_cluster_id(c.get("cluster_id")):
            issues.append({"kind": "invalid_cluster_record"})
            continue
        cid = c["cluster_id"]
        if cid in seen_clusters:
            issues.append({"cluster_id": cid, "kind": "duplicate_cluster_id"})
        seen_clusters.add(cid)
        phrasings = c.get("phrasings", [])
        if not isinstance(phrasings, list) or not phrasings or not all(map(_valid_phrasing, phrasings)):
            issues.append({"cluster_id": cid, "kind": "invalid_phrasings"})
            phrasings = [p for p in phrasings if _valid_phrasing(p)] if isinstance(phrasings, list) else []
        vids = [p["variant_id"] for p in phrasings]
        if len(vids) != len(set(vids)) or 0 not in vids:
            issues.append({"cluster_id": cid, "kind": "invalid_variant_ids"})
        structurally_complete = len(vids) == 3 and set(vids) == {0, 1, 2}
        if type(c.get("complete")) is not bool or c["complete"] != structurally_complete:
            issues.append({"cluster_id": cid, "kind": "complete_flag_mismatch"})
        complete_count += structurally_complete
        if _answers(c.get("answer")) is None:
            issues.append({"cluster_id": cid, "kind": "invalid_cluster_answers"})
        screening = c.get("screening", [])
        if (not isinstance(screening, list) or any(not isinstance(p, dict)
                or not isinstance(p.get("text"), str) or type(p.get("passed")) is not bool for p in screening)):
            issues.append({"cluster_id": cid, "kind": "invalid_screening_log"})
            screening = [p for p in screening if isinstance(p, dict)] if isinstance(screening, list) else []
        accepted = {p["text"] for p in screening if p.get("passed") is True and isinstance(p.get("text"), str)}
        for p in phrasings:
            key = cid, p["variant_id"]
            expected.add(key)
            q = indexed.get(key)
            if q is None or q.get("question") != p["text"]:
                issues.append({"cluster_id": cid, "variant_id": key[1], "kind": "query_text_mismatch"})
            if q is not None and (_answers(q.get("answer")) is None or _answers(q["answer"]) != _answers(c.get("answer"))):
                issues.append({"cluster_id": cid, "variant_id": key[1], "kind": "answer_metadata_mismatch"})
            if key[1] != 0 and p["text"] not in accepted:
                issues.append({"cluster_id": cid, "variant_id": key[1], "kind": "retained_paraphrase_not_passed"})
        for left, right in combinations(phrasings, 2):
            a, b = (set(normalize_question(p["text"]).split()) for p in (left, right))
            similarity = len(a & b) / len(a | b) if a | b else 1.0
            item = {"cluster_id": cid, "variants": [left["variant_id"], right["variant_id"]],
                    "token_jaccard": similarity}
            siblings.append(similarity)
            if similarity >= .9:
                near_duplicates.append(item)
    if expected != set(indexed):
        issues.append({"kind": "query_phrasing_key_mismatch", "extra_query_keys": len(set(indexed) - expected),
                       "missing_query_keys": len(expected - set(indexed))})
    return {"issues": issues, "n_clusters": len(clusters), "n_queries": len(queries),
            "complete_clusters": complete_count,
            "sibling_pairs": len(siblings), "sibling_token_jaccard_median": float(np.median(siblings)) if siblings else None,
            "sibling_pairs_jaccard_at_least_0_9": near_duplicates,
            "interpretation": "lexical diagnostic only; answer metadata agreement does not establish semantic equivalence"}


def audit_archive(root, name):
    root = Path(root)
    _, split_of = _split_mapping(root / "v0.3", name)
    old_queries = read_rows(root / "v0.3" / name / "queries.jsonl")
    old_clusters = read_rows(root / "v0.3" / name / "clusters.jsonl")
    fresh_queries = read_rows(root / "v0.4/fresh_src" / name / "queries.jsonl")
    fresh_clusters = read_rows(root / "v0.4/fresh_src" / name / "clusters.jsonl")
    records = [{**r, "split": split_of[r["cluster_id"]]} for r in old_queries]
    records += [{**r, "split": "fresh"} for r in fresh_queries]
    overlap = duplicate_questions(records)
    group_reports = {}
    for split in ("train", "val", "test", "fresh"):
        queries = [r for r in records if r["split"] == split]
        clusters = fresh_clusters if split == "fresh" else [c for c in old_clusters if split_of[c["cluster_id"]] == split]
        group_reports[split] = audit_records(queries, clusters)
        group_reports[split]["variants_per_cluster"] = dict(sorted(Counter(len(c["phrasings"]) for c in clusters).items()))
    return {"dataset": name, "splits": group_reports, "normalized_duplicates": overlap,
            "all_metadata_checks_pass": all(not r["issues"] for r in group_reports.values()),
            "no_exact_normalized_cross_split_overlap": not overlap["cross_split"],
            "semantic_deduplication": "not established; requires semantic review with original source data"}


def blind_review_sample(root, names=("nq", "hotpotqa"), per_dataset=30, seed=20261016):
    """Outcome-blind sample for two independent human raters, plus separate key.

    Sample clusters uniformly, then one retained paraphrase; incomplete clusters
    without a paraphrase cannot be reviewed and are counted separately. Randomize
    left/right; omit gold answers, outcomes, actions, model scores and screening.
    """
    if type(per_dataset) is not int or per_dataset < 0:
        raise ValueError("per_dataset must be a nonnegative integer")
    names = tuple(names)
    if len(names) != len(set(names)):
        raise ValueError("dataset names must be unique")
    rng = np.random.default_rng(seed)
    tasks, key, eligibility = [], [], {}
    for name in names:
        clusters = read_rows(Path(root) / "v0.4/fresh_src" / name / "clusters.jsonl")
        seen = set()
        for cluster in clusters:
            if not isinstance(cluster, dict) or not _valid_cluster_id(cluster.get("cluster_id")):
                raise ValueError(f"{name}: invalid review cluster record")
            phrasings = cluster.get("phrasings")
            if (not isinstance(phrasings, list) or not phrasings or not all(map(_valid_phrasing, phrasings))):
                raise ValueError(f"{name}/{cluster['cluster_id']}: invalid review phrasings")
            vids = [p["variant_id"] for p in phrasings]
            if len(vids) != len(set(vids)) or 0 not in vids or cluster["cluster_id"] in seen:
                raise ValueError(f"{name}/{cluster['cluster_id']}: duplicate cluster or invalid review variant IDs")
            seen.add(cluster["cluster_id"])
        candidates = [c for c in clusters if len(c["phrasings"]) > 1]
        eligibility[name] = {"eligible_clusters": len(candidates), "no_retained_paraphrase": len(clusters) - len(candidates)}
        for j in rng.choice(len(candidates), min(per_dataset, len(candidates)), replace=False):
            c = candidates[j]
            original = next(p for p in c["phrasings"] if p["variant_id"] == 0)
            others = [p for p in c["phrasings"] if p["variant_id"] != 0]
            paraphrase = others[int(rng.integers(len(others)))]
            swap = bool(rng.integers(2))
            left, right = (paraphrase, original) if swap else (original, paraphrase)
            review_id = f"R{len(tasks) + 1:03d}"
            tasks.append({"review_id": review_id, "left_question": left["text"], "right_question": right["text"],
                          "same_information_need": None, "added_or_removed_constraint": None,
                          "grammatical_and_answerable": None, "rater_id": None, "notes": ""})
            key.append({"review_id": review_id, "dataset": name, "cluster_id": c["cluster_id"],
                        "paraphrase_variant_id": paraphrase["variant_id"], "original_side": "right" if swap else "left"})
    rng.shuffle(tasks)
    return tasks, {"seed": seed, "eligibility": eligibility, "items": key,
                   "status": "pending human ratings; never infer agreement or equivalence from this template"}
