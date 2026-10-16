"""Identity, metadata, and outcome-blind human-review sampling safeguards."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiment_ver2.data import DataValidationError
from experiment_ver2.robustness.integrity import (
    audit_archive, audit_records, blind_review_sample, duplicate_questions,
)


def cluster(cid="c1", size=3):
    texts = [f"Where was person {cid} born?", f"What is the birthplace of person {cid}?",
             f"Which city is person {cid} from originally?"]
    return {"cluster_id": cid, "complete": size == 3, "answer": ["London"],
            "phrasings": [{"variant_id": i, "text": text} for i, text in enumerate(texts[:size])],
            "screening": [{"text": text, "passed": True} for text in texts[1:size]]}


def queries_for(records):
    return [{"cluster_id": c["cluster_id"], "variant_id": p["variant_id"],
             "question": p["text"], "answer": list(c["answer"])}
            for c in records for p in c["phrasings"]]


def kinds(report):
    return {issue["kind"] for issue in report["issues"]}


def write_rows(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_valid_complete_and_incomplete_records_keep_all_queries():
    records = [cluster("a"), cluster("b", 2), cluster("c", 1)]
    report = audit_records(queries_for(records), records)
    assert report["issues"] == []
    assert report["n_clusters"] == 3
    assert report["n_queries"] == 6
    assert report["complete_clusters"] == 1
    assert report["sibling_pairs"] == 4
    assert "does not establish semantic equivalence" in report["interpretation"]


@pytest.mark.parametrize("corruption,expected", [
    ("question", "query_text_mismatch"),
    ("answer", "answer_metadata_mismatch"),
    ("screening", "retained_paraphrase_not_passed"),
    ("empty_cluster_answer", "invalid_cluster_answers"),
])
def test_query_and_screening_disagreements_are_reported(corruption, expected):
    records = [cluster()]
    queries = queries_for(records)
    if corruption == "question":
        queries[1]["question"] = "A different information need?"
    elif corruption == "answer":
        queries[1]["answer"] = ["Paris"]
    elif corruption == "screening":
        records[0]["screening"][0]["passed"] = False
    else:
        records[0]["answer"] = ["???"]
    assert expected in kinds(audit_records(queries, records))


def test_missing_and_extra_queries_are_counted_without_dropping_disagreements():
    records = [cluster()]
    queries = queries_for(records)[1:]
    queries.append({"cluster_id": "orphan", "variant_id": 0, "question": "Where is this?", "answer": ["Paris"]})
    report = audit_records(queries, records)
    coverage = next(i for i in report["issues"] if i["kind"] == "query_phrasing_key_mismatch")
    assert coverage["missing_query_keys"] == coverage["extra_query_keys"] == 1
    assert "query_text_mismatch" in kinds(report)


def test_duplicate_cluster_and_query_keys_cannot_pass():
    record = cluster()
    queries = queries_for([record])
    report = audit_records(queries + [deepcopy(queries[0])], [record, deepcopy(record)])
    assert {"duplicate_query_key", "duplicate_cluster_id"} <= kinds(report)


@pytest.mark.parametrize("invalid", [-1, 3, True, "1", None])
def test_retained_variants_require_integer_ids_zero_one_two(invalid):
    record = cluster(size=2)
    record["phrasings"][1]["variant_id"] = invalid
    report = audit_records(queries_for([record]), [record])
    assert "invalid_phrasings" in kinds(report)
    assert "invalid_query_record" in kinds(report)


def test_duplicate_or_missing_original_variant_is_an_issue():
    for variants in ((0, 1, 1), (1, 2)):
        record = cluster(size=len(variants))
        for phrasing, variant in zip(record["phrasings"], variants):
            phrasing["variant_id"] = variant
        assert "invalid_variant_ids" in kinds(audit_records(queries_for([record]), [record]))


@pytest.mark.parametrize("complete", [None, 1, "true", False])
def test_complete_flag_must_be_boolean_and_agree_with_structure(complete):
    record = cluster()
    if complete is None:
        del record["complete"]
    else:
        record["complete"] = complete
    report = audit_records(queries_for([record]), [record])
    assert "complete_flag_mismatch" in kinds(report)
    assert report["complete_clusters"] == 1


def test_malformed_screening_log_is_reported_instead_of_silently_accepting():
    record = cluster()
    queries = queries_for([record])
    record["screening"] = {"passed": True}
    assert {"invalid_screening_log", "retained_paraphrase_not_passed"} <= kinds(audit_records(queries, [record]))


def test_duplicate_normalization_distinguishes_within_cluster_and_cross_split():
    def row(split, cid, variant, text):
        return {"split": split, "cluster_id": cid, "variant_id": variant, "question": text}
    records = [row("train", "same", 0, "Ｗｈｅｒｅ is it?"),
               row("train", "same", 1, "where\u200b is IT!!!"),
               row("train", "a", 0, "Who is she?"), row("train", "b", 1, "WHO is she"),
               row("val", "c", 0, "When did it occur?"), row("fresh", "d", 2, "WHEN DID IT OCCUR")]
    collisions = duplicate_questions(records)
    assert len(collisions["within_cluster"]) == 1
    assert len(collisions["cross_cluster"]) == 2
    assert len(collisions["cross_split"]) == 1
    assert {r["split"] for r in collisions["cross_split"][0]["rows"]} == {"val", "fresh"}


def test_review_sample_is_reproducible_blind_and_covers_eligible_clusters(tmp_path):
    records = [cluster("a"), cluster("b", 2), cluster("c", 1)]
    # Unexpected sensitive fields must not be copied into the rater template.
    for c in records:
        c.update(model="secret_model", F1=.99, chosen_action="A2", score=.9)
    path = tmp_path / "v0.4/fresh_src/demo/clusters.jsonl"
    write_rows(path, records)
    before = path.read_bytes()
    tasks, key = blind_review_sample(tmp_path, names=("demo",), per_dataset=10, seed=77)
    assert (tasks, key) == blind_review_sample(tmp_path, names=("demo",), per_dataset=10, seed=77)
    assert path.read_bytes() == before
    assert len(tasks) == len(key["items"]) == 2
    assert key["eligibility"]["demo"] == {"eligible_clusters": 2, "no_retained_paraphrase": 1}
    assert {item["cluster_id"] for item in key["items"]} == {"a", "b"}
    allowed = {"review_id", "left_question", "right_question", "same_information_need",
               "added_or_removed_constraint", "grammatical_and_answerable", "rater_id", "notes"}
    by_id = {task["review_id"]: task for task in tasks}
    originals = {c["cluster_id"]: c["phrasings"][0]["text"] for c in records}
    for task in tasks:
        assert set(task) == allowed
        assert task["same_information_need"] is task["rater_id"] is None
        assert "London" not in json.dumps(task)
        assert "secret_model" not in json.dumps(task)
    for item in key["items"]:
        assert by_id[item["review_id"]][item["original_side"] + "_question"] == originals[item["cluster_id"]]


@pytest.mark.parametrize("invalid", [-1, 1.5, True])
def test_review_size_rejects_invalid_counts(tmp_path, invalid):
    with pytest.raises(ValueError, match="nonnegative integer"):
        blind_review_sample(tmp_path, per_dataset=invalid)


def test_review_missing_original_or_duplicate_cluster_fails_explicitly(tmp_path):
    path = tmp_path / "v0.4/fresh_src/demo/clusters.jsonl"
    record = cluster()
    record["phrasings"] = record["phrasings"][1:]
    write_rows(path, [record])
    with pytest.raises(ValueError, match="invalid review variant IDs"):
        blind_review_sample(tmp_path, names=("demo",))
    record = cluster()
    write_rows(path, [record, deepcopy(record)])
    with pytest.raises(ValueError, match="duplicate cluster"):
        blind_review_sample(tmp_path, names=("demo",))


def test_archive_audit_rejects_overlapping_split_ids(tmp_path):
    path = tmp_path / "v0.3/splits.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"demo": {"train": ["a"], "val": ["b"], "test": ["a"]}}), encoding="utf-8")
    with pytest.raises(DataValidationError, match="overlapping cluster"):
        audit_archive(tmp_path, "demo")


@pytest.mark.parametrize("name,queries,complete,near", [("nq", 412, 128, 8), ("hotpotqa", 443, 143, 5)])
def test_committed_archive_metadata_and_literal_overlap_audit_is_read_only(name, queries, complete, near):
    root = Path(__file__).resolve().parents[2] / "runs"
    paths = [root / "v0.3/splits.json"]
    paths += [root / prefix / name / filename for prefix in ("v0.3", "v0.4/fresh_src")
              for filename in ("queries.jsonl", "clusters.jsonl")]
    if not all(path.exists() for path in paths):
        pytest.skip("committed cluster archives not available")
    before = {path: path.read_bytes() for path in paths}
    report = audit_archive(root, name)
    assert report["all_metadata_checks_pass"]
    assert report["no_exact_normalized_cross_split_overlap"]
    assert report["normalized_duplicates"] == {"within_cluster": [], "cross_cluster": [], "cross_split": []}
    assert report["splits"]["fresh"]["n_queries"] == queries
    assert report["splits"]["fresh"]["complete_clusters"] == complete
    assert len(report["splits"]["fresh"]["sibling_pairs_jaccard_at_least_0_9"]) == near
    assert "not established" in report["semantic_deduplication"]
    assert before == {path: path.read_bytes() for path in paths}
