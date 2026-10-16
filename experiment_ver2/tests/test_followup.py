"""Prospective evidence controls; all generated answers here are toy fixtures."""

from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from experiment_ver2.followup import actions, collect, review, sampling, passages, runtime, evaluate
from experiment_ver2.followup.io import digest, read_json, read_jsonl, write_jsonl


class ToyBackend:
    def __init__(self, followup="find missing person", fail=False):
        self.followup, self.fail, self.calls = followup, fail, []

    def retrieve(self, question):
        self.calls.append(("retrieve", question))
        offset = 100 if question == self.followup else 0
        hits = [(str(i + offset), float(50 - i)) for i in range(50)]
        return {"bm25": hits, "dense": hits, "fused": hits}

    def passages(self, ids):
        return {d: ("title " + d, ("content " + d + " ") * 300) for d in ids}

    def vectors(self, ids):
        return np.eye(len(ids), dtype=float)

    def rerank(self, question, text):
        self.calls.append(("rerank", question, len(text)))
        # Original question prefers new passage 101; followup order differs.
        return [100 if question != self.followup and title == "title 101" else 50 - i for i, (title, _) in enumerate(text)]

    def chat(self, messages, **kwargs):
        self.calls.append(("chat", messages, kwargs))
        if self.fail:
            raise RuntimeError("toy backend deliberately unavailable")
        followup = "Search query:" in messages[0]["content"]
        return self.followup if followup else "gold", {"input_tokens": 20, "output_tokens": 3}


def tasks():
    return [{"review_id": f"R{i}", "left_question": "Who invented thing " + str(i),
             "right_question": "Who created thing " + str(i)} for i in range(3)]


def ratings(rater, votes=(True, True, False)):
    form = review.blank_forms(tasks())
    for row, vote in zip(form, votes):
        row.update({"rater_id": rater, "status": "reviewed", "same_information_need": vote,
                    "added_or_removed_constraint": False, "grammatical_and_answerable": True,
                    "notes": "different entity" if not vote else "",
                    "attestations": {k: True for k in review.ATTEST}})
    return form


def query():
    return {"dataset": "nq", "cluster_id": "nq-00001", "variant_id": 0, "question": "Who invented this?", "answer": ["gold"]}


def test_action_intervention_is_real_rerank_on_original_query():
    old, new = ToyBackend(), ToyBackend()
    a = actions.run_action(query()["question"], "A2", old)
    b = actions.run_action(query()["question"], "A2-union", new)
    assert a["doc_ids"] == ["0", "1", "2", "100", "101"]
    assert b["doc_ids"] == ["101", "0", "1", "2", "3"]
    assert a["ce_passes"] == 40 and b["ce_passes"] == 60
    assert a["llm_calls"] == b["llm_calls"] == 2
    assert ("rerank", new.followup, 20) in old.calls
    assert ("rerank", query()["question"], 40) in new.calls
    assert a["n_new"] == 2 and b["n_new"] == 1
    assert "F1" not in a
    assert actions.score_outcome(a, ["gold"])["F1"] == 1
    assert a["latency_total"] >= a["component_seconds"] >= 0


@pytest.mark.parametrize("action", actions.ACTIONS)
def test_actions_share_answer_decoding_prompt_and_context_cap(action):
    backend = ToyBackend()
    result = actions.run_action(query()["question"], action, backend)
    call = [c for c in backend.calls if c[0] == "chat"][-1]
    assert call[2] == {"temperature": 0.0, "seed": 42, "max_tokens": 32}
    assert actions.ac.ANSWER_PROMPT in call[1][0]["content"]
    assert call[1][0]["content"].count("Passage ") == 5
    assert len(call[1][0]["content"].split()) < 1200
    assert len(result["doc_ids"]) == 5


@pytest.mark.parametrize("action", ["A2", "A2-union"])
def test_empty_followup_is_costed_fallback_but_error_is_not_success(action):
    result = actions.run_action(query()["question"], action, ToyBackend(followup=""))
    assert result["fallback"] and result["retrievals"] == 1 and result["llm_calls"] == 2
    assert result["doc_ids"] == result["a1_top_ids"]
    with pytest.raises(RuntimeError):
        actions.run_action(query()["question"], action, ToyBackend(fail=True))


def test_union_keeps_original_candidates_and_deduplicates():
    assert actions.union_ids(["a", "b", "c"], ["c", "d", "a"]) == ["a", "b", "c", "d"]


def test_completed_action_collection_has_exact_matrix_and_no_label_input_to_backend(tmp_path):
    backend = ToyBackend()
    out = tmp_path / "run"
    rows = collect.collect([query()], backend, out, {"source": "toy"})
    assert len(rows) == 3 and {r["action"] for r in rows} == set(actions.ACTIONS)
    assert read_json(out / "status.json")["status"] == "action_collection_complete"
    assert not read_json(out / "status.json")["confirmatory_evaluation_complete"]
    assert len(read_jsonl(out / "journal.jsonl")) == 6
    previous = len(backend.calls)
    assert collect.collect([query()], backend, out, {"source": "toy"}, resume=True) == rows
    assert len(backend.calls) == previous
    with pytest.raises(ValueError, match="identity"):
        collect.collect([query()], backend, out, {"source": "changed"}, resume=True)
    with pytest.raises(ValueError):
        collect.collect([query()], backend, out, {"source": "toy"})


def test_failed_attempt_preserved_on_resume(tmp_path):
    out = tmp_path / "run"
    with pytest.raises(RuntimeError):
        collect.collect([query()], ToyBackend(fail=True), out, {"source": "toy"})
    assert read_json(out / "status.json")["status"] == "interrupted"
    assert not (out / "collector.lock").exists()
    rows = collect.collect([query()], ToyBackend(), out, {"source": "toy"}, resume=True)
    assert len(rows) == 3
    events = read_jsonl(out / "journal.jsonl")
    assert sum(r["event"] == "failed" for r in events) == 1
    assert read_json(out / "status.json")["attempts"] == 4


def test_incomplete_attempt_is_recorded_failed_then_retried(tmp_path):
    out = tmp_path / "run"
    with pytest.raises(RuntimeError):
        collect.collect([query()], ToyBackend(fail=True), out, {"source": "toy"})
    journal = read_jsonl(out / "journal.jsonl")
    # Simulate a crash after durable start but before either terminal record.
    write_jsonl(out / "journal.jsonl", journal[:1])
    rows = collect.collect([query()], ToyBackend(), out, {"source": "toy"}, resume=True)
    assert len(rows) == 3
    events = read_jsonl(out / "journal.jsonl")
    assert events[1]["event"] == "failed" and "previous process stopped" in events[1]["error"]
    assert events[2]["attempt"] == 2


def test_persistent_lock_never_silently_starts_another_collector(tmp_path):
    out = tmp_path / "run"
    collect.collect([query()], ToyBackend(), out, {"source": "toy"})
    (out / "collector.lock").write_text('{"pid":123}')
    with pytest.raises(ValueError, match="inspect the recorded PID"):
        collect.collect([query()], ToyBackend(), out, {"source": "toy"}, resume=True)
    assert (out / "collector.lock").exists()


def test_order_fixed_before_outcomes_and_each_query_actions_interleaved():
    a = [query(), {**query(), "cluster_id": "nq-00002", "question": "Who made another?"}]
    order = collect.schedule(a, 1)
    for i in range(0, len(order), 3):
        assert len({r["cluster_id"] for r in order[i:i + 3]}) == 1
        assert {r["action"] for r in order[i:i + 3]} == set(actions.ACTIONS)
    assert order == collect.schedule(a, 1)
    with pytest.raises(ValueError):
        collect.schedule([{**query(), "F1": .9}], 1)


def test_blank_human_forms_never_count_as_completed():
    a, b = review.blank_forms(tasks()), review.blank_forms(tasks())
    result = review.audit(tasks(), a, b)
    assert result["status"] == "incomplete" and len(result["pending"]) == 3
    assert result["agreement"]["all_criteria_valid"]["cohen_kappa"] is None


def test_completed_human_review_and_degenerate_kappa():
    result = review.audit(tasks(), ratings("Alice"), ratings("Bob"))
    assert result["status"] == "complete_recorded_human_review"
    assert result["valid_pairs"] == 2 and result["invalid_pairs"] == 1
    assert result["agreement"]["all_criteria_valid"]["cohen_kappa"] == 1
    assert result["agreement"]["added_or_removed_constraint"]["cohen_kappa"] is None


@pytest.mark.parametrize("change", ["same_rater", "alter_text", "outcome", "nonhuman", "missing_vote"])
def test_review_rejects_invalid_or_unblinded_records(change):
    a, b = ratings("Alice"), ratings("Bob")
    if change == "same_rater": b = ratings("Alice")
    if change == "alter_text": b[0]["left_question"] += " answer is X"
    if change == "outcome": b[0]["F1"] = 1
    if change == "nonhuman": b[0]["attestations"]["human_review"] = False
    if change == "missing_vote": b[0]["same_information_need"] = None
    with pytest.raises(ValueError): review.audit(tasks(), a, b)


def test_adjudication_requires_third_person_and_reason():
    a, b = ratings("Alice"), ratings("Bob", (False, True, False))
    result = review.audit(tasks(), a, b)
    assert result["unresolved_adjudication"] == ["R0"]
    decision = {"review_id": "R0", "rater_id": "Charlie", "reason": "entity preserved",
                "human_review": True, "outcome_blind": True,
                "same_information_need": True, "added_or_removed_constraint": False, "grammatical_and_answerable": True}
    assert review.audit(tasks(), a, b, [decision])["status"] == "complete_recorded_human_review"
    with pytest.raises(ValueError): review.audit(tasks(), a, b, [{**decision, "rater_id": "Alice"}])
    with pytest.raises(ValueError): review.audit(tasks(), a, b, [{**decision, "reason": ""}])


def test_abstention_is_preserved_and_needs_adjudication():
    a, b = ratings("Alice"), ratings("Bob")
    a[0].update({"status": "abstained", "notes": "ambiguous", **{k: None for k in review.FIELDS}})
    result = review.audit(tasks(), a, b)
    assert result["abstentions"] == [1, 0] and result["status"] == "incomplete"
    assert result["agreement"]["all_criteria_valid"]["n"] == 2


def test_sampling_order_fingerprint_and_all_variant_exclusion():
    pool = [{"dataset": "nq", "cluster_id": f"nq-{i:05d}", "source_index": i, "question": q, "answer": ["a"]}
            for i, q in enumerate(("old question", "old paraphrase", "new first", "new second", "NEW FIRST?"))]
    exposure = {"originals": {"nq-00000": {"source_index": 0, "question": "old question", "source_id": None}},
                "normalized_texts": ["old question", "old paraphrase"]}
    rows, audit = sampling.sample_pool(pool, exposure, 2, 42)
    assert {r["source_index"] for r in rows} == {2, 3}
    assert audit["excluded"] == {"previous_original_id": 1, "previous_normalized_text": 1, "duplicate_in_pool": 1}
    assert sampling.sample_pool(pool, exposure, 2, 42)[0] == rows
    with pytest.raises(ValueError, match="order/content"):
        sampling.sample_pool(list(reversed(pool)), exposure, 2, 42)
    with pytest.raises(ValueError, match="only 2"):
        sampling.sample_pool(pool, exposure, 3, 42)


@pytest.mark.parametrize("dataset", ["nq", "hotpotqa"])
def test_real_exposure_inventory_covers_450_originals_without_outcome_values(dataset):
    root = Path(__file__).resolve().parents[2] / "runs"
    result = sampling.exposure_inventory(root, dataset)
    assert result["n_exposed_originals"] == 450
    assert all(r["source_index"] >= 0 for r in result["originals"].values())
    assert all("F1" not in r and "EM" not in r for r in result["originals"].values())


def test_passage_store_refuses_different_source_and_duplicate_ids(tmp_path):
    source = tmp_path / "corpus.tsv"
    source.write_text('id\ttext\ttitle\n1\ttext here\tTitle\n2\tsecond\tOther\n', encoding="utf-8")
    out = tmp_path / "passages.sqlite"
    assert passages.build(source, out, digest(source)) == 2
    with sqlite3.connect(out) as db:
        assert dict(db.execute("select key,value from metadata"))["status"] == "complete"
        assert db.execute("select text from passages where id=?", ("1",)).fetchone()[0] == "text here"
    with pytest.raises(ValueError): passages.build(source, out, digest(source))
    with pytest.raises(ValueError): passages.build(source, tmp_path / "bad.sqlite", "0" * 64)


def test_preflight_missing_assets_is_blocked_never_success():
    result = runtime.preflight({})
    assert result["status"] == "blocked_missing_runtime_assets"
    assert not all(c["passed"] for c in result["checks"])


def test_pilot_evaluation_requires_full_new_matrix_and_rescores(tmp_path):
    queries = [query(), {**query(), "cluster_id": "nq-00002", "question": "Other question?"}]
    rows = collect.collect(queries, ToyBackend(), tmp_path / "run", {"source": "toy"})
    result = evaluate.evaluate(queries, rows, n_boot=100)
    assert result["datasets"]["nq"]["n_queries"] == 2
    assert result["datasets"]["nq"]["A2_union_F1_comparisons"]["A2"]["difference"] == 0
    assert not result["confirmatory_complete"]
    with pytest.raises(ValueError, match="all three"):
        evaluate.evaluate(queries, rows[:-1], n_boot=100)
    changed = deepcopy(rows)
    changed[0]["F1"] = 0
    with pytest.raises(ValueError, match="scores disagree"):
        evaluate.evaluate(queries, changed, n_boot=100)
    changed = deepcopy(rows)
    changed[0].pop("trace")
    with pytest.raises(ValueError, match="not substitutes"):
        evaluate.evaluate(queries, changed, n_boot=100)


def test_server_rejects_nonlocal_port_encoding(tmp_path):
    with pytest.raises(ValueError, match="port"):
        runtime.LlamaServer({"port": "8089@external.example"}, tmp_path / "log")


def test_unfinished_passage_store_stays_marked_building(tmp_path):
    source = tmp_path / "corpus.tsv"
    source.write_text('id\ttext\ttitle\n1\tx\tA\n1\ty\tB\n', encoding="utf-8")
    with pytest.raises(sqlite3.IntegrityError):
        passages.build(source, tmp_path / "bad.sqlite", digest(source))
    with sqlite3.connect(tmp_path / "bad.sqlite") as db:
        status = dict(db.execute("select key,value from metadata")).get("status")
        assert status != "complete"
