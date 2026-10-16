"""Adversarial evidence tests; toy answers here are never benchmark results."""
from copy import deepcopy
import struct
import numpy as np
import pytest

from experiment_ver2.followup import confirmation as cf, review, paraphrases as pp, runtime, planning, freeze
from experiment_ver2.followup import actions, validation, collect
from experiment_ver2.followup.io import digest, read_json, read_jsonl, write_json, write_jsonl, seal
from experiment_ver2.tests.test_followup import ToyBackend, query
from experiment_ver2.data import Split


def outcome(action="A2", q=None):
    q = q or query()
    return {**actions.score_outcome(actions.run_action(q["question"], action, ToyBackend()), q["answer"]),
            **{k: q[k] for k in ("dataset", "cluster_id", "variant_id")}}


@pytest.mark.parametrize("action", actions.ACTIONS)
def test_full_outcome_trace_is_verifiable(action):
    validation.validate_outcome(outcome(action), query()["question"], query()["answer"])


@pytest.mark.parametrize("mutation", ["empty_trace", "wrong_prompt", "changed_context", "timing", "counter", "tokens", "decoding", "raw_answer", "signals"])
def test_outcome_evidence_tampering_rejected(mutation):
    row = outcome()
    if mutation == "empty_trace": row["trace"] = []
    elif mutation == "wrong_prompt": row["trace"][-1]["messages"][0]["content"] = "another question"
    elif mutation == "changed_context": row["doc_ids"][0] = "999"
    elif mutation == "timing": row["stage_seconds"]["generation"] += 1
    elif mutation == "counter": row["ce_passes"] -= 1
    elif mutation == "tokens": row["output_tokens"] = -1
    elif mutation == "decoding": row["trace"][-1]["decoding"]["seed"] = 13
    elif mutation == "raw_answer": row["raw_answer"] = "different"
    elif mutation == "signals": row["signals"]["S1"] = float("nan")
    with pytest.raises(ValueError): validation.validate_outcome(row, query()["question"], query()["answer"])


@pytest.mark.parametrize("followup", ["", query()["question"]])
def test_valid_fallback_evidence(followup):
    q = query()
    row = actions.score_outcome(actions.run_action(q["question"], "A2", ToyBackend(followup)), q["answer"])
    validation.validate_outcome(row, q["question"], q["answer"])


def review_fixture():
    queries = [query(), {**query(), "variant_id": 1, "question": "Who created this?"},
               {**query(), "variant_id": 2, "question": "Name the inventor of this?"}]
    tasks, key = pp.blind_tasks(queries)
    forms = []
    for person in ("human-a", "human-b"):
        form = review.blank_forms(tasks)
        for r in form:
            r.update(rater_id=person, status="reviewed", same_information_need=True,
                     added_or_removed_constraint=False, grammatical_and_answerable=True,
                     attestations={k: True for k in review.ATTEST})
        forms.append(form)
    return queries, tasks, key, forms, review.audit(tasks, *forms)


def test_new_review_binding_and_recomputation(tmp_path):
    queries, tasks, key, forms, report = review_fixture()
    assert cf.review_subset(queries, report, key, tasks).tolist() == [True] * 3
    write_json(tmp_path / "report.json", report)
    for label, rows in zip(("A", "B"), forms): write_jsonl(tmp_path / f"rater_{label}_raw.jsonl", rows)
    assert review.verified_audit(tmp_path / "report.json", tasks) == report
    report["decisions"][key["items"][0]["review_id"]]["valid"] = False
    write_json(tmp_path / "report.json", report)
    with pytest.raises(ValueError, match="differs"): review.verified_audit(tmp_path / "report.json", tasks)


@pytest.mark.parametrize("change", ["duplicate", "wrong_pair", "changed_text", "wrong_hash", "nonboolean", "missing_pair"])
def test_review_cannot_be_reused_or_swapped(change):
    queries, tasks, key, forms, report = review_fixture()
    if change == "duplicate": key["items"][1]["review_id"] = key["items"][0]["review_id"]
    elif change == "wrong_pair": key["items"][0]["paraphrase_variant_id"], key["items"][1]["paraphrase_variant_id"] = key["items"][1]["paraphrase_variant_id"], key["items"][0]["paraphrase_variant_id"]
    elif change == "changed_text": tasks[0]["left_question"] = "Unrelated question?"
    elif change == "wrong_hash": key["items"][0]["left_question_sha256"] = "0" * 64
    elif change == "nonboolean": report["decisions"][key["items"][0]["review_id"]]["valid"] = 1
    elif change == "missing_pair": key["items"].pop()
    with pytest.raises(ValueError): cf.review_subset(queries, report, key, tasks)


def test_confirmation_excludes_invalid_paraphrase_before_primary_allocation():
    queries, _, _, _, _ = review_fixture()
    queries.append({**query(), "cluster_id": "nq-00002", "question": "Another original?"})
    lock = read_json("experiment_ver2/followup/prepared/nq/policy_lock.json")
    rows = [outcome(a, q) for q in queries for a in ("A1", "A2")]
    report, decisions = cf.evaluate_locked(queries, rows, lock, np.array([True, True, False, True]), n_boot=100)
    assert report["primary_queries"] == 3
    assert report["invalid_paraphrases_retained_in_sensitivity"] == 1
    assert not any(d["variant_id"] == 2 for d in decisions)
    assert report["descriptive_subsets"]["full_pipeline_including_invalid_paraphrases"]["n_queries"] == 4
    assert "reused test" not in report["primary_inference"]["scope"]
    with pytest.raises(ValueError, match="every original"):
        cf.evaluate_locked(queries, rows, lock, np.array([False, True, False, True]), n_boot=100)


class ToyParaphrases:
    def __init__(self, fail_at=None): self.n = 0; self.fail_at = fail_at; self.prompts = []
    def chat(self, messages, **kwargs):
        self.n += 1
        self.prompts.append(messages)
        if self.n == self.fail_at: raise RuntimeError("deliberate test interruption")
        # Deterministic and distinct normalized variants, without gold.
        return "Who created object " + str(kwargs["seed"]), {"input_tokens": 10, "output_tokens": 5}
    def measure(self, original, candidates):
        return {"nli_fwd": [[.9, .05, .05]] * 5, "nli_bwd": [[.9, .05, .05]] * 5,
                "cosines": [.9] * 5, "entailment_index": 0}


def original_fixture():
    return [{"dataset": "nq", "cluster_id": "nq-00005", "source_index": 5, "question": "Who invented this device?", "answer": ["secretgold"]}], {
        "dataset": "nq", "originals": {}, "normalized_texts": []}


def test_paraphrase_generation_resume_and_no_gold_in_prompts(tmp_path):
    originals, exposure = original_fixture()
    backend = ToyParaphrases(fail_at=3)
    with pytest.raises(RuntimeError): pp.run(originals, exposure, backend, tmp_path / "run", {"test": True})
    after = ToyParaphrases()
    queries = pp.run(originals, exposure, after, tmp_path / "run", {"test": True}, resume=True)
    assert after.n == 3 and len(queries) == 3
    assert "secretgold" not in str(backend.prompts + after.prompts)
    assert any(r["event"] == "failed" for r in read_jsonl(tmp_path / "run/journal.jsonl"))
    assert pp.run(originals, exposure, ToyParaphrases(), tmp_path / "run", {"test": True}, resume=True) == queries
    with pytest.raises(ValueError, match="identity"): pp.run(originals, exposure, ToyParaphrases(), tmp_path / "run", {"test": False}, resume=True)


def test_screening_preserves_original_when_no_candidate_survives(tmp_path):
    originals, exposure = original_fixture()
    exposure["normalized_texts"] = ["who created object " + str(i) for i in range(42, 47)]
    queries = pp.run(originals, exposure, ToyParaphrases(), tmp_path / "run", {"test": True})
    assert len(queries) == 1 and queries[0]["variant_id"] == 0
    assert read_json(tmp_path / "run/status.json")["n_incomplete_clusters"] == 1


def test_sample_seal_blocks_changed_exposure(tmp_path):
    originals, exposure = original_fixture()
    protocol = tmp_path / "protocol.md"; protocol.write_text("locked")
    folder = tmp_path / "sample"; folder.mkdir()
    write_jsonl(folder / "originals.jsonl", originals)
    write_json(folder / "exposure_inventory.json", exposure)
    write_json(folder / "manifest.json", {"dataset": "nq", "n": 1, "sample_sha256": digest(folder / "originals.jsonl"),
        "protocol_sha256": digest(protocol), "outcomes_generated": False, "confirmatory_complete": False})
    seal(folder)
    assert len(pp.load_sample(folder, protocol)[0]) == 1
    write_json(folder / "exposure_inventory.json", {**exposure, "normalized_texts": ["changed"]})
    with pytest.raises(ValueError, match="seal"): pp.load_sample(folder, protocol)


def test_manifest_bytes_and_path_validation(tmp_path):
    root = tmp_path / "index"; root.mkdir(); (root / "segment").write_bytes(b"example")
    manifest = tmp_path / "manifest.json"
    data = {"schema": 1, "dataset": "nq", "kind": "bm25", "format": "lucene-bm25", "corpus_sha256": "1" * 64,
            "source": {"url": "https://example.org/index", "revision": "fixed-version-1"}, "files": {"segment": digest(root / "segment")}}
    write_json(manifest, data)
    config = {"dataset": "nq", "bm25_index": str(root), "corpus_sha256": "1" * 64,
              "index_manifests": {"bm25": {"path": str(manifest), "sha256": digest(manifest)}}}
    assert runtime.verify_index_manifest(config, "bm25")["files"] == 1
    (root / "segment").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"): runtime.verify_index_manifest(config, "bm25")
    data["files"] = {"../manifest.json": "0" * 64}; write_json(manifest, data)
    config["index_manifests"]["bm25"]["sha256"] = digest(manifest)
    with pytest.raises(ValueError, match="relative"): runtime.verify_index_manifest(config, "bm25")


def test_gguf_identity_rejects_wrong_architecture(tmp_path):
    def string(s): return struct.pack("<Q", len(s)) + s.encode()
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + string("general.architecture") + struct.pack("<I", 8) + string("wrong"))
    with pytest.raises(ValueError, match="architecture"): runtime.gguf_identity(path)


def test_offline_preflight_never_reports_ready_for_missing_models():
    assert pp.preflight({})["status"] == "blocked_missing_paraphrase_assets"
    assert runtime.preflight({})["status"] == "blocked_missing_runtime_assets"


def test_planning_identical_policies_cannot_create_power():
    split = Split(np.zeros((4, 4)), np.array([[0, 1]] * 4), np.ones((4, 2)), np.zeros((4, 2)),
                  tuple(("nq", str(i), 0) for i in range(4)), tuple("abcd"), actions=("A1", "A2"))
    scores = {m: np.arange(4) for m in cf.PRIMARY_METHODS}
    result = planning.simulate(split, scores, {m: False for m in scores}, [20], repetitions=100)
    assert result["20"]["simulated_statistic_sd"] == 0
    assert result["20"]["variance_scenarios"]["1.0"]["shift_distribution_detection_probability"]["0.01"] is None


def test_freeze_rejects_wrong_seed_before_collection(tmp_path):
    write_jsonl(tmp_path / "queries.jsonl", [query()])
    write_jsonl(tmp_path / "originals.jsonl", original_fixture()[0])
    write_json(tmp_path / "manifest.json", {"dataset": "nq", "n": cf.SAMPLE_SIZES["nq"], "seed": 42})
    with pytest.raises(ValueError, match="seed"):
        freeze.validate_inputs({"queries": tmp_path / "queries.jsonl", "originals": tmp_path / "originals.jsonl", "sample_manifest": tmp_path / "manifest.json"})


def test_collection_requires_original_journal_and_full_matrix(tmp_path):
    folder = tmp_path / "run"
    queries = [query()]
    rows = collect.collect(queries, ToyBackend(), folder, {"source": "toy fixture"})
    assert validation.validate_collection(folder, queries) == rows
    rows[0]["latency_total"] += 1
    rows[0]["unattributed_seconds"] += 1
    write_jsonl(folder / "outcomes.jsonl", rows)
    status = read_json(folder / "status.json")
    status["outcomes_sha256"] = digest(folder / "outcomes.jsonl")
    write_json(folder / "status.json", status)
    with pytest.raises(ValueError, match="journal"): validation.validate_collection(folder, queries)


def test_complete_toy_confirmation_chain_and_precollection_binding(tmp_path, monkeypatch):
    """Tiny count override is test-only; production CLI still requires 1800/3500."""
    originals, exposure = original_fixture()
    sample = tmp_path / "sample"; sample.mkdir()
    protocol = tmp_path / "protocol.md"; protocol.write_text("toy protocol")
    write_jsonl(sample / "originals.jsonl", originals)
    write_json(sample / "exposure_inventory.json", exposure)
    write_json(sample / "manifest.json", {"dataset": "nq", "n": 1, "seed": 20261018,
        "sample_sha256": digest(sample / "originals.jsonl"), "exposure_sha256": digest(sample / "exposure_inventory.json"),
        "protocol_sha256": digest(protocol), "outcomes_generated": False, "confirmatory_complete": False})
    seal(sample)
    generated = tmp_path / "paraphrases"
    queries = pp.run(originals, exposure, ToyParaphrases(), generated,
                     {"sample_manifest_sha256": digest(sample / "manifest.json"), "protocol_sha256": digest(protocol)})
    tasks = read_jsonl(generated / "blind_review_tasks.jsonl")
    audited = tmp_path / "human"; audited.mkdir()
    forms = []
    for letter in ("A", "B"):
        form = review.blank_forms(tasks)
        for row in form:
            row.update(rater_id="toy-fixture-" + letter, status="reviewed", same_information_need=True,
                       added_or_removed_constraint=False, grammatical_and_answerable=True,
                       attestations={k: True for k in review.ATTEST})
        write_jsonl(audited / f"rater_{letter}_raw.jsonl", form)
        forms.append(form)
    write_json(audited / "report.json", review.audit(tasks, *forms))
    policy_dir = tmp_path / "policy"; policy_dir.mkdir()
    lock = policy_dir / "policy_lock.json"
    write_json(lock, read_json("experiment_ver2/followup/prepared/nq/policy_lock.json"))
    write_json(policy_dir / "pilot_manifest.json", {"policy_lock_sha256": digest(lock), "protocol_sha256": digest(protocol)})
    monkeypatch.setitem(cf.SAMPLE_SIZES, "nq", 1)
    frozen = tmp_path / "frozen"
    monkeypatch.setattr("sys.argv", ["freeze", "--sample", str(sample), "--paraphrases", str(generated),
        "--lock", str(lock), "--protocol", str(protocol), "--human-report", str(audited / "report.json"), "--out", str(frozen)])
    freeze.main()
    run = tmp_path / "run"
    identity = {"queries_file_sha256": digest(generated / "queries.jsonl"), "protocol_sha256": digest(protocol),
                "confirmation_binding_sha256": digest(frozen / "binding.json")}
    collect.collect(queries, ToyBackend(), run, identity)
    args = ["--queries", str(generated / "queries.jsonl"), "--run", str(run), "--protocol", str(protocol),
            "--confirmation-bundle", str(frozen), "--out", str(tmp_path / "report")]
    cf.main(args)
    assert read_json(tmp_path / "report/report.json")["primary_queries"] == 3
    # Outcome generation without the review binding cannot acquire it later.
    original_identity = read_json(run / "run_identity.json")
    original_identity["inputs"].pop("confirmation_binding_sha256")
    write_json(run / "run_identity.json", original_identity)
    with pytest.raises(ValueError, match="before this action collection"): cf.main(args)
