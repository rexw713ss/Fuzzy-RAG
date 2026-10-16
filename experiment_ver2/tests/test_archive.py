"""Archive integrity, action identity, fresh-set isolation, and real-cache coverage."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiment_ver2.archive import load_archive, normalize_question
from experiment_ver2.data import DataValidationError, load_dataset
from main_logic.calibration import calibrate


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


@pytest.fixture
def archive(tmp_path):
    old = tmp_path / "v0.3"
    old.mkdir()
    splits = {"demo": {"train": ["train-a", "train-b"], "val": ["val"], "test": ["old-test"]}}
    (old / "splits.json").write_text(json.dumps(splits), encoding="utf8")
    queries, scaled, old_outcomes, new_outcomes = [], [], [], []
    training = []
    for c, cluster in enumerate(("train-a", "train-b", "val", "old-test")):
        block = []
        for variant in range(3):
            ids = {"cluster_id": cluster, "variant_id": variant}
            queries.append({**ids, "question": f"Original {cluster} phrasing {variant}?"})
            x = [.1 + c * .1 + variant * .01] * 4
            scaled.append({**ids, "dataset": "demo", **dict(zip(("S1", "S2", "S3", "S4"), x))})
            block.append(x)
            for i in range(3):
                row = {**ids, "action": f"A{i}", "F1": .2 * i, "EM": 0., "latency_total": 1. + i,
                       "llm_calls": 2 if i == 2 else 1}
                old_outcomes.append(row)
                if i:
                    new_outcomes.append(row)
        if c < 2:
            training.append(np.asarray(block))
    write_rows(old / "demo" / "queries.jsonl", queries)
    write_rows(old / "demo" / "outcomes.jsonl", old_outcomes)
    write_rows(old / "signals_scaled.jsonl", scaled)
    write_rows(tmp_path / "v0.4" / "demo" / "outcomes.jsonl", new_outcomes)
    scalers = {"demo": {signal: {"p5": 1., "p95": 3.} for signal in ("S1", "S2", "S3", "S4")}}
    (old / "scalers.json").write_text(json.dumps(scalers), encoding="utf8")
    (old / "jitter.json").write_text(json.dumps({"demo": calibrate(training)}), encoding="utf8")
    fresh_queries, raw, fresh_outcomes = [], [], []
    for variant in range(3):
        ids = {"cluster_id": "fresh", "variant_id": variant}
        fresh_queries.append({**ids, "question": f"Fresh question wording {variant}?"})
        raw.append({**ids, **{signal: 1.4 + variant * .1 for signal in ("S1", "S2", "S3", "S4")}})
        for i in (1, 2):
            fresh_outcomes.append({**ids, "action": f"A{i}", "F1": .2 * i, "EM": False,
                                   "latency_total": 1. + i, "llm_calls": 2 if i == 2 else 1})
    write_rows(tmp_path / "v0.4" / "fresh_src" / "demo" / "queries.jsonl", fresh_queries)
    write_rows(tmp_path / "v0.4" / "fresh_src" / "demo" / "signals.jsonl", raw)
    write_rows(tmp_path / "v0.4" / "fresh" / "demo" / "outcomes.jsonl", fresh_outcomes)
    return tmp_path


def test_explicit_archive_mode_preserves_two_action_identity_and_frozen_scaling(archive):
    dataset = load_archive(archive, "demo")
    assert dataset.actions == dataset.train.actions == ("A1", "A2")
    assert dataset.train.f1.shape == (6, 2)
    assert dataset.test.cost.shape == (3, 2)
    np.testing.assert_allclose(dataset.train.f1, [[.2, .4]] * 6)
    np.testing.assert_allclose(dataset.test.X[:, 0], [.2, .25, .3], atol=1e-9)
    assert dataset.metadata["preprocessing_mode"] == "archived_frozen_preprocessing"
    assert dataset.metadata["scalers_independently_refitted"] is False
    assert dataset.metadata["jitter"]["saved_metadata_verified"]
    assert dataset.metadata["fresh_disjointness"]["cluster_overlap_original_all_splits"] == 0
    assert len(dataset.metadata["input_sha256"]) == 9
    # Ordinary raw-data loading must not quietly gain this compatibility fallback.
    with pytest.raises(DataValidationError, match="raw signal cache"):
        load_dataset(archive / "v0.3", "demo")


def test_v03_preserves_three_actions_and_rejects_fresh(archive):
    dataset = load_archive(archive, "demo", "v0.3", "old_test")
    assert dataset.actions == ("A0", "A1", "A2")
    assert dataset.test.f1.shape == (3, 3)
    with pytest.raises(DataValidationError, match="v0.3 requires old_test"):
        load_archive(archive, "demo", "v0.3", "fresh")


def test_changing_fresh_features_or_labels_cannot_change_training_calibration(archive):
    original = load_archive(archive, "demo")
    raw_path = archive / "v0.4" / "fresh_src" / "demo" / "signals.jsonl"
    rows = read_rows(raw_path)
    for row in rows:
        for signal in ("S1", "S2", "S3", "S4"):
            row[signal] = 1000
    write_rows(raw_path, rows)
    path = archive / "v0.4" / "fresh" / "demo" / "outcomes.jsonl"
    rows = read_rows(path)
    for row in rows:
        row["F1"] = 1 - row["F1"]
    write_rows(path, rows)
    changed = load_archive(archive, "demo")
    np.testing.assert_array_equal(changed.train.X, original.train.X)
    np.testing.assert_array_equal(changed.widths, original.widths)
    assert changed.metadata["scalers"] == original.metadata["scalers"]
    assert changed.metadata["jitter"] == original.metadata["jitter"]
    assert changed.metadata["fresh_disjointness"]["clipped_fraction"]["S1"] == 1.


@pytest.mark.parametrize("field", ["widths", "n_clusters", "n_skipped"])
def test_saved_jitter_or_training_count_disagreement_fails(archive, field):
    path = archive / "v0.3" / "jitter.json"
    document = json.loads(path.read_text(encoding="utf8"))
    document["demo"][field] = [.25] * 4 if field == "widths" else 100
    path.write_text(json.dumps(document), encoding="utf8")
    with pytest.raises(DataValidationError, match="disagrees with TRAIN"):
        load_archive(archive, "demo")


def test_invalid_scaler_fails(archive):
    path = archive / "v0.3" / "scalers.json"
    document = json.loads(path.read_text(encoding="utf8"))
    document["demo"]["S1"]["p95"] = 0
    path.write_text(json.dumps(document), encoding="utf8")
    with pytest.raises(DataValidationError, match="p95 must"):
        load_archive(archive, "demo")


@pytest.mark.parametrize("part", ["queries", "signals"])
def test_fresh_coverage_mismatch_fails(archive, part):
    path = archive / "v0.4" / "fresh_src" / "demo" / f"{part}.jsonl"
    write_rows(path, read_rows(path)[1:])
    with pytest.raises(DataValidationError, match="coverage mismatch"):
        load_archive(archive, "demo")


@pytest.mark.parametrize("part", ["queries", "signals"])
def test_duplicate_fresh_instances_fail(archive, part):
    path = archive / "v0.4" / "fresh_src" / "demo" / f"{part}.jsonl"
    rows = read_rows(path)
    write_rows(path, rows + [rows[0]])
    with pytest.raises(DataValidationError, match="duplicate instance"):
        load_archive(archive, "demo")


def test_missing_a2_is_error_never_invented_a0(archive):
    path = archive / "v0.4" / "fresh" / "demo" / "outcomes.jsonl"
    write_rows(path, read_rows(path)[:-1])
    with pytest.raises(DataValidationError, match="missing outcome.*A2"):
        load_archive(archive, "demo")


def test_v04_rejects_unexpected_a0_row(archive):
    path = archive / "v0.4" / "demo" / "outcomes.jsonl"
    rows = read_rows(path)
    rows[0]["action"] = "A0"
    write_rows(path, rows)
    with pytest.raises(DataValidationError, match="unknown action 'A0'"):
        load_archive(archive, "demo")


def test_fresh_cluster_overlap_fails_without_filtering(archive):
    path = archive / "v0.4" / "fresh_src" / "demo" / "queries.jsonl"
    rows = read_rows(path)
    for row in rows:
        row["cluster_id"] = "old-test"
    write_rows(path, rows)
    with pytest.raises(DataValidationError, match="fresh/original cluster overlap"):
        load_archive(archive, "demo")


def test_fresh_normalized_question_overlap_covers_original_test_too(archive):
    path = archive / "v0.4" / "fresh_src" / "demo" / "queries.jsonl"
    rows = read_rows(path)
    rows[0]["question"] = "  ORIGINAL OLD-TEST phrasing 0!!!\u200b "
    write_rows(path, rows)
    with pytest.raises(DataValidationError, match="normalized-question overlap.*no rows were dropped"):
        load_archive(archive, "demo")


def test_question_normalization_unicode_case_punctuation_and_formatting():
    assert normalize_question(" Ｗｈｅｒｅ\u200b is IT?!  ") == "where is it"


def test_archive_loading_does_not_change_inputs(archive):
    before = {p: p.read_bytes() for p in archive.rglob("*") if p.is_file()}
    load_archive(archive, "demo")
    assert before == {p: p.read_bytes() for p in archive.rglob("*") if p.is_file()}


@pytest.mark.parametrize("name,expected", [("nq", (506, 165, 170, 412)),
                                           ("hotpotqa", (531, 179, 176, 443))])
def test_committed_real_archives_validate_without_external_data(name, expected):
    root = Path(__file__).resolve().parents[2] / "runs"
    if not root.is_dir():
        pytest.skip("committed run archives not available")
    old = load_archive(root, name, "v0.3", "old_test")
    old_v4 = load_archive(root, name, "v0.4", "old_test")
    fresh = load_archive(root, name, "v0.4", "fresh")
    assert (len(old.train), len(old.val), len(old.test), len(fresh.test)) == expected
    assert old.train.f1.shape[1] == 3
    assert old_v4.train.f1.shape[1] == fresh.train.f1.shape[1] == 2
    np.testing.assert_array_equal(old.train.X, fresh.train.X)
    np.testing.assert_array_equal(old_v4.train.f1, fresh.train.f1)
    assert not set(old.train.groups) & set(fresh.test.groups)
    assert fresh.metadata["fresh_disjointness"]["normalized_question_overlap_original_all_splits"] == 0
