"""Data-integrity and train/held-out isolation checks for cached reanalysis."""

import json

import numpy as np
import pytest

from experiment_ver2.data import DataValidationError, discover_datasets, load_dataset


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


@pytest.fixture
def cache(tmp_path):
    dataset = tmp_path / "demo"
    dataset.mkdir()
    splits = {"demo": {"train": ["train-a", "train-b"], "val": ["val-a"], "test": ["test-a"]}}
    (tmp_path / "splits.json").write_text(json.dumps(splits), encoding="utf8")
    queries, signals, outcomes = [], [], []
    for c, cluster in enumerate(["train-a", "train-b", "val-a", "test-a"]):
        for variant in range(3):
            ids = {"cluster_id": cluster, "variant_id": variant}
            queries.append({**ids, "question": f"Question {c}, phrasing {variant}?"})
            signals.append({**ids, **{f"S{i + 1}": (c + variant * 0.03) / (i + 4)
                                      for i in range(4)}})
            for action in range(3):
                outcomes.append({**ids, "action": f"A{action}", "F1": action * 0.3,
                                 "EM": 0, "latency_total": action + 1.0,
                                 "input_tokens": 50 + action})
    write_jsonl(dataset / "queries.jsonl", queries)
    write_jsonl(dataset / "signals.jsonl", signals)
    write_jsonl(dataset / "outcomes.jsonl", outcomes)
    return tmp_path


def test_valid_cache_alignment_namespace_and_provenance(cache):
    before = {str(p): p.read_bytes() for p in cache.rglob("*") if p.is_file()}
    dataset = load_dataset(cache, "demo")
    assert discover_datasets(cache) == ("demo",)
    assert dataset.train.X.shape == (6, 4)
    assert dataset.test.f1.shape == (3, 3)
    np.testing.assert_allclose(dataset.test.f1, [[0, 0.3, 0.6]] * 3)
    assert dataset.test.keys[0] == ("demo", "test-a", 0)
    assert dataset.test.groups.shape == (3,)
    assert len(set(dataset.test.groups)) == 1
    assert dataset.test.secondary["input_tokens"].shape == (3, 3)
    assert dataset.metadata["calibration_split"] == "train"
    assert len(dataset.metadata["input_sha256"]) == 4
    assert all(len(digest) == 64 for digest in dataset.metadata["input_sha256"].values())
    assert dataset.metadata["counts"]["train"]["complete_clusters"] == 2
    assert before == {str(p): p.read_bytes() for p in cache.rglob("*") if p.is_file()}


def test_held_out_values_cannot_change_scaler_widths_or_training_features(cache):
    first = load_dataset(cache, "demo")
    path = cache / "demo" / "signals.jsonl"
    rows = read_jsonl(path)
    for row in rows:
        if not row["cluster_id"].startswith("train"):
            for signal in ("S1", "S2", "S3", "S4"):
                row[signal] = 1e6 + row["variant_id"] * 500
    write_jsonl(path, rows)
    # Stale derived artifacts have no authority over calibration.
    (cache / "jitter.json").write_text("invalid stale cache", encoding="utf8")
    (cache / "signals_scaled.jsonl").write_text("invalid stale cache", encoding="utf8")
    second = load_dataset(cache, "demo")
    assert first.metadata["scalers"] == second.metadata["scalers"]
    np.testing.assert_array_equal(first.widths, second.widths)
    np.testing.assert_array_equal(first.train.X, second.train.X)


def test_raw_signal_cache_is_required(cache):
    (cache / "demo" / "signals.jsonl").unlink()
    with pytest.raises(DataValidationError, match="raw signal cache"):
        load_dataset(cache, "demo")


@pytest.mark.parametrize("stem", ["queries", "signals", "outcomes"])
def test_duplicate_rows_fail(cache, stem):
    path = cache / "demo" / f"{stem}.jsonl"
    rows = read_jsonl(path)
    write_jsonl(path, rows + [rows[0]])
    with pytest.raises(DataValidationError, match="duplicate"):
        load_dataset(cache, "demo")


@pytest.mark.parametrize("stem", ["queries", "signals", "outcomes"])
def test_missing_rows_fail(cache, stem):
    path = cache / "demo" / f"{stem}.jsonl"
    write_jsonl(path, read_jsonl(path)[1:])
    with pytest.raises(DataValidationError, match="missing|coverage"):
        load_dataset(cache, "demo")


def test_overlapping_splits_fail_before_analysis(cache):
    path = cache / "splits.json"
    splits = json.loads(path.read_text(encoding="utf8"))
    splits["demo"]["test"].append("train-a")
    path.write_text(json.dumps(splits), encoding="utf8")
    with pytest.raises(DataValidationError, match="overlapping cluster"):
        load_dataset(cache, "demo")


@pytest.mark.parametrize("field,value", [("F1", -0.1), ("F1", 1.1), ("F1", float("nan")),
                                         ("EM", float("inf")), ("EM", 2),
                                         ("latency_total", 0), ("latency_total", -1),
                                         ("input_tokens", -1)])
def test_invalid_outcomes_fail(cache, field, value):
    path = cache / "demo" / "outcomes.jsonl"
    rows = read_jsonl(path)
    rows[0][field] = value
    write_jsonl(path, rows)
    with pytest.raises(DataValidationError, match=field):
        load_dataset(cache, "demo")


def test_partial_secondary_counter_is_not_silently_dropped(cache):
    path = cache / "demo" / "outcomes.jsonl"
    rows = read_jsonl(path)
    del rows[0]["input_tokens"]
    write_jsonl(path, rows)
    with pytest.raises(DataValidationError, match="missing field input_tokens"):
        load_dataset(cache, "demo")


def test_nonfinite_raw_signals_fail(cache):
    path = cache / "demo" / "signals.jsonl"
    rows = read_jsonl(path)
    rows[0]["S3"] = float("nan")
    write_jsonl(path, rows)
    with pytest.raises(DataValidationError, match="S3 must be a finite"):
        load_dataset(cache, "demo")


def test_optional_retrieval_cache_must_align(cache):
    queries = read_jsonl(cache / "demo" / "queries.jsonl")
    path = cache / "demo" / "retrieval.jsonl"
    write_jsonl(path, queries)
    assert len(load_dataset(cache, "demo").metadata["input_sha256"]) == 5
    write_jsonl(path, queries[:-1])
    with pytest.raises(DataValidationError, match="retrieval.jsonl: instance coverage"):
        load_dataset(cache, "demo")


def test_incomplete_clusters_are_counted_and_optional_filter_is_explicit(cache):
    for stem in ("queries", "signals", "outcomes"):
        path = cache / "demo" / f"{stem}.jsonl"
        rows = [r for r in read_jsonl(path)
                if not (r["cluster_id"] == "train-b" and r["variant_id"] == 2)]
        write_jsonl(path, rows)
    retained = load_dataset(cache, "demo")
    filtered = load_dataset(cache, "demo", complete_only=True)
    assert len(retained.train) == 5
    assert len(filtered.train) == 3
    assert retained.metadata["counts"]["train"]["incomplete_clusters_in_source"] == 1
    assert retained.metadata["jitter"]["training_complete_clusters"] == 1


def test_variant_id_json_types_are_preserved(cache):
    for stem in ("queries", "signals", "outcomes"):
        path = cache / "demo" / f"{stem}.jsonl"
        rows = read_jsonl(path)
        for row in rows:
            row["variant_id"] = str(row["variant_id"])
        write_jsonl(path, rows)
    assert load_dataset(cache, "demo").test.keys[0][-1] == "0"


def test_wrong_dataset_namespace_fails(cache):
    path = cache / "demo" / "signals.jsonl"
    rows = read_jsonl(path)
    rows[0]["dataset"] = "other"
    write_jsonl(path, rows)
    with pytest.raises(DataValidationError, match="dataset namespace"):
        load_dataset(cache, "demo")


def test_duplicate_json_fields_are_rejected(cache):
    path = cache / "splits.json"
    path.write_text('{"demo": {}, "demo": {}}', encoding="utf8")
    with pytest.raises(DataValidationError, match="duplicate JSON field"):
        load_dataset(cache, "demo")
