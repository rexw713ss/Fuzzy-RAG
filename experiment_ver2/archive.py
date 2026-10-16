"""Explicit compatibility mode for the committed v0.3/v0.4 run archives.

v0.3 raw signals were not archived. Their scaled values and documented TRAIN
scalers are frozen inputs, not independently re-fitted preprocessing. TRAIN
jitter is recomputed and checked against the saved metadata. Fresh v0.4 raw
signals use those same scalers; fresh outcomes never enter preprocessing.
"""

from hashlib import sha256
import unicodedata
from pathlib import Path

import numpy as np

from experiment_ver2.data import (
    ACTIONS, SECONDARY, SIGNALS, SPLITS, Dataset, Split, DataValidationError,
    _dataset_directory, _indexed, _key, _number, _read_json, _read_jsonl,
    _same_keys, _sort_key, _split_mapping,
)


def normalize_question(text):
    """NFKC, casefold, drop invisible format characters, collapse punctuation/space."""
    chars = []
    for char in unicodedata.normalize("NFKC", text).casefold():
        category = unicodedata.category(char)
        if category == "Cf":
            continue
        chars.append(" " if category.startswith("P") else char)
    return " ".join("".join(chars).split())


def _queries(path, name):
    queries = _indexed(_read_jsonl(path), name, str(path))
    clusters = {}
    for key, row in queries.items():
        question = row.get("question")
        if not isinstance(question, str) or not normalize_question(question):
            raise DataValidationError(f"{path}: missing/empty question for {key!r}")
        clusters.setdefault(key[1], []).append(key)
    for cluster, keys in clusters.items():
        if len(keys) > 3:
            raise DataValidationError(f"{path}: cluster {cluster!r} has more than 3 variants")
    return queries, clusters


def _outcomes(path, name, queries, actions):
    rows = _read_jsonl(path)
    secondary = tuple(counter for counter in SECONDARY if any(counter in row for row in rows))
    outcomes = {}
    for row in rows:
        key = _key(row, name, str(path))
        action = row.get("action")
        if action not in actions:
            raise DataValidationError(f"{path}: unknown action {action!r}; expected {actions!r}")
        pair = (key, action)
        if pair in outcomes:
            raise DataValidationError(f"{path}: duplicate outcome {pair!r}")
        where = f"{path}: {pair!r}"
        _number(row, "F1", where, low=0, high=1)
        # Boolean EM is semantically valid, although the committed caches use floats.
        if type(row.get("EM")) is bool:
            row = {**row, "EM": float(row["EM"])}
        _number(row, "EM", where, low=0, high=1)
        _number(row, "latency_total", where, positive=True)
        for counter in secondary:
            _number(row, counter, where, low=0)
        outcomes[pair] = row
    _same_keys(queries, {key for key, _ in outcomes}, str(path))
    for key in queries:
        for action in actions:
            if (key, action) not in outcomes:
                raise DataValidationError(f"{path}: missing outcome {key!r}, action {action}")
    return outcomes, secondary


def _scalers(path, name):
    document = _read_json(path)
    if not isinstance(document, dict) or not isinstance(document.get(name), dict):
        raise DataValidationError(f"{path}: missing scalers for {name}")
    scalers = document[name]
    if set(scalers) != set(SIGNALS):
        raise DataValidationError(f"{path}: scalers must contain exactly {SIGNALS!r}")
    for signal, record in scalers.items():
        if not isinstance(record, dict):
            raise DataValidationError(f"{path}: malformed scaler for {signal}")
        low = _number(record, "p5", f"{path}: {signal}")
        high = _number(record, "p95", f"{path}: {signal}")
        if high < low:
            raise DataValidationError(f"{path}: {signal} p95 must be >= p5")
    return scalers


def _check_jitter(path, name, scaled, clusters, split_of):
    document = _read_json(path)
    if not isinstance(document, dict) or not isinstance(document.get(name), dict):
        raise DataValidationError(f"{path}: missing jitter metadata for {name}")
    saved = document[name]
    differences, used, skipped = [], 0, 0
    for cluster, keys in clusters.items():
        if split_of[cluster] != "train":
            continue
        if len(keys) != 3:
            skipped += 1
            continue
        used += 1
        block = np.asarray([scaled[key] for key in keys])
        differences.extend([abs(block[0] - block[1]), abs(block[0] - block[2]),
                            abs(block[1] - block[2])])
    if not used:
        raise DataValidationError(f"{name}: no complete TRAIN cluster for jitter verification")
    p75 = np.percentile(differences, 75, axis=0)
    median = np.median(differences, axis=0)
    widths, widths_median = np.clip(p75, .20, .25), np.clip(median, .20, .25)

    def close(field, expected):
        try:
            value = np.asarray(saved[field], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise DataValidationError(f"{path}: missing/invalid {field} for {name}") from exc
        if value.shape != np.shape(expected) or not np.allclose(value, expected, rtol=0, atol=1e-12):
            raise DataValidationError(f"{path}: {name} saved {field} disagrees with TRAIN scaled signals")

    close("widths", widths)
    close("widths_median", widths_median)
    for field, expected in (("n_clusters", used), ("n_skipped", skipped)):
        if type(saved.get(field)) is not int or saved[field] != expected:
            raise DataValidationError(f"{path}: {name} {field} disagrees with TRAIN cluster counts")
    if type(saved.get("all_at_floor")) is not bool or saved["all_at_floor"] != bool(np.all(p75 <= .20)):
        raise DataValidationError(f"{path}: {name} all_at_floor disagrees with TRAIN jitter")
    signal_metadata = saved.get("signals")
    if not isinstance(signal_metadata, dict) or set(signal_metadata) != set(SIGNALS):
        raise DataValidationError(f"{path}: missing per-signal jitter metadata for {name}")
    for j, signal in enumerate(SIGNALS):
        record = signal_metadata[signal]
        if not isinstance(record, dict):
            raise DataValidationError(f"{path}: invalid jitter record for {name}/{signal}")
        for field, expected in (("jitter", p75[j]), ("jitter_median", median[j]), ("width", widths[j])):
            value = _number(record, field, f"{path}: {name}/{signal}")
            if abs(value - expected) > 1e-12:
                raise DataValidationError(f"{path}: {name}/{signal} {field} disagrees with TRAIN signals")
    return widths, {"widths": widths.tolist(), "p75": p75.tolist(),
                    "widths_median": widths_median.tolist(), "training_complete_clusters": used,
                    "training_incomplete_clusters_skipped": skipped, "saved_metadata_verified": True}


def _make_split(keys, queries, features, outcomes, actions, secondary):
    keys = tuple(sorted(keys, key=_sort_key))
    if not keys:
        raise DataValidationError("archive split contains no queries")

    def matrix(field):
        return np.asarray([[outcomes[(key, action)][field] for action in actions] for key in keys], dtype=float)

    return Split(X=np.asarray([features[key] for key in keys]), f1=matrix("F1"),
                 cost=matrix("latency_total"), em=matrix("EM"), keys=keys,
                 questions=tuple(queries[key]["question"] for key in keys),
                 secondary={counter: matrix(counter) for counter in secondary},
                 actions=actions, signals=SIGNALS)


def _counts(split):
    _, sizes = np.unique(split.groups, return_counts=True)
    return {"instances": len(split), "clusters": len(sizes),
            "complete_clusters": int(np.sum(sizes == 3)),
            "incomplete_clusters_in_source": int(np.sum(sizes < 3))}


def load_archive(root=Path("runs"), name=None, version="v0.4", test_set="fresh"):
    """Load a published archive with explicit, auditable frozen preprocessing.

    v0.3 accepts ``old_test`` and preserves A0/A1/A2. v0.4 accepts ``fresh``
    or ``old_test`` and preserves ONLY A1/A2. All four signals are retained.
    Fresh cluster IDs and normalized questions must be disjoint from all three
    original splits. Any overlap raises a descriptive error; nothing is dropped.
    This loader never reads reported performance files and never writes inputs.
    """
    root = Path(root).resolve()
    if version not in ("v0.3", "v0.4"):
        raise DataValidationError("archive version must be v0.3 or v0.4")
    if test_set not in ("old_test", "fresh") or (version == "v0.3" and test_set != "old_test"):
        raise DataValidationError("v0.3 requires old_test; v0.4 accepts fresh or old_test")
    old = root / "v0.3"
    directory = _dataset_directory(old, name)
    _, split_of = _split_mapping(old, name)
    queries_path = directory / "queries.jsonl"
    queries, clusters = _queries(queries_path, name)
    if set(clusters) != set(split_of):
        raise DataValidationError("archive split/query cluster coverage mismatch")
    scaled_path = old / "signals_scaled.jsonl"
    all_scaled = _read_jsonl(scaled_path)
    if any(not isinstance(row.get("dataset"), str) for row in all_scaled):
        raise DataValidationError(f"{scaled_path}: every row requires a dataset namespace")
    scaled_rows = _indexed([row for row in all_scaled if row["dataset"] == name], name, str(scaled_path))
    _same_keys(queries, scaled_rows, str(scaled_path))
    scaled = {key: [_number(row, signal, f"{scaled_path}: {key!r}", low=0, high=1)
                    for signal in SIGNALS] for key, row in scaled_rows.items()}
    scaler_path, jitter_path = old / "scalers.json", old / "jitter.json"
    scalers = _scalers(scaler_path, name)
    widths, jitter = _check_jitter(jitter_path, name, scaled, clusters, split_of)
    actions = ACTIONS if version == "v0.3" else ("A1", "A2")
    outcome_path = root / version / name / "outcomes.jsonl"
    outcomes, secondary = _outcomes(outcome_path, name, queries, actions)
    paths = [old / "splits.json", queries_path, scaled_path, scaler_path, jitter_path, outcome_path]
    train = _make_split([key for key in queries if split_of[key[1]] == "train"],
                        queries, scaled, outcomes, actions, secondary)
    val = _make_split([key for key in queries if split_of[key[1]] == "val"],
                      queries, scaled, outcomes, actions, secondary)
    fresh_audit = None
    if test_set == "old_test":
        test = _make_split([key for key in queries if split_of[key[1]] == "test"],
                           queries, scaled, outcomes, actions, secondary)
    else:
        fresh_dir = root / "v0.4" / "fresh_src" / name
        fresh_query_path = fresh_dir / "queries.jsonl"
        fresh_queries, fresh_clusters = _queries(fresh_query_path, name)
        overlap = set(clusters) & set(fresh_clusters)
        if overlap:
            raise DataValidationError(f"fresh/original cluster overlap: {len(overlap)}; example {next(iter(overlap))!r}")
        old_normalized = {normalize_question(row["question"]): key for key, row in queries.items()}
        overlaps = [(old_normalized[normalize_question(row["question"])], key)
                    for key, row in fresh_queries.items() if normalize_question(row["question"]) in old_normalized]
        if overlaps:
            raise DataValidationError(f"fresh/original normalized-question overlap: {len(overlaps)} fresh rows; "
                                      f"example original/fresh keys {overlaps[0]!r}; no rows were dropped")
        raw_path = fresh_dir / "signals.jsonl"
        raw_rows = _indexed(_read_jsonl(raw_path), name, str(raw_path))
        _same_keys(fresh_queries, raw_rows, str(raw_path))
        raw = {key: np.asarray([_number(row, signal, f"{raw_path}: {key!r}") for signal in SIGNALS])
               for key, row in raw_rows.items()}
        p5 = np.asarray([scalers[signal]["p5"] for signal in SIGNALS])
        p95 = np.asarray([scalers[signal]["p95"] for signal in SIGNALS])
        unbounded = {key: (values - p5) / (p95 - p5 + 1e-9) for key, values in raw.items()}
        fresh_scaled = {key: np.clip(values, 0, 1) for key, values in unbounded.items()}
        fresh_outcome_path = root / "v0.4" / "fresh" / name / "outcomes.jsonl"
        fresh_outcomes, fresh_secondary = _outcomes(fresh_outcome_path, name, fresh_queries, actions)
        if fresh_secondary != secondary:
            raise DataValidationError("original/fresh outcome secondary-counter schemas differ")
        test = _make_split(fresh_queries, fresh_queries, fresh_scaled, fresh_outcomes, actions, secondary)
        raw_scaled_values = np.asarray(list(unbounded.values()))
        fresh_audit = {
            "cluster_overlap_original_all_splits": 0, "normalized_question_overlap_original_all_splits": 0,
            "question_normalization": "NFKC + casefold + remove Unicode Cf + punctuation/whitespace collapse",
            "clipped_fraction": {signal: float(np.mean((raw_scaled_values[:, j] < 0) |
                                                       (raw_scaled_values[:, j] > 1)))
                                 for j, signal in enumerate(SIGNALS)},
            "raw_signal_source": str(raw_path),
        }
        paths += [fresh_query_path, raw_path, fresh_outcome_path]
    splits = {"train": train, "val": val, "test": test}
    metadata = {
        "root": str(root), "dataset": name, "archive_version": version, "test_set": test_set,
        "actions": list(actions), "signals": list(SIGNALS), "cost_key": "latency_total",
        "preprocessing_mode": "archived_frozen_preprocessing",
        "calibration_split": "train", "scaler_origin": "v0.3 archived scalers, documented as TRAIN-fitted",
        "raw_training_signals_available": False, "scalers_independently_refitted": False,
        "preprocessing_limitation": "Original raw signals were not archived. The original scaler fit "
                                    "cannot be independently verified or repeated; archived scaled values "
                                    "and documented TRAIN scaler parameters are frozen inputs.",
        "jitter_recomputed_from_training_scaled_signals": True,
        "scalers": scalers, "jitter": jitter, "counts": {s: _counts(v) for s, v in splits.items()},
        "fresh_disjointness": fresh_audit,
        "test_status": "previously reported archived test; exploratory reanalysis",
        "input_sha256": {str(path.relative_to(root)).replace("\\", "/"): sha256(path.read_bytes()).hexdigest()
                         for path in paths},
    }
    return Dataset(name=name, widths=widths, metadata=metadata, actions=actions, signals=SIGNALS, **splits)
