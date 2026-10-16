"""Read-only, validated access to the original pipeline's cached experiments.

The canonical inputs are splits.json and each dataset's queries.jsonl,
signals.jsonl (RAW S1--S4), and outcomes.jsonl. Existing scaled signals, scalers,
and jitter files are deliberately ignored: calibration is fitted on TRAIN only.
IDs retain their JSON type and carry a dataset namespace in every returned key.
"""

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np


SIGNALS = ("S1", "S2", "S3", "S4")
ACTIONS = ("A0", "A1", "A2")
SPLITS = ("train", "val", "test")
SECONDARY = ("retrievals", "ce_passes", "llm_calls", "input_tokens", "output_tokens")


class DataValidationError(ValueError):
    """A cached experiment is incomplete, inconsistent, or invalid."""


@dataclass
class Split:
    X: np.ndarray
    f1: np.ndarray
    cost: np.ndarray
    em: np.ndarray
    keys: tuple
    questions: tuple
    secondary: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self):
        return len(self.keys)

    @property
    def groups(self):
        """One-dimensional, collision-free cluster identifiers for resampling."""
        return np.asarray([json.dumps(k[:2], ensure_ascii=False) for k in self.keys])


SplitData = Split


@dataclass
class Dataset:
    name: str
    train: Split
    val: Split
    test: Split
    widths: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def splits(self):
        return {s: getattr(self, s) for s in SPLITS}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DataValidationError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _parse(text, where):
    try:
        return json.loads(text, object_pairs_hook=_unique_object)
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"{where}: {exc}") from exc


def _read_json(path):
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise DataValidationError(f"Cannot read required cache {path}: {exc}") from exc
    return _parse(text, str(path))


def _read_jsonl(path):
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise DataValidationError(f"Cannot read required cache {path}: {exc}") from exc
    result = []
    for number, line in enumerate(lines, 1):
        if line.strip():
            row = _parse(line, f"{path}:{number}")
            if not isinstance(row, dict):
                raise DataValidationError(f"{path}:{number}: expected a JSON object")
            result.append(row)
    if not result:
        raise DataValidationError(f"{path}: cache contains no rows")
    return result


def _identifier(value, where):
    if type(value) not in (str, int) or (isinstance(value, str) and not value.strip()):
        raise DataValidationError(f"{where}: ID must be a non-empty string or integer")
    return value


def _key(row, dataset, where):
    if "dataset" in row and row["dataset"] != dataset:
        raise DataValidationError(f"{where}: dataset namespace {row['dataset']!r} != {dataset!r}")
    try:
        return (dataset, _identifier(row["cluster_id"], where),
                _identifier(row["variant_id"], where))
    except KeyError as exc:
        raise DataValidationError(f"{where}: missing identifier {exc.args[0]}") from exc


def _indexed(rows, dataset, where):
    result = {}
    for row in rows:
        key = _key(row, dataset, where)
        if key in result:
            raise DataValidationError(f"{where}: duplicate instance {key!r}")
        result[key] = row
    return result


def _same_keys(reference, other, where):
    missing, extra = set(reference) - set(other), set(other) - set(reference)
    if missing or extra:
        example = next(iter(missing or extra))
        raise DataValidationError(
            f"{where}: instance coverage mismatch: {len(missing)} missing, "
            f"{len(extra)} extra; example {example!r}")


def _number(row, field_name, where, *, low=None, high=None, positive=False):
    if field_name not in row:
        raise DataValidationError(f"{where}: missing field {field_name}")
    value = row[field_name]
    if type(value) not in (int, float) or not np.isfinite(value):
        raise DataValidationError(f"{where}: {field_name} must be a finite number")
    value = float(value)
    if positive and value <= 0:
        raise DataValidationError(f"{where}: {field_name} must be positive")
    if (low is not None and value < low) or (high is not None and value > high):
        raise DataValidationError(f"{where}: {field_name} outside [{low}, {high}]")
    return value


def _split_mapping(root, name):
    document = _read_json(root / "splits.json")
    if not isinstance(document, dict) or name not in document:
        raise DataValidationError(f"splits.json has no dataset {name!r}")
    splits = document[name]
    if not isinstance(splits, dict) or set(splits) != set(SPLITS):
        raise DataValidationError(f"{name}: splits must contain exactly train, val, test")
    seen = {}
    for split in SPLITS:
        clusters = splits[split]
        if not isinstance(clusters, list) or not clusters:
            raise DataValidationError(f"{name}/{split}: expected a non-empty cluster list")
        for cluster in clusters:
            _identifier(cluster, f"{name}/{split}")
            if cluster in seen:
                raise DataValidationError(
                    f"{name}: duplicate or overlapping cluster {cluster!r} "
                    f"in {seen[cluster]} and {split}")
            seen[cluster] = split
    return splits, seen


def discover_datasets(root):
    """Dataset names declared in splits.json, without assuming NQ/HotpotQA."""
    root = Path(root)
    document = _read_json(root / "splits.json")
    if not isinstance(document, dict) or not document:
        raise DataValidationError("splits.json must contain a non-empty dataset mapping")
    for name in document:
        _dataset_directory(root, name)
    return tuple(sorted(document))


def _dataset_directory(root, name):
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise DataValidationError(f"invalid dataset name {name!r}")
    directory = (root / name).resolve()
    if directory.parent != root.resolve():
        raise DataValidationError(f"dataset path escapes input root: {name!r}")
    return directory


def _sort_key(key):
    # The type tag distinguishes integer 1 from string "1" without coercing IDs.
    return tuple((type(value).__name__, str(value)) for value in key)


def load_dataset(root, name, cost_key="latency_total", complete_only=False):
    """Validate cached rows, then fit train-only p5/p95 scaling and jitter.

    Incomplete clusters (one or two variants) are retained by default for quality
    estimates and counted explicitly. Jitter uses only complete training clusters.
    ``complete_only=True`` excludes incomplete clusters from every split after
    validating the entire cache. Every retained query must have all three actions.
    No input file is created or modified. Secondary counters are optional, but a
    counter that appears anywhere must be present and valid in every outcome row.
    """
    root = Path(root).resolve()
    directory = _dataset_directory(root, name)
    _, split_of = _split_mapping(root, name)
    paths = [root / "splits.json"]
    indexed = {}
    for stem in ("queries", "signals"):
        path = directory / f"{stem}.jsonl"
        if stem == "signals" and not path.is_file():
            raise DataValidationError(
                f"Missing raw signal cache {path}. Restore the original signals.jsonl; "
                "signals_scaled.jsonl is not a valid substitute for train-only calibration.")
        indexed[stem] = _indexed(_read_jsonl(path), name, path.name)
        paths.append(path)
    queries, signals = indexed["queries"], indexed["signals"]
    _same_keys(queries, signals, "signals.jsonl")

    by_cluster = {}
    for key, query in queries.items():
        by_cluster.setdefault(key[1], []).append(key)
        if not isinstance(query.get("question"), str) or not query["question"].strip():
            raise DataValidationError(f"queries.jsonl: missing/empty question for {key!r}")
    if set(by_cluster) != set(split_of):
        missing = set(split_of) - set(by_cluster)
        extra = set(by_cluster) - set(split_of)
        raise DataValidationError(
            f"split/query cluster coverage mismatch: {len(missing)} missing queries, "
            f"{len(extra)} clusters not assigned to a split")
    for cluster, keys in by_cluster.items():
        if len(keys) > 3:
            raise DataValidationError(f"cluster {cluster!r} has {len(keys)} variants; at most 3 expected")

    # Retrieval itself is not required for offline routing, but if present it must
    # refer to exactly the same instances. This catches stale partial caches.
    retrieval_path = directory / "retrieval.jsonl"
    if retrieval_path.is_file():
        retrieval = _indexed(_read_jsonl(retrieval_path), name, retrieval_path.name)
        _same_keys(queries, retrieval, retrieval_path.name)
        for key, row in retrieval.items():
            if "question" in row and row["question"] != queries[key]["question"]:
                raise DataValidationError(f"retrieval.jsonl: question mismatch for {key!r}")
        paths.append(retrieval_path)

    outcome_path = directory / "outcomes.jsonl"
    outcome_rows = _read_jsonl(outcome_path)
    paths.append(outcome_path)
    secondary_names = [s for s in SECONDARY if any(s in row for row in outcome_rows)]
    outcomes = {}
    for row in outcome_rows:
        key = _key(row, name, outcome_path.name)
        action = row.get("action")
        if action not in ACTIONS:
            raise DataValidationError(f"outcomes.jsonl: unknown action {action!r} for {key!r}")
        pair = (key, action)
        if pair in outcomes:
            raise DataValidationError(f"outcomes.jsonl: duplicate outcome {pair!r}")
        where = f"outcomes.jsonl {pair!r}"
        _number(row, "F1", where, low=0, high=1)
        _number(row, "EM", where, low=0, high=1)
        _number(row, cost_key, where, positive=True)
        for counter in secondary_names:
            _number(row, counter, where, low=0)
        outcomes[pair] = row
    _same_keys(queries, {key for key, _ in outcomes}, "outcomes.jsonl")
    for key in queries:
        for action in ACTIONS:
            if (key, action) not in outcomes:
                raise DataValidationError(f"outcomes.jsonl: missing outcome {key!r}, action {action}")

    # Validate all raw rows before excluding incomplete clusters.
    raw_by_key = {key: [_number(row, signal, f"signals.jsonl {key!r}")
                        for signal in SIGNALS] for key, row in signals.items()}
    keys_by_split = {
        split: sorted((key for key in queries if split_of[key[1]] == split
                       and (not complete_only or len(by_cluster[key[1]]) == 3)), key=_sort_key)
        for split in SPLITS
    }
    for split, keys in keys_by_split.items():
        if not keys:
            raise DataValidationError(f"{name}/{split}: no queries remain after filtering")
    raw = {split: np.asarray([raw_by_key[key] for key in keys], dtype=float)
           for split, keys in keys_by_split.items()}
    p5, p95 = np.percentile(raw["train"], [5, 95], axis=0)
    scaled = {split: np.clip((values - p5) / (p95 - p5 + 1e-9), 0, 1)
              for split, values in raw.items()}
    train_index = {key: i for i, key in enumerate(keys_by_split["train"])}
    differences = []
    train_complete = 0
    for cluster, cluster_keys in by_cluster.items():
        if split_of[cluster] == "train" and len(cluster_keys) == 3:
            train_complete += 1
            block = scaled["train"][[train_index[key] for key in cluster_keys]]
            differences.extend([abs(block[0] - block[1]), abs(block[0] - block[2]),
                                abs(block[1] - block[2])])
    if not differences:
        raise DataValidationError(
            f"{name}: no complete training clusters; jitter calibration requires 3 variants")
    jitter = np.percentile(np.asarray(differences), 75, axis=0)
    widths = np.clip(jitter, 0.20, 0.25)
    splits = {}
    for split, keys in keys_by_split.items():
        def matrix(field_name):
            return np.asarray([[outcomes[(key, action)][field_name] for action in ACTIONS]
                               for key in keys], dtype=float)

        splits[split] = Split(
            X=scaled[split], f1=matrix("F1"), cost=matrix(cost_key), em=matrix("EM"),
            keys=tuple(keys), questions=tuple(queries[key]["question"] for key in keys),
            secondary={counter: matrix(counter) for counter in secondary_names})
    metadata = {
        "root": str(root), "dataset": name, "cost_key": cost_key,
        "raw_signal_source": str(directory / "signals.jsonl"),
        "calibration_split": "train", "complete_only": bool(complete_only),
        "scalers": {signal: {"p5": float(p5[i]), "p95": float(p95[i])}
                    for i, signal in enumerate(SIGNALS)},
        "constant_training_signals": [signal for i, signal in enumerate(SIGNALS)
                                      if p95[i] - p5[i] <= 1e-9],
        "jitter": {"widths": widths.tolist(), "p75": jitter.tolist(),
                   "all_at_floor": bool(np.all(jitter <= 0.20)),
                   "capped_signals": [s for i, s in enumerate(SIGNALS) if jitter[i] > 0.25],
                   "training_complete_clusters": train_complete},
        "counts": {split: {
            "instances": len(keys_by_split[split]),
            "clusters": len({key[1] for key in keys_by_split[split]}),
            "complete_clusters": sum(split_of[c] == split and len(ks) == 3
                                     for c, ks in by_cluster.items()),
            "incomplete_clusters_in_source": sum(split_of[c] == split and len(ks) < 3
                                                for c, ks in by_cluster.items()),
        } for split in SPLITS},
        "input_sha256": {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                         for path in paths},
    }
    return Dataset(name=name, widths=widths, metadata=metadata, **splits)
