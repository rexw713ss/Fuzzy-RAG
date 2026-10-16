"""Interleaved prospective action collection with a durable, auditable journal.

An interrupted action may be retried; all prior attempts remain in the journal.
Completed rows are reused ONLY to resume this exact locked run, not as freshly
measured timings. A different input, config, protocol or source needs a new run.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback
import time
import platform

import numpy as np

from experiment_ver2.followup.actions import ACTIONS, run_action, score_outcome
from experiment_ver2.followup.io import read_json, read_jsonl, write_json, write_jsonl, digest, json_digest, new_directory
from experiment_ver2.followup.runtime import preflight, LlamaServer, LocalBackend


def validate_queries(rows):
    if not rows:
        raise ValueError("queries must not be empty")
    seen = set()
    for q in rows:
        key = (q.get("dataset"), q.get("cluster_id"), q.get("variant_id"))
        if q.get("dataset") not in ("nq", "hotpotqa") or not isinstance(key[1], str) or type(key[2]) is not int or key[2] not in (0, 1, 2) or key in seen:
            raise ValueError("each query needs a unique namespaced cluster/variant ID")
        if not isinstance(q.get("question"), str) or not q["question"].strip():
            raise ValueError("nonempty query text required")
        if not isinstance(q.get("answer"), list) or not q["answer"] or any(not isinstance(a, str) or not a.strip() for a in q["answer"]):
            raise ValueError("nonempty gold answer list required for post-generation scoring")
        if any(k in q for k in ("F1", "EM", "action", "latency_total", "outcomes")):
            raise ValueError("input queries cannot contain previous action outcomes")
        seen.add(key)
    return rows


def schedule(queries, seed):
    validate_queries(queries)
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer schedule seed required")
    rng = np.random.default_rng(seed)
    result = []
    for i in rng.permutation(len(queries)):
        q = queries[int(i)]
        for action in rng.permutation(ACTIONS):
            result.append({"query_index": int(i), "dataset": q["dataset"], "cluster_id": q["cluster_id"],
                           "variant_id": q["variant_id"], "action": str(action)})
    return result


def _key(item):
    return item["dataset"], item["cluster_id"], item["variant_id"], item["action"]


def _append(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def collect(queries, backend, output, run_identity, seed=20261019, resume=False):
    """Dependency-injected collector; tests use explicit toy backends only."""
    order = schedule(queries, seed)
    output = Path(output)
    identity = {"inputs": run_identity, "queries_sha256": json_digest(queries), "schedule_sha256": json_digest(order),
                "seed": seed, "actions": list(ACTIONS), "schema": 1}
    if resume:
        if read_json(output / "run_identity.json") != identity or read_json(output / "schedule.json") != order:
            raise ValueError("resume identity mismatch; changed inputs or code require a new output directory")
    else:
        new_directory(output)
        write_json(output / "run_identity.json", identity)
        write_json(output / "schedule.json", order)
    lock = output / "collector.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ValueError("collector lock exists; inspect the recorded PID/process before removing any stale lock") from error
    try:
        os.write(descriptor, json.dumps({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}).encode())
    finally:
        os.close(descriptor)
    journal = output / "journal.jsonl"
    done, attempts, pending = {}, Counter(), {}
    allowed = {_key(item) for item in order}
    try:
        for record in read_jsonl(journal) if journal.exists() else []:
            key = tuple(record["key"])
            if key not in allowed:
                raise ValueError("journal contains an unplanned action")
            if record["event"] == "start":
                if key in done or key in pending or record["attempt"] != attempts[key] + 1:
                    raise ValueError("inconsistent journal attempt sequence")
                attempts[key] += 1
                pending[key] = record["attempt"]
            elif record["event"] in ("complete", "failed"):
                if key not in pending or record["attempt"] != pending[key]:
                    raise ValueError("journal completion has no matching attempt")
                pending.pop(key)
                if record["event"] == "complete":
                    if key in done or _key(record["outcome"]) != key:
                        raise ValueError("duplicate or misaligned completed journal row")
                    done[key] = record["outcome"]
            else:
                raise ValueError("unknown journal event")
        with journal.open("a", encoding="utf-8", newline="\n") as stream:
            # Preserve evidence that prior starts had no completion before retry.
            for key, attempt in list(pending.items()):
                _append(stream, {"event": "failed", "key": key, "attempt": attempt,
                                 "error": "previous process stopped before durable completion; wall-clock result unavailable"})
            for position, task in enumerate(order):
                key = _key(task)
                if key in done:
                    continue
                attempts[key] += 1
                header = {"key": key, "attempt": attempts[key], "schedule_position": position,
                          "utc": datetime.now(timezone.utc).isoformat()}
                _append(stream, {**header, "event": "start"})
                q = queries[task["query_index"]]
                try:
                    result = score_outcome(run_action(q["question"], task["action"], backend), q["answer"])
                    result.update({k: q[k] for k in ("dataset", "cluster_id", "variant_id")})
                    _append(stream, {**header, "event": "complete", "outcome": result})
                    done[key] = result
                except BaseException as error:
                    _append(stream, {**header, "event": "failed", "error_type": type(error).__name__,
                                     "error": str(error), "traceback": traceback.format_exc()})
                    write_json(output / "status.json", {"status": "interrupted", "completed": len(done),
                               "planned": len(order), "failed_key": key, "error_type": type(error).__name__})
                    raise
        rows = [done[_key(item)] for item in order]
        write_jsonl(output / "outcomes.jsonl", rows)
        write_json(output / "status.json", {"status": "action_collection_complete", "completed": len(done),
                   "planned": len(order), "queries": len(queries), "attempts": sum(attempts.values()),
                   "outcomes_sha256": digest(output / "outcomes.jsonl"), "human_review_complete": False,
                   "confirmatory_evaluation_complete": False})
        return read_jsonl(output / "outcomes.jsonl")
    finally:
        lock.unlink()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--confirmation-bundle", type=Path,
                    help="required for confirmation: sample/policy/human review frozen before collection")
    a = ap.parse_args(argv)
    config = read_json(a.config)
    queries = validate_queries(read_jsonl(a.queries))
    if {q["dataset"] for q in queries} != {config["dataset"]}:
        raise ValueError("queries and runtime dataset differ")
    check = preflight(config)
    if check["status"] != "paths_and_packages_ready":
        print(json.dumps(check, ensure_ascii=False, indent=2))
        return 2
    root = Path(__file__).resolve().parents[1]
    sources = list((root / "followup").glob("*.py"))
    sources += [root.parent / "main_logic" / p for p in ("actions.py", "answers.py", "dense.py", "signals.py")]
    identity = {"config_sha256": digest(a.config), "protocol_sha256": digest(a.protocol),
                "queries_file_sha256": digest(a.queries),
                "source_sha256": {p.relative_to(root.parent).as_posix(): digest(p) for p in sources},
                "runtime_preflight": check, "python": platform.python_version(), "platform": platform.platform(),
                "runtime_measurement_scope": "new action calls on local CPU retrieval/rerank plus GPU generator; JVM startup paid per retrieval; historical batched timing not comparable"}
    if a.confirmation_bundle:
        from experiment_ver2.followup.freeze import load_binding
        load_binding(a.confirmation_bundle, a.queries, a.protocol)
        identity["confirmation_binding_sha256"] = digest(a.confirmation_bundle / "binding.json")
    expected_identity = {"inputs": identity, "queries_sha256": json_digest(queries),
                         "schedule_sha256": json_digest(schedule(queries, 20261019)),
                         "seed": 20261019, "actions": list(ACTIONS), "schema": 1}
    if a.resume:
        if read_json(a.out / "run_identity.json") != expected_identity:
            raise ValueError("resume input/source identity changed; refusing to launch expensive backends")
    elif a.out.exists() and any(a.out.iterdir()):
        raise ValueError("output is nonempty; refusing to launch expensive backends")
    # Server log lives beside the collection directory, allowing collect() to
    # keep its own immutable directory semantics and exact resume identity.
    a.out.parent.mkdir(parents=True, exist_ok=True)
    log = a.out.parent / (a.out.name + f"-llama-{time_id()}.log")
    before_server = time.perf_counter()
    with LlamaServer(config, log) as server:
        server_seconds = time.perf_counter() - before_server
        before_backend = time.perf_counter()
        backend = LocalBackend(config, server)
        backend_seconds = time.perf_counter() - before_backend
        try:
            # Unscored, fixed warmup; not borrowed from the held-out sample.
            warmup = run_action("What is the capital city of France?", "A1", backend)
            write_json(log.with_suffix(".warmup.json"), warmup)
            write_json(log.with_suffix(".setup.json"), {"server_startup_seconds": server_seconds,
                       "backend_initialization_and_integrity_seconds": backend_seconds,
                       "warmup_action_seconds": warmup["latency_total"], "preflight": check,
                       "scope": "startup/hash checks and fixed warmup excluded from measured per-query action cost"})
            collect(queries, backend, a.out, identity, resume=a.resume)
        finally:
            backend.close()
    return 0


def time_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")


if __name__ == "__main__":
    raise SystemExit(main())
