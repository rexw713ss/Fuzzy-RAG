"""Generate and screen paraphrases of an already frozen prospective sample.

This module never samples questions, downloads models, rates semantic review,
or substitutes cached action answers for a newly executed experiment. All
originals survive screening, including clusters with zero accepted rewrites.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from main_logic import paraphrase as pp
from experiment_ver2.archive import normalize_question
from experiment_ver2.followup.io import (digest, json_digest, new_directory, read_json,
                                        read_jsonl, seal, write_json, write_jsonl)
from experiment_ver2.followup.runtime import LlamaServer, verify_generator
from experiment_ver2.followup.review import task_digest


BLIND_SEED = 20261022


def text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def settings():
    return {"prompt": pp.PROMPT, "n_candidates": pp.N_CANDIDATES,
            "temperature": pp.TEMPERATURE, "max_tokens": pp.MAX_TOKENS,
            "seeds": list(range(pp.SEED, pp.SEED + pp.N_CANDIDATES)),
            "per_cluster": pp.PER_CLUSTER, "cosine_min": pp.COSINE_MIN,
            "nli_model": pp.NLI_MODEL, "nli_revision": pp.NLI_REVISION,
            "embedding_model": pp.EMBED_MODEL, "embedding_revision": pp.EMBED_REVISION,
            "gate_device": "cpu", "local_files_only": True,
            "blind_order_seed": BLIND_SEED}


def validate_originals(originals, exposure, dataset):
    if dataset not in ("nq", "hotpotqa") or not originals or exposure.get("dataset") != dataset:
        raise ValueError("nonempty matching dataset sample and exposure inventory required")
    seen_ids, seen_texts = set(), set()
    excluded_ids, excluded_texts = set(exposure["originals"]), set(exposure["normalized_texts"])
    for row in originals:
        cid, text = row.get("cluster_id"), row.get("question")
        source_index = row.get("source_index")
        if type(source_index) is not int or source_index < 0 or cid != f"{dataset}-{source_index:05d}" or cid in seen_ids or row.get("dataset") != dataset:
            raise ValueError("sample requires unique source-aligned original IDs")
        if not isinstance(text, str) or not normalize_question(text):
            raise ValueError("nonempty original question required")
        normalized = normalize_question(text)
        if cid in excluded_ids or normalized in excluded_texts or normalized in seen_texts:
            raise ValueError("frozen original overlaps exposure or another sample original")
        answers = row.get("answer")
        if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
            raise ValueError("nonempty gold list required only for the answer-leak screen")
        if any(k in row for k in ("F1", "EM", "action", "outcomes", "latency_total", "variant_id")):
            raise ValueError("frozen originals cannot contain outcomes or variants")
        seen_ids.add(cid)
        seen_texts.add(normalized)
    return originals


def load_sample(directory, protocol):
    """Validate sampling.py's seal before spending generation compute."""
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    checksums = read_json(directory / "SHA256.json")
    for name in ("manifest.json", "originals.jsonl", "exposure_inventory.json"):
        if checksums.get(name) != digest(directory / name):
            raise ValueError(f"frozen sample seal mismatch: {name}")
    if manifest.get("sample_sha256") != digest(directory / "originals.jsonl"):
        raise ValueError("frozen original sample fingerprint mismatch")
    if manifest.get("protocol_sha256") != digest(protocol):
        raise ValueError("sample was frozen against a different protocol")
    if manifest.get("outcomes_generated") is not False or manifest.get("confirmatory_complete") is not False:
        raise ValueError("sampling manifest must describe an outcome-free sample")
    originals, exposure = read_jsonl(directory / "originals.jsonl"), read_json(directory / "exposure_inventory.json")
    validate_originals(originals, exposure, manifest.get("dataset"))
    if type(manifest.get("n")) is not int or manifest["n"] != len(originals):
        raise ValueError("frozen original sample count mismatch")
    return originals, exposure, manifest


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def preflight(config):
    """Only generation/gate assets are required; retrieval assets are irrelevant."""
    checks, snapshots = [], {}
    try:
        checks.append({"check": "generator_identity", "passed": True, "detail": verify_generator(config)})
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        checks.append({"check": "generator_identity", "passed": False, "error": str(error)})
    port = config.get("port", 8089)
    checks.append({"check": "local_port", "passed": type(port) is int and 1 <= port <= 65535})
    for package in ("torch", "transformers", "numpy"):
        checks.append({"check": "package:" + package, "passed": importlib.util.find_spec(package) is not None})
    for field in ("model", "llama_server"):
        path, expected = config.get(field), config.get(field + "_sha256")
        exists = isinstance(path, str) and bool(path) and Path(path).is_file()
        observed = digest(path) if exists and _sha256(expected) else None
        checks.append({"check": field, "path": path, "expected": expected, "observed": observed,
                       "passed": exists and _sha256(expected) and expected == observed})
    cache = config.get("model_cache")
    for label, model, revision in (("nli", pp.NLI_MODEL, pp.NLI_REVISION),
                                   ("embedder", pp.EMBED_MODEL, pp.EMBED_REVISION)):
        snapshot = Path(cache) / ("models--" + model.replace("/", "--")) / "snapshots" / revision if isinstance(cache, str) and cache else None
        files = sorted(p for p in snapshot.rglob("*") if p.is_file()) if snapshot and snapshot.is_dir() else []
        names = {p.relative_to(snapshot).as_posix() for p in files}
        usable = ("config.json" in names and "tokenizer_config.json" in names
                  and bool(names & {"tokenizer.json", "spm.model", "sentencepiece.bpe.model", "vocab.json"})
                  and bool(names & {"model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"}))
        checks.append({"check": label + "_pinned_local_snapshot", "path": str(snapshot) if snapshot else None,
                       "revision": revision, "passed": usable})
        if usable:
            # Include tokenizer/config and every weight shard in the run identity.
            snapshots[label] = {p.relative_to(snapshot).as_posix(): digest(p) for p in files}
    return {"status": "paraphrase_assets_ready" if all(c["passed"] for c in checks) else "blocked_missing_paraphrase_assets",
            "checks": checks, "snapshot_sha256": snapshots,
            "scope": "read-only local asset/package check; loading and generation must still succeed; no retrieval indices or downloads needed"}


class LocalGateBackend:
    def __init__(self, config, server):
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        opts = {"cache_dir": config["model_cache"], "local_files_only": True}
        self.server = server
        self.nli = (AutoTokenizer.from_pretrained(pp.NLI_MODEL, revision=pp.NLI_REVISION, **opts),
                    AutoModelForSequenceClassification.from_pretrained(pp.NLI_MODEL, revision=pp.NLI_REVISION, **opts).to("cpu").eval())
        self.embedder = (AutoTokenizer.from_pretrained(pp.EMBED_MODEL, revision=pp.EMBED_REVISION, **opts),
                         AutoModel.from_pretrained(pp.EMBED_MODEL, revision=pp.EMBED_REVISION, **opts).to("cpu").eval())

    def chat(self, messages, **kwargs):
        return self.server.chat(messages, **kwargs)

    def measure(self, original, candidates):
        # Gold answers are deliberately not an argument to either model.
        originals = [original] * len(candidates)
        fwd = pp.nli_probs(originals, candidates, self.nli)
        bwd = pp.nli_probs(candidates, originals, self.nli)
        embeddings = pp.embed([original] + list(candidates), self.embedder)
        return {"nli_fwd": fwd.tolist(), "nli_bwd": bwd.tolist(),
                "cosines": (embeddings[1:] @ embeddings[0]).tolist(),
                "entailment_index": pp.entailment_index(self.nli[1].config.id2label)}


def _gate_evidence(evidence, n):
    entail = evidence.get("entailment_index")
    fwd, bwd, cosine = (np.asarray(evidence[k], float) for k in ("nli_fwd", "nli_bwd", "cosines"))
    if type(entail) is not int or entail not in (0, 1, 2) or fwd.shape != (n, 3) or bwd.shape != (n, 3) or cosine.shape != (n,):
        raise ValueError("gate evidence has an invalid label index or shape")
    for probs in (fwd, bwd):
        if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)) or not np.allclose(probs.sum(1), 1, atol=1e-5):
            raise ValueError("NLI evidence must contain finite class probabilities")
    if not np.isfinite(cosine).all() or np.any(abs(cosine) > 1 + 1e-6):
        raise ValueError("cosine evidence must be finite and lie in [-1, 1]")
    return fwd, bwd, cosine, entail


def screen_clusters(originals, generated, measured, exposure):
    """Recompute gates and global duplicate exclusions in frozen sample order."""
    original_texts = {normalize_question(q["question"]) for q in originals}
    previous_texts = set(exposure["normalized_texts"])
    retained_texts, clusters, queries = set(), [], []
    for q in originals:
        cid = q["cluster_id"]
        candidates = generated[cid]
        texts = [c["text"] for c in candidates]
        fwd, bwd, cosines, entail = _gate_evidence(measured[cid], pp.N_CANDIDATES)
        gates = pp.screen(q["question"], q["answer"], texts, fwd, bwd, cosines, entail)
        selected = []
        for index, gate in enumerate(gates):
            gate.update({"candidate_index": index, "nli_fwd_probabilities": fwd[index].tolist(),
                         "nli_bwd_probabilities": bwd[index].tolist(), "entailment_index": entail,
                         "original_gates_passed": gate["passed"], "selected": False})
            normalized = normalize_question(gate["text"])
            if gate["passed"]:
                reason = ("previous_exposure_text" if normalized in previous_texts else
                          "sample_original_text" if normalized in original_texts else
                          "retained_paraphrase_text" if normalized in retained_texts else None)
                if reason:
                    gate.update({"passed": False, "reason": reason})
            if gate["passed"] and len(selected) < pp.PER_CLUSTER:
                gate["selected"] = True
                selected.append(gate["text"])
                retained_texts.add(normalized)
            gate["selection_reason"] = "selected" if gate["selected"] else gate["reason"] or "first_two_already_selected"
        phrasings = [{"variant_id": i, "text": text} for i, text in enumerate([q["question"]] + selected)]
        cluster = {k: v for k, v in q.items() if k != "question"}
        cluster.update({"phrasings": phrasings, "complete": len(selected) == pp.PER_CLUSTER, "screening": gates})
        clusters.append(cluster)
        queries.extend({**q, "variant_id": p["variant_id"], "question": p["text"]} for p in phrasings)
    return clusters, queries


def blind_tasks(queries, seed=BLIND_SEED):
    originals = {(q["dataset"], q["cluster_id"]): q["question"] for q in queries if q["variant_id"] == 0}
    pairs = [q for q in queries if q["variant_id"] != 0]
    rng = np.random.default_rng(seed)
    tasks, key = [], []
    for position, index in enumerate(rng.permutation(len(pairs))):
        q = pairs[int(index)]
        original = originals[(q["dataset"], q["cluster_id"])]
        left_original = bool(rng.integers(2))
        left, right = (original, q["question"]) if left_original else (q["question"], original)
        # Dataset-prefixed IDs avoid collisions when two review files are joined.
        review_id = f"{q['dataset']}-new-review-{position + 1:05d}"
        tasks.append({"review_id": review_id, "left_question": left, "right_question": right})
        key.append({"review_id": review_id, "dataset": q["dataset"], "cluster_id": q["cluster_id"],
                    "paraphrase_variant_id": q["variant_id"], "original_side": "left" if left_original else "right",
                    "left_question_sha256": text_digest(left), "right_question_sha256": text_digest(right)})
    return tasks, {"schema": 1, "seed": seed, "items": key,
                   "tasks_canonical_sha256": task_digest(tasks) if tasks else None, "queries_canonical_sha256": json_digest(queries)}


def _append(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _generation(backend, question, index):
    messages = pp.messages(question)
    started = time.perf_counter()
    raw, info = backend.chat(messages, temperature=pp.TEMPERATURE, seed=pp.SEED + index, max_tokens=pp.MAX_TOKENS)
    if not isinstance(raw, str) or not isinstance(info, dict):
        raise ValueError("generator must return raw text and usage information")
    if any(type(info.get(k)) is not int or info[k] < 0 for k in ("input_tokens", "output_tokens")):
        raise ValueError("generator must return nonnegative integer token usage")
    return {"candidate_index": index, "seed": pp.SEED + index, "raw": raw, "text": pp.clean(raw),
            "messages": messages, "temperature": pp.TEMPERATURE, "max_tokens": pp.MAX_TOKENS,
            "seconds": time.perf_counter() - started, "usage": info}


def _check_generation(row, question, index):
    if (row.get("candidate_index") != index or row.get("seed") != pp.SEED + index
            or row.get("messages") != pp.messages(question) or row.get("temperature") != pp.TEMPERATURE
            or row.get("max_tokens") != pp.MAX_TOKENS or not isinstance(row.get("raw"), str)
            or row.get("text") != pp.clean(row["raw"])):
        raise ValueError("journal generation does not match the locked prompt/settings")
    if type(row.get("seconds")) not in (int, float) or not np.isfinite(row["seconds"]) or row["seconds"] < 0:
        raise ValueError("invalid generation elapsed time")
    if any(type(row.get("usage", {}).get(k)) is not int or row["usage"][k] < 0 for k in ("input_tokens", "output_tokens")):
        raise ValueError("invalid generation token usage")


def run(originals, exposure, backend, output, run_identity, resume=False):
    """Injected backends are for tests; CLI uses only the concrete local runtime."""
    validate_originals(originals, exposure, originals[0]["dataset"] if originals else None)
    output = Path(output)
    identity = {"inputs": run_identity, "originals_sha256": json_digest(originals),
                "exposure_sha256": json_digest(exposure), "settings": settings(), "schema": 1}
    if resume:
        if read_json(output / "run_identity.json") != identity:
            raise ValueError("resume identity mismatch; changed sample/config/code requires a new run")
    else:
        new_directory(output)
        write_json(output / "run_identity.json", identity)
    lock = output / "paraphrases.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise ValueError("paraphrase lock exists; inspect its PID before removing a stale lock") from error
    try:
        os.write(fd, json.dumps({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}).encode())
    finally:
        os.close(fd)
    journal = output / "journal.jsonl"
    allowed = {(q["cluster_id"], "generate", i) for q in originals for i in range(pp.N_CANDIDATES)}
    allowed |= {(q["cluster_id"], "measure") for q in originals}
    done, attempts, pending = {}, Counter(), {}
    try:
        for record in read_jsonl(journal) if journal.exists() else []:
            key = tuple(record["key"])
            if key not in allowed:
                raise ValueError("journal contains an unplanned paraphrase task")
            if record["event"] == "start":
                if key in pending or key in done or record["attempt"] != attempts[key] + 1:
                    raise ValueError("inconsistent journal attempt sequence")
                attempts[key] += 1
                pending[key] = record["attempt"]
            elif record["event"] in ("complete", "failed"):
                if key not in pending or record["attempt"] != pending.pop(key):
                    raise ValueError("journal terminal event lacks a matching start")
                if record["event"] == "complete":
                    done[key] = record["result"]
            else:
                raise ValueError("unknown journal event")
        with journal.open("a", encoding="utf-8", newline="\n") as stream:
            for key, attempt in pending.items():
                _append(stream, {"event": "failed", "key": key, "attempt": attempt,
                                 "error": "previous process stopped before durable completion; result unavailable"})

            def execute(key, callback):
                if key in done:
                    return done[key]
                attempts[key] += 1
                header = {"key": key, "attempt": attempts[key], "utc": datetime.now(timezone.utc).isoformat()}
                _append(stream, {**header, "event": "start"})
                try:
                    result = callback()
                    _append(stream, {**header, "event": "complete", "result": result})
                    done[key] = result
                    return result
                except BaseException as error:
                    _append(stream, {**header, "event": "failed", "error_type": type(error).__name__,
                                     "error": str(error), "traceback": traceback.format_exc()})
                    write_json(output / "status.json", {"status": "interrupted", "completed_tasks": len(done),
                                                       "planned_tasks": len(allowed), "failed_key": key})
                    raise

            generated, measured = {}, {}
            for q in originals:
                cid = q["cluster_id"]
                generated[cid] = []
                for i in range(pp.N_CANDIDATES):
                    candidate = execute((cid, "generate", i), lambda q=q, i=i: _generation(backend, q["question"], i))
                    _check_generation(candidate, q["question"], i)
                    generated[cid].append(candidate)
                texts = [c["text"] for c in generated[cid]]

                def measure(q=q, texts=texts):
                    started = time.perf_counter()
                    evidence = backend.measure(q["question"], texts)
                    _gate_evidence(evidence, pp.N_CANDIDATES)
                    return {**evidence, "seconds": time.perf_counter() - started,
                            "question_sha256": text_digest(q["question"]), "candidates_sha256": json_digest(texts)}

                measured[cid] = execute((cid, "measure"), measure)
                if (measured[cid].get("question_sha256") != text_digest(q["question"])
                        or measured[cid].get("candidates_sha256") != json_digest(texts)):
                    raise ValueError("journal gate evidence differs from its generated candidates")
        clusters, queries = screen_clusters(originals, generated, measured, exposure)
        tasks, key = blind_tasks(queries)
        write_jsonl(output / "candidates.jsonl", [{"dataset": q["dataset"], "cluster_id": q["cluster_id"],
                   "question": q["question"], "candidates": generated[q["cluster_id"]],
                   "gate_evidence": measured[q["cluster_id"]]} for q in originals])
        write_jsonl(output / "clusters.jsonl", clusters)
        write_jsonl(output / "queries.jsonl", queries)
        write_jsonl(output / "blind_review_tasks.jsonl", tasks)
        write_json(output / "blind_review_key.json", key)
        write_json(output / "status.json", {"status": "paraphrase_generation_complete", "n_originals": len(originals),
                   "n_queries": len(queries), "n_complete_clusters": sum(c["complete"] for c in clusters),
                   "n_incomplete_clusters": sum(not c["complete"] for c in clusters), "n_candidates": len(originals) * pp.N_CANDIDATES,
                   "n_review_pairs": len(tasks), "attempts": sum(attempts.values()),
                   "queries_sha256": digest(output / "queries.jsonl"), "tasks_sha256": digest(output / "blind_review_tasks.jsonl"),
                   "key_sha256": digest(output / "blind_review_key.json"), "human_review_complete": False,
                   "action_outcomes_generated": False, "confirmatory_complete": False,
                   "scope": "automated candidate generation/gates only; every original retained; human review and action collection still required"})
    finally:
        lock.unlink()
    seal(output)
    return read_jsonl(output / "queries.jsonl")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=Path, help="sampling.py output directory; never resampled")
    ap.add_argument("--protocol", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    a = ap.parse_args(argv)
    config = read_json(a.config)
    if not a.preflight_only and any(p is None for p in (a.sample, a.protocol, a.out)):
        ap.error("generation requires --sample, --protocol and --out")
    if not a.preflight_only:
        originals, exposure, manifest = load_sample(a.sample, a.protocol)
        if config.get("dataset") != manifest["dataset"]:
            raise ValueError("sample and runtime dataset differ")
        if not a.resume:
            # Validate separation before initializing any expensive models.
            new_directory(a.out, (a.sample, a.protocol, a.config))
    check = preflight(config)
    if a.preflight_only or check["status"] != "paraphrase_assets_ready":
        print(json.dumps(check, ensure_ascii=False, indent=2))
        return 0 if check["status"] == "paraphrase_assets_ready" else 2
    root = Path(__file__).resolve().parents[2]
    source_names = ("experiment_ver2/followup/paraphrases.py", "experiment_ver2/followup/runtime.py",
                    "experiment_ver2/followup/io.py", "experiment_ver2/archive.py", "main_logic/paraphrase.py", "main_logic/answers.py")
    identity = {"config_sha256": digest(a.config), "protocol_sha256": digest(a.protocol),
                "sample_manifest_sha256": digest(a.sample / "manifest.json"),
                "source_sha256": {name: digest(root / name) for name in source_names},
                "runtime_preflight": check}
    expected = {"inputs": identity, "originals_sha256": json_digest(originals),
                "exposure_sha256": json_digest(exposure), "settings": settings(), "schema": 1}
    if a.resume and read_json(a.out / "run_identity.json") != expected:
        raise ValueError("resume identity mismatch; refusing to launch expensive backends")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    with LlamaServer(config, a.out.parent / f"{a.out.name}-paraphrase-llama-{stamp}.log") as server:
        backend = LocalGateBackend(config, server)
        run(originals, exposure, backend, a.out, identity, resume=a.resume)
    print(f"Generated frozen sample paraphrases; human review and action outcomes remain pending: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
