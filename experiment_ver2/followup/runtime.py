"""Concrete local runtime for newly executed outcomes; no cached-answer fallback.

PyTorch encoder/reranker stay on CPU to avoid competing with the 5GB GPU used
by llama.cpp. BM25 uses its separately configured Python/Java environment.
This changes throughput from the historical batched runs; report new timings
only against other actions measured with this SAME runtime.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import struct
import subprocess
import time
from urllib.parse import urlparse
import urllib.request

from experiment_ver2.followup.io import digest, read_json, write_json


PATH_FIELDS = ("llama_server", "model", "bm25_python", "bm25_index", "dense_index", "passage_store", "model_cache", "java_home")
PINNED_HF = {
    "encoder": ("facebook/contriever-msmarco", "abe8c1493371369031bcb1e02acb754cf4e162fa"),
    "reranker": ("cross-encoder/ms-marco-MiniLM-L-6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a"),
}
GENERATOR_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _source(source):
    if not isinstance(source, dict):
        raise ValueError("source must identify an HTTPS URL and immutable revision")
    parsed = urlparse(source.get("url", ""))
    revision = source.get("revision")
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or not isinstance(revision, str) or not revision.strip()
            or revision.lower() in {"main", "master", "latest", "unknown"}):
        raise ValueError("source must identify an HTTPS URL and immutable revision")


def verify_index_manifest(config, kind, verify_hashes=True):
    """Validate every index byte before any pickle, FAISS, or Lucene loader runs.

    The caller must obtain the expected manifest digest from a trusted source.
    Hash matching proves consistency, not that an arbitrary manifest is authentic.
    """
    if kind not in ("dense", "bm25") or config.get("dataset") not in ("nq", "hotpotqa"):
        raise ValueError("invalid index kind or dataset")
    declaration = config.get("index_manifests", {}).get(kind, {})
    manifest_path = declaration.get("path")
    if not isinstance(manifest_path, str) or not _sha(declaration.get("sha256")):
        raise ValueError(f"{kind} manifest path and SHA256 are required")
    root = Path(config[f"{kind}_index"]).resolve(strict=True)
    manifest_path = Path(manifest_path).resolve(strict=True)
    if not root.is_dir() or root == manifest_path or root in manifest_path.parents:
        raise ValueError("index root must be a directory and manifest must be outside it")
    observed = digest(manifest_path) if verify_hashes else None
    if verify_hashes and observed != declaration["sha256"]:
        raise ValueError(f"{kind} manifest SHA256 mismatch")
    manifest = read_json(manifest_path)
    expected_format = ("contriever-pickle-shards" if config["dataset"] == "nq" else "faiss-flat-ip") if kind == "dense" else "lucene-bm25"
    if (manifest.get("schema") != 1 or manifest.get("dataset") != config["dataset"]
            or manifest.get("kind") != kind or manifest.get("format") != expected_format):
        raise ValueError("index manifest schema/dataset/kind/format mismatch")
    _source(manifest.get("source"))
    if not _sha(config.get("corpus_sha256")) or manifest.get("corpus_sha256") != config["corpus_sha256"]:
        raise ValueError("index manifest corpus SHA256 mismatch")
    if kind == "dense":
        encoder = manifest.get("encoder", {})
        if (encoder.get("model_id"), encoder.get("revision")) != PINNED_HF["encoder"]:
            raise ValueError("dense index encoder differs from the pinned query encoder")
        if config["dataset"] == "nq" and manifest.get("trusted_pickle_source") is not True:
            raise ValueError("NQ pickle shards require explicit trusted_pickle_source=true provenance")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("index manifest files must be a nonempty relative-path/SHA256 map")
    seen = set()
    for relative, expected in files.items():
        if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
                or relative.startswith("/") or any(p in ("", ".", "..") for p in relative.split("/"))
                or relative.casefold() in seen or not _sha(expected)):
            raise ValueError("invalid or ambiguous relative file path/SHA256 in index manifest")
        seen.add(relative.casefold())
        path = (root / relative).resolve(strict=True)
        if root not in path.parents or not path.is_file():
            raise ValueError("index manifest file escapes declared index root or is not a file")
        if verify_hashes and digest(path) != expected:
            raise ValueError(f"index file SHA256 mismatch: {relative}")
    actual = set()
    for path in root.rglob("*"):
        # In particular, never traverse a symlink/junction to an unlisted tree.
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("index tree must not contain symlinks or junctions")
        if path.is_file():
            if root not in path.resolve(strict=True).parents:
                raise ValueError("index file escapes declared root")
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files):
        raise ValueError("index directory and manifest must match exactly (missing or additional files)")
    if kind == "dense" and config["dataset"] == "nq":
        shards = [name for name in files if "/" not in name and name.startswith("passages_")]
        if len(shards) != 16 or any(p.is_dir() for p in root.glob("passages_*")):
            raise ValueError("historical NQ dense index requires exactly 16 top-level passages_* shards")
    if kind == "dense" and config["dataset"] == "hotpotqa" and not {"index", "docid"} <= set(files):
        raise ValueError("HotpotQA dense index requires index and docid files")
    return {"manifest_sha256": observed, "files": len(files), "source": manifest["source"],
            "hashes_checked": verify_hashes, "provenance": "declared; expected hashes must come from a trusted source"}


def verify_hf_snapshot(cache, model_id, revision):
    """Check the pinned offline snapshot and all shard names without importing HF."""
    snapshot = Path(cache) / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
    if not snapshot.is_dir():
        raise ValueError(f"missing pinned HF snapshot: {model_id}@{revision}")
    required = {"config.json", "tokenizer_config.json"}
    for filename in required:
        if not (snapshot / filename).is_file():
            raise ValueError(f"missing pinned HF asset: {filename}")
    if not any((snapshot / name).is_file() for name in ("tokenizer.json", "vocab.txt")):
        raise ValueError("missing pinned tokenizer vocabulary")
    weight_files = [p for p in (snapshot / "model.safetensors", snapshot / "pytorch_model.bin") if p.is_file()]
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = snapshot / index_name
        if index_path.is_file():
            mapping = read_json(index_path).get("weight_map", {})
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("empty or malformed HF weight shard map")
            for name in set(mapping.values()):
                if (not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name
                        or not (snapshot / name).is_file()):
                    raise ValueError("missing or invalid pinned HF weight shard")
            weight_files.extend(snapshot / name for name in set(mapping.values()))
    if not weight_files or any(p.stat().st_size == 0 for p in weight_files):
        raise ValueError("missing or empty pinned HF model weights")
    return {"model_id": model_id, "revision": revision, "snapshot": str(snapshot),
            "files_sha256": {p.relative_to(snapshot).as_posix(): digest(p) for p in sorted(snapshot.rglob("*")) if p.is_file()},
            "scope": "offline snapshot presence and byte fingerprints, not upstream authenticity or full tensor deserialization"}


def gguf_identity(path):
    """Read bounded GGUF metadata only; never deserialize tensors or executable data."""
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    wanted = {"general.architecture", "general.name", "general.file_type", "llama.block_count", "llama.embedding_length"}
    values = {}
    with Path(path).open("rb") as stream:
        size = Path(path).stat().st_size

        def exact(n):
            if n < 0 or n > 64 * 1024 * 1024 or stream.tell() + n > size:
                raise ValueError("truncated or excessive GGUF metadata")
            value = stream.read(n)
            if len(value) != n:
                raise ValueError("truncated GGUF metadata")
            return value

        def number(fmt):
            return struct.unpack("<" + fmt, exact(struct.calcsize("<" + fmt)))[0]

        def string(keep=True):
            length = number("Q")
            if not keep:
                if length > 64 * 1024 * 1024 or stream.tell() + length > size:
                    raise ValueError("truncated or excessive GGUF string")
                stream.seek(length, 1)
                return None
            return exact(length).decode("utf-8")

        def value(kind, keep=False):
            if kind in scalar:
                return number(scalar[kind])
            if kind == 8:
                return string(keep)
            if kind != 9:
                raise ValueError("unknown GGUF metadata type")
            element, count = number("I"), number("Q")
            if count > 10_000_000 or element == 9:
                raise ValueError("excessive or nested GGUF metadata array")
            if element in scalar:
                length = count * struct.calcsize("<" + scalar[element])
                if stream.tell() + length > size:
                    raise ValueError("truncated GGUF metadata array")
                stream.seek(length, 1)
            elif element == 8:
                for _ in range(count):
                    string(False)
            else:
                raise ValueError("unknown GGUF array type")
            return None

        if exact(4) != b"GGUF" or number("I") not in (2, 3):
            raise ValueError("generator must be a GGUF v2/v3 file")
        number("Q")  # Tensor count; tensors remain unread.
        count = number("Q")
        if count > 100_000:
            raise ValueError("excessive GGUF metadata count")
        keys = set()
        for _ in range(count):
            key = string()
            if key in keys:
                raise ValueError("duplicate GGUF metadata key")
            keys.add(key)
            result = value(number("I"), key in wanted)
            if key in wanted:
                values[key] = result
    expected = {"general.architecture": "llama", "general.file_type": 15,
                "llama.block_count": 32, "llama.embedding_length": 4096}
    if any(values.get(k) != v for k, v in expected.items()):
        raise ValueError("GGUF architecture/size/quantization does not match Llama 8B Q4_K_M")
    return values


def verify_generator(config, verify_hashes=True):
    identity = config.get("generator_identity", {})
    if identity.get("model_id") != GENERATOR_MODEL_ID or identity.get("quantization") != "Q4_K_M":
        raise ValueError("generator declaration must be Llama-3.1-8B-Instruct Q4_K_M")
    _source(identity.get("source"))
    # A source revision plus bytes is reproducible, but not proof it was the old model.
    historical = identity.get("historical_model_sha256")
    if not _sha(historical) or historical != config.get("model_sha256") or not isinstance(identity.get("historical_evidence"), str) or not identity["historical_evidence"].strip():
        raise ValueError("original-experiment model SHA256 and its provenance evidence are required")
    for field in ("model", "llama_server"):
        if not _sha(config.get(field + "_sha256")) or not Path(config.get(field, "")).is_file():
            raise ValueError(f"missing {field} path or SHA256")
        if verify_hashes and digest(config[field]) != config[field + "_sha256"]:
            raise ValueError(f"{field} SHA256 mismatch")
    if config.get("llama_server_release") != "b11505":
        raise ValueError("historical generator runtime requires declared llama.cpp release b11505")
    return {"metadata": gguf_identity(config["model"]), "identity": identity,
            "scope": "GGUF structure and declared provenance; no signed historical manifest is available in the repository"}


def preflight(config, verify_hashes=True):
    checks = []
    checks.append({"check": "dataset", "passed": config.get("dataset") in ("nq", "hotpotqa")})
    port = config.get("port", 8089)
    checks.append({"check": "local_port", "passed": type(port) is int and 1 <= port <= 65535})
    checksum = config.get("corpus_sha256")
    checks.append({"check": "declared_corpus_sha256", "passed": _sha(checksum)})
    for field in PATH_FIELDS:
        path = config.get(field)
        exists = isinstance(path, str) and bool(path) and Path(path).exists()
        if exists:
            exists = Path(path).is_dir() if field in ("bm25_index", "dense_index", "model_cache", "java_home") else Path(path).is_file()
        checks.append({"check": field, "path": path, "passed": exists})
    for package in ("torch", "transformers", "numpy"):
        checks.append({"check": f"package:{package}", "passed": importlib.util.find_spec(package) is not None})
    for field in ("model", "llama_server", "passage_store"):
        expected = config.get(f"{field}_sha256")
        valid = _sha(expected)
        observed = None
        if verify_hashes and valid and Path(config.get(field, "")).is_file():
            observed = digest(config[field])
        checks.append({"check": f"{field}_sha256", "passed": valid and (not verify_hashes or observed == expected),
                       "expected": expected, "observed": observed})
    validators = [("generator_identity", lambda: verify_generator(config, verify_hashes=False))]
    validators += [(f"index_manifest:{kind}", lambda kind=kind: verify_index_manifest(config, kind, verify_hashes)) for kind in ("dense", "bm25")]
    validators += [(f"pinned_hf:{kind}", lambda model=model, revision=revision: verify_hf_snapshot(config.get("model_cache", ""), model, revision)) for kind, (model, revision) in PINNED_HF.items()]
    for name, validate in validators:
        try:
            checks.append({"check": name, "passed": True, "detail": validate()})
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            checks.append({"check": name, "passed": False, "error": str(error)})
    passed = all(c["passed"] for c in checks)
    status = "paths_and_packages_ready" if passed and verify_hashes else "blocked_unverified_runtime_assets" if passed else "blocked_missing_runtime_assets"
    return {"status": status,
            "checks": checks, "hashes_checked": verify_hashes,
            "scope": "read-only checks; hashes establish consistency with supplied manifests, not authentic historical provenance; backend initialization and warmup must still succeed; this is not a completed experiment",
            "timing_scope": "CPU dense encoding/exact search/CE; configured llama.cpp GPU offload; separate JVM startup paid each retrieval; hash verification and model initialization are setup costs outside per-action latency"}


def _popen(args, **kwargs):
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.Popen(args, **kwargs)


class LlamaServer:
    def __init__(self, config, log_path):
        self.config, self.log_path, self.proc, self.log = config, Path(log_path), None, None
        if type(config.get("port", 8089)) is not int or not 1 <= config.get("port", 8089) <= 65535:
            raise ValueError("port must be an integer in 1..65535")
        self.url = f"http://127.0.0.1:{config.get('port', 8089)}"

    def __enter__(self):
        import socket
        c = self.config
        verify_generator(c)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(.5)
            if probe.connect_ex(("127.0.0.1", c.get("port", 8089))) == 0:
                raise RuntimeError("configured local port is already in use; refusing to connect to an unverified model server")
        self.log = self.log_path.open("x", encoding="utf-8")
        try:
            self.proc = _popen([c["llama_server"], "-m", c["model"], "-ngl", "31", "-c", "4096",
                                "-fa", "on", "-np", "1", "--host", "127.0.0.1", "--port", str(c.get("port", 8089))],
                               stdout=self.log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"llama-server exited: {self.proc.returncode}")
                try:
                    with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                        if r.status == 200:
                            return self
                except (OSError, TimeoutError):
                    pass
                time.sleep(.5)
            raise TimeoutError("llama-server initialization exceeded 600 seconds")
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *args):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        if self.log:
            self.log.close()

    def chat(self, messages, temperature, seed, max_tokens):
        body = json.dumps({"messages": messages, "temperature": temperature, "seed": seed,
                           "max_tokens": max_tokens, "cache_prompt": False, "stream": False}).encode("utf-8")
        req = urllib.request.Request(self.url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            response = json.load(r)
        choice = response["choices"][0]
        usage = response["usage"]
        return choice["message"]["content"], {"input_tokens": usage["prompt_tokens"],
                  "output_tokens": usage["completion_tokens"], "finish_reason": choice.get("finish_reason"),
                  "server_timings": response.get("timings")}


class LocalBackend:
    def __init__(self, config, server):
        # Always repeat the byte check at the loader boundary, including when a
        # caller constructs LocalBackend directly instead of using the CLI.
        # Keep this before imports/model loads and, especially, pickle loading.
        self.integrity = {kind: verify_index_manifest(config, kind) for kind in ("dense", "bm25")}
        if not _sha(config.get("passage_store_sha256")) or digest(config["passage_store"]) != config["passage_store_sha256"]:
            raise ValueError("passage store fingerprint missing or changed")
        for model, revision in PINNED_HF.values():
            verify_hf_snapshot(config["model_cache"], model, revision)
        from main_logic import actions as ac, dense as dn
        from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
        if (dn.MODEL, dn.REVISION) != PINNED_HF["encoder"] or (ac.CROSS_ENCODER, ac.CROSS_ENCODER_REVISION) != PINNED_HF["reranker"]:
            raise ValueError("source HF model pins changed; version the followup protocol first")
        self.config, self.server = config, server
        opts = {"cache_dir": config["model_cache"], "local_files_only": True}
        self.db = sqlite3.connect(Path(config["passage_store"]).resolve().as_uri() + "?mode=ro", uri=True)
        try:
            metadata = dict(self.db.execute("SELECT key,value FROM metadata"))
            if metadata.get("status") != "complete" or metadata.get("source_sha256") != config.get("corpus_sha256"):
                raise ValueError("passage store incomplete or inconsistent with declared corpus fingerprint")
            self.encoder = (AutoTokenizer.from_pretrained(dn.MODEL, revision=dn.REVISION, **opts),
                            AutoModel.from_pretrained(dn.MODEL, revision=dn.REVISION, **opts).to("cpu").eval())
            self.ce = (AutoTokenizer.from_pretrained(ac.CROSS_ENCODER, revision=ac.CROSS_ENCODER_REVISION, **opts),
                       AutoModelForSequenceClassification.from_pretrained(ac.CROSS_ENCODER, revision=ac.CROSS_ENCODER_REVISION, **opts).to("cpu").eval())
            self.index = (dn.EmbeddingIndex.from_shards(config["dense_index"]) if config["dataset"] == "nq"
                          else dn.EmbeddingIndex.from_faiss(config["dense_index"]))
        except BaseException:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def retrieve(self, question):
        from main_logic import dense as dn, signals as sg
        # A separate BM25 process avoids incompatible Java/Python dependencies.
        # JVM startup is deliberately paid here and included in wall-clock cost.
        env = {**os.environ, "JAVA_HOME": self.config["java_home"], "PYTHONUTF8": "1"}
        result = subprocess.run([self.config["bm25_python"], "-m", "experiment_ver2.followup.bm25_worker",
                                 "--index", self.config["bm25_index"]], input=json.dumps({"question": question}),
                                text=True, encoding="utf-8", capture_output=True, check=True, timeout=300, env=env,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        bm25 = json.loads(result.stdout)["hits"]
        dense = self.index.search(dn.encode([question], self.encoder), k=sg.POOL)[0]
        return {"bm25": bm25, "dense": dense, "fused": sg.fuse(dict(bm25), dict(dense))}

    def passages(self, ids):
        output = {}
        for doc_id in ids:
            row = self.db.execute("SELECT title,text FROM passages WHERE id=?", (doc_id,)).fetchone()
            if row is None:
                raise KeyError(f"passage missing: {doc_id}")
            output[doc_id] = row
        return output

    def vectors(self, ids):
        return self.index.vectors(ids)

    def rerank(self, question, passages):
        from main_logic.actions import cross_encode
        return cross_encode(question, passages, self.ce)

    def chat(self, messages, **kwargs):
        return self.server.chat(messages, **kwargs)


def main():
    ap = argparse.ArgumentParser(description="Read-only preflight; never generates answers")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        ap.error("refuse to overwrite prior preflight report")
    result = preflight(read_json(a.config))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out, result)
    print(result["status"])
    return 0 if result["status"] == "paths_and_packages_ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
