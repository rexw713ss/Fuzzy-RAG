"""Pipeline steps 4-5: retrieve every phrasing of every cluster, fuse, extract S1-S4
(spec v0.3, sections 2 and 7.2).

Run from the repo root, in the main environment, after scripts.build_clusters:
    .venv\\Scripts\\python.exe -m scripts.retrieve_clusters --dataset nq [--outdir DIR]

Every phrasing of every cluster (incomplete ones too) is one query instance. BM25 runs in the BM25
environment, which this script calls as a subprocess (scripts.bm25_search); dense search, fusion
and the signals run here, exactly as in scripts.smoke_retrieval. hybrid() is shared with step 8,
which uses it for A2's second retrieval.

Reads OUTDIR/clusters.jsonl (default data/stage1/<dataset>). Writes to OUTDIR:
  queries.jsonl      one line per query instance: cluster_id, variant_id, question, answer
  bm25.jsonl         BM25 top-50 per query instance (from the BM25 environment), + .timing.json
  retrieval.jsonl    per query instance: both retrievers' top-50, the fused top-50,
                     fused_hit_at_5 (a gold answer in the fused top 5; null for yes/no answers)
                     and retrieval_latency (seconds; see hybrid)
  embeddings.npy     (instances, 10, 768) float32: the top-10 fused passages' vectors (for S3)
  signals.jsonl      per query instance: cluster_id, variant_id, raw S1-S4
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from main_logic import answers as an
from main_logic import corpus as cp
from main_logic import dense as dn
from main_logic import signals as sg
from scripts import datasets as ds

REPO = Path(__file__).resolve().parents[1]
BM25_PYTHON = REPO / ".venv-bm25" / "Scripts" / "python.exe"
JAVA_HOME = r"C:\Program Files\Microsoft\jdk-25.0.4.101-hotspot"
YES_NO = ({"yes"}, {"no"})


def read_jsonl(path):
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_yes_no(answers):
    return {a.lower() for a in answers} in YES_NO


def hybrid(queries, dataset, workdir, name, index=None):
    """BM25 + dense top-50 and the fused top-50 for each query (section 2.1).

    queries: dicts with "question" and "answer" (the BM25 script reads both). Files go to
    workdir/<name>_queries.jsonl and <name>_bm25.jsonl. index: an already loaded dense index, or
    None to load it. Returns (rows, index, timing); each row has bm25, dense, fused and
    retrieval_latency: the per-query share of the batched BM25 search plus question encoding
    and dense search (index loading and the BM25 JVM start are one-off and not counted).
    """
    workdir = Path(workdir)
    qpath, bpath = workdir / f"{name}_queries.jsonl", workdir / f"{name}_bm25.jsonl"
    write_jsonl(qpath, [{"question": q["question"], "answer": q["answer"]} for q in queries])
    timing = {}

    env = {**os.environ, "PYTHONUTF8": "1", "JAVA_HOME": os.environ.get("JAVA_HOME", JAVA_HOME)}
    subprocess.run([str(BM25_PYTHON), "-m", "scripts.bm25_search", str(qpath), str(bpath),
                    "--dataset", dataset, "--k", str(sg.POOL)], cwd=REPO, env=env, check=True)
    bm25 = read_jsonl(bpath)
    if [b["question"] for b in bm25] != [q["question"] for q in queries]:
        raise ValueError("BM25 output does not match the queries")
    timing["bm25_search_s"] = json.loads(Path(str(bpath) + ".timing.json").read_text())["search_s"]

    t = time.time()
    encoder = dn.load_encoder()
    timing["load_encoder_s"] = time.time() - t
    t = time.time()
    qemb = dn.encode([q["question"] for q in queries], encoder)
    timing["encode_s"] = time.time() - t
    del encoder
    if index is None:
        t = time.time()
        index = ds.load_index(dataset)
        timing["load_index_s"] = time.time() - t
    t = time.time()
    dense = index.search(qemb, k=sg.POOL)
    timing["dense_search_s"] = time.time() - t

    per_query = (timing["bm25_search_s"] + timing["encode_s"] + timing["dense_search_s"]) / len(queries)
    timing["per_query_s"] = per_query
    rows = [{"bm25": b["hits"], "dense": d,
             "fused": sg.fuse({doc: s for doc, s in b["hits"]}, {doc: s for doc, s in d}),
             "retrieval_latency": per_query}
            for b, d in zip(bm25, dense)]
    return rows, index, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=ds.NAMES, required=True)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.outdir or ds.DATA / "stage1" / args.dataset)

    queries = [{"cluster_id": c["cluster_id"], "variant_id": p["variant_id"],
                "question": p["text"], "answer": c["answer"]}
               for c in read_jsonl(out / "clusters.jsonl") for p in c["phrasings"]]
    write_jsonl(out / "queries.jsonl", queries)
    print(f"{len(queries)} query instances", flush=True)

    hits, index, timing = hybrid(queries, args.dataset, out, "initial")

    # Signals (section 2).
    rows, sigs, embs = [], [], []
    for q, h in zip(queries, hits):
        fused = h["fused"]
        emb = index.vectors([doc for doc, _ in fused[:sg.K]])
        sig = sg.compute_signals(fused, [doc for doc, _ in h["bm25"][:sg.K]],
                                 [doc for doc, _ in h["dense"][:sg.K]], emb)
        ids = {"cluster_id": q["cluster_id"], "variant_id": q["variant_id"]}
        rows.append({**ids, "question": q["question"], **h})
        sigs.append({**ids, **sig})
        embs.append(np.asarray(emb, dtype=np.float32))
    del index

    # Is a gold answer in the fused top 5? (Easy / hard, for the stage 1 check.)
    t = time.time()
    texts = cp.load_passages(ds.CORPUS[args.dataset], {doc for r in rows for doc, _ in r["fused"][:5]})
    timing["read_texts_s"] = time.time() - t
    for r, q in zip(rows, queries):
        r["fused_hit_at_5"] = (None if is_yes_no(q["answer"]) else
                               any(an.has_answer(q["answer"], texts[doc][1])
                                   for doc, _ in r["fused"][:5]))

    write_jsonl(out / "retrieval.jsonl", rows)
    write_jsonl(out / "signals.jsonl", sigs)
    np.save(out / "embeddings.npy", np.stack(embs))
    print(json.dumps({"dataset": args.dataset, "instances": len(rows), "timing": timing}, indent=2))


if __name__ == "__main__":
    main()
