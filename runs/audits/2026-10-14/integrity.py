"""Audit 1 (Oct 14): recompute stored values from raw files and compare. Read-only on data\\full.

Run from D:\\Han\\rex_rag\\Fuzzy-RAG:
    .venv\\Scripts\\python.exe ..\\data\\audit_2026-10-14\\integrity.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(r"D:\Han\rex_rag\Fuzzy-RAG")))
from main_logic import actions as ac
from main_logic import answers as an
from main_logic import signals as sg

ROOT = Path(r"D:\Han\rex_rag\data\full")


def rj(p):
    with open(p, encoding="utf8") as f:
        return [json.loads(l) for l in f if l.strip()]


scalers = json.loads((ROOT / "scalers.json").read_text())
scaled = {(r["dataset"], r["cluster_id"], r["variant_id"]): r for r in rj(ROOT / "signals_scaled.jsonl")}
splits = json.loads((ROOT / "splits.json").read_text())

for d in ("nq", "hotpotqa"):
    print(f"\n===== {d}")
    out = ROOT / d
    queries = rj(out / "queries.jsonl")
    ret = rj(out / "retrieval.jsonl")
    sigs = rj(out / "signals.jsonl")
    emb = np.load(out / "embeddings.npy")
    clusters = {c["cluster_id"]: c for c in rj(out / "clusters.jsonl")}

    # 1. alignment of queries / retrieval / signals / embeddings
    keys_q = [(q["cluster_id"], q["variant_id"]) for q in queries]
    assert keys_q == [(r["cluster_id"], r["variant_id"]) for r in ret], "retrieval order"
    assert keys_q == [(r["cluster_id"], r["variant_id"]) for r in sigs], "signals order"
    assert len(emb) == len(queries)
    assert all(q["question"] == r["question"] for q, r in zip(queries, ret))
    print("instances", len(queries), "| alignment OK")

    # 2. recompute fusion and signals
    maxdiff = np.zeros(4)
    fuse_bad = 0
    overlap50, overlap10, bm_short, dn_short = [], [], 0, 0
    for r, s, e in zip(ret, sigs, emb):
        bm, dn = {a: b for a, b in r["bm25"]}, {a: b for a, b in r["dense"]}
        bm_short += len(bm) < 50
        dn_short += len(dn) < 50
        f = sg.fuse(bm, dn)
        if [x for x, _ in f] != [x for x, _ in r["fused"]]:
            fuse_bad += 1
        overlap50.append(len(set(bm) & set(dn)))
        overlap10.append(len({x for x, _ in r["bm25"][:10]} & {x for x, _ in r["dense"][:10]}))
        rec = sg.compute_signals(r["fused"], [x for x, _ in r["bm25"][:10]],
                                 [x for x, _ in r["dense"][:10]], e)
        maxdiff = np.maximum(maxdiff, [abs(rec[k] - s[k]) for k in ("S1", "S2", "S3", "S4")])
    print("fusion mismatches", fuse_bad, "| bm25<50:", bm_short, "dense<50:", dn_short)
    print("max |recomputed - stored| S1..S4:", maxdiff)
    print("BM25/dense overlap in top-50: mean %.1f, zero-overlap %.1f%%; top-10 mean %.2f, zero %.1f%%"
          % (np.mean(overlap50), 100 * np.mean(np.array(overlap50) == 0),
             np.mean(overlap10), 100 * np.mean(np.array(overlap10) == 0)))
    # embedding norms (S3 input sanity)
    print("embedding norm mean %.2f min %.2f" % (np.linalg.norm(emb, axis=2).mean(),
                                                np.linalg.norm(emb, axis=2).min()))

    # 3. scaled signals
    raw = np.array([[s[k] for k in ("S1", "S2", "S3", "S4")] for s in sigs])
    mx = 0
    for j, k in enumerate(("S1", "S2", "S3", "S4")):
        v, _ = sg.apply_scaler(raw[:, j], scalers[d][k])
        st = np.array([scaled[(d, s["cluster_id"], s["variant_id"])][k] for s in sigs])
        mx = max(mx, np.abs(v - st).max())
    print("max scaled diff", mx)
    train = set(splits[d]["train"])
    for j, k in enumerate(("S1", "S2", "S3", "S4")):
        tr = raw[[s["cluster_id"] in train for s in sigs], j]
        print(f"  {k}: scaler p5/p95 {scalers[d][k]['p5']:.4f}/{scalers[d][k]['p95']:.4f}  "
              f"recomputed {np.percentile(tr, 5):.4f}/{np.percentile(tr, 95):.4f}  "
              f"raw range {raw[:, j].min():.3f}..{raw[:, j].max():.3f}")
    # split disjointness
    sp = splits[d]
    assert not (set(sp["train"]) & set(sp["val"])) and not (set(sp["train"]) & set(sp["test"])) \
        and not (set(sp["val"]) & set(sp["test"]))
    print("splits disjoint; sizes", {k: len(v) for k, v in sp.items()})

    # 4. outcomes
    rows = rj(out / "outcomes.jsonl")
    gens = {(g["cluster_id"], g["variant_id"], g["action"]): g for g in rj(out / "generations.jsonl")}
    sel = {(s["cluster_id"], s["variant_id"]): s for s in rj(out / "selections.jsonl")}
    ans = {q["cluster_id"]: q["answer"] for q in queries}
    qtext = {(q["cluster_id"], q["variant_id"]): q["question"] for q in queries}
    bad_f1 = bad_lat = bad_doc = 0
    for r in rows:
        k = (r["cluster_id"], r["variant_id"])
        g = gens[k + (r["action"],)]
        a = ac.first_line(g["raw"])
        bad_f1 += (a != r["answer"]) or abs(an.f1_score(a, ans[k[0]]) - r["F1"]) > 1e-12 \
            or abs(an.exact_match(a, ans[k[0]]) - r["EM"]) > 1e-12
        tot = r["retrieval_latency"] + r["rerank_latency"] + r["rewrite_latency"] + r["generation_latency"]
        bad_lat += abs(tot - r["latency_total"]) > 1e-9
        bad_doc += r["doc_ids"] != sel[k][r["action"]]["doc_ids"]
        # the prompt must be built from the ORIGINAL phrasing's question for this instance
        assert ("Question: " + qtext[k] + "\nAnswer:") in g["prompt_key"].encode().decode("unicode_escape") \
            or json.loads(g["prompt_key"])[0]["content"].endswith("Question: " + qtext[k] + "\nAnswer:")
    print("outcome rows", len(rows), "| F1/EM mismatches", bad_f1, "| latency mismatches", bad_lat,
          "| doc mismatches", bad_doc)
    c = Counter(r["action"] for r in rows)
    print("rows per action", dict(c), "| A2 fallbacks", sum(r["fallback"] for r in rows))
    # does each action's prompt actually contain its own docs? (A0 vs A1 identity)
    same_set = same_order = 0
    for k, s in sel.items():
        same_set += set(s["A0"]["doc_ids"]) == set(s["A1"]["doc_ids"])
        same_order += s["A0"]["doc_ids"] == s["A1"]["doc_ids"]
    print(f"A0 vs A1: same 5 docs {same_set}/{len(sel)}, same order {same_order}")
    # cluster completeness and answer-type
    print("answers sample:", Counter(r["answer"][:30] for r in rows if r["action"] == "A0").most_common(5))
    lens = [len(r["answer"].split()) for r in rows]
    print("answer words mean %.2f, >10 words %d, empty %d"
          % (np.mean(lens), sum(l > 10 for l in lens), sum(l == 0 for l in lens)))
    reused = sum(g["reused"] for g in gens.values())
    print("generations reused", reused, "of", len(gens))
