"""Audit 1 (Oct 15): recompute the v0.4 outcome rows from the raw phase files. Read-only.

Run from D:\\Han\\rex_rag\\Fuzzy-RAG:
    .venv\\Scripts\\python.exe ..\\data\\audit_2026-10-15\\integrity.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(r"D:\Han\rex_rag\Fuzzy-RAG")))
from main_logic import actions as ac
from main_logic import answers as an
from scripts.outcomes import prompt_passages

DATA = Path(r"D:\Han\rex_rag\data")


def rj(p):
    p = Path(p)
    if not p.exists():
        return []
    with open(p, encoding="utf8") as f:
        return [json.loads(l) for l in f if l.strip()]


def k2(r):
    return (r["cluster_id"], r["variant_id"])


SETS = {("nq", "old"): (DATA / "full/nq", DATA / "v4/nq"),
        ("hotpotqa", "old"): (DATA / "full/hotpotqa", DATA / "v4/hotpotqa"),
        ("nq", "fresh"): (DATA / "v4/fresh_src/nq", DATA / "v4/fresh/nq"),
        ("hotpotqa", "fresh"): (DATA / "v4/fresh_src/hotpotqa", DATA / "v4/fresh/hotpotqa")}

timing = {}
for (d, name), (src, out) in SETS.items():
    print(f"\n===== {d} / {name}")
    queries = {k2(q): q for q in rj(src / "queries.jsonl")}
    initial = {k2(r): r for r in rj(src / "retrieval.jsonl")}
    a1 = {k2(r): r for r in rj(out / "a1.jsonl")}
    fol = {k2(r): r for r in rj(out / "followups.jsonl")}
    sec = {k2(r): r for r in rj(out / "second_retrieval.jsonl")}
    sel = {k2(r): r for r in rj(out / "selections.jsonl")}
    gens = {k2(r) + (r["action"],): r for r in rj(out / "generations.jsonl")}
    texts = {r["id"]: (r["title"], r["text"]) for r in rj(out / "passages.jsonl")}
    rows = rj(out / "outcomes.jsonl")
    old_a1 = {k2(r): r for r in rj(src / "outcomes.jsonl") if r["action"] == "A1"}
    old_sel = {k2(r): r for r in rj(src / "selections.jsonl")}

    by = {}
    for r in rows:
        by.setdefault(k2(r), {})[r["action"]] = r
    c = Counter(tuple(sorted(v)) for v in by.values())
    print("instances", len(by), "| action sets", dict(c), "| queries in src", len(queries))
    dup = len(rows) - sum(len(v) for v in by.values())
    print("duplicate rows", dup)

    bad = Counter()
    for k, acts in by.items():
        q = queries[k]
        r1, r2 = acts["A1"], acts["A2"]
        # --- A1
        if a1[k]["reused"]:
            o = old_a1[k]
            bad["A1 reused row != v0.3 row"] += any(r1[x] != o[x] for x in
                                                    ("doc_ids", "answer", "F1", "EM", "latency_total"))
            bad["A1 reused docs != v0.3 selection"] += r1["doc_ids"] != old_sel[k]["A1"]["doc_ids"]
        else:
            g = gens[k + ("A1",)]
            msgs = ac.answer_messages(q["question"], prompt_passages(a1[k]["doc_ids"], texts))
            bad["A1 prompt mismatch"] += json.dumps(msgs, ensure_ascii=False) != g["prompt_key"]
            top = [a1[k]["pool"][i] for i in ac.rerank_order(a1[k]["ce_scores"])[:5]]
            bad["A1 top5 != CE order"] += top != a1[k]["doc_ids"]
            ans = ac.first_line(g["raw"])
            bad["A1 F1/EM"] += (ans != r1["answer"] or abs(an.f1_score(ans, q["answer"]) - r1["F1"]) > 1e-12
                                or abs(an.exact_match(ans, q["answer"]) - r1["EM"]) > 1e-12)
            tot = r1["retrieval_latency"] + r1["rerank_latency"] + r1["generation_latency"]
            bad["A1 latency sum"] += abs(tot - r1["latency_total"]) > 1e-9
            bad["A1 pool != fused top20"] += a1[k]["pool"] != [x for x, _ in initial[k]["fused"][:20]]
        bad["A1 docs != a1.jsonl"] += r1["doc_ids"] != a1[k]["doc_ids"]
        # --- A2
        s, f = sel[k], fol[k]
        bad["followup question mismatch"] += f["question"] != q["question"]
        bad["followup usable flag"] += f["usable"] != ac.rewrite_usable(q["question"], f["followup"])
        if f["usable"]:
            bad["second-hop query != followup"] += sec[k]["followup"] != f["followup"]
            bad["selection marked fallback though usable"] += s["fallback"]
            ranked = [s["second_pool"][i] for i in ac.rerank_order(s["second_ce_scores"])]
            ctx, n_new = ac.second_hop_context(a1[k]["doc_ids"], ranked)
            bad["A2 context rebuild"] += ctx != s["doc_ids"] or n_new != s["n_new"]
            bad["second pool != fused top20"] += s["second_pool"] != [x for x, _ in sec[k]["fused"][:20]]
        else:
            bad["fallback ctx != A1"] += s["doc_ids"] != a1[k]["doc_ids"]
        g = gens[k + ("A2-V1",)]
        msgs = ac.answer_messages(q["question"], prompt_passages(s["doc_ids"], texts))
        bad["A2 prompt mismatch"] += json.dumps(msgs, ensure_ascii=False) != g["prompt_key"]
        ans = ac.first_line(g["raw"])
        bad["A2 F1/EM"] += (ans != r2["answer"] or abs(an.f1_score(ans, q["answer"]) - r2["F1"]) > 1e-12
                            or abs(an.exact_match(ans, q["answer"]) - r2["EM"]) > 1e-12)
        bad["A2 docs != selection"] += r2["doc_ids"] != s["doc_ids"]
        tot = r2["retrieval_latency"] + r2["rerank_latency"] + r2["followup_latency"] + r2["generation_latency"]
        bad["A2 latency sum"] += abs(tot - r2["latency_total"]) > 1e-9
        exp_ret = initial[k]["retrieval_latency"] + (sec[k]["retrieval_latency"] if f["usable"] else 0)
        bad["A2 retrieval latency"] += abs(exp_ret - r2["retrieval_latency"]) > 1e-9
        bad["A2 followup latency"] += abs(f["seconds"] - r2["followup_latency"]) > 1e-9
        bad["A2 rerank latency"] += abs(a1[k]["rerank_latency"] + s["second_rerank_latency"]
                                        - r2["rerank_latency"]) > 1e-9
        # A2 paid for A1's generation? (must not)
        bad["A2 gen latency == A1 gen latency (not fallback)"] += (
            f["usable"] and abs(r2["generation_latency"] - r1["generation_latency"]) < 1e-12)
    print("mismatch counts (all should be 0):", {k: v for k, v in bad.items()} or "none")

    # generation reuse and how the A2 generation time was measured
    g2 = [gens[k + ("A2-V1",)] for k in by]
    print("A2 generations reused from cache:", sum(g["reused"] for g in g2), "of", len(g2),
          "| fallbacks:", sum(not fol[k]["usable"] for k in by))
    print("A2 context: n_new distribution", dict(Counter(sel[k]["n_new"] for k in by)))
    # identical A1/A2 prompts (A2 context == A1 context) though not a fallback
    same = sum(sel[k]["doc_ids"] == a1[k]["doc_ids"] for k in by)
    print("A2 context identical to A1's (incl. fallbacks):", same)

    # speed comparability: generation seconds per prompt token, A1 vs A2
    def rate(rs):
        return np.array([r["generation_latency"] / max(r["input_tokens"], 1) for r in rs])
    A1s = [by[k]["A1"] for k in by]
    A2gen = [gens[k + ("A2-V1",)] for k in by if not gens[k + ("A2-V1",)]["reused"]]
    t1 = np.array([r["generation_latency"] for r in A1s])
    t2 = np.array([g["seconds"] for g in A2gen])
    in1 = np.array([r["input_tokens"] for r in A1s])
    in2 = np.array([g["input_tokens"] for g in A2gen])
    out1 = np.array([r["output_tokens"] for r in A1s])
    out2 = np.array([g["output_tokens"] for g in A2gen])
    # least-squares fit of seconds = a + b*in + c*out per group, to compare server speed
    def fit(t, i, o):
        X = np.c_[np.ones_like(i), i, o]
        return np.linalg.lstsq(X, t, rcond=None)[0]
    print("A1 gen: mean %.2f s, in %.0f, out %.1f; fit a,b(ms/in tok),c(ms/out tok) = %s"
          % (t1.mean(), in1.mean(), out1.mean(), np.round(fit(t1, in1, out1) * [1, 1000, 1000], 2)))
    print("A2 gen (non-reused): mean %.2f s, in %.0f, out %.1f; fit = %s"
          % (t2.mean(), in2.mean(), out2.mean(), np.round(fit(t2, in2, out2) * [1, 1000, 1000], 2)))
    fs = np.array([fol[k]["seconds"] for k in by])
    print("follow-up call: mean %.2f s, out tokens %.1f; A2 total %.2f vs A1 %.2f"
          % (fs.mean(), np.mean([fol[k]["output_tokens"] or 0 for k in by]),
             np.mean([by[k]["A2"]["latency_total"] for k in by]), t1.mean() + 0))
    print("mean F1 A1 %.3f A2 %.3f" % (np.mean([by[k]["A1"]["F1"] for k in by]),
                                         np.mean([by[k]["A2"]["F1"] for k in by])))
    # follow-up restates the question? token-overlap share of the follow-up found in the question
    def toks(x):
        return set(an.normalize(x).split()) if hasattr(an, "normalize") else set(x.lower().split())
    ov = [len(toks(fol[k]["followup"] or "") & toks(queries[k]["question"])) / max(len(toks(fol[k]["followup"] or "")), 1)
          for k in by]
    print("follow-up tokens also in the question: median %.2f, share >= 0.8: %.1f%%"
          % (np.median(ov), 100 * np.mean(np.array(ov) >= 0.8)))
