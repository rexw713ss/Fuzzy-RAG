"""Audit 2 (Oct 14): is the 'research failed' conclusion valid? Read-only on data\\full.

Uses train + val only for anything signal-related; the test split appears only in the
power calculation, which uses always-A* policies (no fitting).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(r"D:\Han\rex_rag\Fuzzy-RAG")))
from main_logic import checks as ck

ROOT = Path(r"D:\Han\rex_rag\data\full")
SIG = ("S1", "S2", "S3", "S4")
DELTA = 0.02
rng = np.random.default_rng(0)


def rj(p):
    with open(p, encoding="utf8") as f:
        return [json.loads(l) for l in f if l.strip()]


def icc(values, groups):
    """One-way ICC(1): share of variance that is between clusters (complete clusters only)."""
    g = {}
    for v, c in zip(values, groups):
        g.setdefault(c, []).append(v)
    arr = np.array([x for x in g.values() if len(x) == 3], dtype=float)
    k, n = 3, len(arr)
    msb = k * ((arr.mean(1) - arr.mean()) ** 2).sum() / (n - 1)
    msw = ((arr - arr.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw)


scaled = {(r["dataset"], r["cluster_id"], r["variant_id"]): r for r in rj(ROOT / "signals_scaled.jsonl")}
splits = json.loads((ROOT / "splits.json").read_text())

for d in ("nq", "hotpotqa"):
    print(f"\n===== {d}")
    rows = rj(ROOT / d / "outcomes.jsonl")
    sel = {(s["cluster_id"], s["variant_id"]): s for s in rj(ROOT / d / "selections.jsonl")}
    by = {}
    for r in rows:
        by.setdefault((r["cluster_id"], r["variant_id"]), {})[r["action"]] = r
    keys = sorted(by)
    split_of = {c: s for s, cs in splits[d].items() for c in cs}
    F = np.array([[by[k][a]["F1"] for a in ("A0", "A1", "A2")] for k in keys])
    L = np.array([[by[k][a]["latency_total"] for a in ("A0", "A1", "A2")] for k in keys])
    G = np.array([[by[k][a]["generation_latency"] for a in ("A0", "A1", "A2")] for k in keys])
    T = np.array([[by[k][a]["input_tokens"] for a in ("A0", "A1", "A2")] for k in keys])
    cl = np.array([k[0] for k in keys])
    S = np.array([[scaled[(d, *k)][s] for s in SIG] for k in keys])
    tv = np.array([split_of[c] in ("train", "val") for c in cl])

    # --- 1. Is generation deterministic w.r.t. passage ORDER? A1 vs A2 with the same 5 docs.
    same_set_diff_order = [(i, k) for i, k in enumerate(keys)
                           if set(sel[k]["A1"]["doc_ids"]) == set(sel[k]["A2"]["doc_ids"])
                           and sel[k]["A1"]["doc_ids"] != sel[k]["A2"]["doc_ids"]]
    idx = [i for i, _ in same_set_diff_order]
    if idx:
        dif = np.abs(F[idx, 1] - F[idx, 2]) > DELTA
        print(f"[order noise] A1/A2 same docs, different order: n={len(idx)}, "
              f"F1 differs by >delta in {dif.mean():.1%}")
    same_exact = [i for i, k in enumerate(keys) if sel[k]["A1"]["doc_ids"] == sel[k]["A2"]["doc_ids"]]
    print(f"[order noise] A1/A2 identical prompt: n={len(same_exact)}, F1 differs "
          f"{(np.abs(F[same_exact, 1] - F[same_exact, 2]) > 1e-12).mean():.1%} (should be 0: reused)")

    # --- 2. Are the routing TARGETS stable across paraphrases? (ICC of per-instance gains)
    gains = {"A1-A0": F[:, 1] - F[:, 0], "A2-A1": F[:, 2] - F[:, 1], "best-A0": F.max(1) - F[:, 0],
             "F1(A0)": F[:, 0]}
    print("[target ICC] share of variance between questions (1 = paraphrase-invariant):")
    for n, v in gains.items():
        print(f"   {n:8s} ICC {icc(v, cl):.2f}")
    # how often does the oracle action agree across the 3 phrasings?
    best = F.max(1, keepdims=True)
    orc = (F >= best - DELTA - 1e-12).argmax(1)
    agree = []
    for c in np.unique(cl):
        o = orc[cl == c]
        if len(o) == 3:
            agree.append(len(set(o)) == 1)
    print(f"[target] oracle action identical across all 3 phrasings: {np.mean(agree):.1%} of clusters")

    # --- 3. Are the SIGNALS stable across paraphrases?
    print("[signal ICC]", {s: round(icc(S[:, j], cl), 2) for j, s in enumerate(SIG)})

    # --- 4. Signal -> target AUC on train+val (independent re-implementation check)
    tgt = {"A1>A0": F[:, 1] > F[:, 0] + DELTA, "A0>A1": F[:, 0] > F[:, 1] + DELTA,
           "A2>A1": F[:, 2] > F[:, 1] + DELTA, "esc": orc != 0}
    for n, y in tgt.items():
        a = [ck.auc(S[tv, j], y[tv]) for j in range(4)]
        print(f"[AUC tv] {n:6s} pos={int(y[tv].sum()):3d} " + " ".join(f"{s}={v:.2f}" for s, v in zip(SIG, a)))

    # --- 5. Cost noise. Generation latency vs input tokens; residual SD vs action gaps.
    g, t = G.ravel(), T.ravel()
    coef = np.polyfit(t, g, 1)
    res = g - np.polyval(coef, t)
    print(f"[cost] gen latency = {coef[1]:.2f} + {coef[0]*1000:.2f} ms/token; R2 "
          f"{1 - res.var() / g.var():.2f}; residual SD {res.std():.2f} s")
    print(f"[cost] mean gen latency A0/A1/A2: {G.mean(0).round(2)}; mean input tokens {T.mean(0).round(0)}")
    print(f"[cost] mean total A0/A1/A2: {L.mean(0).round(3)}; per-instance SD of (A1-A0) "
          f"{(L[:, 1] - L[:, 0]).std():.2f} s vs mean {(L[:, 1] - L[:, 0]).mean():.3f} s")
    print(f"[cost] share of instances where A1 is CHEAPER than A0: {(L[:, 1] < L[:, 0]).mean():.1%}")
    # rerank (true marginal) cost
    rr = np.array([by[k]["A1"]["rerank_latency"] for k in keys])
    rw = np.array([by[k]["A2"]["rewrite_latency"] for k in keys])
    print(f"[cost] mean rerank {rr.mean():.3f} s, rewrite {rw.mean():.2f} s")

    # --- 6. Power: with 60 test clusters, what F1 difference is detectable?
    te = ~tv
    def paired_sd(diff):
        cs = [diff[te & (cl == c)].mean() for c in np.unique(cl[te])]
        return np.std(cs, ddof=1), len(cs)
    for n, diff in (("A1-A0", F[:, 1] - F[:, 0]), ("A2-A1", F[:, 2] - F[:, 1]),
                    ("oracle-A1", F[np.arange(len(F)), orc] - F[:, 1])):
        sd, m = paired_sd(diff)
        print(f"[power] {n:9s} test mean {diff[te].mean():+.3f}; cluster SD {sd:.3f}; "
              f"80%-power MDE ~ {2.8 * sd / np.sqrt(m):.3f} F1")

    # --- 7. Upper bound for any router at the always-A1 budget, if it chose perfectly among
    # A0/A1 only vs with A2 (oracle restricted to cost <= always-A1, greedy by gain/cost).
    base_cost = L[:, 1].mean()
    for n, acts in (("A0/A1 oracle", [0, 1]),):
        f = F[:, acts].max(1).mean()
        print(f"[headroom] {n}: F1 {f:.3f} vs always-A1 {F[:, 1].mean():.3f} (all instances)")
    # abstentions
    ab = np.array([any(w in by[k]["A0"]["answer"].lower() for w in
                       ("not specified", "not mentioned", "no answer", "no information", "not provided",
                        "unknown", "not stated", "cannot")) for k in keys])
    print(f"[answers] A0 abstains on {ab.mean():.1%}; F1(A1)-F1(A0) on abstentions "
          f"{(F[ab, 1] - F[ab, 0]).mean():+.3f}, F1(A2)-F1(A0) {(F[ab, 2] - F[ab, 0]).mean():+.3f}")
