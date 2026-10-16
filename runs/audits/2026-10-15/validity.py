"""Audit 2 (Oct 15): does step 9 reproduce, and is the v0.4 'routing = random mix' result valid?
Read-only. Run from D:\\Han\\rex_rag\\Fuzzy-RAG:
    .venv\\Scripts\\python.exe ..\\data\\audit_2026-10-15\\validity.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(r"D:\Han\rex_rag\Fuzzy-RAG")))
from main_logic import baselines as bl
from main_logic import checks as ck
from main_logic import evaluation as ev
from main_logic import fuzzy as fz
from main_logic import learned as lr
from main_logic import routing as rt
from scripts import evaluate_v4 as e4

DATA = Path(r"D:\Han\rex_rag\data")
FULL, V4 = DATA / "full", DATA / "v4"
DELTA = 0.02
rng = np.random.default_rng(0)


def rj(p):
    with open(p, encoding="utf8") as f:
        return [json.loads(l) for l in f if l.strip()]


def k2(r):
    return (r["cluster_id"], r["variant_id"])


def icc(values, groups):
    g = {}
    for v, c in zip(values, groups):
        g.setdefault(c, []).append(v)
    arr = np.array([x for x in g.values() if len(x) == 3], dtype=float)
    n = len(arr)
    msb = 3 * ((arr.mean(1) - arr.mean()) ** 2).sum() / (n - 1)
    msw = ((arr - arr.mean(1, keepdims=True)) ** 2).sum() / (n * 2)
    return (msb - msw) / (msb + 2 * msw)


def cluster_boot(x, groups, B=2000):
    """mean and 95% cluster-bootstrap CI of per-instance values x."""
    cl = {}
    for v, c in zip(x, groups):
        cl.setdefault(c, []).append(v)
    sums = np.array([sum(v) for v in cl.values()])
    ns = np.array([len(v) for v in cl.values()])
    idx = rng.integers(0, len(sums), (B, len(sums)))
    means = sums[idx].sum(1) / ns[idx].sum(1)
    return float(np.mean(x)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


for d in ("nq", "hotpotqa"):
    print(f"\n================ {d}")
    saved = json.loads((V4 / "results" / f"{d}.json").read_text())

    # ---------- 1. reproduce step 9 in memory
    report, *_ = e4.evaluate_dataset(d, FULL, V4)
    new = json.loads(json.dumps(report, default=float))
    diffs = []
    for s in ("fresh", "old_test"):
        for n, m in saved[s]["results"].items():
            for x in ("F1", "cost"):
                if abs(m[x]["mean"] - new[s]["results"][n][x]["mean"]) > 1e-9:
                    diffs.append((s, n, x))
            if m["counts"] != new[s]["results"][n]["counts"]:
                diffs.append((s, n, "counts"))
    print("[reproduce] differences vs saved results:", diffs or "none",
          "| B4 theta", new["fits"]["B4"]["theta"])

    # ---------- shared setup (same inputs as evaluate_v4)
    splits = json.loads((FULL / "splits.json").read_text())[d]
    split_of = {c: s for s, cs in splits.items() for c in cs}
    rows = rj(V4 / d / "outcomes.jsonl")
    frows = rj(V4 / "fresh" / d / "outcomes.jsonl")
    scaled = {k2(r): r for r in rj(FULL / "signals_scaled.jsonl") if r["dataset"] == d}
    scalers = json.loads((FULL / "scalers.json").read_text())[d]
    fsc, _ = e4.scale_fresh(rj(V4 / "fresh_src" / d / "signals.jsonl"), scalers)
    scaled.update(fsc)
    jit = json.loads((FULL / "jitter.json").read_text())[d]
    w3 = [jit["signals"][s]["width"] for s in fz.SIGNALS_V4]
    sig3 = {k: tuple(v[s] for s in fz.SIGNALS_V4) for k, v in scaled.items()}
    qs = {k2(q): q["question"] for q in rj(FULL / d / "queries.jsonl")}
    qs.update({k2(q): q["question"] for q in rj(V4 / "fresh_src" / d / "queries.jsonl")})
    qtype = {}
    if d == "hotpotqa":
        for p in (FULL / d / "clusters.jsonl", V4 / "fresh_src" / d / "clusters.jsonl"):
            qtype.update({c["cluster_id"]: c["type"] for c in rj(p)})
    keys = {k2(r) for r in rows}
    T = {s: e4.table_for(rows, [k for k in keys if split_of[k[0]] == s]) for s in ("train", "val", "test")}
    T["tv"] = e4.table_for(rows, [k for k in keys if split_of[k[0]] in ("train", "val")])
    T["fresh"] = e4.table_for(frows, {k2(r) for r in frows})

    # ---------- 2. independent B4 fit (own grid loop, fz.escalate directly)
    def b4(t):
        return np.array([fz.escalate(sig3[k], w3, signals=fz.SIGNALS_V4)[0] for k in t.instances])
    val = T["val"]
    budget = (val.cost[:, 0].mean() + val.cost[:, 1].mean()) / 2
    sv = b4(val)
    best = None
    for i in range(101):
        th = i / 100
        a = (sv >= th).astype(int)
        q, c = val.f1[np.arange(len(a)), a].mean(), val.cost[np.arange(len(a)), a].mean()
        if c <= budget and (best is None or (-q, c, th) < (-best[0], best[1], best[2])):
            best = (q, c, th)
    fr = T["fresh"]
    sf = b4(fr)
    a = (sf >= best[2]).astype(int)
    print("[independent B4] theta %.2f (saved %.2f); fresh F1 %.3f cost %.2f, A2 share %.2f"
          % (best[2], saved["fits"]["B4"]["theta"], fr.f1[np.arange(len(a)), a].mean(),
             fr.cost[np.arange(len(a)), a].mean(), a.mean()))
    print("   B4 score range: val %.3f..%.3f (distinct %d), fresh %.3f..%.3f"
          % (sv.min(), sv.max(), len(np.unique(sv.round(6))), sf.min(), sf.max()))

    # ---------- 3. is "A2 beats A1" a property of the QUESTION (routable) or of the phrasing (noise)?
    for name in ("tv", "fresh"):
        t = T[name]
        cl = [k[0] for k in t.instances]
        gain = t.f1[:, 1] - t.f1[:, 0]
        win = (gain > DELTA).astype(float)
        print(f"[{name}] ICC across paraphrases: gain {icc(gain, cl):.2f}, win {icc(win, cl):.2f}, "
              f"F1(A1) {icc(t.f1[:, 0], cl):.2f}; wins {int(win.sum())}, losses "
              f"{int((gain < -DELTA).sum())} of {len(t)}")

    # ---------- 4. how much of the oracle headroom is reachable by ANY question-level router?
    # sibling router: escalate an instance iff its two paraphrases' mean gain > delta, i.e. a
    # perfect predictor of the question's effect estimated from other phrasings (an upper bound
    # for routing on the question, not on noise). Complete clusters of the fresh set.
    for name in ("tv", "fresh"):
        t = T[name]
        idx = {}
        for i, k in enumerate(t.instances):
            idx.setdefault(k[0], []).append(i)
        comp = [r for r in idx.values() if len(r) == 3]
        sel = np.concatenate(comp)
        gain = t.f1[:, 1] - t.f1[:, 0]
        a_sib = np.zeros(len(t), int)
        a_clu = np.zeros(len(t), int)
        for r in comp:
            for i in r:
                others = [j for j in r if j != i]
                a_sib[i] = gain[others].mean() > DELTA
            a_clu[r] = gain[r].mean() > DELTA
        f = lambda a: (t.f1[sel, a[sel]].mean(), t.cost[sel, a[sel]].mean(), a[sel].mean())
        orc = (gain > DELTA).astype(int)
        c1, c2 = t.cost[sel, 0].mean(), t.cost[sel, 1].mean()
        def mix(c):
            lam = np.clip((c - c1) / (c2 - c1), 0, 1)
            return (1 - lam) * t.f1[sel, 0].mean() + lam * t.f1[sel, 1].mean()
        print(f"[{name} headroom, {len(sel)} inst] always-A1 {t.f1[sel,0].mean():.3f}, always-A2 "
              f"{t.f1[sel,1].mean():.3f}")
        for lab, a in (("instance oracle", orc), ("cluster-mean oracle (in-sample)", a_clu),
                       ("sibling router (out-of-sample)", a_sib)):
            q, c, p = f(a)
            print(f"   {lab:32s} F1 {q:.3f} cost {c:.2f} A2 {p:.0%} | random mix at that cost {mix(c):.3f}"
                  f" -> gain over mix {q - mix(c):+.3f}")

    # ---------- 5. predictive check on the fresh set (diagnostic; the routers are already fixed)
    t = T["fresh"]
    y = t.f1[:, 1] > t.f1[:, 0] + DELTA + 1e-12
    m = bl.signal_matrix(t, sig3, 3)
    model = lr.fit([qs[k] for k in T["train"].instances], ev.oracle_actions(T["train"]), n_classes=2)
    qo = lr.scores(model, [qs[k] for k in t.instances])
    aucs = {s: ck.auc(m[:, j], y) for j, s in enumerate(fz.SIGNALS_V4)}
    aucs["B4"] = ck.auc(sf, y)
    aucs["query-only"] = ck.auc(qo, y)
    if qtype:
        isbridge = np.array([qtype[k[0]] == "bridge" for k in t.instances], float)
        aucs["is-bridge"] = ck.auc(isbridge, y)
    print("[fresh AUC 'A2 beats A1'] positives %d/%d: " % (y.sum(), len(y))
          + ", ".join(f"{k} {v:.2f}" for k, v in aucs.items()))
    # bootstrap CI of B4's AUC on fresh
    cl = np.array([k[0] for k in t.instances])
    uc = np.unique(cl)
    boots = []
    for _ in range(500):
        pick = rng.choice(uc, len(uc))
        ii = np.concatenate([np.flatnonzero(cl == c) for c in pick])
        if 0 < y[ii].sum() < len(ii):
            boots.append(ck.auc(sf[ii], y[ii]))
    print("   B4 AUC 95%% CI [%.2f, %.2f]" % tuple(np.percentile(boots, [2.5, 97.5])))

    if qtype:
        for name in ("tv", "fresh"):
            t = T[name]
            ty = np.array([qtype[k[0]] for k in t.instances])
            for tt in ("bridge", "comparison"):
                s = ty == tt
                g = t.f1[s, 1] - t.f1[s, 0]
                print(f"   [{name}] {tt:10s} n={s.sum():4d}: A1 {t.f1[s,0].mean():.3f} A2 "
                      f"{t.f1[s,1].mean():.3f} gain {g.mean():+.3f}, A2 wins {np.mean(g > DELTA):.0%}")
            # a 'type router': escalate bridge only
            a = (ty == "bridge").astype(int)
            r_ = np.arange(len(t))
            print(f"   [{name}] route bridge->A2, comparison->A1: F1 {t.f1[r_, a].mean():.3f}, cost "
                  f"{t.cost[r_, a].mean():.2f} (always-A2 {t.f1[:,1].mean():.3f} at {t.cost[:,1].mean():.2f})")

    # ---------- 6. power on the fresh set
    groups = [k[0] for k in t.instances] if False else None
    t = T["fresh"]
    cl = [k[0] for k in t.instances]
    a = (sf >= best[2]).astype(int)
    f1_b4 = t.f1[np.arange(len(t)), a]
    lam = np.clip((t.cost[np.arange(len(t)), a].mean() - t.cost[:, 0].mean())
                  / (t.cost[:, 1].mean() - t.cost[:, 0].mean()), 0, 1)
    mixv = (1 - lam) * t.f1[:, 0] + lam * t.f1[:, 1]
    diff = f1_b4 - mixv
    cs = {}
    for v, c in zip(diff, cl):
        cs.setdefault(c, []).append(v)
    cm = np.array([np.mean(v) for v in cs.values()])
    sd = cm.std(ddof=1)
    print(f"[power fresh] B4 - mix {diff.mean():+.3f}; cluster SD {sd:.3f}, {len(cm)} clusters; "
          f"80%-power MDE ~ {2.8 * sd / np.sqrt(len(cm)):.3f} F1")

    # ---------- 7. flip-rate context: a random router with B4's A2 share
    p = a.mean()
    print(f"[flip context] B4 escalates {p:.0%} on fresh; a random router at that share flips "
          f"{2 * p * (1 - p):.0%} of pairs; B4 RFR {saved['fresh']['results']['B4']['RFR']['mean']:.2f}")

    # ---------- 8. timing consistency (old splits: A1 from the v0.3 server session, A2 from v0.4)
    print("[budget] val always-A1 %.2f s, always-A2 %.2f s, midpoint %.2f s" %
          (val.cost[:, 0].mean(), val.cost[:, 1].mean(), budget))
