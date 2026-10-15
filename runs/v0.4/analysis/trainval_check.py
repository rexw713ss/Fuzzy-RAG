"""Train + val check of the v0.4 A2 (no test data): headroom and signal AUC for "A2 beats A1".
Run from D:\\Han\\rex_rag\\Fuzzy-RAG:  .venv\\Scripts\\python.exe ..\\data\\v4\\analysis\\trainval_check.py
"""
import json
from pathlib import Path

import numpy as np

from main_logic import baselines as bl
from main_logic import checks as ck
from main_logic import fuzzy as fz
from main_logic import routing as rt

FULL = Path(r"D:\Han\rex_rag\data\full")
V4 = Path(r"D:\Han\rex_rag\data\v4")
D = 0.02


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf8")]


for d in ("nq", "hotpotqa"):
    rows = jl(V4 / d / "outcomes.jsonl")
    t = rt.OutcomeTable.from_rows(rows, actions=rt.ACTIONS_V4)
    old = {(r["cluster_id"], r["variant_id"]): r["F1"] for r in jl(FULL / d / "outcomes.jsonl")
           if r["action"] == "A2"}
    f1o = np.array([old[k] for k in t.instances])
    sc = {(r["cluster_id"], r["variant_id"]): r for r in jl(FULL / "signals_scaled.jsonl")
          if r["dataset"] == d}
    s = np.array([[sc[k][x] for x in fz.SIGNALS] for k in t.instances])
    jit = json.loads((FULL / "jitter.json").read_text())[d]
    w3 = [jit["signals"][x]["width"] for x in fz.SIGNALS_V4]
    b4 = bl.fuzzy_scores(s[:, 1:], w3, fz.SIGNALS_V4)
    beats = t.f1[:, 1] > t.f1[:, 0] + D
    loses = t.f1[:, 1] < t.f1[:, 0] - D
    print(f"\n== {d}: {len(t)} train+val instances")
    print("mean F1 A1 %.3f  new A2 %.3f  old A2 %.3f   oracle{A1,new A2} %.3f  oracle{A1,old A2} %.3f" % (
        t.f1[:, 0].mean(), t.f1[:, 1].mean(), f1o.mean(), t.f1.max(1).mean(),
        np.maximum(t.f1[:, 0], f1o).mean()))
    print("new A2 beats A1: %d, loses: %d (old A2 beats %d)" % (
        beats.sum(), loses.sum(), (f1o > t.f1[:, 0] + D).sum()))
    print("mean latency A1 %.2f  A2 %.2f" % (t.cost[:, 0].mean(), t.cost[:, 1].mean()))
    for name, y in (("A2 beats A1", beats), ("A2 loses to A1", loses)):
        aucs = {x: ck.auc(s[:, j], y) for j, x in enumerate(fz.SIGNALS)}
        aucs["B4(S2-S4)"] = ck.auc(b4, y)
        print(f"AUC {name:15s}", "  ".join(f"{k} {v:.3f}" for k, v in aucs.items()))
