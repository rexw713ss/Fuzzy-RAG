"""v0.4 A2 pilot report (REDESIGN_2026-10-15 section 4.3) and the variant choice fixed in
RUNLOG_2026-10-15_v04 section 0: V2 if its pooled F1 gain over A1 beats V1's by >= 0.01,
else V1; pass bar: the chosen variant's pooled gain >= 0.04.

Run from the repo root after scripts/run_v4.sh has finished the pilot for both datasets:
    .venv\\Scripts\\python.exe -m scripts.pilot_report [--pilot DIR] [--full DIR]
Compares A1, the old (v0.3) A2 and the new A2 variants on the pilot instances: F1, EM,
Recall@5, latency, second-hop use; for HotpotQA also how many gold supporting pages the five
passages hold. Writes PILOT/report.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scripts import datasets as ds
from scripts.retrieve_clusters import read_jsonl

MARGIN_V2 = 0.01        # V2 must beat V1's gain by this much (RUNLOG section 0)
PASS_BAR = 0.04         # chosen variant's pooled F1 gain over A1


def key_of(r):
    return (r["cluster_id"], r["variant_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default=None)
    ap.add_argument("--full", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    pilot = Path(args.pilot or ds.DATA / "v4" / "pilot")
    full = Path(args.full or ds.DATA / "full")

    report, pooled = {}, {}
    for d in ds.NAMES:
        rows = read_jsonl(pilot / d / "outcomes.jsonl")
        keys = {key_of(r) for r in rows}
        old = [r for r in read_jsonl(full / d / "outcomes.jsonl")
               if r["action"] == "A2" and key_of(r) in keys]
        by = {}
        for r in rows:
            by.setdefault(r["action"], []).append(r)
        by["A2-old"] = old
        res = {}
        for a, rs in sorted(by.items()):
            rec = [r["Recall@5"] for r in rs if r["Recall@5"] is not None]
            res[a] = {"n": len(rs), "F1": float(np.mean([r["F1"] for r in rs])),
                      "EM": float(np.mean([r["EM"] for r in rs])),
                      "Recall@5": float(np.mean(rec)) if rec else None,
                      "latency": float(np.mean([r["latency_total"] for r in rs]))}
            if a.startswith("A2-V"):
                res[a]["fallback"] = int(sum(r["fallback"] for r in rs))
                res[a]["mean_new_passages"] = float(np.mean([r["n_new"] for r in rs]))
                res[a]["followup_s"] = float(np.mean([r["followup_latency"] for r in rs]))
                res[a]["generation_s"] = float(np.mean([r["generation_latency"] for r in rs]))
            pooled.setdefault(a, []).extend(r["F1"] for r in rs)
        f1 = {a: {key_of(r): r["F1"] for r in rs} for a, rs in by.items()}
        for a in ("A2-V1", "A2-V2", "A2-old"):
            if a in f1:
                res[a]["beats_A1"] = sum(f1[a][k] > f1["A1"][k] + 0.02 for k in keys)
                res[a]["loses_to_A1"] = sum(f1[a][k] < f1["A1"][k] - 0.02 for k in keys)
        if d == "hotpotqa":
            src = json.loads(ds.QUESTIONS[d].read_text(encoding="utf-8"))
            gold = {q["_id"]: {t for t, _ in q["supporting_facts"]} for q in src}
            clusters = {c["cluster_id"]: c for c in read_jsonl(full / d / "clusters.jsonl")}
            titles = {p["id"]: p["title"] for p in read_jsonl(pilot / d / "passages.jsonl")}
            titles.update({p["id"]: p["title"] for p in read_jsonl(full / d / "passages.jsonl")})
            for a, rs in by.items():
                hits = [len(gold[clusters[r["cluster_id"]]["_id"]]
                            & {titles.get(i) for i in r["doc_ids"]}) for r in rs]
                res[a]["gold_pages_2_1_0"] = [hits.count(2), hits.count(1), hits.count(0)]
        report[d] = res

    gain = {v: float(np.mean(pooled[f"A2-{v}"]) - np.mean(pooled["A1"])) for v in ("V1", "V2")}
    chosen = "V2" if gain["V2"] - gain["V1"] >= MARGIN_V2 else "V1"
    report["decision"] = {"pooled_gain_over_A1": gain,
                          "old_A2_gain": float(np.mean(pooled["A2-old"]) - np.mean(pooled["A1"])),
                          "rule": f"V2 if gain(V2) - gain(V1) >= {MARGIN_V2}, else V1; "
                                  f"pass if the chosen gain >= {PASS_BAR}",
                          "chosen": chosen, "passes_bar": gain[chosen] >= PASS_BAR}
    (pilot / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
