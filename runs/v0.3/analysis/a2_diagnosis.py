"""Why is A2 weak? Read-only check on cached train + val rows (test untouched).

Run from D:\\Han\\rex_rag\\Fuzzy-RAG:
    .venv\\Scripts\\python.exe ..\\data\\full\\analysis\\a2_diagnosis.py
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(r"D:\Han\rex_rag\data\full")
HOTPOT_SRC = Path(r"D:\Han\rex_rag\data\hotpot_dev_fullwiki_v1.json")
USE = {"train", "val"}
D = 0.02


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


splits = json.loads((ROOT / "splits.json").read_text())
gold_titles = None

for d in ("nq", "hotpotqa"):
    split_of = {c: s for s, cs in splits[d].items() for c in cs}
    keep = lambda c: split_of.get(c) in USE
    out = {}
    for r in jl(ROOT / d / "outcomes.jsonl"):
        if keep(r["cluster_id"]):
            out.setdefault((r["cluster_id"], r["variant_id"]), {})[r["action"]] = r
    sel = {(r["cluster_id"], r["variant_id"]): r for r in jl(ROOT / d / "selections.jsonl")
           if keep(r["cluster_id"])}
    clusters = {c["cluster_id"]: c for c in jl(ROOT / d / "clusters.jsonl")}
    keys = [k for k in out if len(out[k]) == 3]
    n = len(keys)
    print(f"\n=== {d}: {n} instances (train + val)")

    f1 = {a: [out[k][a]["F1"] for k in keys] for a in ("A0", "A1", "A2")}
    mean = lambda xs: sum(xs) / len(xs)
    print("mean F1  A0 %.3f  A1 %.3f  A2 %.3f   oracle{A1,A2} %.3f" % (
        mean(f1["A0"]), mean(f1["A1"]), mean(f1["A2"]),
        mean([max(a, b) for a, b in zip(f1["A1"], f1["A2"])])))
    print("A2 > A1 + d: %d   A1 > A2 + d: %d   tie: %d" % (
        sum(b > a + D for a, b in zip(f1["A1"], f1["A2"])),
        sum(a > b + D for a, b in zip(f1["A1"], f1["A2"])),
        sum(abs(a - b) <= D for a, b in zip(f1["A1"], f1["A2"]))))

    fb = sum(sel[k]["A2"]["fallback"] for k in keys)
    same_set = sum(set(sel[k]["A2"]["doc_ids"]) == set(sel[k]["A1"]["doc_ids"]) for k in keys)
    new = [len(set(sel[k]["A2"]["doc_ids"]) - set(sel[k]["A1"]["doc_ids"])) for k in keys]
    print("A2 fallback to A1: %d (%.1f%%)" % (fb, 100 * fb / n))
    print("A2 top-5 == A1 top-5 (as a set): %d (%.1f%%)" % (same_set, 100 * same_set / n))
    print("new passages in A2 top-5 vs A1:", dict(sorted(Counter(new).items())))

    # Where does A1 fail: retrieval (answer string absent from its 5 passages) or generation?
    rec = lambda k, a: out[k][a]["Recall@5"]
    wrong = [k for k in keys if out[k]["A1"]["F1"] < 0.5 and rec(k, "A1") is not None]
    miss = [k for k in wrong if rec(k, "A1") == 0]
    print("A1 wrong (F1 < 0.5, Recall@5 defined): %d;  answer absent from passages: %d (%.0f%%), "
          "present but answered wrong: %d (%.0f%%)" % (
              len(wrong), len(miss), 100 * len(miss) / max(len(wrong), 1),
              len(wrong) - len(miss), 100 * (len(wrong) - len(miss)) / max(len(wrong), 1)))
    rescued = [k for k in miss if rec(k, "A2") == 1]
    print("  of the A1 retrieval misses, A2 finds the answer string: %d (%.0f%%); "
          "A2 then answers right (F1 >= 0.5): %d" % (
              len(rescued), 100 * len(rescued) / max(len(miss), 1),
              sum(out[k]["A2"]["F1"] >= 0.5 for k in rescued)))
    defined = [k for k in keys if rec(k, "A1") is not None]
    print("mean Recall@5  A0 %.3f  A1 %.3f  A2 %.3f  (n=%d, yes/no excluded)" % tuple(
        [mean([rec(k, a) for k in defined]) for a in ("A0", "A1", "A2")] + [len(defined)]))

    if d == "hotpotqa":
        if gold_titles is None:
            src = json.loads(HOTPOT_SRC.read_text(encoding="utf-8"))
            gold_titles = {q["_id"]: {t for t, _ in q["supporting_facts"]} for q in src}
        titles = {p["id"]: p["title"] for p in jl(ROOT / d / "passages.jsonl")}
        for typ in ("bridge", "comparison"):
            ks = [k for k in keys if clusters[k[0]]["type"] == typ]
            print(f"-- {typ}: {len(ks)} instances;  mean F1 A1 %.3f  A2 %.3f" % (
                mean([out[k]["A1"]["F1"] for k in ks]), mean([out[k]["A2"]["F1"] for k in ks])))
            for a in ("A1", "A2"):
                hits = Counter()
                for k in ks:
                    g = gold_titles[clusters[k[0]]["_id"]]
                    got = {titles.get(i) for i in sel[k][a]["doc_ids"]}
                    hits[len(g & got)] += 1
                print(f"   {a} top-5 holds gold supporting pages: 2 of 2: {hits[2]}  "
                      f"1 of 2: {hits[1]}  0 of 2: {hits[0]}")
