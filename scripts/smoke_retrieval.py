"""End-to-end retrieval smoke test: BM25 + Contriever -> fusion -> S1-S4 (spec v0.3, 2, 7.2).

Why: before stage 1 builds paraphrase clusters on top of retrieval, check that every piece works
together on real questions: both retrievers find answers at roughly their published rates, the
two id spaces line up, fusion and all four signals compute on real data, and no signal is
degenerate. Uses dev questions (NQ-open, or HotpotQA with --dataset hotpotqa); these are NOT the
experiment's sample (spec v0.3 §9.3).

Run from the repo root, after scripts.bm25_search has written BM25 hits for the same questions:
    .venv\\Scripts\\python.exe -m scripts.smoke_retrieval BM25.jsonl QUESTIONS OUTDIR [--dataset hotpotqa]

HotpotQA's yes/no answers (comparison questions) cannot be found in a passage by string match, so
those questions are left out of Recall@k and the hard/easy split, and counted.

The signal summary previews the section 8.2 stop conditions on RAW (unscaled) signals for single
questions. It is not stage 1, which needs paraphrase clusters and fitted scalers.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from main_logic import answers as an
from main_logic import corpus as cp
from main_logic import dense as dn
from main_logic import signals as sg
from scripts import datasets as ds

KS = (1, 5, 10, 20, 50)
YES_NO = ({"yes"}, {"no"})
SIGNALS = ("S1", "S2", "S3", "S4")


def ranks(x):
    """Average ranks (1-based), ties sharing their mean rank."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def auc(score, positive):
    """P(score of a random positive > score of a random negative), ties counted half."""
    positive = np.asarray(positive, dtype=bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks(score)[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bm25")
    ap.add_argument("questions")
    ap.add_argument("outdir")
    ap.add_argument("--dataset", choices=ds.NAMES, default="nq")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    timing = {}

    with open(args.bm25, encoding="utf8") as f:
        bm25 = [json.loads(line) for line in f]
    gold = cp.load_questions(args.questions, len(bm25))
    for b, g in zip(bm25, gold):
        if b["question"] != g["question"]:
            raise ValueError(f"question mismatch at qid {b['qid']}")
    questions = [b["question"] for b in bm25]

    # Dense retrieval: encode all questions, then one exact pass over all 21M passages.
    t = time.time()
    encoder = dn.load_encoder()
    qemb = dn.encode(questions, encoder)
    timing["encode_questions_s"] = time.time() - t
    del encoder

    t = time.time()
    index = ds.load_index(args.dataset)
    timing["load_index_s"] = time.time() - t

    t = time.time()
    dense = index.search(qemb, k=sg.POOL)
    timing["dense_search_s"] = time.time() - t
    timing["dense_search_ms_per_question"] = timing["dense_search_s"] / len(questions) * 1000

    # Fusion and signals, exactly as signals.py defines them.
    t = time.time()
    rows = []
    for b, d in zip(bm25, dense):
        bm25_top = {doc: s for doc, s in b["hits"]}
        dense_top = {doc: s for doc, s in d}
        fused = sg.fuse(bm25_top, dense_top)
        emb = index.vectors([doc for doc, _ in fused[:sg.K]])
        sig = sg.compute_signals(fused, [doc for doc, _ in b["hits"][:sg.K]],
                                 [doc for doc, _ in d[:sg.K]], emb)
        rows.append({"qid": b["qid"], "question": b["question"], "bm25": b["hits"],
                     "dense": d, "fused": fused, "signals": sig})
    timing["fuse_and_signals_s"] = time.time() - t
    del index

    # Passage texts for every retrieved id, in one pass over the corpus file.
    t = time.time()
    needed = {doc for r in rows for key in ("bm25", "dense", "fused") for doc, _ in r[key]}
    texts = cp.load_passages(ds.CORPUS[args.dataset], needed)
    timing["read_texts_s"] = time.time() - t

    # Yes/no answers cannot be string-matched in a passage; leave them out of recall.
    scored = [{a.lower() for a in g["answer"]} not in YES_NO for g in gold]
    recall = {}
    for key in ("bm25", "dense", "fused"):
        flags = [[an.has_answer(g["answer"], texts[doc][1]) for doc, _ in r[key]]
                 for r, g, ok in zip(rows, gold, scored) if ok]
        recall[key] = an.recall_at(flags, KS)
        if key == "fused":
            for r, fl in zip([r for r, ok in zip(rows, scored) if ok], flags):
                r["fused_hit_at_5"] = any(fl[:5])

    # Signal summary: a preview of the section 8.2 stop conditions on raw values.
    S = np.array([[r["signals"][s] for s in SIGNALS] for r, ok in zip(rows, scored) if ok])
    hard = np.array([not r["fused_hit_at_5"] for r, ok in zip(rows, scored) if ok])
    summary = {}
    for j, name in enumerate(SIGNALS):
        v = S[:, j]
        _, counts = np.unique(np.round(v, 12), return_counts=True)
        summary[name] = {
            "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
            "sd": float(v.std()), "non_finite": int((~np.isfinite(v)).sum()),
            "most_common_share": float(counts.max() / len(v)),
            "mean_easy": float(v[~hard].mean()), "mean_hard": float(v[hard].mean()),
            "auc_hard": auc(v, hard),
        }
    pearson = np.corrcoef(S.T)
    spearman = np.corrcoef(np.array([ranks(S[:, j]) for j in range(4)]))

    report = {
        "dataset": args.dataset, "n_questions": len(rows),
        "n_scored": int(sum(scored)), "n_yes_no_skipped": int(len(rows) - sum(scored)),
        "source": f"{Path(args.questions).name}, first n (smoke test only)",
        "dense": {"model": dn.MODEL, "revision": dn.REVISION, "search": "exact inner product",
                  "index": ds.EMBEDDINGS[args.dataset].name},
        "bm25": {"k1": 0.9, "b": 0.4, "index": ds.BM25_INDEX[args.dataset].name},
        "recall": recall, "share_hard_fused_at_5": float(hard.mean()),
        "signals_raw": summary,
        "pearson": {f"{a}-{b}": float(pearson[i, j]) for i, a in enumerate(SIGNALS)
                    for j, b in enumerate(SIGNALS) if i < j},
        "spearman": {f"{a}-{b}": float(spearman[i, j]) for i, a in enumerate(SIGNALS)
                     for j, b in enumerate(SIGNALS) if i < j},
        "timing": timing,
    }
    with open(out / "retrieval.jsonl", "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
