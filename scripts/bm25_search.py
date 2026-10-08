"""Run BM25 for a file of questions and write the top-k hits (spec v0.3, sections 2.1, 7.2).

Runs in the BM25 environment, from the repo root:
    .venv-bm25\\Scripts\\python.exe -m scripts.bm25_search QUESTIONS OUT.jsonl [--dataset hotpotqa]

QUESTIONS: NQ-open .jsonl or HotpotQA .json (corpus.load_questions). --dataset picks the index
(default nq). OUT.jsonl: one line per question, in input order:
{"qid", "question", "hits": [[doc_id, score], ...]}.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from main_logic import corpus as cp
from main_logic.bm25 import BM25
from scripts import datasets as ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=None, help="use only the first n questions")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dataset", choices=ds.NAMES, default="nq")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    qs = [q["question"] for q in cp.load_questions(args.questions, args.n)]

    bm25 = BM25(ds.BM25_INDEX[args.dataset])
    t = time.time()
    hits = bm25.search(qs, k=args.k, threads=args.threads)
    dt = time.time() - t

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf8") as f:
        for i, (q, h) in enumerate(zip(qs, hits)):
            f.write(json.dumps({"qid": i, "question": q, "hits": h}, ensure_ascii=False) + "\n")
    short = sum(len(h) < args.k for h in hits)
    print(f"BM25 k1={bm25.k1} b={bm25.b}: {len(qs)} questions, k={args.k}, "
          f"{dt:.1f}s ({dt / max(len(qs), 1) * 1000:.0f} ms/question), "
          f"{short} with fewer than {args.k} hits -> {args.out}")


if __name__ == "__main__":
    main()
