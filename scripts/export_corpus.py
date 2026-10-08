"""Export the passages stored in a Lucene index to a TSV in the DPR layout (spec v0.3, section 7.2).

Why: HotpotQA's abstracts come only inside pyserini's prebuilt BM25 index, which the main
environment cannot read (pyserini lives in .venv-bm25). Writing them out once as id / text / title,
the same layout as DPR's psgs_w100.tsv, lets corpus.py read either corpus unchanged, for the
embedding check, the smoke test and later the generator's evidence. The text is exactly what BM25
indexed and what the prebuilt dense index was encoded from (both are the BEIR HotpotQA corpus).

Runs in the BM25 environment, from the repo root:
    .venv-bm25\\Scripts\\python.exe -m scripts.export_corpus INDEX_DIR OUT.tsv.gz
"""

import argparse
import csv
import gzip
import json
import sys
import time

from pyserini.search.lucene import LuceneSearcher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("out")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    searcher = LuceneSearcher(args.index)
    n = searcher.num_docs
    t = time.time()
    seen = set()
    with gzip.open(args.out, "wt", encoding="utf8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["id", "text", "title"])
        for i in range(n):
            d = json.loads(searcher.doc(i).raw())
            if d["_id"] in seen:
                raise ValueError(f"duplicate passage id {d['_id']}")
            seen.add(d["_id"])
            w.writerow([d["_id"], d["text"], d["title"]])
            if (i + 1) % 1_000_000 == 0:
                print(f"{i + 1:,} / {n:,} [{time.time() - t:.0f} s]", flush=True)
    print(f"wrote {n:,} passages to {args.out} in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
