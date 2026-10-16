"""One query using the pinned local Lucene index in its own BM25 environment."""

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    a = ap.parse_args()
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(a.index)
    searcher.set_bm25(.9, .4)
    query = json.load(sys.stdin)["question"]
    result = searcher.search(query, k=50)
    print(json.dumps({"hits": [[h.docid, float(h.score)] for h in result]}))


if __name__ == "__main__":
    main()
