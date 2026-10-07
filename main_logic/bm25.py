"""BM25 retrieval over the DPR Wikipedia passages with Lucene via pyserini (spec v0.3, 2.1, 7.2).

Uses the prebuilt index lucene-inverted.wikipedia-dpr-100w.20260508.deb4c7b (21,015,324 passages,
same ids as psgs_w100.tsv; provenance in data/SOURCES.txt).

pyserini needs Java (JAVA_HOME) and runs in its own environment, .venv-bm25 on Python 3.13,
because it does not support the main venv's Python 3.14. It is imported only when a searcher is
built, so the rest of main_logic never needs it.
"""

K1 = 0.9    # pyserini / Anserini defaults, standard for this index; the spec does not set
B = 0.4     # them (v0.3 section 9.3), so they are recorded here explicitly


class BM25:
    def __init__(self, index_dir, k1=K1, b=B):
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(str(index_dir))
        self.searcher.set_bm25(k1, b)
        self.k1, self.b = k1, b

    def __len__(self):
        return self.searcher.num_docs

    def search(self, queries, k=50, threads=8):
        """Top-k (doc_id, score) per query, score descending. Lucene breaks ties by doc order."""
        qids = [str(i) for i in range(len(queries))]
        hits = self.searcher.batch_search(list(queries), qids, k=k, threads=threads)
        return [[(h.docid, float(h.score)) for h in hits[q]] for q in qids]
