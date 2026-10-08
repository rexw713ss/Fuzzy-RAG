"""Where each dataset's corpus and indexes live (spec v0.3, section 7.2), for the scripts.

One place for the paths, so the scripts cannot disagree; tests/test_config.py checks them against
config.yaml's data section.
"""

from pathlib import Path

DATA = Path(r"D:\Han\rex_rag\data")
NAMES = ("nq", "hotpotqa")

CORPUS = {
    "nq": DATA / "dpr" / "psgs_w100.tsv.gz",
    "hotpotqa": DATA / "hotpotqa" / "corpus.tsv.gz",
}
EMBEDDINGS = {
    "nq": DATA / "contriever-msmarco" / "wikipedia_embeddings",
    "hotpotqa": DATA / "hotpotqa" / "faiss-flat.beir-v1.0.0-hotpotqa.contriever-msmarco.20230124",
}
BM25_INDEX = {
    "nq": DATA / "bm25" / "lucene-inverted.wikipedia-dpr-100w.20260508.deb4c7b",
    "hotpotqa": DATA / "hotpotqa" / "lucene-inverted.beir-v1.0.0-hotpotqa.flat.20221116.505594",
}


def load_index(name):
    """The dense EmbeddingIndex of a dataset: Meta's pickle shards (NQ) or the FAISS file (HotpotQA)."""
    from main_logic import dense as dn

    if name == "nq":
        return dn.EmbeddingIndex.from_shards(EMBEDDINGS["nq"])
    if name == "hotpotqa":
        return dn.EmbeddingIndex.from_faiss(EMBEDDINGS["hotpotqa"])
    raise ValueError(f"unknown dataset {name!r}; expected one of {NAMES}")
