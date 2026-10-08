"""Dense retrieval with Contriever (spec v0.3, sections 2.1, 7.2).

Passage vectors are precomputed contriever-msmarco embeddings (provenance in data/SOURCES.txt):
  - NQ: Meta's embeddings of the 21,015,324 DPR passages, 16 pickle shards, float16;
  - HotpotQA: pyserini's FAISS flat index of the 5,233,329 BEIR HotpotQA abstracts, float32.
    Only the file is used, not the FAISS library: it is a 45-byte header followed by the raw
    vectors, read here with numpy.
Queries are encoded here with the same checkpoint, pinned to one revision. As in Contriever, an embedding is the mean of the last
hidden states over real tokens, left unnormalized, and the score is the inner product.

Search is exact: every query is scored against every passage. This is provisional until open
decision O-4 (exact vs approximate search) is settled; an ANN index would replace
EmbeddingIndex.search and nothing else.
"""

import pickle
import struct
from pathlib import Path

import numpy as np

MODEL = "facebook/contriever-msmarco"
REVISION = "abe8c1493371369031bcb1e02acb754cf4e162fa"   # Hugging Face commit, data/SOURCES.txt
MAX_LENGTH = 512        # Contriever's default passage and question length
DIM = 768
CHUNK = 262_144         # rows scored per matmul: ~0.8 GB as float32
FAISS_HEADER = 45       # bytes before the vectors in a FAISS IndexFlat file


def passage_text(title, text):
    """A DPR passage as Contriever encodes it: title, one space, text."""
    return f"{title} {text}"


def load_encoder(device=None):
    """Tokenizer and model for the pinned checkpoint, in eval mode on `device`.

    torch and transformers are imported here, not at module level, so the search code and its
    tests run without them.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModel.from_pretrained(MODEL, revision=REVISION).to(device).eval()
    return tok, model


def encode(texts, encoder, batch_size=64, max_length=MAX_LENGTH):
    """Contriever embeddings: float32 array (len(texts), 768), mean-pooled, unnormalized."""
    import torch

    tok, model = encoder
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            x = tok(list(texts[i:i + batch_size]), padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt").to(device)
            h = model(**x).last_hidden_state
            m = x["attention_mask"].unsqueeze(-1).to(h.dtype)
            out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, DIM), dtype=np.float32)


class EmbeddingIndex:
    """Passage vectors in blocks: float16 shards in RAM for DPR (~32 GB), one float32 memory map
    for HotpotQA (16 GB, paged in by the OS on the first search).

    The blocks are never concatenated: joining 16 shards of 2 GB would briefly need twice the
    memory.

    Ids: with ids=None the passage ids are "1", "2", ... in block order, as in the DPR corpus
    and Meta's shards (checked at load), so an id maps to a row by arithmetic (row = int(id) - 1)
    instead of a 21M-entry dictionary. Otherwise `ids` lists the id of every row (HotpotQA's
    Wikipedia page ids, which are not consecutive).
    """

    def __init__(self, blocks, ids=None):
        self.blocks = [b if isinstance(b, np.memmap) else np.asarray(b) for b in blocks]
        for b in self.blocks:
            if b.ndim != 2 or b.shape[1] != DIM:
                raise ValueError(f"every block must be (n, {DIM}), got {b.shape}")
        sizes = [len(b) for b in self.blocks]
        self.starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self.n = int(self.starts[-1])
        self.ids = None
        if ids is not None:
            self.ids = [str(i) for i in ids]
            if len(self.ids) != self.n:
                raise ValueError(f"{len(self.ids)} ids for {self.n} vectors")
            self._rows = {d: r for r, d in enumerate(self.ids)}
            if len(self._rows) != self.n:
                raise ValueError("passage ids are not unique")

    @classmethod
    def from_shards(cls, directory, pattern="passages_*"):
        """Load Meta's pickle shards. Pickle can execute code, so load only the verified release."""
        files = sorted(Path(directory).glob(pattern))
        if not files:
            raise FileNotFoundError(f"no shards matching {pattern} in {directory}")
        blocks, expected = [], 1
        for f in files:
            with open(f, "rb") as fh:
                ids, vecs = pickle.load(fh)
            if ids[0] != str(expected) or ids[-1] != str(expected + len(ids) - 1):
                raise ValueError(f"{f.name}: ids {ids[0]}..{ids[-1]} are not consecutive "
                                 f"from {expected}")
            blocks.append(vecs)
            expected += len(ids)
        return cls(blocks)

    @classmethod
    def from_faiss(cls, directory):
        """Load a pyserini FAISS flat index directory: `index` (the vectors) and `docid` (one
        passage id per line, in row order). The vectors are memory-mapped, not copied.

        Only an inner-product flat index of float32 vectors of dimension 768 is accepted;
        anything else raises rather than being misread.
        """
        directory = Path(directory)
        with open(directory / "index", "rb") as f:
            head = f.read(FAISS_HEADER)
        magic, d, ntotal, _, _, trained, metric, n_codes = struct.unpack("<4siqqq?iq", head)
        if magic != b"IxFI":
            raise ValueError(f"not a FAISS flat inner-product index (magic {magic!r})")
        if d != DIM or metric != 0 or not trained or n_codes != ntotal * d:
            raise ValueError(f"unexpected FAISS header: d={d}, metric={metric}, "
                             f"trained={trained}, ntotal={ntotal}, codes={n_codes}")
        size = (directory / "index").stat().st_size
        if size != FAISS_HEADER + 4 * ntotal * d:
            raise ValueError(f"index file is {size} bytes, expected "
                             f"{FAISS_HEADER + 4 * ntotal * d} for {ntotal} vectors")
        vecs = np.memmap(directory / "index", dtype="<f4", mode="r", offset=FAISS_HEADER,
                         shape=(ntotal, d))
        with open(directory / "docid", encoding="utf8") as f:
            ids = [line.rstrip("\n") for line in f]
        return cls([vecs], ids=ids)

    def __len__(self):
        return self.n

    def row_of(self, doc_id):
        if self.ids is None:
            return int(doc_id) - 1
        try:
            return self._rows[str(doc_id)]
        except KeyError:
            raise KeyError(f"passage id {doc_id!r} not in the index") from None

    def id_of(self, row):
        return str(int(row) + 1) if self.ids is None else self.ids[int(row)]

    def vectors(self, doc_ids):
        """float32 vectors for the given passage ids, in the given order (S3 needs the top-10)."""
        rows = np.array([self.row_of(d) for d in doc_ids], dtype=np.int64)
        if rows.size and (rows.min() < 0 or rows.max() >= self.n):
            raise KeyError(f"passage id outside 1..{self.n}")
        blk = np.searchsorted(self.starts, rows, side="right") - 1
        out = np.empty((len(rows), DIM), dtype=np.float32)
        for i, (b, r) in enumerate(zip(blk, rows)):
            out[i] = self.blocks[b][r - self.starts[b]]
        return out

    def search(self, queries, k=50, chunk=CHUNK):
        """Exact top-k passages by inner product for each query.

        queries: float32 (n_queries, 768). Returns one list per query of (doc_id, score), score
        descending, ties broken by lower row (for DPR, the lower doc id; for HotpotQA, the
        earlier line in `docid`). Scoring all queries in one call reads every vector once, so
        batch queries rather than calling per query.
        """
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != DIM:
            raise ValueError(f"queries must be (n, {DIM}), got {q.shape}")
        k = min(k, self.n)
        best_s = np.full((len(q), k), -np.inf, dtype=np.float32)
        best_r = np.full((len(q), k), -1, dtype=np.int64)

        for b, block in enumerate(self.blocks):
            for c in range(0, len(block), chunk):
                s = q @ block[c:c + chunk].astype(np.float32).T          # (n_queries, rows)
                cols = _best_columns(s, min(k, s.shape[1]))
                cand_s = np.concatenate([best_s, np.take_along_axis(s, cols, 1)], axis=1)
                cand_r = np.concatenate([best_r, cols + self.starts[b] + c], axis=1)
                keep = np.lexsort((cand_r, -cand_s), axis=1)[:, :k]     # score desc, lower row
                best_s = np.take_along_axis(cand_s, keep, 1)
                best_r = np.take_along_axis(cand_r, keep, 1)

        return [[(self.id_of(r), float(s)) for s, r in zip(s_row, r_row)]
                for s_row, r_row in zip(best_s, best_r)]


def _best_columns(s, kk):
    """Column indices of the kk highest scores in each row, ties going to the lower column.

    np.argpartition alone picks arbitrarily among scores tied at the cutoff. Taking everything
    strictly above the kk-th score, then filling up with tied columns from the left, makes the
    choice follow the documented tie-break.
    """
    cut = s.shape[1] - kk
    t = np.partition(s, cut, axis=1)[:, cut]                  # kk-th highest per row
    above = s > t[:, None]
    tied = s == t[:, None]
    room = kk - above.sum(axis=1)
    take = above | (tied & (np.cumsum(tied, axis=1) <= room[:, None]))
    return np.nonzero(take)[1].reshape(len(s), kk)
