"""Tests for exact dense search (spec v0.3, sections 2.1, 7.2): sharded float16 vectors (DPR) and
a FAISS flat file with an id table (HotpotQA).

The encoder (torch + the contriever-msmarco download) is not unit-tested here: it is checked
end to end by scripts/verify_embeddings.py, which compares our passage embeddings with Meta's.
"""

import pickle
import struct

import numpy as np
import pytest

from main_logic import dense as dn


def random_blocks(sizes, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((n, dn.DIM)).astype(np.float16) for n in sizes]


def brute_force(blocks, q, k):
    """Reference: full score matrix, full sort, same tie-break (score desc, then lower row)."""
    allv = np.concatenate(blocks).astype(np.float32)
    out = []
    for s in q @ allv.T:
        order = np.lexsort((np.arange(len(s)), -s))[:k]
        out.append([(str(r + 1), float(s[r])) for r in order])
    return out


@pytest.mark.parametrize("chunk", [7, 64, 10_000])
@pytest.mark.parametrize("k", [1, 5, 50])
def test_search_matches_brute_force_across_blocks_and_chunks(chunk, k):
    blocks = random_blocks([40, 33, 57])
    idx = dn.EmbeddingIndex(blocks)
    q = np.random.default_rng(1).standard_normal((6, dn.DIM)).astype(np.float32)
    got = idx.search(q, k=k, chunk=chunk)
    want = brute_force(blocks, q, k)
    for g, w in zip(got, want):
        assert [d for d, _ in g] == [d for d, _ in w]
        assert [s for _, s in g] == pytest.approx([s for _, s in w], rel=1e-5)


def test_ties_break_to_the_lower_id():
    v = np.zeros((5, dn.DIM), dtype=np.float16)
    v[:, 0] = [1, 2, 2, 1, 2]                      # ids 2, 3, 5 tie on the top score
    idx = dn.EmbeddingIndex([v[:2], v[2:]])
    q = np.zeros((1, dn.DIM), dtype=np.float32)
    q[0, 0] = 1.0
    assert [d for d, _ in idx.search(q, k=4, chunk=2)[0]] == ["2", "3", "5", "1"]


def test_k_larger_than_corpus_returns_everything():
    idx = dn.EmbeddingIndex(random_blocks([3, 2]))
    q = np.ones((1, dn.DIM), dtype=np.float32)
    assert len(idx.search(q, k=50)[0]) == 5


def test_scores_are_inner_products_not_cosines():
    """Contriever scores by dot product of unnormalized vectors."""
    v = np.zeros((2, dn.DIM), dtype=np.float16)
    v[0, 0], v[1, 0] = 1.0, 3.0                     # same direction, different length
    q = np.zeros((1, dn.DIM), dtype=np.float32)
    q[0, 0] = 2.0
    hits = dn.EmbeddingIndex([v]).search(q, k=2)[0]
    assert hits == [("2", pytest.approx(6.0)), ("1", pytest.approx(2.0))]


def test_vectors_lookup_across_blocks_keeps_order():
    blocks = random_blocks([4, 3, 5])
    idx = dn.EmbeddingIndex(blocks)
    allv = np.concatenate(blocks).astype(np.float32)
    ids = ["12", "1", "5", "4", "8"]
    got = idx.vectors(ids)
    assert np.array_equal(got, allv[[int(i) - 1 for i in ids]])
    with pytest.raises(KeyError):
        idx.vectors(["13"])
    with pytest.raises(KeyError):
        idx.vectors(["0"])


def test_bad_shapes_rejected():
    with pytest.raises(ValueError):
        dn.EmbeddingIndex([np.zeros((3, 10), dtype=np.float16)])
    idx = dn.EmbeddingIndex(random_blocks([3]))
    with pytest.raises(ValueError):
        idx.search(np.zeros((2, 10), dtype=np.float32))


def write_shard(path, first_id, vecs):
    with open(path, "wb") as f:
        pickle.dump(([str(first_id + i) for i in range(len(vecs))], vecs), f)


def test_from_shards_loads_consecutive_ids(tmp_path):
    blocks = random_blocks([3, 4])
    write_shard(tmp_path / "passages_00", 1, blocks[0])
    write_shard(tmp_path / "passages_01", 4, blocks[1])
    idx = dn.EmbeddingIndex.from_shards(tmp_path)
    assert len(idx) == 7
    assert np.array_equal(idx.vectors(["5"])[0], blocks[1][1].astype(np.float32))


def test_from_shards_rejects_a_gap_in_ids(tmp_path):
    blocks = random_blocks([3, 4])
    write_shard(tmp_path / "passages_00", 1, blocks[0])
    write_shard(tmp_path / "passages_01", 5, blocks[1])     # id 4 missing
    with pytest.raises(ValueError, match="not consecutive"):
        dn.EmbeddingIndex.from_shards(tmp_path)


def test_from_shards_needs_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        dn.EmbeddingIndex.from_shards(tmp_path)


def test_passage_text_is_title_space_text():
    assert dn.passage_text("Aaron", "Aaron is a prophet") == "Aaron Aaron is a prophet"


# --- FAISS flat files with an id table (HotpotQA) ------------------------------------------------

def write_faiss(directory, vecs, ids, metric=0, d=None):
    """A FAISS IndexFlatIP file laid out byte for byte as faiss writes it, plus the docid file."""
    directory.mkdir(parents=True, exist_ok=True)
    n, dim = vecs.shape
    d = dim if d is None else d
    head = struct.pack("<4siqqq?iq", b"IxFI", d, n, 1 << 20, 1 << 20, True, metric, n * d)
    assert len(head) == dn.FAISS_HEADER
    (directory / "index").write_bytes(head + vecs.astype("<f4").tobytes())
    (directory / "docid").write_text("".join(f"{i}\n" for i in ids), encoding="utf8")


def test_from_faiss_reads_vectors_and_ids(tmp_path):
    v = np.random.default_rng(5).standard_normal((4, dn.DIM)).astype(np.float32)
    write_faiss(tmp_path, v, ["303", "35370504", "12", "7"])
    idx = dn.EmbeddingIndex.from_faiss(tmp_path)
    assert len(idx) == 4
    assert np.array_equal(idx.vectors(["12", "303"]), v[[2, 0]])
    assert idx.row_of("35370504") == 1 and idx.id_of(3) == "7"


def test_from_faiss_search_returns_page_ids(tmp_path):
    v = np.random.default_rng(6).standard_normal((50, dn.DIM)).astype(np.float32)
    ids = [str(1000 + 7 * i) for i in range(50)]
    write_faiss(tmp_path, v, ids)
    q = np.random.default_rng(7).standard_normal((3, dn.DIM)).astype(np.float32)
    got = dn.EmbeddingIndex.from_faiss(tmp_path).search(q, k=5, chunk=16)
    ref = brute_force([v], q, 5)
    # brute_force numbers rows 1..n; map those to the page ids of the docid file.
    assert [[d for d, _ in row] for row in got] == [[ids[int(d) - 1] for d, _ in row]
                                                    for row in ref]
    assert np.allclose([[s for _, s in row] for row in got],
                       [[s for _, s in row] for row in ref], rtol=1e-5)


def test_ties_go_to_the_earlier_row_not_the_smaller_id(tmp_path):
    """Identical vectors: the earlier line of docid wins, even when its id is larger."""
    v = np.ones((3, dn.DIM), dtype=np.float32)
    write_faiss(tmp_path, v, ["900", "5", "40"])
    hits = dn.EmbeddingIndex.from_faiss(tmp_path).search(np.ones((1, dn.DIM)), k=3)[0]
    assert [d for d, _ in hits] == ["900", "5", "40"]


def test_unknown_id_is_an_error(tmp_path):
    write_faiss(tmp_path, np.zeros((2, dn.DIM), dtype=np.float32), ["1", "2"])
    with pytest.raises(KeyError, match="not in the index"):
        dn.EmbeddingIndex.from_faiss(tmp_path).vectors(["3"])


def test_duplicate_or_missing_ids_are_errors(tmp_path):
    write_faiss(tmp_path / "dup", np.zeros((2, dn.DIM), dtype=np.float32), ["1", "1"])
    with pytest.raises(ValueError, match="not unique"):
        dn.EmbeddingIndex.from_faiss(tmp_path / "dup")
    write_faiss(tmp_path / "short", np.zeros((2, dn.DIM), dtype=np.float32), ["1"])
    with pytest.raises(ValueError, match="1 ids for 2 vectors"):
        dn.EmbeddingIndex.from_faiss(tmp_path / "short")


def test_wrong_metric_or_dimension_is_refused(tmp_path):
    write_faiss(tmp_path / "l2", np.zeros((2, dn.DIM), dtype=np.float32), ["1", "2"], metric=1)
    with pytest.raises(ValueError, match="unexpected FAISS header"):
        dn.EmbeddingIndex.from_faiss(tmp_path / "l2")
    write_faiss(tmp_path / "small", np.zeros((2, 8), dtype=np.float32), ["1", "2"])
    with pytest.raises(ValueError, match="unexpected FAISS header"):
        dn.EmbeddingIndex.from_faiss(tmp_path / "small")


def test_truncated_file_is_refused(tmp_path):
    write_faiss(tmp_path, np.zeros((3, dn.DIM), dtype=np.float32), ["1", "2", "3"])
    f = tmp_path / "index"
    f.write_bytes(f.read_bytes()[:-4])
    with pytest.raises(ValueError, match="bytes, expected"):
        dn.EmbeddingIndex.from_faiss(tmp_path)
