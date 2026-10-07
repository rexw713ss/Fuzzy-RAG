"""Tests for exact dense search over sharded float16 vectors (spec v0.3, sections 2.1, 7.2).

The encoder (torch + the contriever-msmarco download) is not unit-tested here: it is checked
end to end by scripts/verify_embeddings.py, which compares our passage embeddings with Meta's.
"""

import pickle

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
