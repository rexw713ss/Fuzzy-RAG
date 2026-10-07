"""Tests for the cluster-level split (spec v0.3, section 7.1)."""

import random

import pytest

from main_logic import splits as sp


@pytest.mark.parametrize("n, sizes", [(20, (12, 4, 4)), (300, (180, 60, 60)), (7, (5, 1, 1)),
                                      (3, (1, 1, 1)), (1, (1, 0, 0)), (0, (0, 0, 0))])
def test_sizes(n, sizes):
    assert sp.split_sizes(n) == sizes
    got = sp.split_clusters({"nq": range(n)})["nq"]
    assert tuple(len(got[s]) for s in sp.SPLITS) == sizes


def test_every_cluster_in_exactly_one_split():
    ids = [f"nq-{i:03d}" for i in range(300)]
    got = sp.split_clusters({"nq": ids})["nq"]
    together = got["train"] + got["val"] + got["test"]
    assert sorted(together) == sorted(ids)
    assert len(set(together)) == len(together)


def test_datasets_are_split_independently():
    """Stratified by dataset: each gets 60/20/20 of its own clusters, and adding or changing
    another dataset does not change its split."""
    nq = [f"nq-{i}" for i in range(20)]
    hq = [f"hq-{i}" for i in range(20)]
    alone = sp.split_clusters({"nq": nq})["nq"]
    both = sp.split_clusters({"nq": nq, "hotpotqa": hq})
    assert both["nq"] == alone
    assert [len(both["hotpotqa"][s]) for s in sp.SPLITS] == [12, 4, 4]
    assert set(both["hotpotqa"]["test"]) <= set(hq)


def test_same_set_in_any_order_gives_the_same_split():
    ids = list(range(50))
    first = sp.split_clusters({"nq": ids})
    shuffled = ids[:]
    random.Random(1).shuffle(shuffled)
    assert sp.split_clusters({"nq": shuffled}) == first
    assert sp.split_clusters({"nq": ids}) == first


def test_default_seed_is_42_and_seed_matters():
    ids = list(range(300))
    assert sp.SEED == 42
    assert sp.split_clusters({"nq": ids}) == sp.split_clusters({"nq": ids}, seed=42)
    assert sp.split_clusters({"nq": ids}, seed=7) != sp.split_clusters({"nq": ids})


def test_exact_split_is_pinned():
    """Guards reproducibility: fails if a numpy update changes the seed-42 shuffle."""
    assert sp.split_clusters({"nq": range(10)}) == {
        "nq": {"train": [0, 2, 3, 5, 6, 7], "val": [4, 9], "test": [1, 8]}}


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        sp.split_clusters({"nq": ["a", "b", "a"]})
