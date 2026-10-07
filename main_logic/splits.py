"""Cluster-level train / validation / test split (spec v0.3, section 7.1; pipeline step 2).

60 / 20 / 20, stratified by dataset, and by cluster — never by individual query — so all
phrasings of a question share one split. Scalers (2.6) and jitter widths (7.3) are fitted on
train, thresholds (3.4) on validation, and test is never touched by either (8.3); splitting by
cluster keeps a question's other phrasings out of the data that tuned the controller.

Choices the spec leaves open (user decisions, Oct 7, 2026):
  - seed 42 (moves to config.yaml at step 0g);
  - validation and test each get round(20%) of a dataset's clusters, train the rest
    (20 -> 12/4/4, 300 -> 180/60/60);
  - ids are sorted before shuffling, so the same set gives the same split in any input order;
  - each dataset is shuffled by its own generator from the same seed, so one dataset's split
    does not depend on any other dataset.

numpy does not promise identical random streams across versions; the split written to
splits.json (pipeline step 2) is the record, and numpy is pinned in requirements.txt.
"""

import numpy as np

SEED = 42
SPLITS = ("train", "val", "test")


def split_sizes(n):
    """(train, val, test) cluster counts for n clusters: 20% each to val and test, rest train."""
    n_val = n_test = round(n / 5)
    return n - n_val - n_test, n_val, n_test


def split_clusters(clusters_by_dataset, seed=SEED):
    """{dataset: cluster ids} -> {dataset: {"train": [...], "val": [...], "test": [...]}}.

    Each list is sorted. Cluster ids must be unique within a dataset.
    """
    out = {}
    for dataset, ids in clusters_by_dataset.items():
        ids = list(ids)
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate cluster ids in dataset {dataset!r}")
        ids = sorted(ids)
        perm = np.random.default_rng(seed).permutation(len(ids))
        shuffled = [ids[i] for i in perm]
        n_train, n_val, _ = split_sizes(len(ids))
        parts = (shuffled[:n_train], shuffled[n_train:n_train + n_val],
                 shuffled[n_train + n_val:])
        out[dataset] = {name: sorted(part) for name, part in zip(SPLITS, parts)}
    return out
