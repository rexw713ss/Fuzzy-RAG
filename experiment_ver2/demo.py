"""Explicit synthetic input for end-to-end mechanics, NOT a research benchmark."""

import json
from pathlib import Path

import numpy as np


def create_demo(root, seed=42, n_clusters=90):
    root = Path(root)
    folder = root / "synthetic"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = [f"synthetic-{i:04}" for i in range(n_clusters)]
    rng.shuffle(ids)
    train_end, val_end = int(.6 * n_clusters), int(.8 * n_clusters)
    splits = {"synthetic": {"train": ids[:train_end], "val": ids[train_end:val_end], "test": ids[val_end:]}}
    (root / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    queries, signals, outcomes = [], [], []
    for cluster in ids:
        centre = rng.uniform(.05, .95, 4)
        for variant in range(3):
            x = np.clip(centre + rng.normal(0, .025, 4), 0, 1)
            key = {"cluster_id": cluster, "variant_id": variant}
            queries.append({**key, "question": f"SYNTHETIC query {cluster} wording {variant}", "answer": ["synthetic"]})
            signals.append({**key, **dict(zip(("S1", "S2", "S3", "S4"), x.tolist()))})
            # Non-monotone rescue pattern is deliberately built into this toy.
            # Success here only verifies the implementation can exploit such a pattern.
            gain = .3 if centre[0] < .4 and centre[1] > .6 else -.08
            q = [.35, .55, .55 + gain]
            for action in range(3):
                outcomes.append({**key, "action": f"A{action}", "F1": q[action], "EM": float(q[action] > .75),
                                 "latency_total": [4.0, 4.05, 6.05][action],
                                 "retrievals": 2 if action == 2 else 1,
                                 "llm_calls": 2 if action == 2 else 1})
    for filename, rows in (("queries.jsonl", queries), ("signals.jsonl", signals), ("outcomes.jsonl", outcomes)):
        (folder / filename).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return root
