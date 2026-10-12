"""Pipeline steps 2, 6, 7 on the pilot, and the stage 1 checks (spec v0.3, sections 7.1, 2.6,
7.3, 8, 8.2).

Run from the repo root, in the main environment, after scripts.retrieve_clusters for both datasets:
    .venv\\Scripts\\python.exe -m scripts.stage1_checks [--root DIR]

ROOT (default data/stage1) holds one folder per dataset with clusters.jsonl, signals.jsonl and
retrieval.jsonl. Steps:
  2. split the clusters 60/20/20 by cluster, per dataset, seed 42      -> splits.json
  6. fit p5/p95 per signal per dataset on TRAIN instances; scale all    -> scalers.json,
                                                                          signals_scaled.jsonl
  7. jitter widths per dataset from TRAIN clusters (complete only)      -> jitter.json
Checks (section 8.2), on all of a dataset's pilot instances: NaN/Inf, identical values,
clipping, Spearman correlation, S3's direction across datasets, widths at the floor; plus the
stage 1 exit question, whether each raw signal separates hard instances (no gold answer in the
fused top 5; yes/no answers left out) from easy ones (AUC).                -> report.json

The pilot's scalers and widths are not written into config.yaml's fitted: section; they come
from 12 training clusters per dataset and are only for these checks. Stage 4 fits the real ones.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from main_logic import calibration as cb
from main_logic import checks as ck
from main_logic import signals as sg
from main_logic import splits as sp
from scripts import datasets as ds


def read_jsonl(path):
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(args.root or ds.DATA / "stage1")

    clusters = {d: read_jsonl(root / d / "clusters.jsonl") for d in ds.NAMES}
    signals = {d: read_jsonl(root / d / "signals.jsonl") for d in ds.NAMES}
    retrieval = {d: read_jsonl(root / d / "retrieval.jsonl") for d in ds.NAMES}

    # Step 2: split by cluster.
    splits = sp.split_clusters({d: [c["cluster_id"] for c in clusters[d]] for d in ds.NAMES})
    (root / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf8")

    scalers, jitter, summary, scaled_rows = {}, {}, {}, []
    for d in ds.NAMES:
        rows = signals[d]
        raw = np.array([[r[s] for s in ck.SIGNALS] for r in rows], dtype=float)
        train = set(splits[d]["train"])
        is_train = np.array([r["cluster_id"] in train for r in rows])

        # Step 6: scalers on training instances only.
        scalers[d] = {s: sg.fit_scaler(raw[is_train, j]) for j, s in enumerate(ck.SIGNALS)}
        scaled, clipped = np.empty_like(raw), {}
        for j, s in enumerate(ck.SIGNALS):
            scaled[:, j], clipped[s] = sg.apply_scaler(raw[:, j], scalers[d][s])
        for r, v in zip(rows, scaled):
            scaled_rows.append({"dataset": d, "cluster_id": r["cluster_id"],
                                "variant_id": r["variant_id"],
                                **{s: float(x) for s, x in zip(ck.SIGNALS, v)}})

        # Step 7: jitter widths from training clusters (incomplete ones skipped and counted).
        by_cluster = {}
        for r, v in zip(rows, scaled):
            if r["cluster_id"] in train:
                by_cluster.setdefault(r["cluster_id"], []).append(v)
        jitter[d] = cb.calibrate([np.array(v) for v in by_cluster.values()])

        # Checks.
        hit = {(r["cluster_id"], r["variant_id"]): r["fused_hit_at_5"] for r in retrieval[d]}
        flags_hit = [hit[(r["cluster_id"], r["variant_id"])] for r in rows]
        scored = np.array([h is not None for h in flags_hit])
        hard = np.array([h is False for h in flags_hit])
        summary[d] = {
            "instances": len(rows),
            "clusters": len(clusters[d]),
            "complete_clusters": sum(c["complete"] for c in clusters[d]),
            "train_val_test_clusters": [len(splits[d][s]) for s in sp.SPLITS],
            "scored_for_hard": int(scored.sum()),
            "share_hard": float(hard[scored].mean()) if scored.any() else float("nan"),
            "signals": {s: {
                "min": float(raw[:, j].min()), "max": float(raw[:, j].max()),
                "mean": float(raw[:, j].mean()), "sd": float(raw[:, j].std()),
                "non_finite": int((~np.isfinite(raw[:, j])).sum()),
                "most_common_share": ck.most_common_share(raw[:, j]),
                "clipped": clipped[s],
                "auc_hard": ck.auc(raw[scored, j], hard[scored]),
            } for j, s in enumerate(ck.SIGNALS)},
            "correlations": ck.correlations(raw),
            "all_at_floor": jitter[d]["all_at_floor"],
        }

    with open(root / "signals_scaled.jsonl", "w", encoding="utf8") as f:
        for r in scaled_rows:
            f.write(json.dumps(r) + "\n")
    (root / "scalers.json").write_text(json.dumps(scalers, indent=2), encoding="utf8")
    (root / "jitter.json").write_text(json.dumps(jitter, indent=2), encoding="utf8")

    findings = ck.flags(summary)
    report = {"summary": summary, "widths": {d: jitter[d]["widths"] for d in ds.NAMES},
              "flags": findings,
              "stop": any(f["level"] == "stop" for f in findings)}
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
