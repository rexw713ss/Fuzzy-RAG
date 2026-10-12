"""Human audit of the paraphrases (spec v0.3, section 7.1: "Audit 50 stratified pairs and report
the agreement rate").

Run from the repo root, in the main environment:
    .venv\\Scripts\\python.exe -m scripts.audit_sheet make  [--root DIR]
    .venv\\Scripts\\python.exe -m scripts.audit_sheet score [--root DIR]

make: draws 25 (original, paraphrase) pairs per dataset (stratified by dataset, seed 42) from the
complete clusters in ROOT/<dataset>/clusters.jsonl (default ROOT data/stage1) and writes
ROOT/audit.csv. Fill in two columns per row with y or n:
  same_meaning          the paraphrase asks exactly what the original asks
  answer_still_correct  the gold answer is still a correct answer to the paraphrase
score: reads the filled audit.csv and reports the agreement rate: the share of pairs with y in
both columns, per dataset and overall, and how many rows are still blank. -> ROOT/audit_score.json

The CSV is UTF-8 with a byte-order mark, so Excel shows non-ASCII characters correctly.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from scripts import datasets as ds

PER_DATASET = 25
SEED = 42
COLUMNS = ("pair_id", "dataset", "cluster_id", "variant_id", "original", "paraphrase",
           "gold_answer", "same_meaning", "answer_still_correct", "notes")


def make(root):
    rows = []
    for d in ds.NAMES:
        with open(root / d / "clusters.jsonl", encoding="utf8") as f:
            clusters = [json.loads(line) for line in f if line.strip()]
        pairs = [(c, p) for c in clusters if c["complete"] for p in c["phrasings"][1:]]
        if len(pairs) < PER_DATASET:
            raise ValueError(f"{d}: only {len(pairs)} pairs, need {PER_DATASET}")
        pick = sorted(np.random.default_rng(SEED).choice(len(pairs), PER_DATASET, replace=False))
        for i in pick:
            c, p = pairs[i]
            rows.append({"dataset": d, "cluster_id": c["cluster_id"],
                         "variant_id": p["variant_id"], "original": c["phrasings"][0]["text"],
                         "paraphrase": p["text"], "gold_answer": " | ".join(c["answer"]),
                         "same_meaning": "", "answer_still_correct": "", "notes": ""})
    path = root / "audit.csv"
    if path.exists():
        raise FileExistsError(f"{path} exists; it may hold your answers. Move it first.")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        for n, r in enumerate(rows, 1):
            w.writerow({"pair_id": n, **r})
    print(f"{len(rows)} pairs -> {path}")


def score(root):
    with open(root / "audit.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for name, part in [(d, [r for r in rows if r["dataset"] == d]) for d in ds.NAMES] + [("all", rows)]:
        filled = [r for r in part if r["same_meaning"].strip() and r["answer_still_correct"].strip()]
        yes = [r for r in filled if r["same_meaning"].strip().lower() == "y"
               and r["answer_still_correct"].strip().lower() == "y"]
        out[name] = {"pairs": len(part), "filled": len(filled), "agree": len(yes),
                     "agreement_rate": len(yes) / len(filled) if filled else None,
                     "rejected_pair_ids": [int(r["pair_id"]) for r in filled if r not in yes]}
    (root / "audit_score.json").write_text(json.dumps(out, indent=2), encoding="utf8")
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("make", "score"))
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(args.root or ds.DATA / "stage1")
    make(root) if args.command == "make" else score(root)


if __name__ == "__main__":
    main()
