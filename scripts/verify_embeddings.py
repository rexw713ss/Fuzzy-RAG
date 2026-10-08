"""Spot-check precomputed contriever-msmarco passage embeddings (spec v0.3, section 7.2).

NQ: Meta's embeddings of the DPR passages. HotpotQA (--dataset hotpotqa): pyserini's FAISS index of
the BEIR abstracts, with texts from data/hotpotqa/corpus.tsv.gz (scripts/export_corpus.py).

Why: we did not compute the 21M passage vectors ourselves, so we do not control how passages were
formatted or truncated. Queries are encoded by us; if our passage formatting differed from Meta's,
query and passage vectors would come from slightly different procedures and every dense score
would be subtly off. This script encodes a fixed random sample of passages ourselves and checks
that each matches Meta's vector.

Pass: cosine >= 0.999 for every sampled passage (Meta stored float16 and likely computed in
float16; we compute in float32, so tiny differences are expected). On failure, formatting
variants are tried on a subset to show which one Meta used.

Run from the repo root:  .venv\\Scripts\\python.exe -m scripts.verify_embeddings [--dataset hotpotqa]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from main_logic import corpus as cp
from main_logic import dense as dn
from scripts import datasets as ds

COS_MIN = 0.999


def cosines(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", choices=ds.NAMES, default="nq")
    ap.add_argument("--out", default=None,
                    help="default: data/checks/verify_embeddings.json (nq) or "
                         "verify_embeddings_hotpotqa.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    suffix = "" if args.dataset == "nq" else f"_{args.dataset}"
    out = args.out or str(ds.DATA / "checks" / f"verify_embeddings{suffix}.json")

    t = time.time()
    index = ds.load_index(args.dataset)
    print(f"loaded {len(index):,} precomputed vectors in {time.time() - t:.0f}s")

    # Sample rows, then name them by id (for DPR, id = row + 1, the same sample as before).
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(len(index), args.n, replace=False))
    ids = [index.id_of(r) for r in rows]

    t = time.time()
    passages = cp.load_passages(ds.CORPUS[args.dataset], ids)
    print(f"read {len(passages)} sampled passages in {time.time() - t:.0f}s")
    meta = index.vectors(ids)

    encoder = dn.load_encoder()
    t = time.time()
    ours = dn.encode([dn.passage_text(*passages[i]) for i in ids], encoder)
    print(f"encoded {len(ids)} passages on {next(encoder[1].parameters()).device} "
          f"in {time.time() - t:.0f}s")

    cos = cosines(ours, meta)
    norm_ratio = np.linalg.norm(ours, axis=1) / np.linalg.norm(meta, axis=1)
    report = {
        "dataset": args.dataset,
        "n": len(ids), "seed": args.seed, "model": dn.MODEL, "revision": dn.REVISION,
        "format": "title + ' ' + text", "max_length": dn.MAX_LENGTH,
        "cosine": {"min": float(cos.min()), "p1": float(np.percentile(cos, 1)),
                   "median": float(np.median(cos)), "mean": float(cos.mean())},
        "norm_ratio": {"min": float(norm_ratio.min()), "max": float(norm_ratio.max())},
        "passed": bool(cos.min() >= COS_MIN),
        "shards_covered": int(len(set(np.searchsorted(index.starts, rows, side="right")))),
    }
    worst = np.argsort(cos)[:3]
    report["worst"] = [{"id": ids[i], "cosine": float(cos[i]),
                        "title": passages[ids[i]][0]} for i in worst]

    if not report["passed"]:
        sub = ids[:100]
        variants = {
            "text only": lambda ti, te: te,
            "lowercase title + text": lambda ti, te: f"{ti} {te}".lower(),
            "title + newline + text": lambda ti, te: f"{ti}\n{te}",
            "title + '. ' + text": lambda ti, te: f"{ti}. {te}",
        }
        report["variants"] = {}
        for name, fmt in variants.items():
            v = dn.encode([fmt(*passages[i]) for i in sub], encoder)
            report["variants"][name] = float(cosines(v, meta[:len(sub)]).min())

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
