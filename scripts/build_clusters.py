"""Pipeline step 1: paraphrase clusters (spec v0.3, section 7.1; protocol in main_logic/paraphrase.py).

Run from the repo root, in the main environment:
    .venv\\Scripts\\python.exe -m scripts.build_clusters --dataset nq --n 20

Three phases, each resumable:
  1. sample n questions from the dataset's dev file (seed 42; the first n of one permutation);
  2. generate 5 candidates per question with llama-server, which this script starts and stops.
     Every finished question is appended to candidates.jsonl at once, and questions already
     there are skipped, so an interrupted run continues where it stopped;
  3. after the server has stopped (the GPU is free), screen every candidate with the NLI,
     cosine and answer gates, keep the first 2 that pass, and write clusters.jsonl.

Output in OUTDIR (default data/stage1/<dataset>):
  candidates.jsonl   per question: the raw LLM replies, seeds, timings
  clusters.jsonl     per question: cluster_id, answers, phrasings (variant 0 = original),
                     complete (3 phrasings or not), and every candidate's gate results
  report.json        counts, failure reasons per gate, settings, timing
  llama-server.log
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

from main_logic import corpus as cp
from main_logic import llm
from main_logic import paraphrase as pp
from scripts import datasets as ds


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate(todo, path, log):
    """Phase 2: candidates for the questions not yet in candidates.jsonl."""
    with llm.Server(log=log) as server, open(path, "a", encoding="utf8") as f:
        for n, q in enumerate(todo, 1):
            cands = []
            for i in range(pp.N_CANDIDATES):
                seed = pp.SEED + i
                raw, info = server.chat(pp.messages(q["question"]), pp.TEMPERATURE, seed,
                                        pp.MAX_TOKENS)
                cands.append({"seed": seed, "raw": raw, "text": pp.clean(raw), **info})
            f.write(json.dumps({**q, "candidates": cands}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {n}/{len(todo)} {q['cluster_id']}", flush=True)


def screen_all(rows):
    """Phase 3: gate results for every candidate of every question."""
    nli = pp.load_nli()
    entail = pp.entailment_index(nli[1].config.id2label)
    origs = [r["question"] for r in rows for _ in r["candidates"]]
    cands = [c["text"] for r in rows for c in r["candidates"]]
    fwd = pp.nli_probs(origs, cands, nli)
    bwd = pp.nli_probs(cands, origs, nli)
    del nli

    embedder = pp.load_embedder()
    eo, ec = pp.embed(origs, embedder), pp.embed(cands, embedder)
    cos = (eo * ec).sum(1)
    del embedder

    out, j = [], 0
    for r in rows:
        n = len(r["candidates"])
        texts = [c["text"] for c in r["candidates"]]
        out.append(pp.screen(r["question"], r["answer"], texts, fwd[j:j + n], bwd[j:j + n],
                             cos[j:j + n], entail))
        j += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=ds.NAMES, required=True)
    ap.add_argument("--n", type=int, required=True, help="originals to sample")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.outdir or ds.DATA / "stage1" / args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    timing = {}

    # 1. Sample.
    pool = cp.load_questions(ds.QUESTIONS[args.dataset])
    sample = []
    for idx in pp.sample(len(pool), args.n):
        q = pool[idx]
        sample.append({"cluster_id": f"{args.dataset}-{idx:05d}", "dataset": args.dataset,
                       "source_index": idx, **q})

    # 2. Generate (resumable).
    cand_path = out / "candidates.jsonl"
    done = {r["cluster_id"] for r in read_jsonl(cand_path)}
    todo = [q for q in sample if q["cluster_id"] not in done]
    t = time.time()
    if todo:
        print(f"generating {pp.N_CANDIDATES} candidates for {len(todo)} questions", flush=True)
        generate(todo, cand_path, out / "llama-server.log")
    timing["generate_s"] = time.time() - t
    by_id = {r["cluster_id"]: r for r in read_jsonl(cand_path)}
    rows = [by_id[q["cluster_id"]] for q in sample]

    # 3. Screen and select.
    t = time.time()
    screened = screen_all(rows)
    timing["screen_s"] = time.time() - t

    reasons, complete = collections.Counter(), 0
    with open(out / "clusters.jsonl", "w", encoding="utf8") as f:
        for r, sc in zip(rows, screened):
            chosen = pp.select(sc)
            phrasings = [r["question"]] + chosen
            complete += len(phrasings) == 1 + pp.PER_CLUSTER
            reasons.update(x["reason"] or "passed" for x in sc)
            f.write(json.dumps({
                "cluster_id": r["cluster_id"], "dataset": r["dataset"],
                "source_index": r["source_index"], "answer": r["answer"],
                **{k: r[k] for k in ("_id", "type", "level") if k in r},
                "phrasings": [{"variant_id": i, "text": p} for i, p in enumerate(phrasings)],
                "complete": len(phrasings) == 1 + pp.PER_CLUSTER,
                "screening": sc,
            }, ensure_ascii=False) + "\n")

    gen = [c for r in rows for c in r["candidates"]]
    report = {
        "dataset": args.dataset, "source": ds.QUESTIONS[args.dataset].name,
        "n_originals": len(rows), "n_complete": complete, "n_incomplete": len(rows) - complete,
        "n_candidates": len(gen), "candidate_outcomes": dict(reasons),
        "pass_rate": reasons["passed"] / max(len(gen), 1),
        "settings": {"prompt": pp.PROMPT, "n_candidates": pp.N_CANDIDATES,
                     "temperature": pp.TEMPERATURE, "max_tokens": pp.MAX_TOKENS,
                     "seed": pp.SEED, "cosine_min": pp.COSINE_MIN,
                     "generator": llm.MODEL.name, "gpu_layers": llm.GPU_LAYERS,
                     "nli": f"{pp.NLI_MODEL}@{pp.NLI_REVISION}",
                     "embedder": f"{pp.EMBED_MODEL}@{pp.EMBED_REVISION}"},
        "generation_s_per_candidate": sum(c["seconds"] for c in gen) / max(len(gen), 1),
        "timing": timing,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
