"""Pipeline step 8: the action-outcome matrix (spec v0.3, sections 1 and 7.4).

Run from the repo root, in the main environment, after scripts.retrieve_clusters, in two
processes so the GPU is free for llama-server during generation:
    .venv\\Scripts\\python.exe -m scripts.outcomes --dataset nq [--outdir DIR] --stop-after selection
    .venv\\Scripts\\python.exe -m scripts.outcomes --dataset nq [--outdir DIR]

Runs A0, A1 and A2 on every query instance once and caches one row per (instance, action), so
every router afterwards is a selection over cached rows. Four phases, each written to disk and
skipped when its output already exists, so an interrupted run continues where it stopped:

  1. rewrites      the A2 rewrite of every question (LLM, temperature 0), cached on the exact
                   question string (section 1.1)                       -> rewrites.jsonl
  2. re-retrieval  hybrid retrieval of every usable rewrite             -> rewrite_retrieval.jsonl
  3. selection     A0: fused top 5. A1: cross-encoder over the fused top 20 -> top 5.
                   A2: merge (section 1.2) -> cross-encoder over the top 30 -> top 5; an
                   unusable rewrite falls back to A1 with a flag         -> selections.jsonl,
                                                                            passages.jsonl
  4. generation    one answer per (instance, action) on its 5 passages, original question,
                   same prompt and decoding for all actions             -> generations.jsonl
Then the rows of section 7.4                                             -> outcomes.jsonl

Costs (section 3.4). Latencies in seconds: retrieval = per-query share of the batched initial
retrieval (+ the second retrieval for A2), rerank = cross-encoder time for this instance, rewrite
and generation = the LLM call's wall-clock time. latency_total is their sum. Secondary counts:
retrievals, ce_passes, llm_calls, input_tokens, output_tokens. Identical generation prompts are
generated once and the row reuses that call's measured latency and tokens.

Passages longer than 200 words are cut to their first 200 words in the generation prompt (DPR
passages are 100 words; only long HotpotQA abstracts are affected), so 5 passages always fit the
4,096-token context.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from main_logic import actions as ac
from main_logic import answers as an
from main_logic import corpus as cp
from main_logic import llm
from main_logic.routing import ACTIONS
from scripts import datasets as ds
from scripts.retrieve_clusters import hybrid, is_yes_no, read_jsonl, write_jsonl

MAX_WORDS = 200
# Spill guard: normal prompt speed on the P2200 is ~160-280 tokens/s; with the model partly in
# Windows shared memory it fell to ~80 (full run, Oct 11-12). Generation stops after SLOW_CALLS
# such calls in a row, so inflated latencies never enter the outcome rows.
MIN_PROMPT_TPS = 120
SLOW_CALLS = 5


def append_jsonl(f, row):
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()


def phase_rewrites(queries, out):
    path = out / "rewrites.jsonl"
    done = {r["question"] for r in (read_jsonl(path) if path.exists() else [])}
    todo = sorted({q["question"] for q in queries} - done)
    if todo:
        print(f"rewrites: {len(todo)} questions", flush=True)
        with llm.Server(log=out / "llama-server-rewrite.log") as server, \
                open(path, "a", encoding="utf8") as f:
            for i, q in enumerate(todo, 1):
                try:
                    raw, info = server.chat(ac.rewrite_messages(q), ac.TEMPERATURE, ac.SEED,
                                            ac.REWRITE_MAX_TOKENS)
                    rewrite, error = ac.first_line(raw), None
                except Exception as e:                  # a failed call is an A2 fallback
                    raw, rewrite, error = None, None, repr(e)
                    info = {"seconds": 0.0, "input_tokens": 0, "output_tokens": 0}
                append_jsonl(f, {"question": q, "raw": raw, "rewrite": rewrite, "error": error,
                                 **info})
                if i % 50 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)
    return {r["question"]: r for r in read_jsonl(path)}


def phase_reretrieval(queries, rewrites, dataset, out):
    path = out / "rewrite_retrieval.jsonl"
    usable = sorted({q["question"] for q in queries
                     if ac.rewrite_usable(q["question"], rewrites[q["question"]]["rewrite"])})
    if not path.exists():
        print(f"re-retrieval: {len(usable)} usable rewrites", flush=True)
        qs = [{"question": rewrites[q]["rewrite"], "answer": ["-"]} for q in usable]
        rows, _, timing = hybrid(qs, dataset, out, "rewrite") if qs else ([], None, {})
        write_jsonl(path, [{"question": q, "rewrite": rewrites[q]["rewrite"], "fused": r["fused"],
                            "retrieval_latency": r["retrieval_latency"]}
                           for q, r in zip(usable, rows)])
        (out / "rewrite_retrieval_timing.json").write_text(json.dumps(timing, indent=2))
    return {r["question"]: r for r in read_jsonl(path)}


def phase_selection(queries, initial, second, dataset, out):
    sel_path, text_path = out / "selections.jsonl", out / "passages.jsonl"
    if sel_path.exists() and text_path.exists():
        return read_jsonl(sel_path), {r["id"]: (r["title"], r["text"]) for r in read_jsonl(text_path)}

    pools = {}
    for q in queries:
        key = (q["cluster_id"], q["variant_id"])
        fused = initial[key]["fused"]
        a2 = second.get(q["question"])
        pools[key] = (fused[:ac.A1_RERANK],
                      ac.merge_pools(fused, a2["fused"]) if a2 else None)
    needed = {doc for a1, a2 in pools.values() for doc, _ in a1 + (a2 or [])}
    t = time.time()
    texts = cp.load_passages(ds.CORPUS[dataset], needed)
    print(f"selection: read {len(needed)} passages in {time.time() - t:.0f} s", flush=True)

    ce = ac.load_cross_encoder()
    rows = []
    for q in queries:
        key = (q["cluster_id"], q["variant_id"])
        a1_pool, a2_pool = pools[key]
        row = {"cluster_id": key[0], "variant_id": key[1],
               "A0": {"doc_ids": [d for d, _ in initial[key]["fused"][:ac.TOP_N]]}}
        for name, pool in (("A1", a1_pool), ("A2", a2_pool)):
            if pool is None:                            # unusable rewrite: A2 falls back to A1
                row["A2"] = {**row["A1"], "fallback": True}
                continue
            ids = [d for d, _ in pool]
            t = time.time()
            scores = ac.cross_encode(q["question"], [texts[d] for d in ids], ce)
            latency = time.time() - t
            order = ac.rerank_order(scores)
            row[name] = {"doc_ids": [ids[i] for i in order[:ac.TOP_N]], "rerank_latency": latency,
                         "ce_passes": len(ids), "fallback": False}
        rows.append(row)
    del ce
    used = {d for r in rows for a in ACTIONS for d in r[a]["doc_ids"]}
    write_jsonl(text_path, [{"id": d, "title": texts[d][0], "text": texts[d][1]} for d in sorted(used)])
    write_jsonl(sel_path, rows)
    return rows, {d: texts[d] for d in used}


def prompt_passages(doc_ids, texts):
    return [(texts[d][0], " ".join(texts[d][1].split()[:MAX_WORDS])) for d in doc_ids]


def phase_generation(queries, selections, texts, out):
    path = out / "generations.jsonl"
    done = {(r["cluster_id"], r["variant_id"], r["action"]) for r in
            (read_jsonl(path) if path.exists() else [])}
    sel = {(s["cluster_id"], s["variant_id"]): s for s in selections}
    todo = [(q, a) for q in queries for a in ACTIONS
            if (q["cluster_id"], q["variant_id"], a) not in done]
    if todo:
        print(f"generation: {len(todo)} calls", flush=True)
        cache, slow_run, pending = {}, 0, []
        for r in (read_jsonl(path) if path.exists() else []):
            cache.setdefault(r["prompt_key"], r)
        with llm.Server(log=out / "llama-server-generate.log") as server, \
                open(path, "a", encoding="utf8") as f:
            for i, (q, a) in enumerate(todo, 1):
                docs = sel[(q["cluster_id"], q["variant_id"])][a]["doc_ids"]
                msgs = ac.answer_messages(q["question"], prompt_passages(docs, texts))
                pkey = json.dumps(msgs, ensure_ascii=False)
                if pkey in cache:
                    c = cache[pkey]
                    raw, info = c["raw"], {k: c[k] for k in ("seconds", "input_tokens", "output_tokens")}
                    reused = True
                else:
                    raw, info = server.chat(msgs, ac.TEMPERATURE, ac.SEED, ac.ANSWER_MAX_TOKENS)
                    reused = False
                    tps = info.get("prompt_tps")
                    slow_run = slow_run + 1 if tps is not None and tps < MIN_PROMPT_TPS else 0
                    if slow_run >= SLOW_CALLS:
                        # The model has spilled out of VRAM (Windows shared memory): latencies
                        # are inflated. The slow rows held in `pending` are dropped, not written;
                        # rerun after llama-server restarts to resume from the last good row.
                        raise RuntimeError(f"{SLOW_CALLS} calls in a row below {MIN_PROMPT_TPS} "
                                           "prompt tokens/s; rerun to resume")
                row = {"cluster_id": q["cluster_id"], "variant_id": q["variant_id"], "action": a,
                       "prompt_key": pkey, "raw": raw, "reused": reused, **info}
                cache.setdefault(pkey, row)
                # Rows wait while a slow run is in progress; a fast call shows it was a passing
                # spike, and they are written then.
                pending.append(row)
                if slow_run == 0:
                    for r in pending:
                        append_jsonl(f, r)
                    pending = []
                if i % 50 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)
            for r in pending:
                append_jsonl(f, r)
    return {(r["cluster_id"], r["variant_id"], r["action"]): r for r in read_jsonl(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=ds.NAMES, required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--stop-after", choices=("selection",), default=None,
                    help="stop after phase 3; run the script again for generation")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.outdir or ds.DATA / "stage1" / args.dataset)

    queries = read_jsonl(out / "queries.jsonl")
    initial = {(r["cluster_id"], r["variant_id"]): r for r in read_jsonl(out / "retrieval.jsonl")}

    rewrites = phase_rewrites(queries, out)
    second = phase_reretrieval(queries, rewrites, args.dataset, out)
    selections, texts = phase_selection(queries, initial, second, args.dataset, out)
    if args.stop_after == "selection":
        # Phases 2-3 used the GPU (Contriever, cross-encoder) and PyTorch keeps that memory
        # until the process exits; generation in the same process pushed llama-server partly
        # out of VRAM (full run, Oct 11: ~10 s per call instead of ~5 s, inflating latency).
        # Run generation in a fresh process, where finished phases are read from disk.
        return
    gens = phase_generation(queries, selections, texts, out)

    rows = []
    sel = {(s["cluster_id"], s["variant_id"]): s for s in selections}
    for q in queries:
        key = (q["cluster_id"], q["variant_id"])
        rw = rewrites[q["question"]]
        for a in ACTIONS:
            s, g = sel[key][a], gens[key + (a,)]
            answer = ac.first_line(g["raw"])
            fallback = s.get("fallback", False)
            retrieval = initial[key]["retrieval_latency"]
            retrievals = 1
            if a == "A2" and not fallback:
                retrieval += second[q["question"]]["retrieval_latency"]
                retrievals = 2
            rewrite_latency = rw["seconds"] if a == "A2" else 0.0
            llm_in = g["input_tokens"] + (rw["input_tokens"] if a == "A2" else 0)
            llm_out = g["output_tokens"] + (rw["output_tokens"] if a == "A2" else 0)
            row = {
                "cluster_id": key[0], "variant_id": key[1], "action": a,
                "doc_ids": s["doc_ids"], "answer": answer,
                "EM": an.exact_match(answer, q["answer"]),
                "F1": an.f1_score(answer, q["answer"]),
                "Recall@5": (None if is_yes_no(q["answer"]) else
                             float(any(an.has_answer(q["answer"], texts[d][1]) for d in s["doc_ids"]))),
                "retrieval_latency": retrieval,
                "rerank_latency": s.get("rerank_latency", 0.0),
                "rewrite_latency": rewrite_latency,
                "generation_latency": g["seconds"],
                "retrievals": retrievals,
                "ce_passes": s.get("ce_passes", 0),
                "llm_calls": 2 if a == "A2" else 1,
                "input_tokens": llm_in, "output_tokens": llm_out,
                "fallback": fallback if a == "A2" else False,
                "rewrite": rw["rewrite"] if a == "A2" else None,
            }
            row["latency_total"] = (row["retrieval_latency"] + row["rerank_latency"]
                                    + row["rewrite_latency"] + row["generation_latency"])
            rows.append(row)
    write_jsonl(out / "outcomes.jsonl", rows)

    summary = {a: {k: sum(r[k] for r in rows if r["action"] == a) / len(queries)
                   for k in ("EM", "F1", "latency_total")} for a in ACTIONS}
    summary["A2_fallbacks"] = sum(r["fallback"] for r in rows)
    summary["generations_reused"] = sum(g["reused"] for g in gens.values())
    print(json.dumps({"dataset": args.dataset, "instances": len(queries), "mean": summary}, indent=2))


if __name__ == "__main__":
    main()
