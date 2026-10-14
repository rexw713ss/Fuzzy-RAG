"""v0.4 pipeline step 8: A1 (answer now) and the stronger A2 (escalate) outcome rows
(REDESIGN_2026-10-15, sections 3-4; user decision Oct 15, 2026).

Run from the repo root, in the main environment, ONE PHASE PER PROCESS (a phase that used
PyTorch keeps GPU memory until its process exits, which slows llama-server; RUNLOG Oct 11 §6):
    .venv\\Scripts\\python.exe -m scripts.escalate --dataset nq --src DIR --outdir DIR --phase a1
    ... --phase followups / second / selection / generation / outcomes
scripts/run_v4.sh runs the phases in order.

SRC holds queries.jsonl and retrieval.jsonl (steps 4-5). If SRC also holds the v0.3 step 8
files (selections.jsonl, generations.jsonl, outcomes.jsonl), A1 is reused from them: A1 is
unchanged in v0.4, so its passages, answers and measured latencies carry over.

Phases (each cached in OUTDIR and resumable):
  a1          A1's pool (fused top 20), cross-encoder scores (saved for later signal checks) and
              top 5. Reused A1: passages and rerank latency from SRC; the recomputed order is
              checked against them.                                         -> a1.jsonl
  followups   the generator reads the question and A1's top 5 and writes one follow-up search
              query (temperature 0, seed 42)                               -> followups.jsonl
  second      hybrid retrieval of every usable follow-up query      -> second_retrieval.jsonl
  selection   cross-encoder over the second pool's top 20 against the follow-up query; A2's
              context = A1's top 3 + the 2 best new passages. Unusable follow-up (failed, empty,
              or the question itself): no second hop, A2 context = A1's top 5, flagged
                                                                            -> selections.jsonl
  generation  A1's answer (if not reused) and A2's answer per variant: V1 = A1's answer prompt,
              V2 = short reasoning then "Answer: ..."                     -> generations.jsonl
  outcomes    one row per (instance, action)                               -> outcomes.jsonl

Pilot (REDESIGN §4.3): --pilot-clusters N takes N complete TRAIN clusters (seed 42) from
--splits; the chosen instances are written to OUTDIR/instances.json by the first phase.

Costs: retrieval = initial + second retrieval (per-query share of the batch); rerank = A1's
cross-encoder time + the second-hop cross-encoder time; followup = the follow-up call;
generation = the answer call. latency_total is their sum. A2 never pays for A1's answer:
routing happens after reranking, before generation (REDESIGN §3).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from main_logic import actions as ac
from main_logic import answers as an
from main_logic import corpus as cp
from main_logic import llm
from scripts import datasets as ds
from scripts.outcomes import MIN_PROMPT_TPS, SLOW_CALLS, append_jsonl, prompt_passages
from scripts.retrieve_clusters import hybrid, is_yes_no, read_jsonl, write_jsonl

PHASES = ("a1", "followups", "second", "selection", "generation", "outcomes")
VARIANTS = ("V1", "V2")
SEED = 42


def load_jsonl(path):
    return read_jsonl(path) if Path(path).exists() else []


def key_of(r):
    return (r["cluster_id"], r["variant_id"])


# ----------------------------------------------------------------------------- instances

def choose_instances(queries, src, splits_path, pilot_clusters, out, splits_only=None):
    """All query instances of SRC; or the pilot's (N complete train clusters, seed 42, fixed in
    OUTDIR/instances.json by the first phase); or those of some splits only (--splits-only, so
    train + val can run before the old test split)."""
    path = out / "instances.json"
    if pilot_clusters:
        if path.exists():
            keep = {tuple(k) for k in json.loads(path.read_text())}
            return [q for q in queries if key_of(q) in keep]
        dataset = src.name
        train = set(json.loads(Path(splits_path).read_text())[dataset]["train"])
        complete = sorted(c["cluster_id"] for c in read_jsonl(src / "clusters.jsonl")
                          if c["complete"] and c["cluster_id"] in train)
        perm = np.random.default_rng(SEED).permutation(len(complete))
        chosen = {complete[i] for i in perm[:pilot_clusters]}
        queries = [q for q in queries if q["cluster_id"] in chosen]
        path.write_text(json.dumps([list(key_of(q)) for q in queries]))
        return queries
    if splits_only:
        splits = json.loads(Path(splits_path).read_text())[src.name]
        keep = {c for s in splits_only for c in splits[s]}
        queries = [q for q in queries if q["cluster_id"] in keep]
    return queries


# ----------------------------------------------------------------------------- passage texts

def load_texts(out):
    return {r["id"]: (r["title"], r["text"]) for r in load_jsonl(out / "passages.jsonl")}


def save_texts(out, texts):
    write_jsonl(out / "passages.jsonl",
                [{"id": d, "title": t, "text": x} for d, (t, x) in sorted(texts.items())])


# ----------------------------------------------------------------------------- phases

def phase_a1(queries, initial, src, dataset, out):
    path = out / "a1.jsonl"
    done = {key_of(r) for r in load_jsonl(path)}
    todo = [q for q in queries if key_of(q) not in done]
    if not todo:
        return
    cached = {key_of(r): r["A1"] for r in load_jsonl(src / "selections.jsonl")}
    pools = {key_of(q): [d for d, _ in initial[key_of(q)]["fused"][:ac.A1_RERANK]] for q in todo}
    t = time.time()
    texts = cp.load_passages(ds.CORPUS[dataset], {d for p in pools.values() for d in p})
    print(f"a1: {len(todo)} instances; read {len(texts)} passages in {time.time() - t:.0f} s",
          flush=True)
    ce = ac.load_cross_encoder()
    keep_texts = load_texts(out)
    mismatches = 0
    with open(path, "a", encoding="utf8") as f:
        for q in todo:
            k = key_of(q)
            ids = pools[k]
            t = time.time()
            scores = ac.cross_encode(q["question"], [texts[d] for d in ids], ce)
            latency = time.time() - t
            top = [ids[i] for i in ac.rerank_order(scores)[:ac.TOP_N]]
            row = {"cluster_id": k[0], "variant_id": k[1], "pool": ids,
                   "ce_scores": [float(s) for s in scores]}
            if k in cached:
                row.update(doc_ids=cached[k]["doc_ids"], rerank_latency=cached[k]["rerank_latency"],
                           ce_passes=cached[k]["ce_passes"], reused=True,
                           order_matches=top == cached[k]["doc_ids"])
                mismatches += top != cached[k]["doc_ids"]
            else:
                row.update(doc_ids=top, rerank_latency=latency, ce_passes=len(ids), reused=False)
            for d in row["doc_ids"]:
                keep_texts[d] = texts[d]
            append_jsonl(f, row)
    save_texts(out, keep_texts)
    print(f"a1: done; reused A1 whose recomputed top 5 differs from the cached one: {mismatches}",
          flush=True)


class SpillGuard:
    """Stops a generation phase after SLOW_CALLS calls in a row below MIN_PROMPT_TPS (the model
    has spilled out of VRAM and latencies are inflated); the rows of such a run are held back
    and dropped. Same rule as scripts.outcomes."""

    def __init__(self):
        self.slow, self.pending = 0, []

    def add(self, f, row, tps):
        self.slow = self.slow + 1 if tps is not None and tps < MIN_PROMPT_TPS else 0
        if self.slow >= SLOW_CALLS:
            raise RuntimeError(f"{SLOW_CALLS} calls in a row below {MIN_PROMPT_TPS} prompt "
                               "tokens/s; rerun to resume")
        self.pending.append(row)
        if self.slow == 0:
            self.flush(f)

    def flush(self, f):
        for r in self.pending:
            append_jsonl(f, r)
        self.pending = []


def phase_followups(queries, out):
    path = out / "followups.jsonl"
    done = {key_of(r) for r in load_jsonl(path)}
    todo = [q for q in queries if key_of(q) not in done]
    if not todo:
        return
    a1 = {key_of(r): r for r in load_jsonl(out / "a1.jsonl")}
    texts = load_texts(out)
    print(f"followups: {len(todo)} calls", flush=True)
    guard = SpillGuard()
    with llm.Server(log=out / "llama-server-followups.log") as server, \
            open(path, "a", encoding="utf8") as f:
        for i, q in enumerate(todo, 1):
            k = key_of(q)
            msgs = ac.followup_messages(q["question"], prompt_passages(a1[k]["doc_ids"], texts))
            try:
                raw, info = server.chat(msgs, ac.TEMPERATURE, ac.SEED, ac.FOLLOWUP_MAX_TOKENS)
                followup, error = ac.first_line(raw), None
            except Exception as e:                      # a failed call: no second hop
                raw, followup, error = None, None, repr(e)
                info = {"seconds": 0.0, "input_tokens": 0, "output_tokens": 0, "prompt_tps": None}
            row = {"cluster_id": k[0], "variant_id": k[1], "question": q["question"],
                   "raw": raw, "followup": followup, "error": error, **info,
                   "usable": ac.rewrite_usable(q["question"], followup)}
            guard.add(f, row, info.get("prompt_tps"))
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
        guard.flush(f)


def phase_second(queries, dataset, out):
    path = out / "second_retrieval.jsonl"
    fol = {key_of(r): r for r in load_jsonl(out / "followups.jsonl")}
    done = {key_of(r) for r in load_jsonl(path)}
    todo = [q for q in queries if key_of(q) not in done and fol[key_of(q)]["usable"]]
    if not todo:
        return
    print(f"second: {len(todo)} follow-up queries", flush=True)
    qs = [{"question": fol[key_of(q)]["followup"], "answer": ["-"]} for q in todo]
    rows, _, timing = hybrid(qs, dataset, out, "second")
    with open(path, "a", encoding="utf8") as f:
        for q, r in zip(todo, rows):
            append_jsonl(f, {"cluster_id": q["cluster_id"], "variant_id": q["variant_id"],
                             "followup": fol[key_of(q)]["followup"], "fused": r["fused"],
                             "retrieval_latency": r["retrieval_latency"]})
    (out / f"second_timing_{len(done)}.json").write_text(json.dumps(timing, indent=2))


def phase_selection(queries, dataset, out):
    path = out / "selections.jsonl"
    done = {key_of(r) for r in load_jsonl(path)}
    todo = [q for q in queries if key_of(q) not in done]
    if not todo:
        return
    a1 = {key_of(r): r for r in load_jsonl(out / "a1.jsonl")}
    fol = {key_of(r): r for r in load_jsonl(out / "followups.jsonl")}
    second = {key_of(r): r for r in load_jsonl(out / "second_retrieval.jsonl")}
    pools = {key_of(q): [d for d, _ in second[key_of(q)]["fused"][:ac.SECOND_RERANK]]
             for q in todo if key_of(q) in second}
    t = time.time()
    texts = cp.load_passages(ds.CORPUS[dataset], {d for p in pools.values() for d in p})
    print(f"selection: {len(todo)} instances, {len(pools)} with a second hop; read "
          f"{len(texts)} passages in {time.time() - t:.0f} s", flush=True)
    ce = ac.load_cross_encoder() if pools else None
    keep_texts = load_texts(out)
    with open(path, "a", encoding="utf8") as f:
        for q in todo:
            k = key_of(q)
            a1_top = a1[k]["doc_ids"]
            row = {"cluster_id": k[0], "variant_id": k[1]}
            if k in pools:
                ids = pools[k]
                t = time.time()
                scores = ac.cross_encode(fol[k]["followup"], [texts[d] for d in ids], ce)
                latency = time.time() - t
                ranked = [ids[i] for i in ac.rerank_order(scores)]
                ctx, n_new = ac.second_hop_context(a1_top, ranked)
                row.update(fallback=False, second_pool=ids,
                           second_ce_scores=[float(s) for s in scores],
                           second_rerank_latency=latency, second_ce_passes=len(ids),
                           doc_ids=ctx, n_new=n_new)
                for d in ctx:
                    if d not in keep_texts:
                        keep_texts[d] = texts[d]
            else:
                row.update(fallback=True, second_rerank_latency=0.0, second_ce_passes=0,
                           doc_ids=list(a1_top), n_new=0)
            append_jsonl(f, row)
    save_texts(out, keep_texts)


def phase_generation(queries, src, out, variants):
    path = out / "generations.jsonl"
    done = {(r["cluster_id"], r["variant_id"], r["action"]) for r in load_jsonl(path)}
    a1 = {key_of(r): r for r in load_jsonl(out / "a1.jsonl")}
    sel = {key_of(r): r for r in load_jsonl(out / "selections.jsonl")}
    texts = load_texts(out)
    tasks = []
    for q in queries:
        k = key_of(q)
        if not a1[k]["reused"]:
            tasks.append((q, "A1"))
        tasks += [(q, f"A2-{v}") for v in variants]
    todo = [(q, a) for q, a in tasks if key_of(q) + (a,) not in done]
    if not todo:
        return
    # Identical prompts are generated once (temperature 0); v0.3's A1 generations seed the cache.
    cache = {}
    # generations_seed.jsonl: answers already measured for the same prompts (the pilot's).
    for r in (load_jsonl(src / "generations.jsonl") + load_jsonl(out / "generations_seed.jsonl")
              + load_jsonl(path)):
        cache.setdefault(r["prompt_key"], r)
    print(f"generation: {len(todo)} calls (variants {variants})", flush=True)
    guard = SpillGuard()
    with llm.Server(log=out / "llama-server-generate.log") as server, \
            open(path, "a", encoding="utf8") as f:
        for i, (q, a) in enumerate(todo, 1):
            k = key_of(q)
            if a == "A1":
                msgs = ac.answer_messages(q["question"], prompt_passages(a1[k]["doc_ids"], texts))
                max_tokens = ac.ANSWER_MAX_TOKENS
            elif a == "A2-V1":
                msgs = ac.answer_messages(q["question"], prompt_passages(sel[k]["doc_ids"], texts))
                max_tokens = ac.ANSWER_MAX_TOKENS
            else:
                msgs = ac.reason_messages(q["question"], prompt_passages(sel[k]["doc_ids"], texts))
                max_tokens = ac.REASON_MAX_TOKENS
            pkey = json.dumps(msgs, ensure_ascii=False)
            if pkey in cache:
                c = cache[pkey]
                raw, reused = c["raw"], True
                info = {x: c.get(x) for x in ("seconds", "input_tokens", "output_tokens", "prompt_tps")}
                tps = None
            else:
                raw, info = server.chat(msgs, ac.TEMPERATURE, ac.SEED, max_tokens)
                reused, tps = False, info.get("prompt_tps")
            row = {"cluster_id": k[0], "variant_id": k[1], "action": a, "prompt_key": pkey,
                   "raw": raw, "reused": reused, **info}
            cache.setdefault(pkey, row)
            guard.add(f, row, tps)
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
        guard.flush(f)


def phase_outcomes(queries, initial, src, out, variants, final):
    a1 = {key_of(r): r for r in load_jsonl(out / "a1.jsonl")}
    fol = {key_of(r): r for r in load_jsonl(out / "followups.jsonl")}
    second = {key_of(r): r for r in load_jsonl(out / "second_retrieval.jsonl")}
    sel = {key_of(r): r for r in load_jsonl(out / "selections.jsonl")}
    gens = {(r["cluster_id"], r["variant_id"], r["action"]): r
            for r in load_jsonl(out / "generations.jsonl")}
    old = {key_of(r): r for r in load_jsonl(src / "outcomes.jsonl") if r["action"] == "A1"}
    texts = load_texts(out)

    def scored(q, docs, answer):
        yn = is_yes_no(q["answer"])
        return {"doc_ids": docs, "answer": answer,
                "EM": an.exact_match(answer, q["answer"]),
                "F1": an.f1_score(answer, q["answer"]),
                "Recall@5": None if yn else float(any(an.has_answer(q["answer"], texts[d][1])
                                                      for d in docs))}

    rows = []
    for q in queries:
        k = key_of(q)
        if a1[k]["reused"]:
            r = dict(old[k])
            r.pop("rewrite_latency", None)
            r.pop("rewrite", None)
            r.update(followup_latency=0.0, fallback=False, followup=None, n_new=0)
            rows.append(r)
        else:
            g = gens[k + ("A1",)]
            r = {"cluster_id": k[0], "variant_id": k[1], "action": "A1",
                 **scored(q, a1[k]["doc_ids"], ac.first_line(g["raw"])),
                 "retrieval_latency": initial[k]["retrieval_latency"],
                 "rerank_latency": a1[k]["rerank_latency"], "followup_latency": 0.0,
                 "generation_latency": g["seconds"], "retrievals": 1,
                 "ce_passes": a1[k]["ce_passes"], "llm_calls": 1,
                 "input_tokens": g["input_tokens"], "output_tokens": g["output_tokens"],
                 "fallback": False, "followup": None, "n_new": 0}
            r["latency_total"] = (r["retrieval_latency"] + r["rerank_latency"]
                                  + r["generation_latency"])
            rows.append(r)
        s, fu = sel[k], fol[k]
        hop = not s["fallback"]
        for v in variants:
            g = gens[k + (f"A2-{v}",)]
            answer = ac.first_line(g["raw"]) if v == "V1" else ac.final_answer(g["raw"])
            r = {"cluster_id": k[0], "variant_id": k[1],
                 "action": "A2" if final == v else f"A2-{v}",
                 **scored(q, s["doc_ids"], answer),
                 "retrieval_latency": initial[k]["retrieval_latency"]
                 + (second[k]["retrieval_latency"] if hop else 0.0),
                 "rerank_latency": a1[k]["rerank_latency"] + s["second_rerank_latency"],
                 "followup_latency": fu["seconds"],
                 "generation_latency": g["seconds"],
                 "retrievals": 2 if hop else 1,
                 "ce_passes": a1[k]["ce_passes"] + s["second_ce_passes"],
                 "llm_calls": 2,
                 "input_tokens": (fu["input_tokens"] or 0) + g["input_tokens"],
                 "output_tokens": (fu["output_tokens"] or 0) + g["output_tokens"],
                 "fallback": not hop, "followup": fu["followup"], "n_new": s["n_new"],
                 "variant": v, "raw": g["raw"]}
            r["latency_total"] = (r["retrieval_latency"] + r["rerank_latency"]
                                  + r["followup_latency"] + r["generation_latency"])
            rows.append(r)
    write_jsonl(out / "outcomes.jsonl", rows)

    labels = sorted({r["action"] for r in rows})
    summary = {a: {m: float(np.mean([r[m] for r in rows if r["action"] == a]))
                   for m in ("EM", "F1", "latency_total")} for a in labels}
    summary["A2_fallbacks"] = sum(fol[key_of(q)]["usable"] is False for q in queries)
    print(json.dumps({"instances": len(queries), "mean": summary}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=ds.NAMES, required=True)
    ap.add_argument("--src", required=True, help="folder with queries.jsonl, retrieval.jsonl")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--phase", choices=PHASES, required=True)
    ap.add_argument("--variants", default="V1", help="comma-separated: V1, V2")
    ap.add_argument("--final", choices=VARIANTS, default=None,
                    help="label this variant's rows 'A2' in outcomes.jsonl")
    ap.add_argument("--pilot-clusters", type=int, default=0)
    ap.add_argument("--splits", default=None, help="splits.json (default: SRC/../splits.json)")
    ap.add_argument("--splits-only", default=None, help="comma-separated splits, e.g. train,val")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    src, out = Path(args.src), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    variants = [v for v in args.variants.split(",") if v]
    if not set(variants) <= set(VARIANTS):
        raise SystemExit(f"unknown variant in {variants}")

    queries = choose_instances(read_jsonl(src / "queries.jsonl"), src,
                               args.splits or src.parent / "splits.json", args.pilot_clusters, out,
                               args.splits_only.split(",") if args.splits_only else None)
    initial = {key_of(r): r for r in read_jsonl(src / "retrieval.jsonl")}
    print(f"{args.dataset}: {len(queries)} instances, phase {args.phase}", flush=True)
    if args.phase == "a1":
        phase_a1(queries, initial, src, args.dataset, out)
    elif args.phase == "followups":
        phase_followups(queries, out)
    elif args.phase == "second":
        phase_second(queries, args.dataset, out)
    elif args.phase == "selection":
        phase_selection(queries, args.dataset, out)
    elif args.phase == "generation":
        phase_generation(queries, src, out, variants)
    else:
        phase_outcomes(queries, initial, src, out, variants, args.final)


if __name__ == "__main__":
    main()
