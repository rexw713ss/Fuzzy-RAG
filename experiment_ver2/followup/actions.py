"""Freshly execute A1, original v0.4 A2 and a candidate-union A2 intervention.

All actions independently pay initial retrieval/rerank and use the same answer
prompt/decoding/context cap. No old answer or latency is reused. The new action
is explicitly named A2-union and cannot silently replace archived A2 labels.
"""

from dataclasses import dataclass
import time

import numpy as np

from main_logic import actions as ac, answers as an, signals as sg


ACTIONS = ("A1", "A2", "A2-union")


@dataclass(frozen=True)
class ActionConfig:
    top_n: int = 5
    initial_pool: int = 20
    second_pool: int = 20
    max_passage_words: int = 200
    temperature: float = 0.0
    seed: int = 42
    answer_tokens: int = 32
    followup_tokens: int = 64


def _hits(row):
    values = row["fused"]
    if len(values) < 5 or len({str(d) for d, _ in values}) != len(values):
        raise ValueError("retrieval needs at least five distinct fused passages")
    if any(not np.isfinite(float(score)) for _, score in values):
        raise ValueError("retrieval scores must be finite")
    return [str(d) for d, _ in values]


def union_ids(initial_ids, second_ids):
    """All candidates from the two top-20 pools, de-duplicated stably.

    Keeping the original candidate pool prevents irrevocable pre-rerank removal
    of A1 evidence. Top-5 selection still may drop it; retention is logged.
    """
    return list(dict.fromkeys(list(initial_ids) + list(second_ids)))


def run_action(question, action, backend, config=ActionConfig(), clock=time.perf_counter):
    """Backend implements retrieve, passages, vectors, rerank and chat.

    Failures propagate to the journal, never become zero-latency success rows.
    Only an empty/unchanged successful follow-up triggers the declared fallback.
    Gold answers are not accepted here and cannot influence retrieval/generation.
    """
    if action not in ACTIONS or not isinstance(question, str) or not question.strip():
        raise ValueError("valid action and nonempty question required")
    if config != ActionConfig():
        raise ValueError("this protocol freezes prompt, decoding, pools and context limits; version the protocol before changing them")
    started = clock()
    trace, stages, llm_calls, input_tokens, output_tokens = [], {}, 0, 0, 0
    ce_passes, retrievals = 0, 0

    def measured(name, fn):
        before = clock()
        value = fn()
        stages[name] = clock() - before
        return value

    def passages(ids):
        result = backend.passages(ids)
        if set(result) != set(ids):
            raise ValueError("missing/extra passage text in backend response")
        if any(not isinstance(t, (list, tuple)) or len(t) != 2 or any(not isinstance(v, str) for v in t) for t in result.values()):
            raise ValueError("passages require (title,text) strings")
        return result

    def rerank(name, query, ids, texts):
        nonlocal ce_passes
        scores = np.asarray(measured(name, lambda: backend.rerank(query, [texts[d] for d in ids])), float)
        if scores.shape != (len(ids),) or not np.isfinite(scores).all():
            raise ValueError("cross-encoder scores must be finite and aligned")
        ce_passes += len(ids)
        trace.append({"stage": name, "query": query, "doc_ids": ids, "scores": scores.tolist()})
        return [ids[i] for i in ac.rerank_order(scores)]

    def prompt_passages(ids, texts):
        return [(texts[d][0], " ".join(texts[d][1].split()[:config.max_passage_words])) for d in ids]

    def chat(name, messages, maximum):
        nonlocal llm_calls, input_tokens, output_tokens
        raw, info = measured(name, lambda: backend.chat(messages, temperature=config.temperature, seed=config.seed, max_tokens=maximum))
        if not isinstance(raw, str):
            raise ValueError("generator must return a string")
        for key in ("input_tokens", "output_tokens"):
            if type(info.get(key)) is not int or info[key] < 0:
                raise ValueError("measured integer token usage required")
        llm_calls += 1
        input_tokens += info["input_tokens"]
        output_tokens += info["output_tokens"]
        trace.append({"stage": name, "messages": messages, "raw": raw, "usage": info,
                      "decoding": {"temperature": config.temperature, "seed": config.seed, "max_tokens": maximum}})
        return raw

    initial = measured("initial_retrieval", lambda: backend.retrieve(question))
    retrievals += 1
    initial_ids = _hits(initial)[:config.initial_pool]
    texts = measured("initial_passage_lookup", lambda: passages(initial_ids))
    ranked = rerank("initial_rerank", question, initial_ids, texts)
    a1_top = ranked[:config.top_n]
    # Features are computed before any follow-up or answer, so no A2 leakage.
    signal_result = measured("signals", lambda: sg.compute_signals(
        initial["fused"], [str(d) for d, _ in initial["bm25"][:sg.K]],
        [str(d) for d, _ in initial["dense"][:sg.K]],
        backend.vectors([str(d) for d, _ in initial["fused"][:sg.K]])))
    followup, raw_followup, fallback, context = None, None, False, a1_top
    second = None
    if action != "A1":
        raw_followup = chat("followup", ac.followup_messages(question, prompt_passages(a1_top, texts)), config.followup_tokens)
        followup = ac.first_line(raw_followup)
        fallback = not ac.rewrite_usable(question, followup)
        if not fallback:
            second = measured("second_retrieval", lambda: backend.retrieve(followup))
            retrievals += 1
            second_ids = _hits(second)[:config.second_pool]
            if action == "A2":
                new_texts = measured("second_passage_lookup", lambda: passages(second_ids))
                texts.update(new_texts)
                second_ranked = rerank("second_rerank", followup, second_ids, texts)
                context, _ = ac.second_hop_context(a1_top, second_ranked)
            else:
                combined = union_ids(initial_ids, second_ids)
                new_texts = measured("union_passage_lookup", lambda: passages(combined))
                texts.update(new_texts)
                context = rerank("union_rerank", question, combined, texts)[:config.top_n]
    if len(context) != config.top_n or len(set(context)) != config.top_n:
        raise ValueError("every action must generate with five distinct passages")
    raw = chat("generation", ac.answer_messages(question, prompt_passages(context, texts)), config.answer_tokens)
    elapsed = clock() - started
    components = sum(stages.values())
    return {"action": action, "answer": ac.first_line(raw), "raw_answer": raw,
            "followup": followup, "raw_followup": raw_followup, "fallback": fallback,
            "doc_ids": context, "a1_top_ids": a1_top,
            "a1_top_retained": len(set(context) & set(a1_top)), "n_new": len(set(context) - set(a1_top)),
            "initial_retrieval": initial, "second_retrieval": second, "signals": signal_result,
            "passages": {d: {"title": texts[d][0], "text": texts[d][1]} for d in texts},
            "trace": trace, "stage_seconds": stages, "latency_total": elapsed,
            "component_seconds": components, "unattributed_seconds": elapsed - components,
            "retrievals": retrievals, "ce_passes": ce_passes, "llm_calls": llm_calls,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "latency_scope": "single-query wall clock including lookup/signals; warmed backend; excludes one-time model/index/server load and router allocation",
            "answer_prompt": ac.ANSWER_PROMPT, "newly_executed": True}


def score_outcome(outcome, answers):
    """Gold enters only after action execution has returned an answer."""
    if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
        raise ValueError("gold answer list required for scoring")
    return {**outcome, "EM": float(an.exact_match(outcome["answer"], answers)),
            "F1": float(an.f1_score(outcome["answer"], answers))}
