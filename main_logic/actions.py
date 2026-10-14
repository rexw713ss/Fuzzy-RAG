"""The three actions (spec v0.3, section 1).

    A0  top-5 of the fused pool -> generate
    A1  top-20 of the fused pool -> cross-encoder -> new top-5 -> generate
    A2  rewrite -> second hybrid retrieval -> merge (section 1.2) -> cross-encoder on the
        top-30 -> new top-5 -> generate

All three end with one generation call on exactly 5 passages, with the same prompt and decoding
(the confound control of section 1). merge_pools is the section 1.2 merge; rewrite_usable decides
when A2 falls back to A1.

Stage 3 pins (spec 9.3; chosen by Claude as defaults when the user asked to finish the
experiment, Oct 11, 2026 -- DECISIONS.md section 14):
  - cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2, scoring (original question, passage
    "title text") pairs; ties keep the earlier pool rank;
  - answer prompt (ANSWER_PROMPT): short answer only; passages numbered with their titles,
    in rank order; the ORIGINAL question for every action;
  - rewriter: the generator LLM with REWRITE_PROMPT, temperature 0, seed 42 (section 1.1);
  - decoding: temperature 0, seed 42; at most 32 answer tokens, 64 rewrite tokens.
torch and transformers are imported only by the cross-encoder functions.

v0.4 (REDESIGN_2026-10-15, user decision Oct 15): A0 is merged into A1, and A2 becomes an
evidence-aware second hop:

    A2  follow-up query (the generator reads the question and A1's top 5) -> second hybrid
        retrieval -> cross-encoder on the second pool's top 20 against the FOLLOW-UP query ->
        A1's top 3 + the 2 best new passages -> generate
        V1: the A1 answer prompt;  V2: a short reasoning step, then "Answer: ..."

followup_messages, reason_messages, final_answer and second_hop_context are the v0.4 pieces;
the v0.3 functions above them are unchanged.
"""

import numpy as np

from main_logic import signals as sg

TOP_N = 5           # passages given to the generator, every action (section 1)
A1_RERANK = 20      # passages the cross-encoder re-scores for A1 (section 1)
A2_RERANK = 30      # passages the cross-encoder re-scores after the A2 merge (section 1)

CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"

ANSWER_PROMPT = ("Answer the question using the passages below. Reply with the answer only: a "
                 "short phrase of a few words, with no explanation. If it is a yes/no question, "
                 "reply yes or no.")
REWRITE_PROMPT = ("Rewrite this question as a search query that will find Wikipedia passages "
                  "answering it. Keep every name and detail in the question. Reply with the "
                  "query only.")
TEMPERATURE = 0.0
SEED = 42
ANSWER_MAX_TOKENS = 32
REWRITE_MAX_TOKENS = 64

# v0.4 A2 (REDESIGN_2026-10-15 section 4.2).
FOLLOWUP_PROMPT = ("Below are a question and passages retrieved for it. Write one search query "
                   "that would find the information still missing to answer the question, such "
                   "as a person, place or work that the passages mention and the question "
                   "depends on. Reply with the query only.")
REASON_PROMPT = ("Answer the question using the passages below. Reply in exactly this format:\n"
                 "Reasoning: <at most two short sentences>\n"
                 "Answer: <a short phrase of a few words, with no explanation>\n"
                 "If it is a yes/no question, the answer is yes or no.")
FOLLOWUP_MAX_TOKENS = 64
REASON_MAX_TOKENS = 192
SECOND_RERANK = 20  # second-hop passages the cross-encoder re-scores (against the follow-up)
A2_KEEP = 3         # A1's top passages kept in A2's context
A2_NEW = 2          # best second-hop passages added (not already in A1's top 5)

_LABELS = ("answer:", "search query:", "query:", "rewritten query:", "rewritten question:")


def answer_messages(question, passages):
    """Chat messages for one generation. passages: [(title, text), ...] in rank order."""
    return [{"role": "user",
             "content": f"{ANSWER_PROMPT}\n\n{_passage_body(passages)}\n\nQuestion: {question}\nAnswer:"}]


def rewrite_messages(question):
    """Chat messages for the A2 rewrite (section 1.1)."""
    return [{"role": "user", "content": f"{REWRITE_PROMPT}\n\nQuestion: {question}"}]


def _passage_body(passages):
    return "\n\n".join(f"Passage {i} ({title}): {text}"
                       for i, (title, text) in enumerate(passages, 1))


def followup_messages(question, passages):
    """v0.4 A2: chat messages asking for one follow-up search query, given A1's passages."""
    return [{"role": "user", "content": f"{FOLLOWUP_PROMPT}\n\n{_passage_body(passages)}\n\n"
                                        f"Question: {question}\nSearch query:"}]


def reason_messages(question, passages):
    """v0.4 A2 variant V2: a short reasoning step before the answer."""
    return [{"role": "user", "content": f"{REASON_PROMPT}\n\n{_passage_body(passages)}\n\n"
                                        f"Question: {question}"}]


def final_answer(text):
    """The answer of a V2 (reasoning) reply: what follows the last "Answer:", or the last
    non-empty line if there is none (e.g. the reply was cut off). Cleaned like first_line."""
    t = text or ""
    i = t.lower().rfind("answer:")
    if i >= 0:
        return first_line(t[i + len("answer:"):])
    lines = [line for line in t.splitlines() if line.strip()]
    return first_line(lines[-1]) if lines else ""


def second_hop_context(a1_top, second_ranked, keep=A2_KEEP, new=A2_NEW, top_n=TOP_N):
    """v0.4 A2 passages: A1's first `keep`, then the best `new` second-hop passages that are not
    in A1's top_n, then A1's next passages if fewer than `new` were found.

    a1_top: A1's top_n doc ids in rank order; second_ranked: second-hop doc ids, best first.
    Returns (doc ids, number of second-hop passages used).
    """
    a1_top = list(a1_top)[:top_n]
    seen = set(a1_top)
    fresh = []
    for d in second_ranked:
        if d not in seen and d not in fresh:
            fresh.append(d)
        if len(fresh) == new:
            break
    ctx = a1_top[:keep] + fresh
    for d in a1_top[keep:]:
        if len(ctx) >= top_n:
            break
        ctx.append(d)
    return ctx[:top_n], len(fresh)


def first_line(text):
    """The reply's first non-empty line, without a leading "Answer:"-style label or
    surrounding quotes. "" if nothing is left."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    s = lines[0]
    if s.lower().startswith(_LABELS):
        s = s.split(":", 1)[1].strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def rerank_order(ce_scores):
    """Pool positions sorted by cross-encoder score, descending; ties keep pool order."""
    s = np.asarray(ce_scores, dtype=float)
    return sorted(range(len(s)), key=lambda i: (-s[i], i))


def load_cross_encoder(device=None):
    """Tokenizer and cross-encoder at the pinned revision, in eval mode."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(CROSS_ENCODER, revision=CROSS_ENCODER_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        CROSS_ENCODER, revision=CROSS_ENCODER_REVISION).to(device).eval()
    return tok, model


def cross_encode(question, passages, ce, batch_size=32):
    """Relevance score of each passage ("title text") for the question, in passage order."""
    import torch

    tok, model = ce
    device = next(model.parameters()).device
    texts = [f"{title} {text}" for title, text in passages]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            x = tok([question] * len(chunk), chunk, padding=True, truncation=True,
                    max_length=512, return_tensors="pt").to(device)
            out.append(model(**x).logits.squeeze(-1).float().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def merge_pools(original, rewrite, k=A2_RERANK):
    """A2 merge rule (section 1.2). Returns the merged top-k as [(doc_id, score), ...].

    original, rewrite: fused pools as returned by signals.fuse, [(doc_id, score), ...].
    1. Union of both pools.
    2. A passage in both keeps max(score_original, score_rewrite), so ids are unique.
    3. Min-max normalize over the whole merged pool (signals' rule: a constant pool maps to 1.0).
    4. Sort descending, ties broken by doc id as in signals.fuse, and keep the top k.

    Step 3 is a straight-line rescaling, so it cannot change which passages make the top k or
    their order; the cross-encoder re-scores them anyway. It is kept because the spec asks for it.
    """
    best = {}
    for doc, score in list(original) + list(rewrite):
        score = float(score)
        if not np.isfinite(score):
            raise ValueError(f"non-finite score {score} for passage {doc}")
        if doc not in best or score > best[doc]:
            best[doc] = score
    if not best:
        return []
    scaled = sg._minmax(best)
    ranked = sorted(scaled.items(), key=lambda x: (-x[1], str(x[0])))
    return ranked[:k]


def rewrite_usable(original, rewrite):
    """False when A2 must fall back to A1 (section 1.2).

    Unusable: the rewrite call failed (rewrite is None), the rewrite is empty, or it is the
    original question once whitespace runs and letter case are ignored (spec v0.3: a rewrite that
    only changes capitalization or spacing is not a rewrite). A fallback still counts as an A2
    instance, with a flag in the results.
    """
    if rewrite is None:
        return False
    norm = " ".join(rewrite.split()).lower()
    return bool(norm) and norm != " ".join(original.split()).lower()
