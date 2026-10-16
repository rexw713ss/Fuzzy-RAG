"""Validate generated evidence against action semantics, not presence flags."""
import numpy as np
from main_logic import actions as ac
from experiment_ver2.followup.actions import ACTIONS, ActionConfig, score_outcome, union_ids
from experiment_ver2.followup.io import read_json, read_jsonl, digest, json_digest


def validate_outcome(row, question, answers):
    action = row.get("action")
    if action not in ACTIONS or row.get("newly_executed") is not True:
        raise ValueError("newly executed known action required")
    trace = row.get("trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError("nonempty generation/reranking trace required")
    stages = {t.get("stage"): t for t in trace}
    if len(stages) != len(trace):
        raise ValueError("duplicate trace stages")
    cfg = ActionConfig()
    texts = row.get("passages", {})

    def context(ids):
        if len(ids) != 5 or len(set(ids)) != 5:
            raise ValueError("five distinct context documents required")
        try:
            return [(texts[d]["title"], " ".join(texts[d]["text"].split()[:cfg.max_passage_words])) for d in ids]
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("context passage text missing or invalid") from exc

    def rank(stage, query, ids):
        t = stages.get(stage, {})
        scores = np.asarray(t.get("scores", []), float)
        if t.get("query") != query or t.get("doc_ids") != ids or scores.shape != (len(ids),) or not np.isfinite(scores).all():
            raise ValueError("rerank query/documents/scores mismatch")
        return [ids[i] for i in ac.rerank_order(scores)]

    def chat(stage, messages, maximum):
        t = stages.get(stage, {})
        if t.get("messages") != messages or not isinstance(t.get("raw"), str):
            raise ValueError("generation trace prompt/raw response mismatch")
        if t.get("decoding") != {"temperature": 0.0, "seed": 42, "max_tokens": maximum}:
            raise ValueError("generation decoding differs from frozen protocol")
        usage = t.get("usage", {})
        if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("input_tokens", "output_tokens")):
            raise ValueError("nonnegative measured token counts required")
        return t["raw"]

    try:
        initial_ids = [str(d) for d, _ in row["initial_retrieval"]["fused"]][:20]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("initial retrieval evidence missing") from exc
    if len(initial_ids) < 5 or len(initial_ids) != len(set(initial_ids)):
        raise ValueError("invalid initial candidate pool")
    a1 = rank("initial_rerank", question, initial_ids)[:5]
    if row.get("a1_top_ids") != a1:
        raise ValueError("A1 context does not follow its ranking")
    expected_trace = {"initial_rerank", "generation"}
    required_times = {"initial_retrieval", "initial_passage_lookup", "initial_rerank", "signals", "generation"}
    expected_context = a1
    ce_count, retrievals = len(initial_ids), 1
    if action != "A1":
        raw_followup = chat("followup", ac.followup_messages(question, context(a1)), 64)
        followup = ac.first_line(raw_followup)
        fallback = not ac.rewrite_usable(question, followup)
        if row.get("raw_followup") != raw_followup or row.get("followup") != followup or row.get("fallback") is not fallback:
            raise ValueError("followup/fallback mismatch")
        expected_trace.add("followup")
        required_times.add("followup")
        if not fallback:
            try:
                second_ids = [str(d) for d, _ in row["second_retrieval"]["fused"]][:20]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("second retrieval evidence missing") from exc
            retrievals = 2
            required_times.add("second_retrieval")
            if action == "A2":
                ranked = rank("second_rerank", followup, second_ids)
                expected_context, _ = ac.second_hop_context(a1, ranked)
                ce_count += len(second_ids)
                expected_trace.add("second_rerank")
                required_times.update(("second_passage_lookup", "second_rerank"))
            else:
                combined = union_ids(initial_ids, second_ids)
                expected_context = rank("union_rerank", question, combined)[:5]
                ce_count += len(combined)
                expected_trace.add("union_rerank")
                required_times.update(("union_passage_lookup", "union_rerank"))
        elif row.get("second_retrieval") is not None:
            raise ValueError("fallback unexpectedly contains second retrieval")
    elif any(row.get(k) is not None for k in ("followup", "raw_followup", "second_retrieval")) or row.get("fallback") is not False:
        raise ValueError("A1 must not contain followup execution")
    if set(stages) != expected_trace or row.get("doc_ids") != expected_context:
        raise ValueError("action trace/context differs from frozen definition")
    raw = chat("generation", ac.answer_messages(question, context(expected_context)), 32)
    if row.get("raw_answer") != raw or row.get("answer") != ac.first_line(raw) or row.get("answer_prompt") != ac.ANSWER_PROMPT:
        raise ValueError("generated answer differs from trace")
    for name, count in (("ce_passes", ce_count), ("retrievals", retrievals), ("llm_calls", 1 + (action != "A1")),
                        ("a1_top_retained", len(set(a1) & set(expected_context))), ("n_new", len(set(expected_context) - set(a1)))):
        if type(row.get(name)) is not int or row[name] != count:
            raise ValueError(f"action accounting mismatch: {name}")
    for name in ("input_tokens", "output_tokens"):
        count = sum(t["usage"][name] for t in trace if "usage" in t)
        if type(row.get(name)) is not int or row[name] != count:
            raise ValueError("token totals disagree with trace")
    timing = row.get("stage_seconds", {})
    if set(timing) != required_times or any(type(v) not in (int, float) or not np.isfinite(v) or v < 0 for v in timing.values()):
        raise ValueError("required finite nonnegative stage timings missing")
    for name in ("latency_total", "component_seconds", "unattributed_seconds"):
        if type(row.get(name)) not in (int, float) or not np.isfinite(row[name]) or row[name] < 0:
            raise ValueError("invalid measured wall-clock cost")
    if row["latency_total"] <= 0 or abs(sum(timing.values()) - row["component_seconds"]) > 1e-8 or abs(row["component_seconds"] + row["unattributed_seconds"] - row["latency_total"]) > 1e-8:
        raise ValueError("stage/component/total wall-clock costs disagree")
    signals = row.get("signals", {})
    if any(s not in signals or not np.isfinite(signals[s]) for s in ("S1", "S2", "S3", "S4")):
        raise ValueError("four finite pre-action signals required")
    rescored = score_outcome(row, answers)
    if any(not np.isfinite(row.get(m, np.nan)) or abs(row[m] - rescored[m]) > 1e-12 for m in ("F1", "EM")):
        raise ValueError("scores disagree with generated answer and gold")
    return row


def validate_collection(directory, queries):
    """Reconcile full exported matrix with the durable planned-attempt journal."""
    from pathlib import Path
    from experiment_ver2.followup.collect import schedule, _key
    directory = Path(directory)
    identity, status = read_json(directory / "run_identity.json"), read_json(directory / "status.json")
    order = schedule(queries, 20261019)
    if (identity.get("seed") != 20261019 or identity.get("queries_sha256") != json_digest(queries)
            or identity.get("schedule_sha256") != json_digest(order)
            or identity.get("actions") != list(ACTIONS) or read_json(directory / "schedule.json") != order):
        raise ValueError("collection query/schedule/action identity changed")
    if (status.get("status") != "action_collection_complete"
            or status.get("outcomes_sha256") != digest(directory / "outcomes.jsonl")
            or status.get("planned") != len(order) or status.get("completed") != len(order)):
        raise ValueError("collection is incomplete or exported outcome fingerprint/count changed")
    rows = read_jsonl(directory / "outcomes.jsonl")
    if [_key(r) for r in rows] != [_key(r) for r in order]:
        raise ValueError("exported outcomes differ from planned full action order")
    expected = {_key(r): r for r in rows}
    completed, active, attempts = {}, {}, {}
    for event in read_jsonl(directory / "journal.jsonl"):
        key = tuple(event["key"])
        if key not in expected:
            raise ValueError("unplanned journal action")
        if event["event"] == "start":
            if key in active or key in completed or event["attempt"] != attempts.get(key, 0) + 1:
                raise ValueError("journal start/attempt sequence inconsistent")
            active[key] = event["attempt"]
            attempts[key] = event["attempt"]
        elif event["event"] in ("complete", "failed"):
            if key not in active or event["attempt"] != active.pop(key):
                raise ValueError("journal terminal event lacks its start")
            if event["event"] == "complete":
                if event.get("outcome") != expected[key]:
                    raise ValueError("exported outcome differs from original durable journal")
                completed[key] = event["outcome"]
        else:
            raise ValueError("unknown journal event")
    if active or set(completed) != set(expected) or status.get("attempts") != sum(attempts.values()):
        raise ValueError("journal is incomplete or attempt count changed")
    for row, task in zip(rows, order):
        q = queries[task["query_index"]]
        validate_outcome(row, q["question"], q["answer"])
    return rows
