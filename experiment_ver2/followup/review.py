"""Blinded independent HUMAN review, with honest pending/abstention states.

The software validates recorded attestations; it cannot verify that the raters
are human or independent. It never supplies ratings or imputes missing votes.
"""

import argparse
from collections import Counter
from pathlib import Path

from experiment_ver2.followup.io import read_json, read_jsonl, write_json, write_jsonl, digest, json_digest, new_directory, seal


FIELDS = ("same_information_need", "added_or_removed_constraint", "grammatical_and_answerable")
ATTEST = ("human_review", "independent_review", "outcome_blind")
IDENTITY = ("review_id", "left_question", "right_question")


def _index(rows):
    if not rows:
        raise ValueError("review set must not be empty")
    indexed = {}
    for row in rows:
        key = row.get("review_id")
        if not isinstance(key, str) or not key or key in indexed:
            raise ValueError("review_id must be unique nonempty strings")
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in IDENTITY):
            raise ValueError("question text and review identity required")
        indexed[key] = row
    return indexed


def blank_forms(tasks):
    _index(tasks)
    return [{**{k: r[k] for k in IDENTITY}, **{k: None for k in FIELDS},
             "rater_id": None, "status": "pending", "notes": "",
             "attestations": {k: False for k in ATTEST}} for r in tasks]


def task_digest(tasks):
    reference = _index(tasks)
    return json_digest([{k: reference[rid][k] for k in IDENTITY} for rid in sorted(reference)])


def verified_audit(report_path, tasks):
    """Recompute from preserved human records, not an editable summary alone."""
    report_path = Path(report_path)
    folder = report_path.parent
    adjudications = folder / "adjudications_raw.jsonl"
    actual = audit(tasks, read_jsonl(folder / "rater_A_raw.jsonl"),
                   read_jsonl(folder / "rater_B_raw.jsonl"),
                   read_jsonl(adjudications) if adjudications.exists() else ())
    if actual != read_json(report_path):
        raise ValueError("human-review report differs from preserved source ratings or task text")
    return actual


def validate_ratings(tasks, rows):
    reference, indexed = _index(tasks), _index(rows)
    if set(indexed) != set(reference):
        raise ValueError("ratings must contain exactly the blinded task IDs")
    raters, reviewed, abstained, pending = set(), {}, [], []
    allowed = set(IDENTITY + FIELDS + ("rater_id", "status", "notes", "attestations"))
    for key, row in indexed.items():
        if set(row) != allowed:
            raise ValueError("unexpected or missing review fields; do not add outcomes, gold answers or model identities")
        if any(row[k] != reference[key][k] for k in IDENTITY):
            raise ValueError("blinded question text or identity was changed")
        if not isinstance(row["notes"], str):
            raise ValueError("notes must be text")
        attest = row["attestations"]
        if not isinstance(attest, dict) or set(attest) != set(ATTEST) or any(type(v) is not bool for v in attest.values()):
            raise ValueError("attestations require exactly three boolean fields")
        status = row["status"]
        if status == "pending":
            if any(row[k] is not None for k in FIELDS):
                raise ValueError("pending records cannot contain ratings")
            pending.append(key)
            continue
        if status not in ("reviewed", "abstained"):
            raise ValueError("status must be pending, reviewed or abstained")
        rater = row["rater_id"]
        if not isinstance(rater, str) or not rater.strip() or rater != rater.strip():
            raise ValueError("completed review requires an explicit rater_id")
        if not all(attest.values()):
            raise ValueError("human, independent and outcome-blind attestations are required")
        raters.add(rater)
        if status == "abstained":
            if not row["notes"].strip() or any(row[k] is not None for k in FIELDS):
                raise ValueError("abstention requires a reason and null ratings")
            abstained.append(key)
        else:
            if any(type(row[k]) is not bool for k in FIELDS):
                raise ValueError("reviewed criteria must be booleans, never inferred from blanks")
            if (not row[FIELDS[0]] or row[FIELDS[1]] or not row[FIELDS[2]]) and not row["notes"].strip():
                raise ValueError("negative semantic/answerability ratings need a reason")
            reviewed[key] = row
    if len(raters) > 1:
        raise ValueError("each independent form must be completed by a single rater")
    return {"rater_id": next(iter(raters), None), "reviewed": reviewed, "abstained": abstained, "pending": pending}


def agreement(left, right):
    if len(left) != len(right):
        raise ValueError("paired votes must align")
    n = len(left)
    if not n:
        return {"n": 0, "agreement": None, "cohen_kappa": None, "undefined_reason": "no paired completed votes"}
    observed = sum(a == b for a, b in zip(left, right)) / n
    a, b = Counter(left), Counter(right)
    expected = sum(a[k] * b[k] for k in set(a) | set(b)) / n**2
    return {"n": n, "agreement": observed, "cohen_kappa": (observed - expected) / (1 - expected) if expected < 1 else None,
            "undefined_reason": "expected agreement is one" if expected == 1 else None}


def _valid(row):
    return row[FIELDS[0]] and not row[FIELDS[1]] and row[FIELDS[2]]


def audit(tasks, left, right, adjudications=()):
    adjudications = list(adjudications)
    ref = _index(tasks)
    a, b = validate_ratings(tasks, left), validate_ratings(tasks, right)
    if a["rater_id"] is not None and a["rater_id"] == b["rater_id"]:
        raise ValueError("two independent forms must have distinct rater IDs")
    paired = sorted(set(a["reviewed"]) & set(b["reviewed"]))
    metrics = {field: agreement([a["reviewed"][k][field] for k in paired], [b["reviewed"][k][field] for k in paired]) for field in FIELDS}
    metrics["all_criteria_valid"] = agreement([_valid(a["reviewed"][k]) for k in paired], [_valid(b["reviewed"][k]) for k in paired])
    resolved, required = {}, set(a["abstained"]) | set(b["abstained"])
    for key in paired:
        if any(a["reviewed"][key][f] != b["reviewed"][key][f] for f in FIELDS):
            required.add(key)
        else:
            resolved[key] = {"valid": _valid(a["reviewed"][key]), "source": "two_rater_agreement"}
    adjudicated = set()
    for row in adjudications:
        key = row.get("review_id")
        if key not in required or key in adjudicated:
            raise ValueError("adjudication must uniquely address a disagreement or abstention")
        allowed = {"review_id", "rater_id", "reason", "human_review", "outcome_blind", *FIELDS}
        if set(row) != allowed or any(type(row.get(f)) is not bool for f in FIELDS):
            raise ValueError("adjudication schema/criteria invalid")
        if row.get("human_review") is not True or row.get("outcome_blind") is not True:
            raise ValueError("adjudication requires human and blinding attestations")
        if not isinstance(row["rater_id"], str) or not row["rater_id"].strip() or row["rater_id"] != row["rater_id"].strip() or row["rater_id"] in (a["rater_id"], b["rater_id"]):
            raise ValueError("adjudicator must be a third distinct rater")
        if not isinstance(row["reason"], str) or not row["reason"].strip():
            raise ValueError("adjudication requires a reason")
        resolved[key] = {"valid": _valid(row), "source": "third_rater_adjudication", "reason": row["reason"]}
        adjudicated.add(key)
    pending = sorted(set(a["pending"]) | set(b["pending"]))
    unresolved = sorted(required - adjudicated)
    complete = not pending and not unresolved and len(resolved) == len(ref)
    return {"status": "complete_recorded_human_review" if complete else "incomplete",
            "tasks_canonical_sha256": task_digest(tasks),
            "ratings_canonical_sha256": {"left": json_digest(left), "right": json_digest(right),
                                          "adjudications": json_digest(adjudications)},
            "n_tasks": len(ref), "rater_ids": [a["rater_id"], b["rater_id"]], "agreement": metrics,
            "pending": pending, "unresolved_adjudication": unresolved, "decisions": resolved,
            "valid_pairs": sum(r["valid"] for r in resolved.values()),
            "invalid_pairs": sum(not r["valid"] for r in resolved.values()),
            "abstentions": [len(a["abstained"]), len(b["abstained"])],
            "inference_limit": "validates self-attested human records only, not actual identities or independence; no automatic deletion of outcomes; sample pair reviews do not validate every variant of a cluster"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("export", "audit"))
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--left", type=Path)
    ap.add_argument("--right", type=Path)
    ap.add_argument("--adjudications", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    tasks = read_jsonl(args.tasks)
    inputs = [p for p in (args.tasks, args.left, args.right, args.adjudications) if p is not None]
    out = new_directory(args.out, inputs)
    if args.mode == "export":
        for filename in ("rater_A.jsonl", "rater_B.jsonl"):
            write_jsonl(out / filename, blank_forms(tasks))
        result = {"status": "pending_human_review", "n_tasks": len(tasks), "completed_ratings": 0,
                  "instructions": "Give one blank form to each independent human. Do not disclose the key, outcomes or policy. Fill rater_id/status/criteria/notes/attestations; never label AI ratings as human."}
    else:
        if args.left is None or args.right is None:
            ap.error("audit requires --left and --right")
        result = audit(tasks, read_jsonl(args.left), read_jsonl(args.right), read_jsonl(args.adjudications) if args.adjudications else ())
        for role, path in (("rater_A", args.left), ("rater_B", args.right), ("adjudications", args.adjudications)):
            if path:
                (out / f"{role}_raw.jsonl").write_bytes(path.read_bytes())
    write_json(out / "report.json", result)
    write_json(out / "provenance.json", {"input_sha256": {str(p): digest(p) for p in inputs}, "schema": 1})
    seal(out)
    print(f"{result['status']}: {out}")
    return 0 if args.mode == "export" or result["status"] == "complete_recorded_human_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
