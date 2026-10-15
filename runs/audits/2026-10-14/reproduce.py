"""Audit 3 (Oct 14): rerun scripts.evaluate.evaluate_dataset in memory (no files written) and
compare with the saved data\\full\\results\\<dataset>.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"D:\Han\rex_rag\Fuzzy-RAG")))
from scripts import evaluate as ev

ROOT = Path(r"D:\Han\rex_rag\data\full")
for d in ("nq", "hotpotqa"):
    new = json.loads(json.dumps(ev.evaluate_dataset(d, ROOT)[0], default=float))
    old = json.loads((ROOT / "results" / f"{d}.json").read_text())
    diffs = []
    for name, m in old["test"].items():
        for k in ("F1", "cost"):
            if abs(m[k]["mean"] - new["test"][name][k]["mean"]) > 1e-9:
                diffs.append((name, k, m[k]["mean"], new["test"][name][k]["mean"]))
        if m["counts"] != new["test"][name]["counts"]:
            diffs.append((name, "counts", m["counts"], new["test"][name]["counts"]))
    fits = {n: (f["theta1"], f["theta2"]) for n, f in new["fits"].items()}
    print(d, "routers", len(new["test"]), "| differences vs saved:", diffs or "none")
    print("   fitted thresholds:", fits)
