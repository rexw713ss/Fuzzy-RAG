"""Empirical variance sensitivity for planning; NOT new experimental outcomes.

Resample exposed source clusters into future batch sizes and recompute quota
allocations. Center the resulting statistic and shift its distribution by an
assumed effect, without fabricating query answers or reporting observed power.
"""
import argparse
from pathlib import Path
import numpy as np

from experiment_ver2.archive import load_archive
from experiment_ver2.run import fit_candidates, score_candidates, freeze_policy
from experiment_ver2.robustness.statistics import reranked_counts
from experiment_ver2.followup.io import new_directory, digest, write_json, seal


def simulate(split, scores, fallbacks, sizes, repetitions=10000, seed=20261023):
    if type(repetitions) is not int or repetitions < 100 or any(type(n) is not int or n < 2 for n in sizes):
        raise ValueError("positive sample sizes and at least 100 simulations required")
    ids, inv = np.unique(split.groups, return_inverse=True)
    gain = split.f1[:, 1] - split.f1[:, 0]
    rng = np.random.default_rng(seed)
    result = {}
    for n in sizes:
        values = []
        for start in range(0, repetitions, 128):
            count = min(128, repetitions - start)
            draws = rng.integers(len(ids), size=(count, n))
            weights = np.zeros((count, len(ids)), dtype=np.int64)
            np.add.at(weights, (np.arange(count)[:, None], draws), 1)
            mult = weights[:, inv]
            allocations = [reranked_counts(scores[m], split.questions, mult, .25, fallback=fallbacks[m])
                           for m in ("fuzzy-calibrated", "ridge")]
            values.extend(((allocations[0] - allocations[1]) @ gain / mult.sum(1)).tolist())
        values = np.array(values)
        noise = values - values.mean()
        degenerate = np.ptp(noise) <= 1e-15
        scenarios = {}
        for inflation in (1.0, 1.25, 1.5):
            distribution = noise * inflation
            lower = np.quantile(distribution, .0125)  # two datasets, two-sided family .05
            power = {str(effect): None if degenerate else float(np.mean(effect + distribution + lower > 0)) for effect in (.005, .01, .02)}
            scenarios[str(inflation)] = {"assumed_sd_multiplier": inflation,
                "shift_distribution_detection_probability": power,
                "approximate_F1_effect_for_80_percent": None if degenerate else float(-lower - np.quantile(distribution, .2)),
                "nominal_positive_null_detection_probability": float(np.mean(distribution + lower > 0))}
        result[str(n)] = {"simulated_statistic_mean": float(values.mean()), "simulated_statistic_sd": float(values.std(ddof=1)),
                          "degenerate_variance_no_power_estimate": bool(degenerate),
                          "variance_scenarios": scenarios}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = new_directory(args.out, (args.archive,))
    results = {}
    for name, sizes in (("nq", (600, 1000, 1500, 1800)), ("hotpotqa", (1300, 2000, 3000, 3500))):
        data = load_archive(args.archive, name, version="v0.4", test_set="fresh")
        candidates = fit_candidates(data)
        validation = score_candidates(candidates, data.val.X, data.widths, data.actions, data.signals)
        fits = {m: freeze_policy(m, validation[m], data.val, .25) for m in ("fuzzy-calibrated", "ridge")}
        prediction = score_candidates(candidates, data.test.X, data.widths, data.actions, data.signals)
        fallbacks = {m: fits[m]["candidate"] == "always-A1" for m in fits}
        scores = {m: np.zeros(len(data.test)) if fallbacks[m] else prediction[m][fits[m]["candidate"]] for m in fits}
        results[name] = {"source_clusters": len(set(data.test.groups)), "source_queries": len(data.test),
                         "future_batches": simulate(data.test, scores, fallbacks, sizes)}
    write_json(out / "report.json", {"status": "exposed_data_planning_simulation_only", "repetitions": 10000,
        "seed": 20261023, "datasets": results,
        "limitations": "empirical cluster distribution with frozen TRAIN/VAL models; full retained variants used because human-valid subset is not yet available; centered location-shift approximation, not nested bootstrap power or new generated data; SD multipliers are sensitivity assumptions, not estimated uncertainty; does not guarantee power under semantic filtering, generation variance or distribution shift"})
    write_json(out / "inputs.json", {"archive": {str(p.relative_to(args.archive)): digest(p) for p in sorted(args.archive.rglob("*"))
                                    if p.is_file() and p.suffix in (".json", ".jsonl")},
                                    "source_sha256": {str(Path(__file__).name): digest(__file__)}})
    seal(out)
    print(out)


if __name__ == "__main__":
    main()
