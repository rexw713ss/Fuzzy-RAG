"""Export scientific figures from saved evidence; never fit or tune a model."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robustness", type=Path, default=Path("experiment_ver2/robustness/evidence"))
    parser.add_argument("--additive", type=Path, default=Path("experiment_ver2/additive_evidence"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory must be new or empty")
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.7), layout="constrained")
    colors = ["#276b88", "#a65728", "#207b5a"]
    for j, dataset in enumerate(("nq", "hotpotqa")):
        r = json.loads((args.robustness / f"{dataset}_robustness.json").read_text(encoding="utf-8"))
        a = json.loads((args.additive / f"{dataset}_report.json").read_text(encoding="utf-8"))
        first = [("81-rule fuzzy", r["inference"]["estimands"]["frozen_matched_fraction"]["ridge"])]
        first += [(label, a["methods"][m]["inference"]["comparisons"]["ridge"])
                  for label, m in (("Additive fixed", "additive-fixed"), ("Additive calibrated", "additive-calibrated"))]
        second = [(label, r["inference"]["estimands"][mode]["random"])
                  for label, mode in (("Frozen / fixed fraction", "frozen_fixed_fraction"),
                                      ("Frozen / matched fraction", "frozen_matched_fraction"),
                                      ("Reranked batch", "reranked_batch"))]
        for i, entries in enumerate((first, second)):
            ax = axes[i, j]
            for k, (label, entry) in enumerate(entries):
                point = entry["difference"] * 100
                lo, hi = [v * 100 for v in entry["percentile_95"]]
                ax.plot([lo, hi], [k, k], color=colors[k], linewidth=2.2)
                ax.plot(point, k, "o", color=colors[k], markersize=6)
            ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
            ax.set_yticks(range(len(entries)), [label for label, _ in entries])
            ax.invert_yaxis()
            ax.set_ylim(2.7, -.7)
            ax.set_xlim(-3, 3.2)
            ax.grid(axis="x", alpha=.18)
            ax.set_xlabel("F1 difference (percentage points), 95% interval")
            ax.set_title(f"{'NQ' if j == 0 else 'HotpotQA'}: " + ("difference vs ridge" if i == 0 else "81-rule fuzzy vs random"), loc="left", fontweight="bold")
    fig.suptitle("Exploratory evidence: higher mean F1 does not establish a robust routing advantage", fontsize=14)
    fig.supxlabel("10,000 cluster resamples; q = 0.25. Unadjusted percentile intervals; previously inspected test.\nTop: frozen assignments. Bottom: three sampling definitions. No new retrieval or generation.", fontsize=10)
    fig.savefig(args.out / "robustness_and_additive.png")
    fig.savefig(args.out / "robustness_and_additive.pdf")
    print(args.out / "robustness_and_additive.png")


if __name__ == "__main__":
    main()
