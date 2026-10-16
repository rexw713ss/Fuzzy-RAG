"""Plot frozen result JSON; never fit policies or choose a favorable budget."""

import argparse
import json
from pathlib import Path

METHODS = ("global", "uncertainty-B3", "uncertainty-B4", "ridge", "fuzzy-fixed", "fuzzy-calibrated")
COLORS = {"global": "#85909e", "uncertainty-B3": "#e59a50", "uncertainty-B4": "#c35d33",
          "ridge": "#2469a8", "fuzzy-fixed": "#39a28c", "fuzzy-calibrated": "#7654a3"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="one archived run output directory")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.root.glob("*_report.json"))]
    if not reports:
        parser.error("no dataset reports found")
    if any(r.get("synthetic", True) for r in reports):
        parser.error("this figure script is reserved for real cached datasets")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(reports), figsize=(6.6 * len(reports), 4.3), squeeze=False)
    for ax, report in zip(axes[0], reports):
        primary = report["budgets"][str(report["primary_budget"])]["methods"]
        for i, method in enumerate(METHODS):
            interval = primary[method]["vs_random_same_A2_count"]["F1"]
            x, lo, hi = [100 * interval[k] for k in ("diff", "lo", "hi")]
            ax.errorbar(x, i, xerr=[[max(x - lo, 0)], [max(hi - x, 0)]], fmt="o",
                        color=COLORS[method], capsize=3, markersize=6)
        ax.axvline(0, linestyle="--", color="#6b7280", linewidth=1)
        ax.set_yticks(range(len(METHODS)), METHODS)
        ax.invert_yaxis()
        ax.set_title(f"{report['dataset'].upper()} | {len(report['signals'])} signals")
        ax.set_xlabel("F1 difference vs same-count random (percentage points)")
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Exploratory cached replay | maximum A2 fraction = 25%", fontsize=14)
    fig.text(.5, .015, "95% paired cluster intervals; unadjusted; frozen model and batch allocation. No new LLM generation.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .93))
    fig.savefig(args.root / "primary_comparison.png", dpi=180)
    fig.savefig(args.root / "primary_comparison.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(reports), figsize=(6.6 * len(reports), 4.5), squeeze=False)
    for ax, report in zip(axes[0], reports):
        for method in METHODS:
            points = sorted(report["budgets"].values(), key=lambda b: b["quota"])
            ax.plot([100 * b["quota"] for b in points],
                    [100 * b["methods"][method]["F1"]["mean"] for b in points],
                    marker="o", markersize=4, color=COLORS[method], label=method)
        for label, linestyle in (("always-A1", "--"), ("always-A2", ":")):
            ax.axhline(100 * report["fixed"][label]["F1"]["mean"], linestyle=linestyle,
                       color="#444444", linewidth=1, label=label)
        ax.set_title(report["dataset"].upper())
        ax.set_xlabel("Maximum A2 allocation (%)")
        ax.set_ylabel("Test F1 (%)")
        ax.set_xticks([0, 10, 25, 50, 100])
        ax.grid(alpha=.2)
    axes[0, -1].legend(fontsize=8, loc="best")
    fig.suptitle("All prespecified budgets | exploratory, not a test-selected frontier", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(args.root / "all_budgets.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
