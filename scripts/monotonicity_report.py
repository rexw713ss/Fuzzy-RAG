"""Reproduce the section 4.2 monotonicity numbers from the repo (spec v0.3, step 0f).

Spec 4.2's values were first measured with an exploratory script; this makes them reproducible
from config + commit (section 7.5). Default: 0.05 grid (20 steps), w = 0.20, main controller.
Several --steps values give the spacing-sensitivity table (20 steps takes about a minute; 30
steps about five).

Run from the repo root:  .venv\\Scripts\\python.exe -m scripts.monotonicity_report
Optional: --steps 5 10 15 20 25 30   --theta1 0.30 --theta2 0.60 (once selected)
Writes data/checks/monotonicity.json.
"""

import argparse
import json
import time
from pathlib import Path

from main_logic import monotonicity as mo

DATA = Path(r"D:\Han\rex_rag\data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[mo.STEPS])
    ap.add_argument("--theta1", type=float)
    ap.add_argument("--theta2", type=float)
    args = ap.parse_args()

    results = []
    for steps in args.steps:
        t0 = time.time()
        r = mo.report(mo.grid_scores(steps), theta1=args.theta1, theta2=args.theta2)
        r = {"steps": steps, "spacing": 1 / steps, **r,
             "per_threshold": {f"{t:.2f}": v for t, v in r["per_threshold"].items()}}
        results.append(r)
        print(f"spacing {1 / steps:.4f}: {r['pairs']:,} pairs, {r['violations']:,} violations "
              f"({100 * r['rate']:.2f}%), max {r['drop_max']:.4f}, mean {r['drop_mean']:.4f}, "
              f"p95 {r['drop_p95']:.4f}; per signal {r['per_signal']}; straddle a threshold "
              f"{r['straddle_any_threshold']:,}; thresholds without flips "
              f"{r['thresholds_without_flips']}/{len(r['per_threshold'])} "
              f"[{time.time() - t0:.0f} s]")
        worst = sorted(r["per_threshold"].items(), key=lambda kv: -kv[1])[:4]
        print("  most downgrades at:", ", ".join(f"theta={t}: {v:,}" for t, v in worst))
        if "selected" in r:
            print("  at the selected thresholds:", r["selected"])

    out = DATA / "checks" / "monotonicity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf8")
    print("wrote", out)


if __name__ == "__main__":
    main()
