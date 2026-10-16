# Additive fuzzy exploratory follow-up

Real v0.4 cached outcomes; no new retrieval/generation. Primary quota .25. All tests previously inspected.

| Dataset | Family | VAL choice | Test F1 | A2 count | Difference vs ridge (95% cluster interval) |
|---|---|---|---:|---:|---|
| nq | additive-fixed | alpha=0.1 | 0.443955 | 103 | -0.004604 [-0.016418, +0.007536] |
| nq | additive-calibrated | alpha=0.1 | 0.446960 | 103 | -0.001599 [-0.014230, +0.011695] |
| hotpotqa | additive-fixed | alpha=0.1 | 0.484575 | 110 | +0.004343 [-0.009191, +0.018033] |
| hotpotqa | additive-calibrated | alpha=0.1 | 0.484575 | 110 | +0.004343 [-0.009191, +0.018033] |

The report JSON includes all 12 comparisons, conservative multiplicity sensitivity intervals, EM, cached cost, routing flips, and subset scores. No new test-based tuning or selection follows this run.
