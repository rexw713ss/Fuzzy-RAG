# Ver2 execution report

Exploratory replay of existing cached outcomes. No new LLM calls were made.

Primary analysis: A1/A2 rescue at a maximum A2 fraction of 0.25.
The quota is an operational call limit, not a latency-matched comparison.

| Dataset | Method | F1 | A2 fraction | Cached cost | F1 difference vs same-count random (95% CI) |
|---|---|---:|---:|---:|---|
| hotpotqa | global | 0.4731 | 0.248 | 5.4399 | +0.0114 [-0.0060, +0.0287] |
| hotpotqa | ridge | 0.4678 | 0.248 | 5.4609 | +0.0061 [-0.0146, +0.0255] |
| hotpotqa | fuzzy-fixed | 0.4663 | 0.248 | 5.4740 | +0.0047 [-0.0088, +0.0182] |
| hotpotqa | fuzzy-calibrated | 0.4663 | 0.248 | 5.4740 | +0.0047 [-0.0088, +0.0182] |
| hotpotqa | uncertainty-B3 | 0.4528 | 0.248 | 5.4574 | -0.0089 [-0.0234, +0.0049] |
| hotpotqa | uncertainty-B4 | 0.4528 | 0.248 | 5.4574 | -0.0089 [-0.0234, +0.0049] |
| nq | global | 0.4407 | 0.000 | 5.6611 | +0.0000 [+0.0000, +0.0000] |
| nq | ridge | 0.4446 | 0.250 | 7.2365 | -0.0031 [-0.0158, +0.0088] |
| nq | fuzzy-fixed | 0.4455 | 0.250 | 7.2192 | -0.0022 [-0.0106, +0.0065] |
| nq | fuzzy-calibrated | 0.4443 | 0.250 | 7.2191 | -0.0034 [-0.0116, +0.0055] |
| nq | uncertainty-B3 | 0.4441 | 0.250 | 7.2296 | -0.0037 [-0.0128, +0.0054] |
| nq | uncertainty-B4 | 0.4441 | 0.250 | 7.2325 | -0.0037 [-0.0128, +0.0054] |

All intervals are exploratory and unadjusted for multiple comparisons.
Zero A2 usage can reflect the quota, validation fallback, or no positive predicted test gains.
Latency values replay cached component sums and exclude newly measured online serving overhead.
Random references integrate allocation randomness analytically; CIs reflect cluster sampling only.
Bootstrap intervals condition on frozen batch assignments and fitted models; they do not rerank resampled batches.
See report JSON for all budgets, validation fits, EM, RFR/HRFR, and paired comparisons.
