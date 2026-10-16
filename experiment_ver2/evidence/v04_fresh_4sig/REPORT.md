# Ver2 execution report

Exploratory replay of existing cached outcomes. No new LLM calls were made.

Primary analysis: A1/A2 rescue at a maximum A2 fraction of 0.25.
The quota is an operational call limit, not a latency-matched comparison.

| Dataset | Method | F1 | A2 fraction | Cached cost | F1 difference vs same-count random (95% CI) |
|---|---|---:|---:|---:|---|
| hotpotqa | global | 0.4731 | 0.248 | 5.4399 | +0.0114 [-0.0060, +0.0287] |
| hotpotqa | ridge | 0.4802 | 0.248 | 5.4788 | +0.0186 [+0.0017, +0.0377] |
| hotpotqa | fuzzy-fixed | 0.4750 | 0.248 | 5.4515 | +0.0133 [+0.0008, +0.0257] |
| hotpotqa | fuzzy-calibrated | 0.4750 | 0.248 | 5.4515 | +0.0133 [+0.0008, +0.0257] |
| hotpotqa | uncertainty-B3 | 0.4555 | 0.248 | 5.4667 | -0.0061 [-0.0194, +0.0072] |
| hotpotqa | uncertainty-B4 | 0.4555 | 0.248 | 5.4695 | -0.0061 [-0.0194, +0.0072] |
| nq | global | 0.4407 | 0.000 | 5.6611 | +0.0000 [+0.0000, +0.0000] |
| nq | ridge | 0.4486 | 0.250 | 7.2413 | +0.0008 [-0.0105, +0.0147] |
| nq | fuzzy-fixed | 0.4470 | 0.250 | 7.2276 | -0.0007 [-0.0101, +0.0096] |
| nq | fuzzy-calibrated | 0.4483 | 0.250 | 7.2252 | +0.0006 [-0.0089, +0.0109] |
| nq | uncertainty-B3 | 0.4450 | 0.250 | 7.2270 | -0.0028 [-0.0117, +0.0063] |
| nq | uncertainty-B4 | 0.4438 | 0.250 | 7.2288 | -0.0040 [-0.0128, +0.0054] |

All intervals are exploratory and unadjusted for multiple comparisons.
Zero A2 usage can reflect the quota, validation fallback, or no positive predicted test gains.
Latency values replay cached component sums and exclude newly measured online serving overhead.
Random references integrate allocation randomness analytically; CIs reflect cluster sampling only.
Bootstrap intervals condition on frozen batch assignments and fitted models; they do not rerank resampled batches.
See report JSON for all budgets, validation fits, EM, RFR/HRFR, and paired comparisons.
