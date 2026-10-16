# Ver2 execution report

Exploratory replay of existing cached outcomes. No new LLM calls were made.

Primary analysis: A1/A2 rescue at a maximum A2 fraction of 0.25.
The quota is an operational call limit, not a latency-matched comparison.

| Dataset | Method | F1 | A2 fraction | Cached cost | F1 difference vs same-count random (95% CI) |
|---|---|---:|---:|---:|---|
| hotpotqa | global | 0.5286 | 0.250 | 5.5532 | -0.0073 [-0.0295, +0.0139] |
| hotpotqa | ridge | 0.5445 | 0.250 | 5.5954 | +0.0086 [-0.0246, +0.0439] |
| hotpotqa | fuzzy-fixed | 0.5345 | 0.250 | 5.5583 | -0.0014 [-0.0262, +0.0241] |
| hotpotqa | fuzzy-calibrated | 0.5345 | 0.250 | 5.5583 | -0.0014 [-0.0262, +0.0241] |
| hotpotqa | uncertainty-B3 | 0.5298 | 0.250 | 5.5320 | -0.0061 [-0.0316, +0.0209] |
| hotpotqa | uncertainty-B4 | 0.5298 | 0.250 | 5.5320 | -0.0061 [-0.0316, +0.0209] |
| nq | global | 0.5279 | 0.000 | 6.1523 | +0.0000 [+0.0000, +0.0000] |
| nq | ridge | 0.5458 | 0.247 | 7.6231 | +0.0176 [-0.0056, +0.0420] |
| nq | fuzzy-fixed | 0.5424 | 0.247 | 7.6607 | +0.0142 [-0.0064, +0.0379] |
| nq | fuzzy-calibrated | 0.5450 | 0.247 | 7.6423 | +0.0168 [-0.0024, +0.0406] |
| nq | uncertainty-B3 | 0.5448 | 0.247 | 7.6419 | +0.0166 [-0.0037, +0.0375] |
| nq | uncertainty-B4 | 0.5507 | 0.247 | 7.6338 | +0.0225 [+0.0018, +0.0454] |

All intervals are exploratory and unadjusted for multiple comparisons.
Zero A2 usage can reflect the quota, validation fallback, or no positive predicted test gains.
Latency values replay cached component sums and exclude newly measured online serving overhead.
Random references integrate allocation randomness analytically; CIs reflect cluster sampling only.
Bootstrap intervals condition on frozen batch assignments and fitted models; they do not rerank resampled batches.
See report JSON for all budgets, validation fits, EM, RFR/HRFR, and paired comparisons.
