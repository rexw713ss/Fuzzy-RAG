# Ver2 execution report

Exploratory replay of existing cached outcomes. No new LLM calls were made.

Primary analysis: A1/A2 rescue at a maximum A2 fraction of 0.25.
The quota is an operational call limit, not a latency-matched comparison.

| Dataset | Method | F1 | A2 fraction | Cached cost | F1 difference vs same-count random (95% CI) |
|---|---|---:|---:|---:|---|
| hotpotqa | global | 0.5383 | 0.250 | 4.8034 | +0.0061 [-0.0035, +0.0183] |
| hotpotqa | ridge | 0.5180 | 0.250 | 4.7878 | -0.0142 [-0.0292, -0.0029] |
| hotpotqa | fuzzy-fixed | 0.5330 | 0.250 | 4.8250 | +0.0008 [-0.0091, +0.0135] |
| hotpotqa | fuzzy-calibrated | 0.5330 | 0.250 | 4.8248 | +0.0008 [-0.0091, +0.0135] |
| hotpotqa | uncertainty-B3 | 0.5283 | 0.250 | 4.8032 | -0.0040 [-0.0168, +0.0092] |
| hotpotqa | uncertainty-B4 | 0.5283 | 0.250 | 4.8032 | -0.0040 [-0.0168, +0.0092] |
| nq | global | 0.5279 | 0.000 | 6.1523 | +0.0000 [+0.0000, +0.0000] |
| nq | ridge | 0.5227 | 0.247 | 6.6268 | -0.0070 [-0.0166, +0.0018] |
| nq | fuzzy-fixed | 0.5398 | 0.247 | 6.6755 | +0.0101 [-0.0076, +0.0292] |
| nq | fuzzy-calibrated | 0.5398 | 0.247 | 6.6769 | +0.0101 [-0.0076, +0.0292] |
| nq | uncertainty-B3 | 0.5362 | 0.247 | 6.6704 | +0.0065 [-0.0125, +0.0260] |
| nq | uncertainty-B4 | 0.5362 | 0.247 | 6.6711 | +0.0065 [-0.0125, +0.0260] |

All intervals are exploratory and unadjusted for multiple comparisons.
Zero A2 usage can reflect the quota, validation fallback, or no positive predicted test gains.
Latency values replay cached component sums and exclude newly measured online serving overhead.
Random references integrate allocation randomness analytically; CIs reflect cluster sampling only.
Bootstrap intervals condition on frozen batch assignments and fitted models; they do not rerank resampled batches.
See report JSON for all budgets, validation fits, EM, RFR/HRFR, and paired comparisons.
