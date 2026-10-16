# Verification record

This record describes engineering and cached-data checks, not confirmation of the research hypothesis.

Environment: Windows 10, Python 3.12.14, NumPy 2.3.5, pytest 9.1.1, regex 2026.9.29, PyYAML 6.0.3. Figures use matplotlib 3.11.2. CPU-only replay; no new model inference or GPU timing.

## Tests

```powershell
.\.venv-core\Scripts\python.exe -m pytest -q -rs
```

Final result: **528 passed, 2 skipped**. The two existing external rule-CSV tests are skipped because `Fuzzy_RAG_personal/etc/rule_base_81.csv` and `rule_base_27.csv` are not in this repository. Internal generated rule-base tests pass. All 156 ver2 tests pass.

The suite covers:

- Raw-cache and explicit archived-cache schemas, duplicate/missing rows, split leakage, action/feature dimensions, immutable inputs, incomplete clusters, fresh-ID/text overlap, and archived jitter consistency.
- Fuzzy and ridge fitting, sparse-rule shrinkage, JSON round trips, boundary numerics, and both 27/81 rule and 2/3 action variants.
- Validation-only selection, maximum quota, optional always-A1 fallback, and deterministic feature-only tie-breaking.
- Test-label and measured-cost mutation invariance for synthetic three-action and real two-action datasets, with both three and four signals.
- Primary fuzzy trace weights reconstruct the recorded predicted gain.
- Minimal input ZIP round trip, failure artifacts, and refusal to overwrite run outputs or write inside original input caches.

## Original result reproduction

```powershell
.\.venv-core\Scripts\python.exe -m experiment_ver2.reproduction --full runs/v0.3 --v4 runs/v0.4 --out experiment_ver2/runs/reproduction-check.json
```

The saved [reproduction record](validation/reproduction.json) contains exact matches for 6,494 numeric values (2,884 NQ; 2,884 HotpotQA; 726 pooled), with maximum absolute error 0. All 32 checked input JSON/JSONL files were unchanged. The integrity audit checks 5,164 outcome rows, rescoring EM/F1 and verifying component costs and old/fresh original identities.

## Ver2 real-data runs

Four complete analyses are preserved under [evidence](evidence/): v0.3 old test, v0.4 old test, v0.4 fresh four signals, and v0.4 fresh three signals. Each has 1,000 cluster-bootstrap resamples, the five fixed quotas, all models and decisions, and input/source hashes. The main figure was visually inspected for readable labels and confidence intervals. The synthetic plumbing run remains separate and is not empirical evidence.

An independent numerical audit recomputed all 240 method × quota × dataset blocks from saved decisions: action counts, quota feasibility, F1/cost means, same-count random expected differences, validation selection and baseline selection agreed. Primary F1 and vs-ridge bootstrap intervals, and 27/81-rule traces, were also checked independently. The regression suite persists the key two-action and trace invariants.

## Limits

- Published fresh-set results were already inspected; this is exploratory replay, not a new blinded test.
- Archived old raw signals are absent, so their scaler provenance cannot be independently reconstructed. Saved TRAIN jitter was independently checked.
- CIs condition on fitted models and observed batch assignments, without model-selection or future-batch uncertainty; they are unadjusted for multiple comparisons.
- A2 count matching is not latency matching. Cached component latency excludes a newly measured online serving path.
- Human paraphrase review and new confirmatory data collection remain research work, not replaced by unit tests.
