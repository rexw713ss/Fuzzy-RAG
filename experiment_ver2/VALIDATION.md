# Verification record

This record describes engineering and cached-data checks, not confirmation of the research hypothesis.

Environment: Windows 10, Python 3.12.14, NumPy 2.3.5, pytest 9.1.1, regex 2026.9.29, PyYAML 6.0.3. Figures use matplotlib 3.11.2. CPU-only replay; no new model inference or GPU timing.

## Tests

```powershell
.\.venv-core\Scripts\python.exe -m pytest -q -rs
```

The current prospective-followup checkpoint has **702 passed, 2 skipped**, recorded in [its raw test output](followup/evidence/tests.txt). The earlier robustness/additive checkpoint had 641 passed, 2 skipped. These are engineering tests, not newly generated research outcomes. The two existing external rule-CSV tests are skipped because `Fuzzy_RAG_personal/etc/rule_base_81.csv` and `rule_base_27.csv` are not in this repository. Internal generated rule-base tests pass.

The suite covers:

- Raw-cache and explicit archived-cache schemas, duplicate/missing rows, split leakage, action/feature dimensions, immutable inputs, incomplete clusters, fresh-ID/text overlap, and archived jitter consistency.
- Fuzzy and ridge fitting, sparse-rule shrinkage, JSON round trips, boundary numerics, and both 27/81 rule and 2/3 action variants.
- Validation-only selection, maximum quota, optional always-A1 fallback, and deterministic feature-only tie-breaking.
- Test-label and measured-cost mutation invariance for synthetic three-action and real two-action datasets, with both three and four signals.
- Primary fuzzy trace weights reconstruct the recorded predicted gain.
- Minimal input ZIP round trip, failure artifacts, and refusal to overwrite run outputs or write inside original input caches.
- Whole-cluster TRAIN/VAL resampling, exact fuzzy sufficient-statistic cache parity, no TEST-label fitting, and fallback/width failure behavior.
- Three test-bootstrap estimands checked against literal duplicated-row calculations, cluster weighting, conservative family intervals, and descriptive subgroups.
- Metadata and split audits, outcome-blind human-review sampling, cluster-level support/ESS and tied-rank diagnostics.
- Additive basis geometry, independent weighted-ridge solve, collinearity, inability to learn a balanced XOR interaction, model serialization, TEST-outcome absence/perturbation invariance, and saved-reference verification.
- Prospective sample/exposure fingerprints, complete human-review source records, unique pair/text binding, pre-collection population freeze, immutable action schedules, preserved failed attempts, interrupted-run recovery, and original journal/export reconciliation.
- A1/A2/A2-union context and decoding reconstruction, empty/wrong traces, cost/token inconsistencies, offline asset/index manifests, original retention after paraphrase screening, and a complete miniature confirmation chain using explicitly toy backends in temporary test directories.

The [followup readiness audit](followup/evidence/readiness.json) verifies real TRAIN pilot inputs and unchanged policy locks, and records missing runtime assets plus incomplete human review. The [planning simulation](followup/evidence/planning_final/report.json) uses exposed cached clusters; it supplies variance sensitivity for future sample sizes, not new answers or evidence of superiority. Full prospective research remains incomplete as detailed in [STATUS](followup/STATUS.zh-TW.md).

## Original result reproduction

```powershell
.\.venv-core\Scripts\python.exe -m experiment_ver2.reproduction --full runs/v0.3 --v4 runs/v0.4 --out experiment_ver2/runs/reproduction-check.json
```

The saved [reproduction record](validation/reproduction.json) contains exact matches for 6,494 numeric values (2,884 NQ; 2,884 HotpotQA; 726 pooled), with maximum absolute error 0. All 32 checked input JSON/JSONL files were unchanged. The integrity audit checks 5,164 outcome rows, rescoring EM/F1 and verifying component costs and old/fresh original identities.

## Ver2 real-data runs

Four complete analyses are preserved under [evidence](evidence/): v0.3 old test, v0.4 old test, v0.4 fresh four signals, and v0.4 fresh three signals. Each has 1,000 cluster-bootstrap resamples, the five fixed quotas, all models and decisions, and input/source hashes. The main figure was visually inspected for readable labels and confidence intervals. The synthetic plumbing run remains separate and is not empirical evidence.

An independent numerical audit recomputed all 240 method × quota × dataset blocks from saved decisions: action counts, quota feasibility, F1/cost means, same-count random expected differences, validation selection and baseline selection agreed. Primary F1 and vs-ridge bootstrap intervals, and 27/81-rule traces, were also checked independently. The regression suite persists the key two-action and trace invariants.

## Limits

The additional [robustness evidence](robustness/evidence/) uses 10,000 test-cluster resamples per dataset, three sampling definitions and 100 independent TRAIN/VAL resampling fits. The original primary candidates, all six policy action arrays and F1 values matched saved evidence. Repeating the robustness run after review produced identical statistical and stability outputs. [Additive follow-up evidence](additive_evidence/) contains two additional families fitted on TRAIN, selected on VAL and evaluated once under its written exploratory protocol. Both datasets and all 12 planned comparisons are preserved, including negative/uncertain findings.

After correcting Windows newline serialization, both runs were repeated without any model or protocol change: every non-manifest result matched the earlier run after newline normalization. All saved evidence checksums match both working files and staged Git blobs. The new PNG/PDF comparison figure was visually inspected. This serialization replay is not another independent statistical experiment.

- Published fresh-set results were already inspected; this is exploratory replay, not a new blinded test.
- Archived old raw signals are absent, so their scaler provenance cannot be independently reconstructed. Saved TRAIN jitter was independently checked.
- Original ver2 CIs condition on fitted models and observed batch assignments and are unadjusted. The new robustness outputs separately examine reranked batches, TRAIN/VAL fitting variation and approximate multiplicity-adjusted intervals; none repairs test reuse or substitutes for a new confirmatory dataset.
- A2 count matching is not latency matching. Cached component latency excludes a newly measured online serving path.
- Human paraphrase review and new confirmatory data collection remain research work, not replaced by unit tests.
