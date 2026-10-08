# FuzzyRoute-RAG

[English](README.md) | [繁體中文](README.zh-TW.md)

Interpretable uncertainty-aware routing for retrieval-augmented generation.

> **Status: research prototype with implemented core modules.** Retrieval, uncertainty signals, fuzzy inference, width calibration, cluster splitting, controller baselines, and evaluation primitives are implemented. Paraphrase construction, complete A0/A1/A2 execution, and answer generation are not yet integrated. The results below are implementation checks and synthetic controller diagnostics, not benchmark answer-quality results.

## Overview

Retrieval-augmented generation (RAG) supplies retrieved passages as context for answer generation. FuzzyRoute-RAG studies how retrieval uncertainty can guide the allocation of additional processing. A Type-1 Mamdani fuzzy inference system combines four retrieval-side signals and produces an escalation score.

The research question is:

> Can fuzzy routing improve the answer-quality/cost trade-off and reduce harmful routing changes across semantically equivalent query formulations?

This is a research objective, not an established result. The controller returns membership degrees and fired rules as a decision trace. Transparent inference does not by itself establish that the selected actions are effective.

## Method

The diagram shows the intended complete pipeline. The shared generator and full action executors remain under development.

~~~mermaid
flowchart TD
    Q["Query"] --> R["BM25 and Contriever"]
    R --> S["Fusion and uncertainty signals"]
    S --> F["Fuzzy escalation score"]
    F --> A0["A0: direct top-5"]
    F --> A1["A1: rerank top-20"]
    F --> A2["A2: rewrite and retrieve"]
    A0 --> G["Shared top-5 context and generator"]
    A1 --> G
    A2 --> G
~~~

### Retrieval and score fusion

- **BM25:** Lucene/Pyserini over the Dense Passage Retrieval (DPR) Wikipedia corpus; $k_1=0.9$ and $b=0.4$.
- **Dense retrieval:** `facebook/contriever-msmarco`; attention-mask-aware mean pooling, unnormalized embeddings, and inner-product scoring.
- **Checkpoint revision:** `abe8c1493371369031bcb1e02acb754cf4e162fa`.
- Each retriever returns its top 50 passages. Per-query min-max scores are averaged; passages absent from one list receive zero for that retriever. The fused pool is truncated to 50.
- Constant-score lists currently normalize to all ones. Fusion and A2 merging break ties by the string representation of the passage ID.
- Dense search is currently exhaustive over 21,015,324 passages. Its computation and memory costs are substantial; the final search backend remains open.

### Uncertainty signals

Raw signals are designed so that larger values favor escalation. This orientation still requires empirical validation, especially for multi-hop questions where diverse evidence may be useful.

Let $s_1\geq s_2\geq\cdots\geq s_k$ be fused scores, with $k=10$.

| Signal | Interpretation | Function |
|---|---|---|
| S1 | Small leading-score margin | `s1_margin` |
| S2 | BM25/dense rank disagreement | `s2_disagreement` |
| S3 | Dispersion of top-10 passage embeddings | `s3_dispersion` |
| S4 | Entropy of top-10 fused scores | `s4_entropy` |

$$
S_1=1-\frac{s_1-s_2}{s_1-s_k+\epsilon},\qquad \epsilon=10^{-9}.
$$

S2 uses extrapolated Rank-Biased Overlap (RBO), including its residual term:

$$
\mathrm{RBO}_{EXT}@k=(1-p)\sum_{d=1}^{k}p^{d-1}A_d+p^kA_k,\qquad
S_2=1-\mathrm{RBO}_{EXT}@k.
$$

$A_d$ is the overlap proportion at depth $d$, and $p=0.9$. Identical rankings yield S2 approximately zero; disjoint rankings yield one.

$$
S_3=1-\frac{2}{k(k-1)}\sum_{i<j}\cos(\mathbf e_i,\mathbf e_j).
$$

S3 normalizes embeddings only for cosine calculation. Dense retrieval still uses unnormalized inner products.

$$
p_i=\frac{\exp(s_i/\tau)}{\sum_{j=1}^{k}\exp(s_j/\tau)},\qquad
S_4=-\frac{\sum_{i=1}^{k}p_i\log p_i}{\log k},\qquad \tau=0.1.
$$

`fit_scaler` estimates the 5th and 95th percentiles (P5/P95) for each signal and dataset using training clusters. `apply_scaler` rescales and clips to $[0,1]$. Callers must enforce training-only fitting; these functions do not inspect split labels. Check near-constant signals before fitting.

### Fuzzy inference and width calibration

The main controller uses minimum AND, clipped output sets, maximum aggregation, and centroid defuzzification over 1,001 output points.

Each input has Low, Medium, and High labels, with crossover centers 0.35 and 0.65. All $3^4=81$ rules are generated from ordinal levels Low = 0, Medium = 1, and High = 2. For their sum $t$:

| Ordinal sum | Consequent | Rules |
|---|---|---:|
| $t\leq3$ | Low | 31 |
| $t=4$ | Medium | 19 |
| $t\geq5$ | High | 31 |

Fixed-width inference uses $w=0.20$. Calibration pools absolute signal differences over all three formulation pairs in each complete training cluster:

$$
w_s=\operatorname{clip}\!\left(P_{75}(|S_{i,s}-S_{j,s}|),\,0.20,\,0.25\right).
$$

The implementation also returns median-based sensitivity widths, flags uncapped jitter above 0.25, and counts skipped incomplete clusters. If every width remains 0.20, calibration has not changed the controller and cannot be credited with a separate improvement.

Implemented single-stage operator ablations use product t-norm, probabilistic-sum aggregation, or zero-order Sugeno output.

### Actions and controller baselines

| Action | Intended processing | Current support |
|---|---|---|
| A0 | Fused top-5 directly to generation | Full executor pending |
| A1 | Rerank top-20, retain top-5, then generate | Full executor pending |
| A2 | Rewrite; second hybrid retrieval; merge/deduplicate; rerank top-30; retain top-5 | Merge and rewrite-usability checks implemented; full executor pending |

A2 merging keeps the maximum fused score for each duplicate ID, rescales the union, and retains 30 candidates. Failed, empty, or case/whitespace-only rewrites are unusable.

All actions are intended to share generator, context size, prompt, and decoding settings. A future fallback executor must include all attempted work in its costs, including failed rewrite calls.

| Baseline | Score and fitting |
|---|---|
| B1: hard weighted threshold | Weighted signal sum; 35 weight vectors on a 0.25 lattice; validation threshold search |
| B2: crisp ordinal | Input cuts at 0.35/0.65; sum 0–8; 45 integer cut-point pairs |
| B3: fixed fuzzy | Mamdani score with all widths 0.20 |
| B4: calibrated fuzzy | Mamdani score with training-calibrated per-signal widths |

A score $e<\theta_1$ selects A0; $\theta_1\leq e<\theta_2$ selects A1; otherwise A2. Current continuous grids use $\theta_1\in[0.20,0.50]$, $\theta_2\in[0.50,0.80]$, and step 0.01, yielding 960 valid pairs.

Selection maximizes mean cached F1 under a mean-cost budget, defaulting to always-A1 latency. The restricted grid can make this budget infeasible; the code raises an error. `pareto_frontier` currently returns a best-quality budget sweep and may repeat policies; it is not a deduplicated nondominated frontier.

## Verified results and evidence boundaries

Checks were rerun on **2026-10-08**, using source commit [`65d5cc4`](https://github.com/rexw713ss/Fuzzy-RAG/commit/65d5cc49eb0a287720cd393c252ecc6fda06d54b).

### Unit tests

Verification environment: Linux, Python 3.12.14, NumPy 2.3.5, pytest 9.1.1, regex 2026.9.29, and PyYAML 6.0.3. This is lightweight module validation, not a reinstall of the recorded Windows GPU environment.

| Result | Count | Explanation |
|---|---:|---|
| Passed | 251 | Implemented modules and synthetic/hand-built test cases |
| Failed | 1 | `test_data_paths_match_the_scripts`: Windows backslashes and forward slashes compare differently under POSIX `Path` |
| Skipped | 1 | `test_rule_base_matches_csv`: external `Fuzzy_RAG_personal/etc/rule_base_81.csv` is absent |

The generated 81-rule count and 31/19/31 distribution have separate passing tests. These checks do not verify large external retrieval indexes or generated-answer quality.

### Controller monotonicity diagnostics

Settings: main Mamdani controller, all widths 0.20, 1,001-point centroid grid, and violation tolerance $10^{-9}$. Each comparison raises exactly one signal by one grid step. Rates are fractions of adjacent comparisons, not fractions of real queries.

| Input spacing | Grid points | Comparisons | Score-decrease violations | Rate | Maximum decrease | Violations crossing any candidate threshold |
|---|---:|---:|---:|---:|---:|---:|
| 0.20 | 1,296 | 4,320 | 64 | 1.48% | 0.006720 | 0 |
| 0.10 | 14,641 | 53,240 | 1,584 | 2.98% | 0.006720 | 0 |
| 0.05 | 194,481 | 740,880 | 44,800 | 6.05% | 0.037344 | 7,072 |

At spacing 0.05, each signal contributes 11,200 violations; mean decrease is 0.014994 and P95 is 0.029791. Of 61 candidate threshold values, 47 have no downgrades on this grid. The illustrative pair $(0.30,0.60)$ produces 484 downward action changes. **This pair was not fitted on real validation data.**

For example, raising S1 from 0.30 to 0.35 while holding the other signals at $(0.40,0.70,0.75)$ changes the score from 0.612179 to 0.574834, reversing A2 to A1 at $\theta_2=0.60$.

The ordinal consequents are monotone, but the defuzzified surface is not globally monotone. Coarse-grid zero action flips do not guarantee stability on finer grids or real paraphrases. Calibrated widths and validation-selected thresholds require separate diagnostics.

### What is not yet measured

No committed benchmark output currently establishes EM/F1 gains, latency savings, calibrated-fuzzy superiority, or paraphrase robustness on Natural Questions or HotpotQA. Retrieval smoke scripts exist, but dataset-level outputs are not published in the current tree. Baseline and evaluation tests use synthetic cached outcomes.

## Installation

Clone the repository and run commands from its root:

~~~bash
git clone https://github.com/rexw713ss/Fuzzy-RAG.git
cd Fuzzy-RAG
~~~

### Lightweight module checks

These tests do not need PyTorch, Java, model downloads, or large passage indexes:

~~~bash
python -m venv .venv-core
source .venv-core/bin/activate
python -m pip install numpy==2.3.5 pytest==9.1.1 regex==2026.9.29 PyYAML==6.0.3
python -m pytest -q -rs
~~~

Use Python 3.12 to match the verification above. On Windows PowerShell, activate with `.venv-core\Scripts\Activate.ps1`. The platform-specific failure and missing CSV above remain known limitations.

### Recorded retrieval environments

Tracked requirement files are development-environment snapshots:

| Environment | Recorded Python | Dependency file |
|---|---|---|
| Main/dense | 3.14.4 | `requirements.txt` |
| BM25 | 3.13.0 | `requirements-bm25.txt` |

Example Windows setup:

~~~powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

py -3.13 -m venv .venv-bm25
.\.venv-bm25\Scripts\python.exe -m pip install -r requirements-bm25.txt
~~~

The main snapshot requests `torch==2.14.1+cu126` from the CUDA 12.6 package index. The BM25 snapshot contains Windows-specific packages. These full environments were not independently reinstalled during the documentation checks and are not portable lock files. Confirm Python, PyTorch/CUDA, and Java compatibility before retrieval; record any substitutions in a reported experiment.

Pyserini requires a compatible JDK and `JAVA_HOME`. Existing development notes record Microsoft OpenJDK 25.0.4. Record the actual Java version used. For Windows console encoding issues, set `$env:PYTHONUTF8 = "1"`.

## Data and path setup

Large data and indexes are excluded by `.gitignore`. The existing repository records:

| Resource | Source | Recorded scope |
|---|---|---|
| DPR passages | [psgs_w100.tsv.gz](https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz) | 21,015,324 passages |
| Contriever-MSMARCO embeddings | [wikipedia_embeddings.tar](https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar) | 16 pickle shards |
| BM25 index | [Lucene DPR index](https://rgw.cs.uwaterloo.ca/pyserini/indexes/lucene/lucene-inverted.wikipedia-dpr-100w.20260508.deb4c7b.tar) | Same passage-ID space |
| NQ-open dev | [NQ-open.dev.jsonl](https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.dev.jsonl) | 3,610 questions; smoke checks only |

<details>
<summary>Recorded file sizes and checksums</summary>

Retained from the existing README; archives were not re-downloaded or re-hashed during this update.

| Resource | Recorded bytes | Recorded checksum |
|---|---:|---|
| DPR passages | 4,694,541,059 | SHA-256: `c39b020c855a2b5c25ffef3abe4a3b6f9b829ad7dbc14ec3d163d34d7c53ea8d` |
| Embedding archive | 32,499,691,520 | No published checksum recorded; use embedding verification |
| BM25 archive | 11,078,635,520 | MD5: `1ef94a97f2ac418577d1e6a9ecf44806` |
| NQ-open dev | Not recorded | SHA-256: `f15567f38099f3615f5b8a685c0aef449c11ad90d3da3735e8d1b98115b40616` |

</details>

The loader uses `pickle`, which can execute code; load only trusted releases. `verify_embeddings` checks sampled embedding-direction agreement and reports norm ratios, not cryptographic integrity. Use the matching `contriever-msmarco` archive rather than the unsupervised `contriever` archive.

Update `config.yaml` under `data` and the current script constants:

| File | Constant |
|---|---|
| `scripts/bm25_search.py` | `INDEX` |
| `scripts/smoke_retrieval.py` | `DATA` |
| `scripts/verify_embeddings.py` | `DATA` |
| `scripts/monotonicity_report.py` | `DATA` |

Scripts are not fully config-driven. Editing YAML alone does not redirect their filesystem access. Keep corpus/index IDs aligned and allow approximately 32 GB RAM for embeddings plus search workspaces.

## Running the available scripts

Examples use PowerShell and separate interpreters. Replace input paths with local paths.

### BM25 retrieval

Input: one JSON object per line containing `question`. Output preserves order and records `qid`, `question`, and hits.

~~~powershell
.\.venv-bm25\Scripts\python.exe -m scripts.bm25_search "path/to/NQ-open.dev.jsonl" "results/smoke/bm25.jsonl" --n 100 --k 50
~~~

### Hybrid retrieval smoke test

Use the same question file, containing `question` and `answer`; `answer` is a list of gold-answer strings.

~~~powershell
.\.venv\Scripts\python.exe -m scripts.smoke_retrieval "results/smoke/bm25.jsonl" "path/to/NQ-open.dev.jsonl" "results/smoke"
~~~

Outputs:

- `retrieval.jsonl`: sparse, dense, fused rankings, raw S1–S4, and answer-hit flags.
- `report.json`: answer-containing-passage Recall@k, signal summaries, correlations, and timing.

Recall uses DPR-style token-sequence answer matching, not generated-answer accuracy or multi-hop supporting-fact recall. Single-formulation smoke checks do not establish paraphrase robustness.

### Embedding checks

~~~powershell
.\.venv\Scripts\python.exe -m scripts.verify_embeddings --n 1000 --seed 0 --out "results/checks/verify_embeddings.json"
~~~

### Monotonicity report

No retrieval data or language model is required, but configure the output path first.

~~~powershell
.\.venv-core\Scripts\python.exe -m scripts.monotonicity_report --steps 5 10 20 --theta1 0.30 --theta2 0.60
~~~

The report is written to `checks/monotonicity.json` under the script's `DATA` directory. The example thresholds are illustrative, not fitted settings.

## Splits, evaluation, and reproducibility

- `splits.py` partitions clusters separately per dataset, nominally 60/20/20, with seed 42. Keep all three formulations together and save the actual split IDs.
- Fit P5/P95 scalers and jitter widths on training only; weights/cuts/thresholds on validation only. Keep test data locked.
- `OutcomeTable` expects an A0/A1/A2 row for every `(cluster_id, variant_id)`, with cached F1 and cost. Use dataset-qualified cluster IDs when combining datasets.
- The current oracle selects the first action in the fixed order A0 < A1 < A2 within $\delta=0.02$ of the best per-instance F1. It does **not** minimize measured latency.
- Routing flip rate (RFR) compares all three pairs per complete cluster. Harmful routing flip rate (HRFR) uses the same all-pairs denominator and counterfactual F1 criteria; HRFR is no greater than RFR. Incomplete clusters are skipped and counted.
- Within-cluster metrics include escalation-score SD, routed-F1 SD, and best-minus-worst F1.
- Confidence intervals and paired comparisons use cluster bootstrap: 1,000 resamples, percentile 95% intervals, seed 42.
- `config.yaml` records implemented constants; tests check agreement. Its `fitted` fields are null. A future pipeline must persist returned scalers, widths, thresholds, configuration, and split IDs.
- Record commit, model revisions, prompts, seeds, hardware, environment versions, and component costs. Include online routing overhead and all attempted action work.
- Exact batched-search timing is not per-query online latency. Treat smoke timings as diagnostics until the latency protocol is fixed.

## Repository map

| Location | Purpose |
|---|---|
| `main_logic/signals.py`, `dense.py`, `bm25.py`, `corpus.py`, `answers.py` | Retrieval, signals, corpus access, answer matching |
| `main_logic/fuzzy.py`, `calibration.py`, `baselines.py`, `routing.py` | Inference, calibration, baseline fitting, policy selection |
| `main_logic/actions.py` | A2 merge and rewrite-usability rules |
| `main_logic/splits.py`, `evaluation.py`, `monotonicity.py` | Splits, cached-outcome metrics, diagnostics |
| `main_logic/config.py`, `config.yaml` | Configuration loading and parameter record |
| `scripts/` | Retrieval smoke, embedding, monotonicity checks |
| `tests/` | Unit tests, synthetic evaluation, config consistency |
| `requirements*.txt` | Recorded development dependencies |

Some source comments reference specification v0.3 and external rule/provenance files not included in the repository.

## Roadmap

- [x] Sparse/dense retrieval and S1–S4 primitives
- [x] Complete fuzzy rule base and operator ablations
- [x] Cluster splitting and width-calibration primitives
- [x] B1–B4 fitting and cached-policy evaluation
- [x] Oracle, RFR/HRFR, within-cluster and bootstrap utilities
- [x] Monotonicity diagnostics and unit tests
- [ ] Portable, fully config-driven paths and environments
- [ ] Repository-contained rule CSV, specification, and provenance
- [ ] Validated paraphrase clusters and pilot signal analysis
- [ ] Cross-encoder, rewrite model, prompts, and shared generator
- [ ] Complete A0/A1/A2 action-outcome collection
- [ ] EM/token-F1, realistic costs, and held-out comparisons
- [ ] Published outputs, confidence intervals, and paper tables

Before scaling generation, confirm on a pilot that A1/A2 improve on A0 for a meaningful subset of queries. Unit tests and synthetic grids do not demonstrate routing effectiveness.

## Citation and license

No accompanying paper citation or software license is provided yet. In a reproducibility report, reference the repository URL and exact commit. Contact the owner before redistribution or code integration; external datasets and model weights retain their own terms.
