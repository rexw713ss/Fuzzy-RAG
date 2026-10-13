# FuzzyRoute-RAG

[English](README.md) | [繁體中文](README.zh-TW.md)

Interpretable uncertainty-aware routing for retrieval-augmented generation.

> **Status (2026-10-13): first full experiment complete; results under review.** The whole pipeline now runs end to end on Natural Questions (NQ) and HotpotQA. It covers paraphrase clusters, retrieval and signals, all three actions with a shared generator, and the evaluation of every controller baseline. At the specification's headline budget (mean latency ≤ always-A1), B2, B3, B4, the ablations and the query-only router all select A1 for every query, because reranking adds almost no latency. Routing therefore makes no difference at that budget. See [Full-run results](#full-run-results-2026-10-12) and [Progress](#progress-and-next-steps). How to evaluate further is still open; the method has not been changed.

## Overview

Retrieval-augmented generation (RAG) supplies retrieved passages as context for answer generation. FuzzyRoute-RAG studies how retrieval uncertainty can guide the allocation of additional processing. A Type-1 Mamdani fuzzy inference system combines four retrieval-side signals and produces an escalation score.

The research question is:

> Can fuzzy routing improve the answer-quality/cost trade-off and reduce harmful routing changes across semantically equivalent query formulations?

This is a research objective, not an established result. The controller returns membership degrees and fired rules as a decision trace. Transparent inference does not by itself establish that the selected actions are effective.

## Method

The diagram shows the complete pipeline used in the full run.

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
- Dense search is exhaustive (exact inner product) over all 21,015,324 NQ passages, and likewise over the HotpotQA abstracts; the full run used exact search. Switching to an approximate index was considered but never adopted.
- **HotpotQA** uses the BEIR version of the HotpotQA Wikipedia abstracts (5,233,329), with Pyserini's prebuilt BM25 and `contriever-msmarco` indexes, because the original corpus host was unavailable. The dense index is read directly from its FAISS file with NumPy; the FAISS library is not needed.

### Uncertainty signals

Raw signals are designed so that larger values favor escalation. The full run partly supports this orientation. AUC is the chance that a hard query (no gold answer in the fused top 5; yes/no questions left out) scores higher than an easy one; 0.5 is chance level.

| | S1 | S2 | S3 | S4 |
|---|---:|---:|---:|---:|
| NQ | 0.54 | 0.62 | 0.55 | 0.59 |
| HotpotQA | 0.43 | 0.67 | 0.57 | 0.49 |

On HotpotQA, S1 points the opposite way and S4 is at chance. All four signals separate hard from easy queries only weakly.

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

| Action | Processing |
|---|---|
| A0 | Fused top-5 directly to generation |
| A1 | Cross-encoder rerank of the fused top-20, keep top-5, then generate |
| A2 | LLM rewrite of the query; second hybrid retrieval; merge/deduplicate; cross-encoder rerank of the top-30; keep top-5; generate |

A2 merging keeps the maximum fused score for each duplicate ID, rescales the union, and retains 30 candidates. Failed, empty, or case/whitespace-only rewrites are unusable. A2 then uses A1's passages and is flagged, and its cost still includes the rewrite call. This happened for 17 NQ instances and 0 HotpotQA instances.

All actions share the generator, context size (5 passages, each cut to 200 words), prompt and decoding:

| Component | Setting |
|---|---|
| Generator | Llama 3.1 8B Instruct, Q4_K_M GGUF, llama.cpp `llama-server` b11505, 31 GPU layers, context 4,096, prompt cache off |
| Decoding | Temperature 0, seed 42; answers ≤ 32 tokens, rewrites ≤ 64 tokens |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2`, revision `233902d25c440f23af6f7d6e94d2946bac0bee0a` |
| Answer scoring | Official SQuAD exact match (EM) and token F1 |
| Cost | End-to-end latency per query: retrieval + rerank + rewrite + generation. Retrievals, cross-encoder passes, LLM calls and input/output tokens are recorded alongside |

Prompts are pinned in `main_logic/actions.py` and `config.yaml` (`actions:`).

| Baseline | Score and fitting |
|---|---|
| B1: hard weighted threshold | Weighted signal sum; 35 weight vectors on a 0.25 lattice; validation threshold search |
| B2: crisp ordinal | Input cuts at 0.35/0.65; sum 0–8; 45 integer cut-point pairs |
| B3: fixed fuzzy | Mamdani score with all widths 0.20 |
| B4: calibrated fuzzy | Mamdani score with training-calibrated per-signal widths |

A score $e<\theta_1$ selects A0; $\theta_1\leq e<\theta_2$ selects A1; otherwise A2. The grids use $\theta_1\in[0.00,0.50]$, $\theta_2\in[0.50,1.00]$ and step 0.01, which gives 2,600 valid pairs. The specification's original ranges ($[0.20,0.50]$ and $[0.50,0.80]$, 960 pairs) were widened on 2026-10-11. Under them, most routers had no threshold pair within the always-A1 budget: every score ≥ 0.80 was forced to A2, and every score < 0.20 to A0.

Selection maximizes mean validation F1 under a mean-cost budget, which defaults to always-A1 latency. Ties go to lower cost, then lower $\theta_1$, then lower $\theta_2$. The query-only reference router is a multinomial logistic regression on 12 question-text features, trained on the training split's oracle actions. `pareto_frontier` returns a best-quality budget sweep and may repeat policies; it is not a deduplicated nondominated frontier.

### Paraphrase clusters

Each cluster is one original question plus two paraphrases from the same generator. The prompt, five candidates, temperature 0.7 and seed 42 are pinned in `main_logic/paraphrase.py`. A candidate is kept only if it passes four checks:

- not a duplicate (case- and punctuation-insensitive);
- bidirectional entailment under `microsoft/deberta-large-mnli`;
- `all-mpnet-base-v2` cosine ≥ 0.75;
- the gold answer does not appear in the paraphrase unless it was already in the original.

Back-translation was dropped. 300 originals per dataset were sampled with seed 42 from NQ-open dev and HotpotQA dev (fullwiki). The pilot's 20 are the first 20 of these 300. Clusters were split 60/20/20 by cluster (seed 42).

## Implementation checks

### Unit tests

**Current (2026-10-13, commit `6cb7b28`):** 329 tests pass on the Windows development environment.

**Earlier independent check (2026-10-08, commit [`65d5cc4`](https://github.com/rexw713ss/Fuzzy-RAG/commit/65d5cc49eb0a287720cd393c252ecc6fda06d54b)), not rerun since:** Linux, Python 3.12.14, NumPy 2.3.5, pytest 9.1.1, regex 2026.9.29, and PyYAML 6.0.3. This was lightweight module validation, not a reinstall of the recorded Windows GPU environment.

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
| 0.05 | 194,481 | 740,880 | 44,800 | 6.05% | 0.037344 | 29,680 |

The last column counts violations crossing any of the 101 threshold values of the current grid (0.00–1.00), recomputed on 2026-10-13. On the old 61-value grid (0.20–0.80) the 0.05 count was 7,072. At spacing 0.05, each signal contributes 11,200 violations; mean decrease is 0.014994 and P95 is 0.029791. Of the 101 threshold values, 83 have no downgrades on this grid (old grid: 47 of 61). The illustrative pair $(0.30,0.60)$ produces 484 downward action changes. **This pair was not fitted on real validation data.** At the thresholds actually selected for B4 in the full run, there are 0 downgrades on real queries.

For example, raising S1 from 0.30 to 0.35 while holding the other signals at $(0.40,0.70,0.75)$ changes the score from 0.612179 to 0.574834, reversing A2 to A1 at $\theta_2=0.60$.

The ordinal consequents are monotone, but the defuzzified surface is not globally monotone. Coarse-grid zero action flips do not guarantee stability on finer grids or real paraphrases. Calibrated widths and validation-selected thresholds require separate diagnostics.

## Full-run results (2026-10-12)

Code at commit `6cb7b28`; 329 unit tests pass (Windows, development environment). Hardware: one Quadro P2200 (5 GB). Outputs are in the data directory, not in this repository: `full/results/{nq,hotpotqa,pooled}.json`, `full/evaluate.log`, and the per-instance outcome rows `full/<dataset>/outcomes.jsonl`.

### Data

| | NQ | HotpotQA |
|---|---:|---:|
| Complete clusters (3 phrasings) | 261 / 300 | 290 / 300 |
| Query instances | 841 | 886 |
| Split, clusters (train / val / test) | 180 / 60 / 60 | 180 / 60 / 60 |
| Test instances | 170 | 176 |
| Outcome rows (instance × action) | 2,523 | 2,658 |
| Calibrated widths S1–S4 | 0.25, 0.229, 0.20, 0.25 | 0.221, 0.20, 0.20, 0.20 |

Stage-1 signal checks (missing values, identical values, clipping, correlation, direction) raised no flags on the full data. HotpotQA's S1–S4 Spearman correlation is 0.90, at the limit but not above it. Calibration changes the controller on both datasets. NQ's S1 and S4 widths hit the 0.25 cap and are flagged as reported.

Oracle shares (δ = 0.02, all instances):

| | A0 is enough | A1 needed | A2 needed | All three actions give the same F1 |
|---|---:|---:|---:|---:|
| NQ | 81.7% | 15.0% | 3.3% | 69% |
| HotpotQA | 88.3% | 9.3% | 2.5% | 81% |

### Test results at the headline budget (mean latency ≤ always-A1)

F1 with 95% cluster-bootstrap intervals for the fixed policies and the oracle; cost is mean latency per query. RFR / HRFR = routing flip rate / harmful routing flip rate across paraphrases.

| Policy | NQ F1 | NQ latency | NQ RFR / HRFR | HotpotQA F1 | HotpotQA latency | HotpotQA RFR / HRFR |
|---|---|---:|---|---|---:|---|
| always-A0 | 0.470 [0.379, 0.566] | 6.07 s | 0 / 0 | 0.510 [0.406, 0.615] | 4.39 s | 0 / 0 |
| always-A1 | 0.528 [0.433, 0.632] | 6.15 s | 0 / 0 | 0.530 [0.427, 0.633] | 4.30 s | 0 / 0 |
| always-A2 | 0.535 [0.434, 0.638] | 8.16 s | 0 / 0 | 0.540 [0.438, 0.646] | 6.40 s | 0 / 0 |
| oracle | 0.605 [0.506, 0.711] | 6.14 s | 0.284 / 0 | 0.587 [0.481, 0.690] | 4.41 s | 0.246 / 0 |
| B1 hard threshold | 0.521 | 6.16 s | 0.173 / 0.154 | 0.497 | 4.30 s | 0.129 / 0.117 |
| B2, B3, B4, three ablations, query-only | 0.528 | 6.15 s | 0 / 0 | 0.530 | 4.30 s | 0 / 0 |

- **Every router except B1 selected always-A1 on both datasets.** Reranking costs ~0.06–0.08 s on average, against ~4.0–5.6 s of generation, so A0 saves almost no latency. A2 costs ~2 s more and can only be afforded by sending other queries to A0. No threshold pair beat always-A1 on validation for the fuzzy routers.
- **B1** mixes A0 and A1. Its validation gain (NQ 0.500 vs 0.491; HotpotQA 0.549 vs 0.538) did not carry over to test. It flips between paraphrases on 13–17% of pairs, and most of those flips are harmful.
- **Paired tests (B4):**
  - Better than always-A0 on NQ F1 (+0.058, significant); not significant on HotpotQA (+0.020).
  - Lower HRFR than B1 on both datasets (significant); better F1 than B1 on HotpotQA (+0.033, significant).
  - Identical to B2, B3, the ablations and query-only.
  - These differences come from always-A1, not from routing.
- **Paper criteria for B4:** "uses at least two actions" fails (one action). "Lower HRFR than the strongest matched baseline (B2)" fails (both 0). The remaining criteria pass only because B4 equals always-A1.
- Monotonicity (§4.2 item 5) at B4's selected thresholds: 0 downgrades on both datasets.

### Budget sweep (thresholds chosen on validation at each budget, scored on test)

Between the always-A1 and always-A2 costs, the fuzzy routers do route. On NQ, B4 reaches **F1 0.542 at 6.78 s** (θ = 0.00 / 0.86), and B3 reaches 0.542 at 6.80 s. Both beat always-A2 (0.535 at 8.16 s) at 1.4 s less per query. B1 reaches 0.533, and B2 0.534–0.535. On HotpotQA, no router beats always-A1 at any budget. These are single-test-split points without confidence intervals. The saved `pareto_*.png` plots join points in budget order rather than as a frontier, so they are hard to read; see [Progress](#progress-and-next-steps).

### Evidence boundaries

- One generator, one hardware setup, 60 test clusters per dataset; intervals are wide (about ±0.1 F1).
- Latency was measured on a desktop GPU. Rows from two slow periods were discarded and regenerated. In the first, the outcome script's own PyTorch models held GPU memory, so the generator ran partly off the GPU; generation now runs in a separate process. In the second, the GPU was stuck in a low-power state. NQ rows after 2026-10-12 ran ~10–15% slower per token but above the speed guard.
- The 50-pair human audit of paraphrase meaning is not yet done, so flip rates assume the automatic paraphrase checks are correct. The sheet (`full/audit.csv`, drawn from the full clusters on 2026-10-13) is waiting to be filled in.

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

Use Python 3.12 to match the Linux check above. On Windows PowerShell, activate with `.venv-core\Scripts\Activate.ps1`. On Linux, the path-comparison failure and the skipped rule CSV test listed above may still occur (not rechecked since 2026-10-08). Modules that need PyTorch or transformers import them inside functions, so the tests run without them.

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
| NQ-open dev | [NQ-open.dev.jsonl](https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.dev.jsonl) | 3,610 questions; source of the 300 NQ originals and of the smoke checks |
| HotpotQA questions | Hugging Face `hotpotqa/hotpot_qa`, converted to the original JSON layout | train 90,447; dev distractor and dev fullwiki 7,405 each; the 300 originals come from dev fullwiki |
| HotpotQA BM25 index | [Lucene BEIR HotpotQA index](https://huggingface.co/datasets/castorini/prebuilt-indexes-beir/resolve/main/lucene-inverted/flat/lucene-inverted.beir-v1.0.0-hotpotqa.flat.20221116.505594.tar.gz) | 5,233,329 abstracts |
| HotpotQA dense index | [FAISS flat, contriever-msmarco](https://rgw.cs.uwaterloo.ca/pyserini/indexes/faiss/faiss-flat.beir-v1.0.0-hotpotqa.contriever-msmarco.20230124.tar.gz) | Same abstracts; ids in its `docid` file |

<details>
<summary>Recorded file sizes and checksums</summary>

Retained from the existing README; archives were not re-downloaded or re-hashed during this update.

| Resource | Recorded bytes | Recorded checksum |
|---|---:|---|
| DPR passages | 4,694,541,059 | SHA-256: `c39b020c855a2b5c25ffef3abe4a3b6f9b829ad7dbc14ec3d163d34d7c53ea8d` |
| Embedding archive | 32,499,691,520 | No published checksum recorded; use embedding verification |
| BM25 archive | 11,078,635,520 | MD5: `1ef94a97f2ac418577d1e6a9ecf44806` |
| NQ-open dev | Not recorded | SHA-256: `f15567f38099f3615f5b8a685c0aef449c11ad90d3da3735e8d1b98115b40616` |
| HotpotQA BM25 archive | 2,019,088,696 | MD5: `3f41d640a8ebbcad4f598140750c24f8` |
| HotpotQA dense archive | 14,889,518,959 | MD5: `38c37708f9927501ca2f7563aa43f407`; verified with `verify_embeddings --dataset hotpotqa` |

</details>

The loader uses `pickle`, which can execute code; load only trusted releases. `verify_embeddings` checks sampled embedding-direction agreement and reports norm ratios, not cryptographic integrity. Use the matching `contriever-msmarco` archive rather than the unsupervised `contriever` archive.

Unpack the archives into the data directory:

~~~bash
mkdir -p bm25 contriever-msmarco hotpotqa
tar -xf lucene-inverted.wikipedia-dpr-100w.20260508.deb4c7b.tar -C bm25
tar -xf wikipedia_embeddings.tar -C contriever-msmarco
tar -xzf lucene-inverted.beir-v1.0.0-hotpotqa.flat.20221116.505594.tar.gz -C hotpotqa
tar -xzf faiss-flat.beir-v1.0.0-hotpotqa.contriever-msmarco.20230124.tar.gz -C hotpotqa
~~~

HotpotQA's abstract texts are stored only inside the BM25 index. Export them once, in the BM25 environment, to a TSV in the DPR layout that the main environment reads:

~~~powershell
.\.venv-bm25\Scripts\python.exe -m scripts.export_corpus "data/hotpotqa/lucene-inverted.beir-v1.0.0-hotpotqa.flat.20221116.505594" "data/hotpotqa/corpus.tsv.gz"
~~~

All script paths live in `scripts/datasets.py` (corpus, dense index and BM25 index per dataset), and `tests/test_config.py` checks them against `config.yaml` under `data`. If the data moves, update both; editing YAML alone does not redirect the scripts' filesystem access. Keep corpus/index IDs aligned and allow approximately 32 GB RAM for embeddings plus search workspaces.

## Running the available scripts

Examples use PowerShell and separate interpreters. Replace input paths with local paths.

### Full experiment pipeline

Run from the repository root with the main environment and `$env:PYTHONUTF8 = "1"`. The generator scripts start and stop `llama-server` themselves; the paths are in `main_logic/llm.py`. Every generating step is resumable: rerunning it continues from its cache files.

~~~powershell
$D = "D:\Han\rex_rag\data\full"
# Step 1: paraphrase clusters (generate, then screen)
.\.venv\Scripts\python.exe -m scripts.build_clusters --dataset nq --n 300 --outdir "$D\nq"
# Steps 4-5: hybrid retrieval and raw signals for every phrasing (calls .venv-bm25 internally)
.\.venv\Scripts\python.exe -m scripts.retrieve_clusters --dataset nq --outdir "$D\nq"
# Steps 2, 6, 7: splits, scalers, jitter widths, stage-1 checks (both datasets)
.\.venv\Scripts\python.exe -m scripts.stage1_checks --root $D
# Step 8: action outcomes. Run the cross-encoder phases first, then generation in a fresh
# process so PyTorch does not keep GPU memory the generator needs
.\.venv\Scripts\python.exe -m scripts.outcomes --dataset nq --outdir "$D\nq" --stop-after selection
.\.venv\Scripts\python.exe -m scripts.outcomes --dataset nq --outdir "$D\nq"
# Step 9: fit routers on validation, report test results, plots
.\.venv\Scripts\python.exe -m scripts.evaluate --root $D
# Paraphrase audit sheet (50 pairs) and its score once filled in
.\.venv\Scripts\python.exe -m scripts.audit_sheet make --root $D
.\.venv\Scripts\python.exe -m scripts.audit_sheet score --root $D
~~~

Repeat the per-dataset steps with `--dataset hotpotqa --outdir "$D\hotpotqa"`. Generation stops by itself, without writing the slow rows, after 5 calls in a row below 120 prompt tokens/s. This keeps latency costs from a throttled GPU out of the results; rerun once the GPU is at full speed.

### BM25 retrieval

Input: NQ-open's JSON Lines file or a HotpotQA `.json` file. Add `--dataset hotpotqa` for HotpotQA (default `nq`); the same option applies to the smoke test and the embedding check. Output preserves order and records `qid`, `question`, and hits.

~~~powershell
.\.venv-bm25\Scripts\python.exe -m scripts.bm25_search "path/to/NQ-open.dev.jsonl" "results/smoke/bm25.jsonl" --n 100 --k 50
~~~

### Hybrid retrieval smoke test

Use the same question file, containing `question` and `answer` (NQ: a list of gold-answer strings; HotpotQA: one string). HotpotQA's yes/no answers cannot be found in a passage by string matching, so those questions are left out of Recall@k and counted.

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
.\.venv\Scripts\python.exe -m scripts.verify_embeddings --dataset hotpotqa --n 1000 --seed 0
~~~

### Monotonicity report

No retrieval data or language model is required, but configure the output path first.

~~~powershell
.\.venv-core\Scripts\python.exe -m scripts.monotonicity_report --steps 5 10 20 --theta1 0.30 --theta2 0.60
~~~

The report is written to `checks/monotonicity.json` under `DATA` in `scripts/datasets.py`. The example thresholds are illustrative, not fitted settings.

## Splits, evaluation, and reproducibility

- `splits.py` partitions clusters separately per dataset, nominally 60/20/20, with seed 42. Keep all three formulations together and save the actual split IDs.
- Fit P5/P95 scalers and jitter widths on training only; weights/cuts/thresholds on validation only. Keep test data locked.
- `OutcomeTable` expects an A0/A1/A2 row for every `(cluster_id, variant_id)`, with cached F1 and cost. Use dataset-qualified cluster IDs when combining datasets.
- The current oracle selects the first action in the fixed order A0 < A1 < A2 within $\delta=0.02$ of the best per-instance F1. It does **not** minimize measured latency.
- Routing flip rate (RFR) compares all three pairs per complete cluster. Harmful routing flip rate (HRFR) uses the same all-pairs denominator and counterfactual F1 criteria; HRFR is no greater than RFR. Incomplete clusters are skipped and counted.
- Within-cluster metrics include escalation-score SD, routed-F1 SD, and best-minus-worst F1.
- Confidence intervals and paired comparisons use cluster bootstrap: 1,000 resamples, percentile 95% intervals, seed 42.
- `config.yaml` records implemented constants; tests check agreement. Its `fitted` fields are still null: the pipeline writes split IDs, scalers and widths to `splits.json`, `scalers.json` and `jitter.json`, and every router's fitted thresholds to `results/<dataset>.json`.
- Record commit, model revisions, prompts, seeds, hardware, environment versions, and component costs. Include online routing overhead and all attempted action work.
- Retrieval latency in the outcome rows is batched search time divided by the number of queries (BM25 search without JVM start, query encoding, dense search). It is not measured one query at a time. Rerank, rewrite and generation latencies are measured per call, with the prompt cache off.

## Repository map

| Location | Purpose |
|---|---|
| `main_logic/signals.py`, `dense.py`, `bm25.py`, `corpus.py`, `answers.py` | Retrieval, signals, corpus access, answer matching |
| `main_logic/fuzzy.py`, `calibration.py`, `baselines.py`, `routing.py` | Inference, calibration, baseline fitting, policy selection |
| `main_logic/actions.py` | A1/A2 settings, answer and rewrite prompts, cross-encoder reranking, A2 merge and rewrite-usability rules |
| `main_logic/llm.py` | `llama-server` wrapper for the generator |
| `main_logic/paraphrase.py`, `checks.py` | Paraphrase generation and screening; stage-1 signal checks |
| `main_logic/learned.py` | Query-only learned router |
| `main_logic/splits.py`, `evaluation.py`, `monotonicity.py` | Splits, cached-outcome metrics, diagnostics |
| `main_logic/config.py`, `config.yaml` | Configuration loading and parameter record |
| `scripts/build_clusters.py`, `retrieve_clusters.py`, `stage1_checks.py`, `outcomes.py`, `evaluate.py`, `audit_sheet.py` | Full experiment pipeline (steps 1–9) |
| `scripts/` (others) | Retrieval smoke, embedding, monotonicity checks; `datasets.py` (all data paths); `export_corpus.py` (Lucene passages to TSV) |
| `tests/` | Unit tests, synthetic evaluation, config consistency |
| `requirements*.txt` | Recorded development dependencies |

Some source comments reference specification v0.3 and external rule/provenance files not included in the repository.

## Progress and next steps

- [x] Sparse/dense retrieval and S1–S4 primitives
- [x] NQ and HotpotQA indexes (HotpotQA from prebuilt BEIR indexes, embeddings verified)
- [x] Complete fuzzy rule base and operator ablations
- [x] Cluster splitting and width-calibration primitives
- [x] B1–B4 fitting and cached-policy evaluation
- [x] Oracle, RFR/HRFR, within-cluster and bootstrap utilities
- [x] Monotonicity diagnostics and unit tests
- [x] Paraphrase clusters and stage-1 signal checks (pilot 2026-10-10, full 2026-10-11)
- [x] Cross-encoder, rewrite and answer prompts, shared generator
- [x] Pilot gate: A1 and A2 improve on A0 for a meaningful subset of queries (passed)
- [x] Complete A0/A1/A2 outcome collection (2026-10-12)
- [x] EM/F1, measured latency, held-out comparisons with confidence intervals and paired tests
- [ ] **Decide how to evaluate beyond the degenerate always-A1 budget.** Options under discussion:
  - a second budget fixed in advance between the always-A1 and always-A2 costs;
  - fixed thresholds taken from the output membership functions;
  - actions whose costs differ more.
- [ ] Human audit of 50 paraphrase pairs (`audit_sheet`)
- [ ] Redraw the Pareto plots as frontiers (monotone best-F1-at-cost), with distinct colors for the fixed policies
- [ ] Portable, fully config-driven paths and environments
- [ ] Repository-contained rule CSV, specification, and provenance
- [ ] Paper tables

Observation for follow-up: B4's escalation scores cluster at the two ends of their range (0.133 = all Low, 0.867 = all High). Median test score is 0.709 on NQ and 0.154 on HotpotQA, so the controller rarely produces middle values.

## Citation and license

No accompanying paper citation or software license is provided yet. In a reproducibility report, reference the repository URL and exact commit. Contact the owner before redistribution or code integration; external datasets and model weights retain their own terms.
