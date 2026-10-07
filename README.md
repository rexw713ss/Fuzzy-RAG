# FuzzyRoute-RAG — code

An interpretable, paraphrase-robust fuzzy escalation controller for adaptive RAG. After one shared
hybrid retrieval (BM25 + Contriever), a type-1 Mamdani controller reads four retrieval-state
signals and picks one of three cost-ordered actions: A0 use the evidence, A1 rerank, A2 rewrite
and re-retrieve.

**The specification is the source of truth:**
`../Fuzzy_RAG_personal/etc/FuzzyRoute-RAG_Experimental_Specification_v0.3.md` (sibling docs
repo). Section numbers in code comments (§2.3, §8.1, …) refer to it. Its §10 lists what is built
and what comes next.

## Layout

```
D:\Han\rex_rag\
  Fuzzy-RAG\              this repo
    main_logic\           the package
      signals.py          §2    fusion, signals S1–S4, rescaling
      fuzzy.py            §3–4  membership functions, 81-rule base, escalation score + trace
      routing.py          §3.4  thresholds → action, grid search, Pareto frontier
      actions.py          §1    A2 merge rule, rewrite fallback
      calibration.py      §7.3  jitter calibration of the band widths (baseline B4)
      splits.py           §7.1  cluster-level 60/20/20 train/val/test split, seed 42
      config.py           §7.5  loads config.yaml
      dense.py            §2.1, §7.2  Contriever encoding, exact search over 21M passages
      bm25.py             §2.1, §7.2  BM25 via pyserini (BM25 environment only)
      corpus.py           §7.2  reading the DPR passages
      answers.py          §7.4  DPR has_answer, Recall@k
    tests\                one file per module
    scripts\              runnable steps (run as modules from the repo root)
    config.yaml           every experiment parameter, plus fitted values once stage 1 runs
    requirements.txt      main environment, exact versions
    requirements-bm25.txt BM25 environment, exact versions
  Fuzzy_RAG_personal\     docs repo: spec, paper draft, rule CSV, run logs
  data\                   corpora and indexes, NOT in git (see Data)
```

`tests/test_fuzzy_controller.py` reads `../Fuzzy_RAG_personal/etc/rule_base_81.csv`, so keep the
two repos side by side under the same folder names; otherwise that test skips.

## Machines

| Machine | GPU | Used for |
|---|---|---|
| Development desktop | Quadro P2200, 5 GB (Pascal, sm_61) | All code, tests, the retrieval index |
| Laptop | RTX 4050, 6 GB | Generation: Llama 3.1 8B Instruct, 4-bit, llama.cpp (stage 3+) |

Both have 64 GB RAM. Exact dense search holds ~32 GB of vectors in RAM.

## Setup (Windows)

### Main environment — Python 3.14.4

```
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`torch` is pinned to the CUDA 12.6 build (`2.14.1+cu126`) because newer CUDA builds dropped Pascal
GPUs; `requirements.txt` adds the PyTorch index for it. Do not `pip install --upgrade torch`.
Check the GPU is usable:

```
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())"
```

Always call `.venv\Scripts\python.exe -m ...` rather than the `.exe` launchers, so a moved venv
never runs the wrong interpreter.

### BM25 environment — Python 3.13 + Java

pyserini does not support Python 3.14, and installing it into the main environment would replace
the cu126 torch with a CPU build, so it has its own environment:

```
py -3.13 -m venv .venv-bm25
.venv-bm25\Scripts\python.exe -m pip install -r requirements-bm25.txt
```

It needs a JDK and `JAVA_HOME`. Tested with Microsoft OpenJDK 25.0.4 (pyserini documents Java 21;
25 works):

```
setx JAVA_HOME "C:\Program Files\Microsoft\jdk-25.0.4.101-hotspot"
```

### Console encoding

The Windows console code page (cp950 here) cannot print many Wikipedia characters. Set
`PYTHONUTF8=1` before running scripts.

## Data

Everything lives in `D:\Han\rex_rag\data\` (outside both repos, not synced). Download, check,
unpack:

| File | Source | Size | Check |
|---|---|---|---|
| `dpr\psgs_w100.tsv.gz` — 21,015,324 DPR Wikipedia passages | `https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz` | 4,694,541,059 B | sha256 `c39b020c855a2b5c25ffef3abe4a3b6f9b829ad7dbc14ec3d163d34d7c53ea8d` |
| `wikipedia_embeddings.tar` — Meta's precomputed **contriever-msmarco** passage embeddings | `https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar` | 32,499,691,520 B | no published checksum; verified by `scripts.verify_embeddings` |
| `lucene-inverted.wikipedia-dpr-100w.20260508.deb4c7b.tar` — prebuilt BM25 index | `https://rgw.cs.uwaterloo.ca/pyserini/indexes/lucene/lucene-inverted.wikipedia-dpr-100w.20260508.deb4c7b.tar` | 11,078,635,520 B | md5 `1ef94a97f2ac418577d1e6a9ecf44806` |
| `nq\NQ-open.dev.jsonl` — smoke tests only | `https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.dev.jsonl` | 3,610 questions | sha256 `f15567f38099f3615f5b8a685c0aef449c11ad90d3da3735e8d1b98115b40616` |

Unpack (Git Bash):

```
mkdir -p bm25 contriever-msmarco
tar -xf lucene-inverted.wikipedia-dpr-100w.20260508.deb4c7b.tar -C bm25
tar -xf wikipedia_embeddings.tar -C contriever-msmarco
```

Expected result: `bm25\lucene-inverted.wikipedia-dpr-100w.20260508.deb4c7b\` and
`contriever-msmarco\wikipedia_embeddings\passages_00` … `passages_15`. The `contriever` and
`contriever-msmarco` tarballs have the same size, so make sure you have the right one: the
verification below fails for the wrong checkpoint. `data\SOURCES.txt` records the provenance of
the local copies.

The embedding shards are Python pickles; loading a pickle can execute code, so load only the
official release.

HotpotQA is not set up yet (its corpus host was unreachable on Oct 6–7, 2026).

## Tests

```
.venv\Scripts\python.exe -m pytest tests
```

Unit tests need only numpy, pytest and `regex`; they use small synthetic data and run in about a
second. `bm25.py` has no unit tests in the main environment (pyserini is not there); it is
exercised end to end by `scripts.bm25_search`.

## Scripts

Run from the repo root.

| Command | What it does |
|---|---|
| `.venv\Scripts\python.exe -m scripts.verify_embeddings` | Encodes 1,000 random passages with contriever-msmarco and checks they match Meta's precomputed vectors (pass: every cosine ≥ 0.999). ~6 min. Report: `data\checks\verify_embeddings.json` |
| `.venv-bm25\Scripts\python.exe -m scripts.bm25_search QUESTIONS.jsonl OUT.jsonl --n 500` | BM25 top-50 per question |
| `.venv\Scripts\python.exe -m scripts.smoke_retrieval BM25.jsonl QUESTIONS.jsonl OUTDIR` | Contriever top-50 (exact), fusion, S1–S4 and answer recall for the same questions. ~8 min for 500 |

The results of these, and what they mean, are in
`../Fuzzy_RAG_personal/RUNLOG_2026-10-07_index.md`.

## Reproducibility

Spec §7.5 requires every reported result to be reproducible from `config.yaml` plus a commit of
this repo, run from a clean tree. `config.yaml` holds every fixed parameter;
`tests/test_config.py` checks each value equals the constant the code uses. Its `fitted:` section
(scalers, calibrated widths, thresholds) is filled by the fitting steps and committed with the
results. Pinned so far:

| What | Pin |
|---|---|
| Python packages | `requirements.txt`, `requirements-bm25.txt` (exact versions) |
| Contriever query encoder | `facebook/contriever-msmarco`, revision `abe8c1493371369031bcb1e02acb754cf4e162fa` (`dense.py`) |
| Passage vectors | Meta's contriever-msmarco release above, spot-checked |
| BM25 | prebuilt index `wikipedia-dpr-100w.20260508.deb4c7b`, k1 = 0.9, b = 0.4 (`bm25.py`) |
| Corpus | `psgs_w100.tsv.gz`, sha256 above |
| Search | exact inner product; ties broken by lower passage id (deterministic) |

Not yet: the generator, cross-encoder, prompts and generation seeds (stage 3), the paraphrase
protocol and question sample (stage 1).
