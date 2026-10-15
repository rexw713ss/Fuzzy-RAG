# Saved run outputs

The results of each full run, plus the small inputs needed to recompute them from cached outcome
rows without any model, index or corpus. Copied from the data directory
(`D:\Han\rex_rag\data`) on 2026-10-15. The originals stay there, and the pipeline scripts still
read and write the data directory.

| Run | Code commit | Folder | Original location |
|---|---|---|---|
| v0.3: A0 / A1 / A2, S1–S4, 81 rules | `6cb7b28` | `v0.3/` | `data\full\` |
| v0.4: A1 / A2, S2–S4, 27 rules, second-hop A2 (V1) | `53d0946` | `v0.4/` | `data\v4\` |
| Stage 1 (pilot clusters) | earlier commits | `stage1/` | `data\stage1\results\` |

## Layout

```
runs/
  v0.3/
    splits.json, scalers.json, jitter.json   splits, p5/p95 scalers, jitter widths (fitted on train)
    signals_scaled.jsonl                     S1–S4 scaled, every instance of both datasets
    <dataset>/clusters.jsonl                 paraphrase clusters, with screening scores
    <dataset>/queries.jsonl                  one row per phrasing (instance)
    <dataset>/outcomes.jsonl                 one row per instance and action: answer, EM, F1, latencies
    reported/                                results as reported: <dataset>.json, pooled.json, pareto_*.png
    analysis/a2_diagnosis.py                 why the v0.3 A2 was weak (REDESIGN_2026-10-15 §4.1)
  v0.4/
    <dataset>/outcomes.jsonl                 A1 and A2 on the 300 original clusters (train, val, old test)
    fresh_src/<dataset>/                     fresh held-out clusters (originals 301–450): clusters,
                                             queries, raw signals, cluster-building report
    fresh/<dataset>/outcomes.jsonl           A1 and A2 on the fresh set (primary test set)
    reported/                                results as reported, evaluate_v4.log, run_all.log
    pilot/                                   pilot report (V1 vs V2) and its log
    dryrun_oldA2/reported/                   step 9 on v0.3's A1 / A2 rows, before the new A2
    analysis/trainval_check.py               new A2 on train + val (RUNLOG §3.1)
  stage1/reported/                           stage-1 results (pilot clusters)
  checks/                                    monotonicity and embedding checks
  audits/2026-10-14, 2026-10-15/             audits of the v0.3 and v0.4 runs
```

`<dataset>` is `nq` or `hotpotqa`.

## Recompute the results

Run from the repository root, in the main environment. No model, index or corpus is needed.

```powershell
git checkout 53d0946   # or a later commit; the v0.4 code reproduces v0.3 exactly
.venv\Scripts\python.exe -m scripts.evaluate --root runs/v0.3
.venv\Scripts\python.exe -m scripts.evaluate_v4 --full runs/v0.3 --v4 runs/v0.4
```

Both write to a `results/` folder next to `reported/` (`runs/v0.3/results/`,
`runs/v0.4/results/`). `.gitignore` ignores `results/`, so a rerun never overwrites the committed
copy. Compare the two.

Checked on 2026-10-15 with commit `53d0946`:
- **v0.4:** every file is identical to `reported/`.
- **v0.3:** every number is identical. The newer code adds one field per fit, `thresholds`, which
  repeats `theta1` / `theta2`.

## Not included

These stay in the data directory. They are large, can be rebuilt, or are third-party text. See
the main README, "Data and path setup".
- Corpora, indexes, embeddings and the generator model.
- Retrieval rows: `retrieval.jsonl`, `second_retrieval.jsonl`, `*_bm25.jsonl`, `embeddings.npy`.
- Raw generations (`generations*.jsonl`), passage texts (`passages.jsonl`), paraphrase candidates
  (`candidates.jsonl`) and server logs.

## Notes

- **Hardcoded paths.**
  - The audit and analysis scripts were written against the data directory and use absolute
    paths (`D:\Han\rex_rag\data\...`). They are kept as the record of what was run; edit `ROOT` /
    `DATA` at the top of each to point them at `runs/`.
  - Some of them also need files that are not copied here (generations, selections, retrieval).
- **Licences.** The clusters and queries are paraphrases of questions from NQ-open (CC BY-SA 3.0)
  and HotpotQA (CC BY-SA 4.0), generated with Llama 3.1 8B Instruct. Credit those datasets when
  you reuse them.
- **Run records.** Per-run decisions and findings are in the project notes (`RUNLOG_*.md`,
  `DECISIONS.md`), not in this repository. The v0.4 audit is `RUNLOG_2026-10-15_v04.md` §5.
