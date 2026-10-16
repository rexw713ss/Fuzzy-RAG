# Preserved real-data replay outputs

Generated from source commit `1b2f1e3`, using the archived inputs introduced by main `667639a`. These are actual cached NQ/HotpotQA outcomes, not synthetic fixtures. All comparisons remain exploratory; no new LLM calls or online latency measurements were made.

| Directory | Purpose |
|---|---|
| `v04_fresh_4sig` | Main four-signal analysis, primary maximum A2 fraction 0.25 |
| `v04_fresh_3sig` | Prespecified S2–S4 feature ablation |
| `v04_old` | v0.4 actions on the original exposed test split |
| `v03_old` | v0.3 actions on the original exposed test split |

Each directory preserves all five quotas, all six router families, validation choices, fitted parameters, per-instance actions, primary fuzzy traces, cluster intervals, and provenance. `manifest.json` records the code commit, local input/source byte hashes and environment. Original caches are unchanged.

The final rerun reproduced every scientific report value and decision exactly; only diagnostic runtime fields changed. Timing and environment-dependent absolute paths are not expected to reproduce byte-for-byte elsewhere. JSON/JSONL/Markdown snapshots here use LF newlines for portable checksums. `SHA256.json` covers every other file in this evidence directory.

Commands to reproduce into new output directories are in [the experiment README](../README.md). Do not overwrite these committed snapshots.
