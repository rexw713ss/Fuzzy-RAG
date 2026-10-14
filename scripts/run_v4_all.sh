#!/usr/bin/env bash
# v0.4 full experiment after the pilot (REDESIGN_2026-10-15 section 5, steps 4-6), unattended.
# Usage (Git Bash, from the repo root): scripts/run_v4_all.sh VARIANT
# Order puts the primary result first:
#   1. A2 on the train + val instances of the 300-original clusters (routers are fitted there);
#   2. the fresh held-out clusters: originals 301-450 of the seed-42 order, built, retrieved,
#      then A1 and A2 (the primary test set);
#   3. A2 on the old test split (secondary: seen in the first run);
#   4. step 9 (scripts.evaluate_v4).
# The pilot's follow-up queries and answers are reused (same prompts, temperature 0).
set -euo pipefail
v=$1
py=.venv/Scripts/python.exe
export PYTHONUTF8=1
D=../data
for d in hotpotqa nq; do
  out=$D/v4/$d
  mkdir -p "$out"
  if [ ! -f "$out/followups.jsonl" ]; then cp "$D/v4/pilot/$d/followups.jsonl" "$out/followups.jsonl"; fi
  if [ ! -f "$out/generations_seed.jsonl" ]; then cp "$D/v4/pilot/$d/generations.jsonl" "$out/generations_seed.jsonl"; fi
  scripts/run_v4.sh "$d" "$D/full/$d" "$out" "$v" --final "$v" --splits-only train,val
done
for d in hotpotqa nq; do
  src=$D/v4/fresh_src/$d
  mkdir -p "$src"
  if [ ! -f "$src/clusters.jsonl" ]; then
    "$py" -m scripts.build_clusters --dataset "$d" --n 150 --offset 300 --outdir "$src"
  fi
  if [ ! -f "$src/signals.jsonl" ]; then
    "$py" -m scripts.retrieve_clusters --dataset "$d" --outdir "$src"
  fi
  scripts/run_v4.sh "$d" "$src" "$D/v4/fresh/$d" "$v" --final "$v"
done
for d in hotpotqa nq; do
  scripts/run_v4.sh "$d" "$D/full/$d" "$D/v4/$d" "$v" --final "$v"
done
"$py" -m scripts.evaluate_v4 > "$D/v4/evaluate_v4.log" 2>&1
echo "v0.4 experiment complete"
