#!/usr/bin/env bash
# v0.4 step 8 driver: every scripts.escalate phase in its own process, in order (REDESIGN_2026-10-15).
# Usage (Git Bash, from the repo root):
#   scripts/run_v4.sh DATASET SRC OUTDIR VARIANTS [extra escalate args...]
# e.g. scripts/run_v4.sh nq ../data/full/nq ../data/v4/pilot/nq V1,V2 --pilot-clusters 34
# Generation stops if the GPU spills (scripts.outcomes spill guard); the driver then waits
# 10 minutes and retries, up to 6 times, and the phase resumes where it stopped.
set -euo pipefail
dataset=$1; src=$2; out=$3; variants=$4; shift 4
py=.venv/Scripts/python.exe
export PYTHONUTF8=1
for phase in a1 followups second selection generation outcomes; do
  for attempt in 1 2 3 4 5 6 7; do
    if "$py" -m scripts.escalate --dataset "$dataset" --src "$src" --outdir "$out" \
         --phase "$phase" --variants "$variants" "$@"; then
      break
    fi
    if [ "$attempt" = 7 ]; then echo "phase $phase failed 7 times" >&2; exit 1; fi
    echo "phase $phase failed (attempt $attempt); retrying in 10 minutes" >&2
    sleep 600
  done
done
