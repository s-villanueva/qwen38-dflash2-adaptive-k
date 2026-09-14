#!/usr/bin/env bash
# Reproducible TP=4 hybrid DFlash2 policy for RadixArk/Qwen3.8-27B.
# This wrapper keeps the base launcher generic and saves the validated policy
# in a dedicated, versionable implementation file.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec env \
  SPECULATIVE_METHOD=dflash \
  SPECULATIVE_TOKENS=12 \
  MAX_NUM_SEQS=64 \
  MAX_NUM_BATCHED_TOKENS=8192 \
  GPU_MEMORY_UTILIZATION=0.60 \
  DYNAMIC_SD_SCHEDULE='[[1,1,12],[2,4,5],[5,64,0]]' \
  SHM_SIZE=32g \
  bash "$SCRIPT_DIR/run_tp4_base.sh"
