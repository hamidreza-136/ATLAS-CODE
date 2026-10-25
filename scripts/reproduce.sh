#!/bin/bash
# Full ATLAS pipeline for one LVLM: extract training attention, train the
# adapter, then extract attention and segment every evaluation benchmark.
#
# Usage:   GPUS="0 1 2 3" bash scripts/reproduce.sh <lvlm> [dataset ...]
#   Default datasets: davis reasonvos revos mevis ytvos_valid mevis_valid
#   Set SKIP_TRAIN=1 to use an existing runs/<lvlm>/adapter.pt.
set -euo pipefail
cd "$(dirname "$0")/.."

LVLM=$1
shift
DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=(davis reasonvos revos mevis ytvos_valid mevis_valid)
fi

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  bash scripts/1_extract_attention.sh "$LVLM" ytvos_train
  read -r -a GPU_LIST <<< "${GPUS:-0}"
  GPU="${GPU_LIST[0]}" bash scripts/2_train_adapter.sh "$LVLM"
fi
for DATASET in "${DATASETS[@]}"; do
  bash scripts/1_extract_attention.sh "$LVLM" "$DATASET"
  bash scripts/3_segment_and_propagate.sh "$LVLM" "$DATASET"
done
