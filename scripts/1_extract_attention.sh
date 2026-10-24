#!/bin/bash
# Stage 1 - Attention Extraction (Sec. 3.2).
#
# Usage:   GPUS="0 1 2 3" bash scripts/1_extract_attention.sh <lvlm> <dataset>
#   lvlm:    qwen2.5-vl-7b | qwen2-vl-7b | llava-ov-7b | internvl3-8b
#   dataset: ytvos_train (training) | davis | reasonvos | revos | mevis | ytvos_valid | mevis_valid
# Output:  cache/<lvlm>/<dataset>/<video>__<exp_id>.pt
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_shards.sh

LVLM=$1
DATASET=$2
run_shards tools/extract_attention.py --lvlm "$LVLM" --dataset "$DATASET" --out "cache/$LVLM/$DATASET"
