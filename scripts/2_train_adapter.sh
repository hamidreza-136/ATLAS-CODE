#!/bin/bash
# Stage 2 - train the attention adaptation network on Ref-YouTube-VOS train (Sec. 3.3).
# Requires: bash scripts/1_extract_attention.sh <lvlm> ytvos_train
#
# Usage:   GPU=0 bash scripts/2_train_adapter.sh <lvlm>
# Output:  runs/<lvlm>/adapter.pt
set -euo pipefail
cd "$(dirname "$0")/.."

LVLM=$1
CUDA_VISIBLE_DEVICES="${GPU:-0}" PYTHONPATH=. "${PYTHON:-python}" tools/train_adapter.py \
  --cache "cache/$LVLM/ytvos_train" --out "runs/$LVLM"
