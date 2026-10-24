#!/bin/bash
# Stages 2 + 3 at test time - attention adaptation, then segmentation and
# propagation with SAM2 (Sec. 3.3-3.4), followed by J&F evaluation.
# Requires: bash scripts/1_extract_attention.sh <lvlm> <dataset>
#           runs/<lvlm>/adapter.pt (trained or downloaded; override with ADAPTER=...)
#
# Usage:   GPUS="0 1 2 3" bash scripts/3_segment_and_propagate.sh <lvlm> <dataset>
# Output:  outputs/<lvlm>/<dataset>/<video>/<exp_id>/<frame>.png
#          outputs/<lvlm>/<dataset>.json   (J, F, J&F)
#          outputs/<lvlm>/<dataset>_submission.zip   (ytvos_valid / mevis_valid servers)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/_shards.sh

LVLM=$1
DATASET=$2
OUT="outputs/$LVLM/$DATASET"
run_shards tools/segment_and_propagate.py --dataset "$DATASET" \
  --cache "cache/$LVLM/$DATASET" --adapter "${ADAPTER:-runs/$LVLM/adapter.pt}" --out "$OUT"

case "$DATASET" in
  ytvos_valid)  # Ref-YouTube-VOS server: Annotations/<video>/<exp_id>/<frame>.png
    (cd "outputs/$LVLM" && ln -sfn "$DATASET" Annotations && zip -qr "${DATASET}_submission.zip" Annotations && rm Annotations)
    echo "Submit outputs/$LVLM/${DATASET}_submission.zip to the Ref-YouTube-VOS server." ;;
  mevis_valid)  # MeViS server: <video>/<exp_id>/<frame>.png at the archive root
    (cd "$OUT" && zip -qr "../${DATASET}_submission.zip" .)
    echo "Submit outputs/$LVLM/${DATASET}_submission.zip to the MeViS server." ;;
  *)
    PYTHONPATH=. "${PYTHON:-python}" tools/evaluate.py --dataset "$DATASET" --pred "$OUT" ;;
esac
