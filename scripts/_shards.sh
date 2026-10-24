# Run a Python tool once per GPU in $GPUS (default "0"), each on its own shard
# of the samples, and fail if any worker fails.
# Usage (after `source`): run_shards tools/<tool>.py [args...]
run_shards() {
  local gpus pids i
  read -r -a gpus <<< "${GPUS:-0}"
  pids=()
  for i in "${!gpus[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpus[$i]}" PYTHONPATH=. "${PYTHON:-python}" "$@" \
      --num-shards "${#gpus[@]}" --shard-id "$i" &
    pids+=($!)
  done
  for i in "${pids[@]}"; do
    wait "$i"
  done
}
