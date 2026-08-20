#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/Xiaohan/Codes/SpecPE/MoSE
GRAPHGPS="$REPO/GraphGPS"
PY=/home/Xiaohan/anaconda3/envs/SPE/bin/python
VERIFY="$GRAPHGPS/run/table1_mose_verify.py"
DATA_DIR="$GRAPHGPS/datasets_table1"
CAMPAIGN=${1:-"$GRAPHGPS/results_table1_repro_20260819"}

PHYSICAL_GPU=0
# A non-empty MIN_FREE_MIB overrides the profiled per-task thresholds below.
# Observed process peaks were <1 GiB for ZINC/CIFAR and 2.654 GiB for PCQM.
MIN_FREE_MIB=${MIN_FREE_MIB:-}
STABLE_SAMPLES=${STABLE_SAMPLES:-10}
POLL_SECONDS=${POLL_SECONDS:-30}
MAX_OOM_RETRIES=${MAX_OOM_RETRIES:-2}
PARTIAL_POLICY=${PARTIAL_POLICY:-stop}

if [[ "$CAMPAIGN" != /* || "$CAMPAIGN" == / ]]; then
  echo "CAMPAIGN must be a safe absolute path: $CAMPAIGN" >&2
  exit 64
fi

mkdir -p "$CAMPAIGN/launcher_logs" "$CAMPAIGN/state" \
  "$CAMPAIGN/quarantine"
exec 9>/tmp/specpe-mose-table1-cuda0.lock
if ! flock -n 9; then
  echo "Another MoSE Table 1 CUDA0 launcher holds the global lock." >&2
  exit 75
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
# OGB 1.3.6 uses torch.load without an explicit weights_only value. The
# processed file is generated locally from the official OGB archive.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

declare -A CFG RUN_NAME LAST_EPOCH METRIC METRIC_AGG TASK_MIN_FREE_MIB
CFG[zinc]="$GRAPHGPS/configs/ZINC/With_Edge_Features/GPSe/+mose.yaml"
RUN_NAME[zinc]="+mose"
LAST_EPOCH[zinc]=1999
METRIC[zinc]=mae
METRIC_AGG[zinc]=argmin
TASK_MIN_FREE_MIB[zinc]=6000

CFG[cifar]="$GRAPHGPS/configs/CIFAR/Main Results/gps+mose.yaml"
RUN_NAME[cifar]="gps+mose"
LAST_EPOCH[cifar]=99
METRIC[cifar]=accuracy
METRIC_AGG[cifar]=argmax
TASK_MIN_FREE_MIB[cifar]=6000

CFG[pcqm]="$GRAPHGPS/configs/PCQM4Mv2/gps+mose.yaml"
RUN_NAME[pcqm]="gps+mose"
LAST_EPOCH[pcqm]=299
METRIC[pcqm]=mae
METRIC_AGG[pcqm]=argmin
TASK_MIN_FREE_MIB[pcqm]=8000

SEEDS=(0 14 48 96)
TASKS=(zinc cifar pcqm)

write_provenance() {
  local dir="$CAMPAIGN/provenance"
  if [[ -f "$dir/recorded.complete" ]]; then
    return
  fi
  mkdir -p "$dir/configs"
  {
    date --iso-8601=seconds
    printf 'repo=%s\n' "$REPO"
    printf 'head=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
    printf 'python=%s\n' "$PY"
    printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
    printf 'seeds=%s\n' "${SEEDS[*]}"
    printf 'min_free_mib_override=%s\n' "${MIN_FREE_MIB:-none}"
    printf 'task_min_free_mib=zinc:%s,cifar:%s,pcqm:%s\n' \
      "${TASK_MIN_FREE_MIB[zinc]}" "${TASK_MIN_FREE_MIB[cifar]}" \
      "${TASK_MIN_FREE_MIB[pcqm]}"
  } >"$dir/run.txt"
  git -C "$REPO" status --short >"$dir/git-status.txt"
  git -C "$REPO" diff >"$dir/source-changes.patch"
  "$PY" -m pip freeze >"$dir/pip-freeze.txt"
  nvidia-smi >"$dir/nvidia-smi.txt"
  cp -- "${CFG[zinc]}" "$dir/configs/zinc+mose.yaml"
  cp -- "${CFG[cifar]}" "$dir/configs/cifar-gps+mose.yaml"
  cp -- "${CFG[pcqm]}" "$dir/configs/pcqm-gps+mose.yaml"
  cp -- "$GRAPHGPS/run/run_table1_mose.sh" "$VERIFY" "$dir/"
  sha256sum \
    "$REPO/hombasis-gt/hombasis-bench/data/zinc-data/zinc_with_homs_c7.json" \
    "$REPO/hombasis-gt/hombasis-bench/data/zinc-data/zinc_with_homs_c8.json" \
    "$REPO/hombasis-gt/hombasis-bench/data/zinc-data/zinc_3to8C_multhom.json" \
    "$REPO/hombasis-gt/image-datasets/data/CIFAR/cifar_v5.json" \
    "$REPO/hombasis-gt/pcqm/data/pcqm_v5.json" \
    >"$dir/count-data.sha256"
  touch "$dir/recorded.complete"
}

log_state() {
  local message=$1
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$message" | \
    tee -a "$CAMPAIGN/launcher.log"
}

wait_for_file() {
  local path=$1
  while [[ ! -s "$path" ]]; do
    log_state "WAIT required file: $path"
    sleep 60
  done
}

require_task_data() {
  local task=$1
  case "$task" in
    zinc)
      wait_for_file "$REPO/hombasis-gt/hombasis-bench/data/zinc-data/zinc_with_homs_c7.json"
      wait_for_file "$REPO/hombasis-gt/hombasis-bench/data/zinc-data/zinc_with_homs_c8.json"
      wait_for_file "$REPO/hombasis-gt/hombasis-bench/data/zinc-data/zinc_3to8C_multhom.json"
      ;;
    cifar)
      wait_for_file "$REPO/hombasis-gt/image-datasets/data/CIFAR/cifar_v5.json"
      wait_for_file "$DATA_DIR/GNNBenchmarkDataset/CIFAR10/processed/train_data.pt"
      ;;
    pcqm)
      wait_for_file "$REPO/hombasis-gt/pcqm/data/pcqm_v5.json"
      wait_for_file "$DATA_DIR/pcqm4m-v2/processed/geometric_data_processed.pt"
      ;;
  esac
}

wait_for_cuda0() {
  local task=$1
  local required=${MIN_FREE_MIB:-${TASK_MIN_FREE_MIB[$task]}}
  local stable=0 line free_mib util
  while (( stable < STABLE_SAMPLES )); do
    line=$(nvidia-smi -i "$PHYSICAL_GPU" \
      --query-gpu=memory.free,utilization.gpu \
      --format=csv,noheader,nounits)
    IFS=',' read -r free_mib util <<<"$line"
    free_mib=${free_mib//[[:space:]]/}
    util=${util//[[:space:]]/}

    if (( free_mib >= required )); then
      ((stable += 1))
    else
      stable=0
    fi
    log_state "CUDA0 task=$task free=${free_mib}MiB required=${required}MiB util=${util}% stable=${stable}/${STABLE_SAMPLES}"
    if (( stable < STABLE_SAMPLES )); then
      sleep "$POLL_SECONDS"
    fi
  done
}

verify_one() {
  local task=$1 seed=$2 run_parent=$3
  "$PY" "$VERIFY" "$run_parent" "${LAST_EPOCH[$task]}" \
    "${METRIC[$task]}" "${METRIC_AGG[$task]}" "$seed"
}

quarantine_run() {
  local task=$1 seed=$2 run_dir=$3 reason=$4
  local stamp target
  stamp=$(date +%Y%m%dT%H%M%S)
  target="$CAMPAIGN/quarantine/${task}.seed${seed}.${reason}.${stamp}"
  mv -- "$run_dir" "$target"
  log_state "QUARANTINE $run_dir -> $target"
}

run_one() {
  local task=$1 seed=$2
  local out_base="$CAMPAIGN/$task"
  local run_parent="$out_base/${RUN_NAME[$task]}"
  local run_dir="$run_parent/$seed"
  local log="$CAMPAIGN/launcher_logs/${task}.seed${seed}.log"
  local marker="$CAMPAIGN/state/${task}.seed${seed}.complete"
  local attempt=0 rc attempt_log

  if verify_one "$task" "$seed" "$run_parent" >/dev/null 2>&1; then
    log_state "SKIP verified complete: task=$task seed=$seed"
    touch "$marker"
    return
  fi

  if [[ -e "$run_dir" ]]; then
    if [[ "$PARTIAL_POLICY" == restart ]]; then
      quarantine_run "$task" "$seed" "$run_dir" partial
    else
      log_state "STOP partial/corrupt run exists: $run_dir"
      exit 73
    fi
  fi

  while true; do
    wait_for_cuda0 "$task"
    attempt=$((attempt + 1))
    attempt_log="${log}.attempt${attempt}"
    log_state "START task=$task seed=$seed attempt=$attempt"
    {
      printf '%s\n' "===== task=$task seed=$seed attempt=$attempt ====="
      date --iso-8601=seconds
      nvidia-smi -i "$PHYSICAL_GPU"
    } | tee -a "$log" "$attempt_log"

    set +e
    (
      cd "$GRAPHGPS"
      "$PY" main.py \
        --cfg "${CFG[$task]}" \
        --repeat 1 \
        out_dir "$out_base" \
        dataset.dir "$DATA_DIR" \
        accelerator cuda:0 \
        seed "$seed" \
        metric_agg "${METRIC_AGG[$task]}" \
        num_workers 0 \
        wandb.use False \
        tensorboard_each_run False \
        tensorboard_agg False \
        train.auto_resume False \
        train.enable_ckpt True \
        train.ckpt_clean True
    ) 2>&1 | tee -a "$log" "$attempt_log"
    rc=${PIPESTATUS[0]}
    set -e

    if (( rc == 0 )); then
      if verify_one "$task" "$seed" "$run_parent" | tee -a "$log"; then
        touch "$marker"
        log_state "COMPLETE task=$task seed=$seed"
        return
      fi
      log_state "STOP process returned 0 but verification failed: task=$task seed=$seed"
      exit 74
    fi

    if grep -Eqi 'out of memory|cudaErrorMemoryAllocation' "$attempt_log" && \
       (( attempt <= MAX_OOM_RETRIES )); then
      if [[ -e "$run_dir" ]]; then
        quarantine_run "$task" "$seed" "$run_dir" oom
      fi
      log_state "RETRY CUDA OOM: task=$task seed=$seed attempt=$attempt"
      continue
    fi

    log_state "FAIL task=$task seed=$seed rc=$rc"
    exit "$rc"
  done
}

write_provenance

for task in "${TASKS[@]}"; do
  require_task_data "$task"
  for seed in "${SEEDS[@]}"; do
    run_one "$task" "$seed"
  done
  run_parent="$CAMPAIGN/$task/${RUN_NAME[$task]}"
  "$PY" "$VERIFY" --aggregate "$run_parent" "${LAST_EPOCH[$task]}" \
    "${METRIC[$task]}" "${METRIC_AGG[$task]}" "${SEEDS[@]}" | \
    tee "$CAMPAIGN/state/${task}.aggregate.json"
done

log_state "ALL TABLE1 MOSE RUNS COMPLETE"
