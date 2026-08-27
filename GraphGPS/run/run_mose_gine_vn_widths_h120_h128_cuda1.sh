#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/Xiaohan/Codes/SpecPE/MoSE_spec
GRAPHGPS="$REPO/GraphGPS"
PY=/home/Xiaohan/anaconda3/envs/SPE/bin/python
CONFIG_DIR="$GRAPHGPS/configs/ZINC/With_Edge_Features/GINe-VN"
DATASET_DIR="$GRAPHGPS/datasets"
CAMPAIGN=${1:-"$GRAPHGPS/results_mose_gine_vn_width_h120_h128_20260825"}
SEEDS=(0 14 48 96)
LABELS=(h120 h128)

declare -A CONFIGS=(
  [h120]="$CONFIG_DIR/+mose-h120.yaml"
  [h128]="$CONFIG_DIR/+mose-h128.yaml"
)

if [[ "$CAMPAIGN" != /* || "$CAMPAIGN" == / ]]; then
  echo "CAMPAIGN must be a safe absolute path: $CAMPAIGN" >&2
  exit 64
fi
if [[ -e "$CAMPAIGN" ]]; then
  echo "Refusing to reuse existing campaign path: $CAMPAIGN" >&2
  exit 73
fi
for label in "${LABELS[@]}"; do
  if [[ ! -f "${CONFIGS[$label]}" ]]; then
    echo "Missing configuration: ${CONFIGS[$label]}" >&2
    exit 66
  fi
done

exec 9>/tmp/specpe-mose-spec-gine-vn-widths-cuda1.lock
if ! flock -n 9; then
  echo "Another MoSE_spec GINE+VN width campaign holds the CUDA1 lock." >&2
  exit 75
fi

mkdir -p "$CAMPAIGN/logs" "$CAMPAIGN/provenance/source" \
  "$CAMPAIGN/runs" "$CAMPAIGN/state"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

SOURCE_FILES=(
  GraphGPS/configs/ZINC/With_Edge_Features/GINe-VN/+mose.yaml
  GraphGPS/configs/ZINC/With_Edge_Features/GINe-VN/+mose-h120.yaml
  GraphGPS/configs/ZINC/With_Edge_Features/GINe-VN/+mose-h128.yaml
  GraphGPS/graphgps/encoder/composed_encoders.py
  GraphGPS/graphgps/encoder/type_dict_encoder.py
  GraphGPS/graphgps/layer/gps_layer.py
  GraphGPS/graphgps/network/gps_model.py
  GraphGPS/graphgps/loader/master_loader.py
  GraphGPS/run/run_mose_gine_vn_widths_h120_h128_cuda1.sh
)

for relative_path in "${SOURCE_FILES[@]}"; do
  mkdir -p "$CAMPAIGN/provenance/source/$(dirname "$relative_path")"
  cp -- "$REPO/$relative_path" \
    "$CAMPAIGN/provenance/source/$relative_path"
done

{
  date --iso-8601=seconds
  printf 'repo=%s\n' "$REPO"
  printf 'git_head=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
  printf 'python=%s\n' "$PY"
  printf 'physical_gpu=1\n'
  printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'logical_accelerator=cuda:0\n'
  printf 'labels=%s\n' "${LABELS[*]}"
  printf 'seeds=%s\n' "${SEEDS[*]}"
  printf 'dataset_dir=%s\n' "$DATASET_DIR"
  printf 'comparison=official-node-mose-baseline-width-only\n'
} >"$CAMPAIGN/provenance/run.txt"

git -C "$REPO" status --short >"$CAMPAIGN/provenance/git-status.txt"
git -C "$REPO" diff >"$CAMPAIGN/provenance/tracked-source.patch"
"$PY" -m pip freeze >"$CAMPAIGN/provenance/pip-freeze.txt"
nvidia-smi >"$CAMPAIGN/provenance/nvidia-smi-at-launch.txt"
(
  cd "$CAMPAIGN/provenance/source"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) >"$CAMPAIGN/provenance/source-sha256.txt"

declare -A PIDS
for label in "${LABELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    key="${label}_seed${seed}"
    seed_out="$CAMPAIGN/runs/$label/seed_${seed}"
    seed_log="$CAMPAIGN/logs/${label}.seed_${seed}.log"
    mkdir -p "$seed_out"
    (
      printf 'START label=%s seed=%s time=%s\n' \
        "$label" "$seed" "$(date --iso-8601=seconds)"
      cd "$GRAPHGPS"
      exec "$PY" main.py \
        --cfg "${CONFIGS[$label]}" \
        --repeat 1 \
        out_dir "$seed_out" \
        dataset.dir "$DATASET_DIR" \
        accelerator cuda:0 \
        seed "$seed" \
        metric_agg argmin \
        num_workers 0 \
        wandb.use False \
        tensorboard_each_run False \
        tensorboard_agg False \
        train.auto_resume False \
        train.enable_ckpt True \
        train.ckpt_best False \
        train.ckpt_clean True
    ) >"$seed_log" 2>&1 &
    PIDS[$key]=$!
    printf '%s\t%s\t%s\t%s\n' \
      "$label" "$seed" "${PIDS[$key]}" "$seed_log" \
      >>"$CAMPAIGN/state/pids.tsv"
  done
done

printf 'All eight runs started at %s\n' "$(date --iso-8601=seconds)" \
  | tee "$CAMPAIGN/state/started.txt"

overall_status=0
for label in "${LABELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    key="${label}_seed${seed}"
    if wait "${PIDS[$key]}"; then
      process_status=0
    else
      process_status=$?
      overall_status=1
    fi
    printf '%s\t%s\t%s\t%s\n' \
      "$label" "$seed" "$process_status" "$(date --iso-8601=seconds)" \
      >>"$CAMPAIGN/state/exit-status.tsv"
  done
done

exit "$overall_status"
