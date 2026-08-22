#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/Xiaohan/Codes/SpecPE/MoSE_spec
GRAPHGPS="$REPO/GraphGPS"
PY=/home/Xiaohan/anaconda3/envs/SPE/bin/python
CONFIG_DIR="$GRAPHGPS/configs/ZINC/With_Edge_Features/GINe"
DATASET_DIR="$GRAPHGPS/datasets"
CACHE_DIR="$REPO/hombasis-gt/hombasis-bench/data/zinc-data/zinc12k_two_root_mose_cheby_k10_edge_v1"
SEEDS=(0 14 48 96)

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 CONFIG_NAME CAMPAIGN_DIR" >&2
  exit 64
fi

CONFIG_NAME=$1
CAMPAIGN=$2
if [[ "$CONFIG_NAME" == */* || "$CONFIG_NAME" != +specmose*.yaml ]]; then
  echo "CONFIG_NAME must be a +specmose*.yaml basename" >&2
  exit 64
fi
CFG="$CONFIG_DIR/$CONFIG_NAME"
if [[ ! -f "$CFG" ]]; then
  echo "Missing configuration: $CFG" >&2
  exit 66
fi
if [[ "$CAMPAIGN" != /* || "$CAMPAIGN" == / ]]; then
  echo "CAMPAIGN must be a safe absolute path: $CAMPAIGN" >&2
  exit 64
fi
if [[ -e "$CAMPAIGN" ]]; then
  echo "Refusing to reuse existing campaign path: $CAMPAIGN" >&2
  exit 73
fi

mkdir -p "$CAMPAIGN/logs" "$CAMPAIGN/provenance/source" \
  "$CAMPAIGN/runs" "$CAMPAIGN/state"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

SOURCE_FILES=(
  "GraphGPS/configs/ZINC/With_Edge_Features/GINe/$CONFIG_NAME"
  GraphGPS/graphgps/config/specmose_edge_config.py
  GraphGPS/graphgps/encoder/specmose_edge_encoder.py
  GraphGPS/graphgps/loader/spectral_edge.py
  GraphGPS/graphgps/loader/master_loader.py
  GraphGPS/graphgps/optimizer/extra_optimizers.py
  GraphGPS/run/run_specmose_gine_variant_cuda1.sh
  hombasis-gt/pact/pact/two_root.py
  hombasis-gt/pact/pact/chebyshev_pair.py
  hombasis-gt/pact/precompute_zinc_two_root.py
  hombasis-gt/pact/precompute_zinc_two_root_chebyshev.py
  hombasis-gt/pact/tests/test_two_root.py
  hombasis-gt/pact/tests/test_chebyshev_pair.py
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
  printf 'seeds=%s\n' "${SEEDS[*]}"
  printf 'config=%s\n' "$CFG"
  printf 'dataset_dir=%s\n' "$DATASET_DIR"
  printf 'cache_dir=%s\n' "$CACHE_DIR"
  printf 'shared_gpu_at_launch=true\n'
} >"$CAMPAIGN/provenance/run.txt"

git -C "$REPO" status --short >"$CAMPAIGN/provenance/git-status.txt"
git -C "$REPO" diff >"$CAMPAIGN/provenance/tracked-source.patch"
"$PY" -m pip freeze >"$CAMPAIGN/provenance/pip-freeze.txt"
nvidia-smi >"$CAMPAIGN/provenance/nvidia-smi-at-launch.txt"
cp -- "$CACHE_DIR/metadata.json" "$CAMPAIGN/provenance/cache-metadata.json"
(
  cd "$CAMPAIGN/provenance/source"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) >"$CAMPAIGN/provenance/source-sha256.txt"

declare -A PIDS
for seed in "${SEEDS[@]}"; do
  seed_out="$CAMPAIGN/runs/seed_${seed}"
  seed_log="$CAMPAIGN/logs/seed_${seed}.log"
  mkdir -p "$seed_out"
  (
    printf 'START seed=%s time=%s\n' "$seed" "$(date --iso-8601=seconds)"
    cd "$GRAPHGPS"
    exec "$PY" main.py \
      --cfg "$CFG" \
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
  PIDS[$seed]=$!
  printf '%s\t%s\t%s\n' "$seed" "${PIDS[$seed]}" "$seed_log" \
    >>"$CAMPAIGN/state/pids.tsv"
done

printf 'All four seeds for %s started at %s\n' \
  "$CONFIG_NAME" "$(date --iso-8601=seconds)"

overall_status=0
for seed in "${SEEDS[@]}"; do
  if wait "${PIDS[$seed]}"; then
    process_status=0
  else
    process_status=$?
    overall_status=1
  fi
  printf '%s\t%s\t%s\n' "$seed" "$process_status" \
    "$(date --iso-8601=seconds)" >>"$CAMPAIGN/state/exit-status.tsv"
done

exit "$overall_status"
