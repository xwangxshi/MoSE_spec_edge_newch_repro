#!/usr/bin/env bash
set -Eeuo pipefail

PY=/home/Xiaohan/anaconda3/envs/SPE/bin/python
SEEDS=(0 14 48 96)
LAUNCHER_REPO=/home/Xiaohan/Codes/SpecPE/MoSE_spec

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 REPO CONFIG_NAME CAMPAIGN" >&2
  exit 64
fi

REPO=$1
CONFIG_NAME=$2
CAMPAIGN=$3

case "$REPO" in
  /home/Xiaohan/Codes/SpecPE/MoSE_spec | \
  /home/Xiaohan/Codes/SpecPE/MoSE_spec_onehot)
    ;;
  *)
    echo "Unexpected repository: $REPO" >&2
    exit 64
    ;;
esac

GRAPHGPS="$REPO/GraphGPS"
CONFIG_DIR="$GRAPHGPS/configs/ZINC/With_Edge_Features/GINe-VN"
CFG="$CONFIG_DIR/$CONFIG_NAME"
DATASET_DIR="$GRAPHGPS/datasets"
CACHE_DIR="$REPO/hombasis-gt/hombasis-bench/data/zinc-data/zinc12k_two_root_mose_cheby_k10_edge_v1"

if [[ "$CONFIG_NAME" == */* || "$CONFIG_NAME" != +specmose-*.yaml ]]; then
  echo "CONFIG_NAME must be a +specmose-*.yaml basename" >&2
  exit 64
fi
if [[ ! -f "$CFG" ]]; then
  echo "Missing configuration: $CFG" >&2
  exit 66
fi
case "$CAMPAIGN" in
  "$GRAPHGPS"/results_specmose_gine_vn_matched_*)
    ;;
  *)
    echo "Unexpected campaign path: $CAMPAIGN" >&2
    exit 64
    ;;
esac
if [[ -e "$CAMPAIGN" ]]; then
  echo "Refusing to reuse campaign path: $CAMPAIGN" >&2
  exit 73
fi

SOURCE_FILES=(
  "GraphGPS/configs/ZINC/With_Edge_Features/GINe-VN/$CONFIG_NAME"
  GraphGPS/graphgps/config/specmose_edge_config.py
  GraphGPS/graphgps/encoder/specmose_edge_encoder.py
  GraphGPS/graphgps/loader/spectral_edge.py
  GraphGPS/graphgps/loader/master_loader.py
  GraphGPS/graphgps/optimizer/extra_optimizers.py
  GraphGPS/graphgps/transform/specmose_virtual_node.py
  GraphGPS/unittests/test_specmose_virtual_node.py
  hombasis-gt/pact/pact/two_root.py
  hombasis-gt/pact/pact/chebyshev_pair.py
  hombasis-gt/pact/precompute_zinc_two_root_chebyshev.py
)

for relative_path in "${SOURCE_FILES[@]}"; do
  if [[ ! -f "$REPO/$relative_path" ]]; then
    echo "Missing source file: $REPO/$relative_path" >&2
    exit 66
  fi
done

mkdir -p "$CAMPAIGN/logs" "$CAMPAIGN/provenance/source" \
  "$CAMPAIGN/provenance/launcher" "$CAMPAIGN/runs" "$CAMPAIGN/state"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

for relative_path in "${SOURCE_FILES[@]}"; do
  mkdir -p "$CAMPAIGN/provenance/source/$(dirname "$relative_path")"
  cp -- "$REPO/$relative_path" \
    "$CAMPAIGN/provenance/source/$relative_path"
done
cp -- "$LAUNCHER_REPO/GraphGPS/run/run_specmose_gine_vn_matched_group_cuda0.sh" \
  "$CAMPAIGN/provenance/launcher/"
cp -- "$LAUNCHER_REPO/GraphGPS/run/queue_specmose_gine_vn_matched_8groups_cuda0.sh" \
  "$CAMPAIGN/provenance/launcher/"

{
  date --iso-8601=seconds
  printf 'repo=%s\n' "$REPO"
  printf 'git_head=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
  printf 'python=%s\n' "$PY"
  printf 'physical_gpu=0\n'
  printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'logical_accelerator=cuda:0\n'
  printf 'seeds=%s\n' "${SEEDS[*]}"
  printf 'config=%s\n' "$CFG"
  printf 'dataset_dir=%s\n' "$DATASET_DIR"
  printf 'cache_dir=%s\n' "$CACHE_DIR"
  printf 'runtime_overrides=device,seed,output,logging,checkpoint_only\n'
} >"$CAMPAIGN/provenance/run.txt"

git -C "$REPO" status --short >"$CAMPAIGN/provenance/git-status.txt"
git -C "$REPO" diff >"$CAMPAIGN/provenance/tracked-source.patch"
"$PY" -m pip freeze >"$CAMPAIGN/provenance/pip-freeze.txt"
nvidia-smi >"$CAMPAIGN/provenance/nvidia-smi-at-launch.txt"
cp -- "$CACHE_DIR/metadata.json" "$CAMPAIGN/provenance/cache-metadata.json"
(
  cd "$CAMPAIGN/provenance"
  find source launcher -type f -print0 | sort -z | xargs -0 sha256sum
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

printf 'All four seeds started at %s\n' "$(date --iso-8601=seconds)"

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
