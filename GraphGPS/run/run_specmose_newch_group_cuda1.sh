#!/usr/bin/env bash
set -Eeuo pipefail

PY=/home/Xiaohan/anaconda3/envs/SPE/bin/python
REPO=/home/Xiaohan/Codes/SpecPE/MoSE_spec_edge_newch
GRAPHGPS="$REPO/GraphGPS"
CFG="$GRAPHGPS/configs/ZINC/With_Edge_Features/GINe-VN/+specmose-newch-k5-h110-f10-t21.yaml"
DATASET_DIR="$GRAPHGPS/datasets"
CACHE_DIR="$REPO/hombasis-gt/hombasis-bench/data/zinc-data/zinc12k_two_root_mose_eq_cheby_k10_edge_v1"
SEEDS=(0 14 48 96)

if [[ "$#" -ne 7 ]]; then
  echo "Usage: $0 TAG CAMPAIGN ZERO_VN HIDDEN_DP OUTPUT_DP FILTER_DP SPEC_WD" >&2
  exit 64
fi

TAG=$1
CAMPAIGN=$2
ZERO_VN=$3
HIDDEN_DP=$4
OUTPUT_DP=$5
FILTER_DP=$6
SPEC_WD=$7

case "$CAMPAIGN" in
  "$GRAPHGPS"/results_specmose_newch_*) ;;
  *) echo "Unexpected campaign path: $CAMPAIGN" >&2; exit 64 ;;
esac
if [[ -e "$CAMPAIGN" ]]; then
  echo "Refusing to reuse campaign: $CAMPAIGN" >&2
  exit 73
fi
if [[ ! -f "$CFG" || ! -f "$CACHE_DIR/metadata.json" ]]; then
  echo "Missing configuration or validated cache" >&2
  exit 66
fi

SOURCE_FILES=(
  GraphGPS/configs/ZINC/With_Edge_Features/GINe-VN/+specmose-newch-k5-h110-f10-t21.yaml
  GraphGPS/graphgps/config/specmose_edge_config.py
  GraphGPS/graphgps/encoder/specmose_edge_encoder.py
  GraphGPS/graphgps/encoder/type_dict_encoder.py
  GraphGPS/graphgps/loader/master_loader.py
  GraphGPS/graphgps/loader/spectral_edge.py
  GraphGPS/graphgps/optimizer/extra_optimizers.py
  GraphGPS/graphgps/transform/specmose_virtual_node.py
  GraphGPS/main.py
  GraphGPS/unittests/test_specmose_regularization.py
  GraphGPS/unittests/test_specmose_virtual_node.py
  hombasis-gt/pact/pact/chebyshev_pair.py
  hombasis-gt/pact/precompute_zinc_two_root_chebyshev.py
  hombasis-gt/pact/tests/test_chebyshev_pair.py
  SPECMOSE_TUNING_NOTES.md
)

mkdir -p "$CAMPAIGN/logs" "$CAMPAIGN/provenance/source" \
  "$CAMPAIGN/runs" "$CAMPAIGN/state"
for relative_path in "${SOURCE_FILES[@]}"; do
  mkdir -p "$CAMPAIGN/provenance/source/$(dirname "$relative_path")"
  cp -- "$REPO/$relative_path" "$CAMPAIGN/provenance/source/$relative_path"
done
cp -- "$CACHE_DIR/metadata.json" "$CAMPAIGN/provenance/cache-metadata.json"
cp -- "$CACHE_DIR/cross_validation.json" \
  "$CAMPAIGN/provenance/cache-cross-validation.json"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

{
  date --iso-8601=seconds
  printf 'tag=%s\n' "$TAG"
  printf 'git_head=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
  printf 'python=%s\n' "$PY"
  printf 'physical_gpu=1\nlogical_accelerator=cuda:0\n'
  printf 'seeds=%s\n' "${SEEDS[*]}"
  printf 'zero_virtual_node_embedding=%s\n' "$ZERO_VN"
  printf 'hidden_dropout=%s\n' "$HIDDEN_DP"
  printf 'output_dropout=%s\n' "$OUTPUT_DP"
  printf 'filter_dropout=%s\n' "$FILTER_DP"
  printf 'specmose_weight_decay=%s\n' "$SPEC_WD"
  printf 'backbone_weight_decay=1e-5\n'
} >"$CAMPAIGN/provenance/run.txt"
git -C "$REPO" status --short >"$CAMPAIGN/provenance/git-status.txt"
git -C "$REPO" diff >"$CAMPAIGN/provenance/tracked-source.patch"
"$PY" -m pip freeze >"$CAMPAIGN/provenance/pip-freeze.txt"
nvidia-smi >"$CAMPAIGN/provenance/nvidia-smi-at-launch.txt"
(
  cd "$CAMPAIGN/provenance"
  find source -type f -print0 | sort -z | xargs -0 sha256sum
) >"$CAMPAIGN/provenance/source-sha256.txt"

declare -A PIDS
for seed in "${SEEDS[@]}"; do
  seed_out="$CAMPAIGN/runs/seed_${seed}"
  seed_log="$CAMPAIGN/logs/seed_${seed}.log"
  mkdir -p "$seed_out"
  (
    printf 'START tag=%s seed=%s time=%s\n' \
      "$TAG" "$seed" "$(date --iso-8601=seconds)"
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
      train.ckpt_clean True \
      specmose_edge.zero_virtual_node_embedding "$ZERO_VN" \
      specmose_edge.hidden_dropout "$HIDDEN_DP" \
      specmose_edge.output_dropout "$OUTPUT_DP" \
      specmose_edge.filter_dropout "$FILTER_DP" \
      specmose_edge.weight_decay "$SPEC_WD"
  ) >"$seed_log" 2>&1 &
  PIDS[$seed]=$!
  printf '%s\t%s\t%s\n' "$seed" "${PIDS[$seed]}" "$seed_log" \
    >>"$CAMPAIGN/state/pids.tsv"
done

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
