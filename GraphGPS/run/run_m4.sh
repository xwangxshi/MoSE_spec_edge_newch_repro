#!/usr/bin/env bash
set -Eeuo pipefail

PY=/home/Xiaohan/anaconda3/envs/SPE/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAPHGPS="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$(cd "$GRAPHGPS/.." && pwd)"

CFG_NAME=+specmose-newch-k5-h116-f10-t21.yaml
CFG="$GRAPHGPS/configs/ZINC/With_Edge_Features/GINe-VN/$CFG_NAME"
DATASET_DIR="$GRAPHGPS/datasets"
CACHE_DIR="$REPO/hombasis-gt/hombasis-bench/data/zinc-data/zinc12k_two_root_mose_eq_cheby_k10_edge_v1"
SEEDS=(0 14 48 96)

# Usage: ./run/run_m4.sh [physical_gpu] [stamp]
PHYSICAL_GPU=${1:-1}
STAMP=${2:-$(date +%Y%m%d_%H%M%S)}
CAMPAIGN="$GRAPHGPS/results_specmose_newch_stage2_m4_h116_zero_hdp10_wd1e3_repro_$STAMP"

if [[ "$PHYSICAL_GPU" != 0 && "$PHYSICAL_GPU" != 1 ]]; then
  echo "physical_gpu must be 0 or 1" >&2
  exit 64
fi
if [[ -e "$CAMPAIGN" ]]; then
  echo "Refusing to reuse existing output: $CAMPAIGN" >&2
  exit 73
fi
if [[ ! -f "$CFG" || ! -f "$CACHE_DIR/metadata.json" ]]; then
  echo "Missing M4 config or preprocessing cache" >&2
  exit 66
fi

cd "$GRAPHGPS"
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 "$PY" \
  run/verify_specmose_model_params.py --cfg "$CFG" --expected 135229

mkdir -p "$CAMPAIGN"/{logs,runs,state}
printf 'physical_gpu=%s\nseeds=%s\nconfig=%s\n' \
  "$PHYSICAL_GPU" "${SEEDS[*]}" "$CFG_NAME" >"$CAMPAIGN/state/launch.txt"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

declare -A PIDS
for SEED in "${SEEDS[@]}"; do
  SEED_OUT="$CAMPAIGN/runs/seed_$SEED"
  SEED_LOG="$CAMPAIGN/logs/seed_$SEED.log"
  mkdir -p "$SEED_OUT"
  (
    echo "START seed=$SEED time=$(date --iso-8601=seconds)"
    exec "$PY" main.py \
      --cfg "$CFG" --repeat 1 \
      out_dir "$SEED_OUT" dataset.dir "$DATASET_DIR" \
      accelerator cuda:0 seed "$SEED" num_workers 0 \
      wandb.use False tensorboard_each_run False tensorboard_agg False \
      train.auto_resume False train.enable_ckpt True \
      train.ckpt_best False train.ckpt_clean True \
      specmose_edge.zero_virtual_node_embedding True \
      specmose_edge.hidden_dropout 0.1 \
      specmose_edge.output_dropout 0.0 \
      specmose_edge.filter_dropout 0.0 \
      specmose_edge.weight_decay 1e-3
  ) >"$SEED_LOG" 2>&1 &
  PIDS[$SEED]=$!
  printf '%s\t%s\n' "$SEED" "${PIDS[$SEED]}" >>"$CAMPAIGN/state/pids.tsv"
done

STATUS=0
for SEED in "${SEEDS[@]}"; do
  if wait "${PIDS[$SEED]}"; then
    CODE=0
  else
    CODE=$?
    STATUS=1
  fi
  printf '%s\t%s\n' "$SEED" "$CODE" >>"$CAMPAIGN/state/exit-status.tsv"
done

echo "M4 campaign: $CAMPAIGN"
exit "$STATUS"
