#!/usr/bin/env bash
set -Eeuo pipefail

RANDOM_REPO=/home/Xiaohan/Codes/SpecPE/MoSE_spec
ONEHOT_REPO=/home/Xiaohan/Codes/SpecPE/MoSE_spec_onehot
GRAPHGPS="$RANDOM_REPO/GraphGPS"
RUNNER="$GRAPHGPS/run/run_specmose_gine_vn_matched_group_cuda0.sh"
QUEUE_STATE="$GRAPHGPS/specmose_gine_vn_matched_8groups_cuda0_20260822.tsv"

REPOSITORIES=(
  "$ONEHOT_REPO"
  "$ONEHOT_REPO"
  "$ONEHOT_REPO"
  "$ONEHOT_REPO"
  "$RANDOM_REPO"
  "$RANDOM_REPO"
  "$RANDOM_REPO"
  "$RANDOM_REPO"
)
INITIALIZATIONS=(
  onehot onehot onehot onehot
  random random random random
)
CONFIGS=(
  +specmose-h110-f10-t21.yaml
  +specmose-h90-f20-t42.yaml
  +specmose-h100-f20-t28.yaml
  +specmose-k6-h92-f27-t28.yaml
  +specmose-h110-f10-t21.yaml
  +specmose-h90-f20-t42.yaml
  +specmose-h100-f20-t28.yaml
  +specmose-k6-h92-f27-t28.yaml
)
TAGS=(
  k5_h110_f10_t21
  k5_h90_f20_t42
  k5_h100_f20_t28
  k6_h92_f27_t28
  k5_h110_f10_t21
  k5_h90_f20_t42
  k5_h100_f20_t28
  k6_h92_f27_t28
)

if [[ -e "$QUEUE_STATE" ]]; then
  echo "Refusing to overwrite queue state: $QUEUE_STATE" >&2
  exit 73
fi

for index in "${!CONFIGS[@]}"; do
  repo=${REPOSITORIES[$index]}
  initialization=${INITIALIZATIONS[$index]}
  config=${CONFIGS[$index]}
  tag=${TAGS[$index]}
  cfg="$repo/GraphGPS/configs/ZINC/With_Edge_Features/GINe-VN/$config"
  campaign="$repo/GraphGPS/results_specmose_gine_vn_matched_${initialization}_${tag}_20260822"
  if [[ ! -f "$cfg" ]]; then
    echo "Missing queue configuration: $cfg" >&2
    exit 66
  fi
  if [[ -e "$campaign" ]]; then
    echo "Refusing to reuse campaign: $campaign" >&2
    exit 73
  fi
done

printf 'queue-started\t%s\n' "$(date --iso-8601=seconds)" >"$QUEUE_STATE"

for index in "${!CONFIGS[@]}"; do
  group=$((index + 1))
  repo=${REPOSITORIES[$index]}
  initialization=${INITIALIZATIONS[$index]}
  config=${CONFIGS[$index]}
  tag=${TAGS[$index]}
  campaign="$repo/GraphGPS/results_specmose_gine_vn_matched_${initialization}_${tag}_20260822"

  printf 'running\t%02d/08\t%s\t%s\t%s\t%s\n' \
    "$group" "$initialization" "$tag" "$config" \
    "$(date --iso-8601=seconds)" >>"$QUEUE_STATE"

  if /usr/bin/bash "$RUNNER" "$repo" "$config" "$campaign"; then
    printf 'completed\t%02d/08\t%s\t%s\t%s\n' \
      "$group" "$initialization" "$tag" \
      "$(date --iso-8601=seconds)" >>"$QUEUE_STATE"
  else
    status=$?
    printf 'failed\t%02d/08\t%s\t%s\tstatus=%s\t%s\n' \
      "$group" "$initialization" "$tag" "$status" \
      "$(date --iso-8601=seconds)" >>"$QUEUE_STATE"
    exit "$status"
  fi
done

printf 'queue-completed\t%s\n' "$(date --iso-8601=seconds)" \
  >>"$QUEUE_STATE"
