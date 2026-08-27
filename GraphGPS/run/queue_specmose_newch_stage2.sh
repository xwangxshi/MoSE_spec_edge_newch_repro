#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/Xiaohan/Codes/SpecPE/MoSE_spec_edge_newch
GRAPHGPS="$REPO/GraphGPS"
RUNNER="$GRAPHGPS/run/run_specmose_newch_stage2_group.sh"
STAMP=${2:-20260826}
PREDECESSOR=${3:-}

if [[ "$#" -lt 1 || "$#" -gt 3 ]]; then
  echo "Usage: $0 PHYSICAL_GPU [STAMP] [PREDECESSOR_SERVICE]" >&2
  exit 64
fi
PHYSICAL_GPU=$1
case "$PHYSICAL_GPU" in
  0|1) ;;
  *) echo "PHYSICAL_GPU must be 0 or 1" >&2; exit 64 ;;
esac

STATE="$GRAPHGPS/specmose_newch_stage2_cuda${PHYSICAL_GPU}_${STAMP}.tsv"
MANIFEST="$GRAPHGPS/specmose_newch_stage2_source_cuda${PHYSICAL_GPU}_${STAMP}.sha256"
H110=+specmose-newch-k5-h110-f10-t21.yaml
H116=+specmose-newch-k5-h116-f10-t21.yaml

TAGS=(
  l1_h110_legacy_hdp10_wd5e4
  l2_h110_legacy_hdp10_wd1e3
  m1_h116_legacy_noreg
  m2_h116_zero_noreg
  m3_h116_legacy_hdp10_wd1e3
  m4_h116_zero_hdp10_wd1e3
)
CONFIGS=("$H110" "$H110" "$H116" "$H116" "$H116" "$H116")
ZERO_VN=(False False False True False True)
HIDDEN_DP=(0.1 0.1 0.0 0.0 0.1 0.1)
OUTPUT_DP=(0.0 0.0 0.0 0.0 0.0 0.0)
FILTER_DP=(0.0 0.0 0.0 0.0 0.0 0.0)
SPEC_WD=(5e-4 1e-3 1e-5 1e-5 1e-3 1e-3)
EXPECTED_PARAMS=(123163 123163 135229 135229 135229 135229)

SOURCE_FILES=(
  GraphGPS/configs/ZINC/With_Edge_Features/GINe-VN/+specmose-newch-k5-h110-f10-t21.yaml
  GraphGPS/configs/ZINC/With_Edge_Features/GINe-VN/+specmose-newch-k5-h116-f10-t21.yaml
  GraphGPS/graphgps/config/specmose_edge_config.py
  GraphGPS/graphgps/encoder/specmose_edge_encoder.py
  GraphGPS/graphgps/encoder/type_dict_encoder.py
  GraphGPS/graphgps/loader/master_loader.py
  GraphGPS/graphgps/loader/spectral_edge.py
  GraphGPS/graphgps/optimizer/extra_optimizers.py
  GraphGPS/graphgps/transform/specmose_virtual_node.py
  GraphGPS/main.py
  GraphGPS/run/queue_specmose_newch_stage2.sh
  GraphGPS/run/run_specmose_newch_stage2_group.sh
  GraphGPS/run/verify_specmose_model_params.py
  hombasis-gt/pact/pact/chebyshev_pair.py
  hombasis-gt/pact/precompute_zinc_two_root_chebyshev.py
  hombasis-gt/hombasis-bench/data/zinc-data/zinc12k_two_root_mose_eq_cheby_k10_edge_v1/metadata.json
  hombasis-gt/hombasis-bench/data/zinc-data/zinc12k_two_root_mose_eq_cheby_k10_edge_v1/cross_validation.json
  SPECMOSE_TUNING_NOTES.md
)

if [[ -e "$STATE" || -e "$MANIFEST" ]]; then
  echo "Refusing to overwrite queue state or source manifest" >&2
  exit 73
fi
for tag in "${TAGS[@]}"; do
  campaign="$GRAPHGPS/results_specmose_newch_stage2_${tag}_${STAMP}"
  if [[ -e "$campaign" ]]; then
    echo "Refusing to reuse campaign: $campaign" >&2
    exit 73
  fi
done

exec 9>"/tmp/specmose-newch-stage2-cuda${PHYSICAL_GPU}.lock"
if ! flock -n 9; then
  echo "Another stage-2 queue holds the CUDA${PHYSICAL_GPU} lock" >&2
  exit 75
fi

printf 'queue-created\tphysical-gpu=%s\t%s\n' \
  "$PHYSICAL_GPU" "$(date --iso-8601=seconds)" >"$STATE"
(
  cd "$REPO"
  sha256sum "${SOURCE_FILES[@]}"
) >"$MANIFEST"

verify_sources() {
  if ! (cd "$REPO" && sha256sum --check --quiet "$MANIFEST"); then
    printf 'source-drift\t%s\n' "$(date --iso-8601=seconds)" >>"$STATE"
    echo "Source drift detected; refusing to mix experiment versions" >&2
    return 1
  fi
}

while [[ -n "$PREDECESSOR" ]] \
    && systemctl --user is-active --quiet "$PREDECESSOR"; do
  printf 'waiting-predecessor\t%s\t%s\n' \
    "$PREDECESSOR" "$(date --iso-8601=seconds)" >>"$STATE"
  sleep 30
done

for first in 0 2 4; do
  verify_sources
  second=$((first + 1))
  declare -A BATCH_PIDS
  for index in "$first" "$second"; do
    tag=${TAGS[$index]}
    campaign="$GRAPHGPS/results_specmose_newch_stage2_${tag}_${STAMP}"
    printf 'starting\t%s\t%s\n' "$tag" "$(date --iso-8601=seconds)" >>"$STATE"
    "$RUNNER" "$tag" "$campaign" "${CONFIGS[$index]}" \
      "$PHYSICAL_GPU" "${ZERO_VN[$index]}" "${HIDDEN_DP[$index]}" \
      "${OUTPUT_DP[$index]}" "${FILTER_DP[$index]}" \
      "${SPEC_WD[$index]}" "${EXPECTED_PARAMS[$index]}" &
    BATCH_PIDS[$index]=$!
  done

  batch_status=0
  for index in "$first" "$second"; do
    tag=${TAGS[$index]}
    if wait "${BATCH_PIDS[$index]}"; then
      status=0
      state=completed
    else
      status=$?
      state=failed
      batch_status=1
    fi
    printf '%s\t%s\tstatus=%s\t%s\n' \
      "$state" "$tag" "$status" "$(date --iso-8601=seconds)" >>"$STATE"
  done
  if [[ "$batch_status" -ne 0 ]]; then
    exit 1
  fi
done
printf 'queue-completed\t%s\n' "$(date --iso-8601=seconds)" >>"$STATE"
