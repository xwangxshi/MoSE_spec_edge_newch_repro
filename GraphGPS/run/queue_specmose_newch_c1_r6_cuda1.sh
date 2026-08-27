#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/Xiaohan/Codes/SpecPE/MoSE_spec_edge_newch
GRAPHGPS="$REPO/GraphGPS"
RUNNER="$GRAPHGPS/run/run_specmose_newch_group_cuda1.sh"
STATE="$GRAPHGPS/specmose_newch_c1_r6_cuda1_20260825.tsv"
MANIFEST="$GRAPHGPS/specmose_newch_c1_r6_source_20260825.sha256"
PREDECESSOR=mose-spec-baseline-gine-vn-h120-h128-cuda1-20260825.service

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
  GraphGPS/run/queue_specmose_newch_c1_r6_cuda1.sh
  GraphGPS/run/run_specmose_newch_group_cuda1.sh
  hombasis-gt/pact/pact/chebyshev_pair.py
  hombasis-gt/pact/precompute_zinc_two_root_chebyshev.py
  hombasis-gt/hombasis-bench/data/zinc-data/zinc12k_two_root_mose_eq_cheby_k10_edge_v1/metadata.json
  hombasis-gt/hombasis-bench/data/zinc-data/zinc12k_two_root_mose_eq_cheby_k10_edge_v1/cross_validation.json
)

TAGS=(c1_legacyvn c2_zerovn r1_hdp10 r2_hdp20 r3_wd1e3 r4_output10 r5_filter10 r6_all10)
ZERO_VN=(False True True True True True True True)
HIDDEN_DP=(0.0 0.0 0.1 0.2 0.1 0.1 0.1 0.1)
OUTPUT_DP=(0.0 0.0 0.0 0.0 0.0 0.1 0.0 0.1)
FILTER_DP=(0.0 0.0 0.0 0.0 0.0 0.0 0.1 0.1)
SPEC_WD=(1e-5 1e-5 5e-4 5e-4 1e-3 5e-4 5e-4 5e-4)

if [[ -e "$STATE" || -e "$MANIFEST" ]]; then
  echo "Refusing to overwrite queue state or source manifest" >&2
  exit 73
fi
for tag in "${TAGS[@]}"; do
  campaign="$GRAPHGPS/results_specmose_newch_${tag}_20260825"
  if [[ -e "$campaign" ]]; then
    echo "Refusing to reuse campaign: $campaign" >&2
    exit 73
  fi
done

printf 'queue-created\t%s\n' "$(date --iso-8601=seconds)" >"$STATE"
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

verify_sources
while systemctl --user is-active --quiet "$PREDECESSOR"; do
  printf 'waiting-predecessor\t%s\t%s\n' \
    "$PREDECESSOR" "$(date --iso-8601=seconds)" >>"$STATE"
  sleep 30
done

for first in 0 2 4 6; do
  verify_sources
  second=$((first + 1))
  declare -A BATCH_PIDS
  for index in "$first" "$second"; do
    tag=${TAGS[$index]}
    campaign="$GRAPHGPS/results_specmose_newch_${tag}_20260825"
    printf 'starting\t%s\t%s\n' "$tag" "$(date --iso-8601=seconds)" >>"$STATE"
    "$RUNNER" "$tag" "$campaign" \
      "${ZERO_VN[$index]}" "${HIDDEN_DP[$index]}" \
      "${OUTPUT_DP[$index]}" "${FILTER_DP[$index]}" \
      "${SPEC_WD[$index]}" &
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
