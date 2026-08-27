#!/usr/bin/env bash
set -Eeuo pipefail

RANDOM_REPO=/home/Xiaohan/Codes/SpecPE/MoSE_spec
ONEHOT_REPO=/home/Xiaohan/Codes/SpecPE/MoSE_spec_onehot
RANDOM_GRAPHGPS="$RANDOM_REPO/GraphGPS"
ONEHOT_GRAPHGPS="$ONEHOT_REPO/GraphGPS"
RANDOM_RUNNER="$RANDOM_GRAPHGPS/run/run_specmose_gine_vn_variant_cuda0.sh"
ONEHOT_RUNNER="$ONEHOT_GRAPHGPS/run/run_specmose_gine_vn_variant_cuda0.sh"
QUEUE_STATE="$RANDOM_GRAPHGPS/specmose_gine_vn_16groups_cuda0_20260822.tsv"

ARCHITECTURES=(
  k5_h110_f10_t21
  k5_h90_f20_t42
  k5_h100_f20_t28
  k6_h92_f27_t28
)
CONFIGS=(
  +specmose.yaml
  +specmose-h90-f20-t42.yaml
  +specmose-h100-f20-t28-p10.yaml
  +specmose-k6-h84-f24-t42.yaml
)
MAX_TOTAL_DEGREES=(5 5 5 6)
NUM_BASES=(21 21 21 28)
NUM_FILTERS=(10 20 20 27)
TEMPLATE_DIMS=(21 42 28 28)
HIDDEN_DIMS=(110 90 100 92)
PATIENCES=(10 20)
INITIALIZATIONS=(random onehot)

if [[ -e "$QUEUE_STATE" ]]; then
  echo "Refusing to overwrite queue state: $QUEUE_STATE" >&2
  exit 73
fi

for arch_index in "${!ARCHITECTURES[@]}"; do
  architecture=${ARCHITECTURES[$arch_index]}
  for initialization in "${INITIALIZATIONS[@]}"; do
    if [[ "$initialization" == random ]]; then
      graphgps=$RANDOM_GRAPHGPS
    else
      graphgps=$ONEHOT_GRAPHGPS
    fi
    for patience in "${PATIENCES[@]}"; do
      campaign="$graphgps/results_specmose_gine_vn_${initialization}_${architecture}_p${patience}_20260822"
      if [[ -e "$campaign" ]]; then
        echo "Refusing to reuse campaign: $campaign" >&2
        exit 73
      fi
    done
  done
done

printf 'queue-started\t%s\n' "$(date --iso-8601=seconds)" >"$QUEUE_STATE"

group=0
for arch_index in "${!ARCHITECTURES[@]}"; do
  architecture=${ARCHITECTURES[$arch_index]}
  config=${CONFIGS[$arch_index]}
  max_total_degree=${MAX_TOTAL_DEGREES[$arch_index]}
  num_basis=${NUM_BASES[$arch_index]}
  num_filters=${NUM_FILTERS[$arch_index]}
  template_dim=${TEMPLATE_DIMS[$arch_index]}
  hidden_dim=${HIDDEN_DIMS[$arch_index]}

  for initialization in "${INITIALIZATIONS[@]}"; do
    if [[ "$initialization" == random ]]; then
      graphgps=$RANDOM_GRAPHGPS
      runner=$RANDOM_RUNNER
    else
      graphgps=$ONEHOT_GRAPHGPS
      runner=$ONEHOT_RUNNER
    fi

    for patience in "${PATIENCES[@]}"; do
      group=$((group + 1))
      campaign="$graphgps/results_specmose_gine_vn_${initialization}_${architecture}_p${patience}_20260822"
      printf 'running\t%02d/16\t%s\t%s\tpatience=%s\t%s\n' \
        "$group" "$architecture" "$initialization" "$patience" \
        "$(date --iso-8601=seconds)" >>"$QUEUE_STATE"

      "$runner" "$config" "$max_total_degree" "$num_basis" \
        "$num_filters" "$template_dim" "$hidden_dim" "$patience" \
        "$campaign"

      printf 'completed\t%02d/16\t%s\t%s\tpatience=%s\t%s\n' \
        "$group" "$architecture" "$initialization" "$patience" \
        "$(date --iso-8601=seconds)" >>"$QUEUE_STATE"
    done
  done
done

printf 'queue-completed\t%s\n' "$(date --iso-8601=seconds)" \
  >>"$QUEUE_STATE"
