#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/Xiaohan/Codes/SpecPE/MoSE_spec
GRAPHGPS="$REPO/GraphGPS"
RUNNER="$GRAPHGPS/run/run_specmose_gine_variant_cuda1.sh"
PREDECESSOR_UNIT=mose-specmose-k6-h84-f24-t42-cuda1-20260821.service
PREDECESSOR_STATUS="$GRAPHGPS/results_specmose_gine_k6_h84_f24_t42_20260821/state/exit-status.tsv"
SEQUENCE_STATE="$GRAPHGPS/specmose_followup_queue_20260821.tsv"

CONFIGS=(
  +specmose-h100-f20-t28-p10.yaml
  +specmose-h100-f20-t28-p20.yaml
  +specmose-h110-f10-t21-p20.yaml
  +specmose-h90-f20-t42-p20.yaml
  +specmose-k6-h84-f24-t42-p20.yaml
)
CAMPAIGNS=(
  "$GRAPHGPS/results_specmose_gine_h100_f20_t28_p10_20260821"
  "$GRAPHGPS/results_specmose_gine_h100_f20_t28_p20_20260821"
  "$GRAPHGPS/results_specmose_gine_h110_f10_t21_p20_20260821"
  "$GRAPHGPS/results_specmose_gine_h90_f20_t42_p20_20260821"
  "$GRAPHGPS/results_specmose_gine_k6_h84_f24_t42_p20_20260821"
)

printf 'waiting\t%s\t%s\n' "$PREDECESSOR_UNIT" \
  "$(date --iso-8601=seconds)" >"$SEQUENCE_STATE"
while systemctl --user is-active --quiet "$PREDECESSOR_UNIT"; do
  sleep 15
done

if [[ ! -f "$PREDECESSOR_STATUS" ]]; then
  echo "Missing predecessor exit-status file: $PREDECESSOR_STATUS" >&2
  exit 75
fi
predecessor_total=$(wc -l <"$PREDECESSOR_STATUS")
predecessor_successes=$(awk -F '\t' '$2 == 0 { count += 1 } END { print count + 0 }' \
  "$PREDECESSOR_STATUS")
if [[ "$predecessor_total" -ne 4 || "$predecessor_successes" -ne 4 ]]; then
  echo "Predecessor did not finish all four seeds successfully" >&2
  exit 76
fi

for index in "${!CONFIGS[@]}"; do
  config=${CONFIGS[$index]}
  campaign=${CAMPAIGNS[$index]}
  printf 'running\t%s\t%s\t%s\n' "$config" "$campaign" \
    "$(date --iso-8601=seconds)" >>"$SEQUENCE_STATE"
  "$RUNNER" "$config" "$campaign"
  printf 'completed\t%s\t%s\t%s\n' "$config" "$campaign" \
    "$(date --iso-8601=seconds)" >>"$SEQUENCE_STATE"
done

printf 'queue-completed\t%s\n' "$(date --iso-8601=seconds)" \
  >>"$SEQUENCE_STATE"
