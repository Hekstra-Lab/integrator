#!/bin/bash
# The crystallographic metrics of each arm as one long-format CSV.
#
# `xtal_table.sh` prints the same numbers for reading in a terminal; this
# writes them for plotting, in the shape figtex reads:
#
#   method,dataset,estimator,metric,value
#
# with `estimator` either `rb` (the Rao-Blackwellized profile fit computed at
# export, from `i_rb`/`var_rb`) or `model` (the integrator's own posterior,
# from `qi_mean`/`qi_var`). Both rows are always written: one training gives
# two sets of intensities and sigmas, `dials.merge` weights by 1/sigma^2, and
# which one wins is not even monotone across arms, so a figure that shows one
# can invert the conclusion.
#
# `metric` is one of i_over_sigma, cc_anom, r_free_f, r_free_i, max_peak.
# Long format rather than a column per metric because the metrics have
# different units and are drawn in different panels.
#
# Usage: bash tidy_xtal.sh <out.csv> <arm> [arm ...]
S=/n/netscratch/hekstra_lab/Lab/laldama
OUT=$1; shift
DATASET=${DATASET:-hewl_9b7c}
[ -n "$OUT" ] || { echo "usage: tidy_xtal.sh <out.csv> <arm> [arm ...]"; exit 1; }

emit() {  # method estimator metric value
  [ -n "$4" ] && [ "$4" != "?" ] && printf '%s,%s,%s,%s,%s\n' "$1" "$DATASET" "$2" "$3" "$4"
}

{
  echo "method,dataset,estimator,metric,value"
  for A in "$@"; do
    for E in rb model; do
      L=$(ls -1t $S/dialsarm_${A}_${E}_[0-9]*.log 2>/dev/null | head -1)
      N="${A}_${E}"
      if [ -z "$L" ] && [ "$E" = rb ]; then
        L=$(ls -1t $S/dialsarm_${A}_[0-9]*.log 2>/dev/null | head -1)
        [ -z "$L" ] && L=$(ls -1t $S/dialsarm_${A//_/}_[0-9]*.log 2>/dev/null | head -1)
        N="$A"
      fi
      [ -n "$L" ] || continue
      emit "$A" "$E" i_over_sigma \
        "$(grep -aE '^I/sigma' "$L" | head -1 | awk '{print $2}')"
      emit "$A" "$E" cc_anom \
        "$(grep -a 'Anomalous correlation' "$L" | head -1 | awk '{print $3}')"
      emit "$A" "$E" r_free_f \
        "$(grep -a "RESULT ${N}_F" "$L" | tail -1 | sed 's/.*R-free = //')"
      emit "$A" "$E" r_free_i \
        "$(grep -a "RESULT ${N}_I" "$L" | tail -1 | sed 's/.*R-free = //')"
      emit "$A" "$E" max_peak \
        "$(grep -a "PEAKS ${N}_F" "$L" | tail -1 | awk '{print $3}' \
           | tr ',' '\n' | sort -g | tail -1)"
    done
  done
} > "$OUT"
echo "$OUT: $(( $(wc -l < "$OUT") - 1 )) rows"
