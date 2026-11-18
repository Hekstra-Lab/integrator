#!/bin/bash
#SBATCH -J pdz2_ml
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 0-08:00
#SBATCH -o /n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model/logs/ml_%x_%j.out
# Prior-free ML intensities through the reference reduction. FLOOR unset keeps the
# 8.71% negatives; FLOOR=0 clamps them, which isolates "removed the Wilson prior"
# from "allowed negative intensities" -- the two are otherwise confounded.
set -euo pipefail
ARM=${ARM:?arm}
FLOOR=${FLOOR:-none}
M=/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model
REF=/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_ref
CRLS=/n/lab_storage/hekstra_lab/people/aldama/micromamba/envs/crls/bin
T=$M/tree_$ARM
source /n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh
mkdir -p $T/scripts $T/logs
cp $REF/scripts/{expt_concat.py,custom_ccsym.py,careless-cdef-ohp-mlpw.sh,stats_for_production.sh,wdm.py} $T/scripts/
ln -sf $REF/5e1y_phases.mtz $T/5e1y_phases.mtz

echo "==== 1) write ML intensities ($ARM, floor=$FLOOR)"
EXTRA=""
[[ "$FLOOR" != "none" ]] && EXTRA="--floor $FLOOR"
micromamba run -n integrator-cuda-dev python $M/split_pdz2_ml.py --tree $T $EXTRA 2>&1 | tail -20

echo "==== 2) expt_concat"
cd $T && micromamba run -n integrator-cuda-dev python scripts/expt_concat.py 0.3 2>&1 | tail -3
[[ -s $T/gain_0,3_from_stills/ld_0,3/cdef_e35_off_P1.mtz ]] || { echo "CONCAT_INCOMPLETE"; exit 1; }

echo "==== 3) careless"
cd $T && bash scripts/careless-cdef-ohp-mlpw.sh 0.3 2>&1 | grep -viE "WARNING|tensorflow|deprecat" | tail -4

echo "==== 4) CCsym"
cd $T && bash scripts/stats_for_production.sh 0.3 2>&1 | tail -2
ARM=$ARM $CRLS/python - <<'PY'
import collections, csv, os
arm = os.environ["ARM"]
p = (f"/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model/tree_{arm}"
     "/gain_0,3_from_stills/careless-cdef-0,3-ohp-mlpw/e35_ccsym.csv")
agg = collections.defaultdict(list); bb = collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open(p)):
    try: v = float(r["CCsym"])
    except ValueError: continue
    agg[r["delay"]].append(v); bb[r["delay"]][int(r["bin"])].append(v)
for d in ("off","50ns","100ns","200ns"):
    if d in agg:
        per = " ".join("%+.3f" % (sum(v)/len(v)) for _, v in sorted(bb[d].items()))
        print("CCsym %-12s %-6s mean %+.3f | %s" % (arm, d, sum(agg[d])/len(agg[d]), per))
PY

echo "==== 5) difference map + peaks"
cd $T && micromamba run -n integrator-cuda-dev python scripts/wdm.py 0.3 2>&1 | tail -2
micromamba run -n integrator-cuda-dev python $M/wdm_peaks_compare.py \
  --arm "laue=$REF/gain_0,3_from_stills/wdm_rescut.mtz" \
  --arm "$ARM=$T/gain_0,3_from_stills/wdm_rescut.mtz" 2>&1 | tail -8
echo "ML_ARM_DONE_$ARM"
