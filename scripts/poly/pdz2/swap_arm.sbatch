#!/bin/bash
#SBATCH -J pdz2_swap
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 0-06:00
#SBATCH -o /n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model/logs/swap_%x_%j.out
# One sigma-swap cell through the REFERENCE reduction: expt_concat -> careless -> CCsym.
# Only the I and SIGI columns differ between cells; everything else is byte-identical.
set -euo pipefail
COMBO=${COMBO:?combo}
M=/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model
REF=/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_ref
CRLS=/n/lab_storage/hekstra_lab/people/aldama/micromamba/envs/crls/bin
TREE=$M/swap_$COMBO
source /n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh
mkdir -p $TREE/scripts $TREE/logs
cp $REF/scripts/{expt_concat.py,custom_ccsym.py,careless-cdef-ohp-mlpw.sh,stats_for_production.sh,wdm.py} $TREE/scripts/
ln -sf $REF/5e1y_phases.mtz $TREE/5e1y_phases.mtz

echo "==== 1) expt_concat ($COMBO)"
cd $TREE && micromamba run -n integrator-cuda-dev python scripts/expt_concat.py 0.3 2>&1 | tail -4
[[ -s $TREE/gain_0,3_from_stills/ld_0,3/cdef_e35_off_P1.mtz ]] || { echo "CONCAT_INCOMPLETE"; exit 1; }

echo "==== 2) careless ($COMBO)"
cd $TREE && bash scripts/careless-cdef-ohp-mlpw.sh 0.3 2>&1 | grep -viE "WARNING|tensorflow|deprecat" | tail -4

echo "==== 3) CCsym ($COMBO)"
cd $TREE && bash scripts/stats_for_production.sh 0.3 2>&1 | tail -3

echo "==== RESULT ($COMBO)"
COMBO=$COMBO $CRLS/python - <<'PY'
import csv, collections, os
combo = os.environ["COMBO"]
p = f"/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model/swap_{combo}/gain_0,3_from_stills/careless-cdef-0,3-ohp-mlpw/e35_ccsym.csv"
rows = list(csv.DictReader(open(p)))
agg = collections.defaultdict(list)
bb = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    try: v = float(r["CCsym"])
    except: continue
    agg[r["delay"]].append(v); bb[r["delay"]][int(r["bin"])].append(v)
for d in ["off", "50ns", "100ns", "200ns"]:
    if d in agg:
        per = " ".join("%+.3f" % (sum(v)/len(v)) for b, v in sorted(bb[d].items()))
        print("CCsym %-6s %-4s mean %+.3f | %s" % (combo, d, sum(agg[d])/len(agg[d]), per))
PY
echo "==== 4) difference map ($COMBO)"
cd $TREE && micromamba run -n integrator-cuda-dev python scripts/wdm.py 0.3 2>&1 | tail -3
echo "SWAP_DONE_$COMBO"
