#this script uses careless to compute statistics of the resultant mtz files. This requires a conda environment with careless in it. Here, the conda environment is tf2.12_cuda11 and the conda environment activation is found in line 13. 

gain=${1}
gain_str=$(echo "$gain" | tr . ,)
expt=e35
tail=ohp-mlpw
merged_mtz_prefix=gain_${gain_str}_from_stills/
out_dir_prefix=./
careless_run=careless-cdef-${gain_str}-${tail}
merged_mtz_dir=${merged_mtz_prefix}${careless_run}/
out_dir=${merged_mtz_dir}/

source /n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh
PY="micromamba run -n crls python"
CCPRED="$PY /n/lab_storage/hekstra_lab/people/aldama/micromamba/envs/crls/bin/careless.ccpred"

cd $merged_mtz_dir

#careless.cchalf merged_${expt}_xval_*.mtz -o ${expt}_cchalf.csv --image ${expt}_cchalf.png

$CCPRED merged_${expt}_predictions_*.mtz -o ${expt}_ccpred.csv --image ${expt}_ccpred.png -m spearman

$PY ../../scripts/custom_ccsym.py merged_${expt}_xval_[4-7].mtz --spacegroup=5 --op=1 -p ${expt}_ccsym -b 10 --cb_op=a+b,a-b,-c  # -m weighted_pearson

#python ../../scripts/custom_ccsym.py merged_${expt}_xval_[4-7].mtz --spacegroup=5 --op=1 -p ${expt}_ccsym_b -b 1 --cb_op=a+b,a-b,-c  # -m weighted_pearson

#python ../../scripts/custom_ccsym.py merged_${expt}_xval_[4-7].mtz --spacegroup=5 --op=1 -p ${expt}_ccsym -b 10 --cb_op=x,y,z  # -m weighted_pearson

# python ../../custom_ccsym-bulk.py ${out_dir}merged_${expt}_xval_[4-7].mtz --spacegroup=5 --op=1 -p ${out_dir}/${expt}_ccsym --cb_op=a+b,a-b,-c
