#!/bin/bash
# Simulate Laue rows of a grid with datsimx: noiseless truth, a noisy copy, a truth table, a manifest.
#
# Each data row of GRID (tab-separated, header line, '#' comments) gives
#   name run mos_spread en_steps mos_doms xtal_size numimg test_shot [extra]
# where the optional `extra` column holds further datsimx arguments ('-' or absent = none).
# and produces, under ROOT/<name>/:
#   truth/    datsimx --noNoise --waveImg --writeMtz output (noiseless Bragg in sim_image,
#             background_<run>.h5, per-pixel h/k/l/wavelength maps, ground_truth.mtz),
#             truth_reflections.csv from truth_table.py
#   noisy/    master, geometry and background copied; shots files carry sim_image only
#             (no h/k/l/wavelength maps), then NOISE_TOOL writes Bragg + background + noise:
#             braggaddnoise (default; numpy, no PSF) or nanobragg (add_noise_nanobragg.py:
#             nanoBragg's own add_noise with PSF and ADC offset, as the laue-dials paper sims)
#   manifest.json  every parameter, the commits of datsimx / easyBragg / integrator,
#             package versions, GPU, wall time per stage, disk per directory
# A row whose truth/ already has truth_reflections.csv is skipped, so a resubmission resumes.
#
#   GRID=scripts/sim_truth/grids/sampling_check.tsv sbatch scripts/sim_truth/simulate.sbatch
#   GRID=... sbatch --array=0-2 scripts/sim_truth/simulate.sbatch     # one row per task
#   GRID=... ROWS=m005_e100_d050 sbatch scripts/sim_truth/simulate.sbatch
# Submit from the integrator checkout ($S/integrator_main), so the commit recorded is what ran.
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -J sim_truth
#SBATCH -o sim_truth_%A_%a.out
set -euo pipefail

MAMBA_SH="${MAMBA_SH:-/n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh}"
ENV_SIM="${ENV_SIM:-simtbx}"
DATSIMX="${DATSIMX:-/n/holylabs/hekstra_lab/Users/laldama/datsimx}"
EASYBRAGG="${EASYBRAGG:-/n/holylabs/hekstra_lab/Users/laldama/easyBragg}"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
GRID="${GRID:?set GRID to a grid .tsv}"
[[ "$GRID" = /* ]] || GRID="$REPO/$GRID"
ROOT="${ROOT:-/n/netscratch/hekstra_lab/Lab/laldama/sims/$(basename "$GRID" .tsv)}"
ROWS="${ROWS:-}"                      # comma-separated names; empty = all rows (or the array task's row)
SPEC="${SPEC:-$DATSIMX/datsimx/from_vukica.lam}"
NABC="${NABC:-100 100 100}"
DEG_PER_IMG="${DEG_PER_IMG:-1}"
NOISE_GAIN="${NOISE_GAIN:-0.7}"       # laue-dials paper sims: gain 0.7, readout 3 ADU, calibration 3 %
NOISE_READOUT="${NOISE_READOUT:-3}"
NOISE_CALIB="${NOISE_CALIB:-0.03}"
NOISE_TOOL="${NOISE_TOOL:-braggaddnoise}"  # braggaddnoise | nanobragg
NOISE_PSF_MM="${NOISE_PSF_MM:-0.1}"     # nanobragg only; laue-dials paper 0.1 mm on 88.54 um pixels
NOISE_ADC="${NOISE_ADC:-10}"            # nanobragg only; laue-dials paper 10 ADU
NJ="${SLURM_CPUS_PER_TASK:-4}"

source "$MAMBA_SH"
micromamba activate "$ENV_SIM"
export HDF5_USE_FILE_LOCKING=FALSE
HERE="$REPO/scripts/sim_truth"

mapfile -t DATA < <(grep -v '^#' "$GRID" | tail -n +2 | grep -v '^[[:space:]]*$')
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  DATA=("${DATA[$SLURM_ARRAY_TASK_ID]}")
elif [[ -n "$ROWS" ]]; then
  sel=(); for r in "${DATA[@]}"; do [[ ",$ROWS," == *",$(cut -f1 <<<"$r"),"* ]] && sel+=("$r"); done; DATA=("${sel[@]}")
fi
echo "grid $GRID -> $ROOT; ${#DATA[@]} row(s); host $(hostname); $(nvidia-smi -L | head -1)"

for row in "${DATA[@]}"; do
  IFS=$'\t' read -r NAME RUN MOS EN DOMS XTAL NUMIMG TEST EXTRA <<<"$row"
  [[ -z "${EXTRA:-}" || "$EXTRA" == "-" ]] && EXTRA=""
  D="$ROOT/$NAME"; T="$D/truth"; N="$D/noisy"
  if [[ -s "$T/truth_reflections.csv" ]]; then echo "== $NAME: done, skipped"; continue; fi
  rm -rf "$T" "$N"; mkdir -p "$T" "$N"
  TOTALDEG=$(python -c "print($NUMIMG * $DEG_PER_IMG)")
  test_flag=(); [[ "$TEST" == 1 ]] && test_flag=(--testShot)
  cmd=(datsimx.sim "$T" --noNoise --waveImg --writeMtz --static --cuda --specFile "$SPEC"
       --Nabc $NABC --run "$RUN" --mosSpread "$MOS" --mosDoms "$DOMS" --enSteps "$EN"
       --xtalSize "$XTAL" --numimg "$NUMIMG" --totalDeg "$TOTALDEG" "${test_flag[@]}" $EXTRA)
  echo "== $NAME: ${cmd[*]}"
  t0=$(date +%s); "${cmd[@]}" > "$D/datsimx.log" 2>&1; t1=$(date +%s)

  python - "$T" "$N" "$RUN" <<'EOF'
import glob, os, shutil, sys
import h5py
src, dst, run = sys.argv[1], sys.argv[2], sys.argv[3]
for f in (f"run_{run}_master.h5", f"geom_run{run}.expt", f"background_{run}.h5", f"commandline_run{run}.txt"):
    shutil.copy2(os.path.join(src, f), dst)
for path in glob.glob(os.path.join(src, f"shots_{run}_*.h5")):
    with h5py.File(path, "r") as fi, h5py.File(os.path.join(dst, os.path.basename(path)), "w") as fo:
        for g in ("sim_image", "Umat", "Bmat", "delta_phi", "mos_spread"):
            fi.copy(g, fo)
EOF
  if [[ "$NOISE_TOOL" == nanobragg ]]; then
    calib_pct=$(python -c "print(100 * $NOISE_CALIB)")
    python "$HERE/add_noise_nanobragg.py" "$N" --run "$RUN" --gain "$NOISE_GAIN" --adc "$NOISE_ADC" \
      --readout "$NOISE_READOUT" --calib-pct "$calib_pct" --psf-mm "$NOISE_PSF_MM" > "$D/add_noise.log" 2>&1
  else
    braggaddnoise --dirname "$N" --run "$RUN" --nj "$NJ" \
      --gain "$NOISE_GAIN" --readout "$NOISE_READOUT" --calib "$NOISE_CALIB" > "$D/braggaddnoise.log" 2>&1
  fi
  t2=$(date +%s)
  python "$HERE/truth_table.py" "$T" --run "$RUN" | tee "$D/truth_table.log"
  t3=$(date +%s)

  python - "$D" <<EOF
import json, platform, subprocess, sys
from importlib.metadata import version
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
git = lambda p: dict(commit=sh(f"git -C {p} rev-parse HEAD"),
                     dirty_tracked=bool(sh(f"git -C {p} status --porcelain --untracked-files=no")))
m = dict(
    name="$NAME", grid="$GRID", run=$RUN,
    datsimx_command=open("$T/commandline_run$RUN.txt").read().strip(),
    params=dict(mos_spread=$MOS, en_steps=$EN, mos_doms=$DOMS, xtal_size=$XTAL, numimg=$NUMIMG,
                total_deg=$TOTALDEG, test_shot=bool($TEST), nabc="$NABC", spec_file="$SPEC"),
    extra_datsimx_args="$EXTRA",
    noise=dict(tool="$NOISE_TOOL", gain=$NOISE_GAIN, readout=$NOISE_READOUT, calib=$NOISE_CALIB,
               psf_mm=($NOISE_PSF_MM if "$NOISE_TOOL" == "nanobragg" else None),
               adc=($NOISE_ADC if "$NOISE_TOOL" == "nanobragg" else 0),
               seeds=("100000*run + shot; calibration 100000*run" if "$NOISE_TOOL" == "nanobragg"
                      else "shot number (braggaddnoise default)")),
    code=dict(datsimx=git("$DATSIMX"), easyBragg=git("$EASYBRAGG"), integrator=git("$REPO"),
              nanoBragg_noise=version("nanoBragg_noise")),
    host=platform.node(), gpu=sh("nvidia-smi -L | head -1"), slurm_job="${SLURM_JOB_ID:-}",
    seconds=dict(datsimx=$t1 - $t0, noise=$t2 - $t1, truth_table=$t3 - $t2),
    disk=dict(truth=sh("du -sh $T | cut -f1"), noisy=sh("du -sh $N | cut -f1")),
)
json.dump(m, open("$D/manifest.json", "w"), indent=1)
print(json.dumps(m["seconds"]), m["disk"])
EOF
done
echo "all rows done"
