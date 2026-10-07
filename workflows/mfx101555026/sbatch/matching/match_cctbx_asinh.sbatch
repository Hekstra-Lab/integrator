#!/bin/bash
#SBATCH --job-name=match_asinh
#SBATCH --account=lcls:prjlumine22
#SBATCH --partition=milano
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/match_asinh_%j.out
#SBATCH --error=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/match_asinh_%j.err

source /sdf/group/lcls/ds/tools/cctbx/psana2_setup.sh

dials.python - <<'PY'
from pathlib import Path
import json
import os
import shutil

BASE = Path("/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx")

ORIG = BASE / "all_runs_269_289_no275_024_rg101_integrated"
WRITEBACK = BASE / "runs/run_20260805-011728_9704/predictions/mfx_refl_writeback"
SCALED = BASE / "PREDICTED_scale_all_r269_r289_asinh_dmin2"
MATCHED = BASE / "matched_cctbx_asinh"

def read_ids(path):
    with path.open() as fh:
        data = json.load(fh)

    return [
        exp["identifier"]
        for exp in data.get("experiment", [])
        if exp.get("identifier")
    ]

# 1. UUIDs that survived ASINH scaling/filtering
survivors = set()

scaled_files = list(SCALED.glob("*.expt"))

for f in scaled_files:
    survivors.update(read_ids(f))

print("Scaled chunk files:", len(scaled_files))
print("Surviving ASINH experiments:", len(survivors))

# 2. Map survivor UUID -> write-back filename
mapping = {}
writeback_files = list(WRITEBACK.glob("*.expt"))

for i, f in enumerate(writeback_files, 1):

    for uid in read_ids(f):
        if uid in survivors:
            mapping[uid] = f.name

    if i % 10000 == 0:
        print(
            f"Scanned {i}/{len(writeback_files)} "
            f"write-back files; mapped {len(mapping)}"
        )

unmapped = survivors - set(mapping)

print("Mapped UUIDs:", len(mapping))
print("Unmapped UUIDs:", len(unmapped))

if unmapped:
    raise RuntimeError("Some survivor UUIDs could not be mapped")

# 3. Rebuild matched cctbx directory
if MATCHED.exists():
    shutil.rmtree(MATCHED)

MATCHED.mkdir(parents=True)

linked = 0
missing = 0

for uid, expt_name in mapping.items():

    refl_name = expt_name[:-5] + ".refl"

    src_expt = ORIG / expt_name
    src_refl = ORIG / refl_name

    if src_expt.is_file() and src_refl.is_file():

        os.symlink(src_expt, MATCHED / expt_name)
        os.symlink(src_refl, MATCHED / refl_name)

        linked += 1

    else:
        missing += 1

print()
print("Matched folder:", MATCHED)
print("Linked cctbx pairs:", linked)
print("Missing original pairs:", missing)
print("Total symlinks:", linked * 2)

print("DONE ASINH")
PY