"""Prepare PDZ2 off-state merges for refinement against the deposited model.

The deposited Laue-DIALS pipeline stops at CCsym and the difference map -- the authors
say so themselves in one_pass-from_off.sh ("This is where laue_dials ends ... can be
merged in careless and refined in phenix to get a model"). So no PDZ2 arm, theirs or
ours, has ever been checked against a physical model. That matters because every metric
the dataset HAS been judged on is a half-dataset correlation, which rewards any error the
two halves share.

This adds only the step they left to the user, keeping their conventions: the off state
(full C2 symmetry, matching the deposited model's space group) in the shared-index merge,
so no change of basis is needed at all.

One R-free set is generated once and applied to every arm by Miller index, so the arms are
compared on the SAME held-out reflections. Generating flags per run -- as the 845
comparison did -- leaves the arms unpaired and wastes most of the sensitivity.

The model cell is reset to the data cell (they differ by ~0.5%, the usual isomorphous
mismatch) so phenix refines against the measured geometry rather than refusing the pair.
"""

import argparse
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import reciprocalspaceship as rs

ap = argparse.ArgumentParser()
ap.add_argument("--arm", action="append", required=True, help="label=/path/to/arm_root")
ap.add_argument("--out", required=True)
ap.add_argument("--pdb", default="/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_ref/5e1y.pdb")
ap.add_argument("--merge-file", default="merged_e35_0.mtz",
                help="0 = off state in the shared-index (C2) setting")
ap.add_argument("--free-fraction", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--regenerate-flags", action="store_true",
                help="draw a NEW free set, unpairing these arms from "
                     "any refinement already on disk")
a = ap.parse_args()

CAR = "gain_0,3_from_stills/careless-cdef-0,3-ohp-mlpw"
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

arms = {}
for spec in a.arm:
    label, root = spec.split("=", 1)
    p = Path(root) / CAR / a.merge_file
    if not p.exists():
        print(f"{label}: missing {p}")
        continue
    arms[label] = rs.read_mtz(str(p))
    print(f"{label}: {len(arms[label])} refl, {arms[label].spacegroup.short_name()}")

if not arms:
    raise SystemExit("no arms found")

# One free set over the union of Miller indices, so every arm is scored on the
# same held-out reflections wherever it has them.
#
# REUSED when it already exists, and that is the point of the file rather
# than a convenience. The flags are drawn over the union of the arms PASSED
# TO THIS RUN, so adding an arm later and regenerating gives a different
# union, a different draw, and refinements that are no longer paired with
# the ones already on disk -- silently, since every file still looks right.
# `--regenerate-flags` forces a new set, which is correct only when every
# arm is being refined again from scratch.
flags_path = out / "shared_free_flags.csv"
if flags_path.exists() and not a.regenerate_flags:
    union = pd.read_csv(flags_path)
    print(f"\nreusing the free set in {flags_path.name}: {len(union)} hkl, "
          f"{union['FreeR_flag'].mean()*100:.2f}% free")
    new_hkl = None
    for ds in arms.values():
        idx = ds.reset_index()[["H", "K", "L"]]
        new_hkl = idx if new_hkl is None else pd.concat([new_hkl, idx])
    new_hkl = new_hkl.drop_duplicates()
    merged_chk = new_hkl.merge(union, on=["H", "K", "L"], how="left")
    missing = int(merged_chk["FreeR_flag"].isna().sum())
    if missing:
        print(f"  {missing} hkl in these arms are not in the stored set and "
              f"will be treated as working reflections")
else:
    union = None
    for ds in arms.values():
        idx = ds.reset_index()[["H", "K", "L"]]
        union = idx if union is None else pd.concat([union, idx])
    union = union.drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(a.seed)
    union["FreeR_flag"] = (rng.random(len(union)) < a.free_fraction).astype(np.int32)
    print(f"\nshared free set: {len(union)} unique hkl, "
          f"{union['FreeR_flag'].mean()*100:.2f}% free")
    union.to_csv(flags_path, index=False)

# the deposited model, re-celled to the data so phenix accepts the pair
st = gemmi.read_structure(a.pdb)
ref_cell = next(iter(arms.values())).cell
print(f"model cell {st.cell} -> data cell {ref_cell}")
st.cell = ref_cell
st.setup_entities()
model_out = out / "model_recelled.pdb"
st.write_pdb(str(model_out))

for label, ds in arms.items():
    d = ds.reset_index()
    merged = pd.merge(d, union, on=["H", "K", "L"], how="left")
    merged["FreeR_flag"] = merged["FreeR_flag"].fillna(0).astype(np.int32)
    keep = ["H", "K", "L", "F", "SigF", "FreeR_flag"]
    new = rs.DataSet(merged[keep], cell=ds.cell, spacegroup=ds.spacegroup)
    new = new.set_index(["H", "K", "L"])
    new["FreeR_flag"] = new["FreeR_flag"].astype("MTZInt")
    new.infer_mtz_dtypes(inplace=True)
    f = out / f"refine_in_{label}.mtz"
    new.write_mtz(str(f))
    nfree = int(new["FreeR_flag"].sum())
    print(f"  {label:14s} -> {f.name}  n={len(new)}  free={nfree} ({nfree/len(new)*100:.1f}%)")

print("\nPREP_DONE")
