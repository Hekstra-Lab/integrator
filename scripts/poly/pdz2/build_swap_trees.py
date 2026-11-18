"""Build four PDZ2 trees that cross our intensities with our sigmas.

The CCsym gap between laue.integrate (0.253 at 200 ns) and the model (0.216) has
been read as an integration-quality gap, but careless weights observations by sigma,
and this model's sigma is a documented weak point elsewhere: it decorrelates from I
in outer shells on SBGrid data, and the HEWL/1118 analysis concluded the weak-data
defect was variance rather than profile. So the gap may live entirely in SIGI.

Four trees, identical in every other column, on the reflections both arms share:

    LL   I from laue    SIGI from laue     (reference, on the matched set)
    ML   I from model   SIGI from laue     (does our I help when weights are theirs?)
    LM   I from laue    SIGI from model    (does our SIGI hurt good intensities?)
    MM   I from model   SIGI from model    (the model arm, on the matched set)

LL and MM also serve as the matched-reflection-set control: the arms differ by
937,365 vs 993,143 reflections overall, so comparing these to the full-set numbers
separates dilution by extra weak reflections from integration quality.

Geometry (BATCH, xcal, ycal, wavelength, PARTIAL) and the background columns are
taken from the reference in all four, so only I and SIGI ever vary. careless uses
xcal, ycal, wavelength, BATCH and dHKL as metadata and ignores BG/SIGBG.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import reciprocalspaceship as rs

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_ref")
ap.add_argument("--model", default="/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model/tree_w21_k7m")
ap.add_argument("--out", default="/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model")
a = ap.parse_args()

REF = Path(a.ref) / "gain_0,3_from_stills"
MOD = Path(a.model) / "gain_0,3_from_stills"
OUT = Path(a.out)
SWEEPS = [f"{p}_{t}" for p in "cdef" for t in ("off", "50ns", "100ns", "200ns")]
COMBOS = {"LL": ("ref", "ref"), "ML": ("mod", "ref"), "LM": ("ref", "mod"), "MM": ("mod", "mod")}
# Position, NOT BATCH. Our BATCH is a dense re-index of the frames we extracted,
# while the reference numbers the original images and has gaps where laue.integrate
# dropped frames, so BATCH_model - BATCH_ref drifts (-4..+3 depending on sweep).
# Matching on (H,K,L,xcal,ycal) is 100% in every sweep and maps each model frame to
# exactly one reference frame (purity 1.0000, zero ambiguous). Taking every column
# but I/SIGI from the reference therefore also repairs BATCH in all four cells.
KEY = ["H", "K", "L", "xr", "yr"]

for c in COMBOS:
    for s in SWEEPS:
        (OUT / f"swap_{c}" / "gain_0,3_from_stills" / f"dials_files_{s}").mkdir(parents=True, exist_ok=True)
    (OUT / f"swap_{c}" / "logs").mkdir(parents=True, exist_ok=True)

tot_ref = tot_mod = tot_keep = 0
print(f"{'sweep':10s} {'ref':>8} {'model':>8} {'matched':>8} {'kept %':>7}")
for s in SWEEPS:
    ref = rs.read_mtz(str(REF / f"dials_files_{s}" / "integrated.mtz"))
    mod = rs.read_mtz(str(MOD / f"dials_files_{s}" / "integrated.mtz"))
    cell, sg = ref.cell, ref.spacegroup

    def flat(d):
        f = d.reset_index()
        f["xr"] = f["xcal"].round(0).astype(int)
        f["yr"] = f["ycal"].round(0).astype(int)
        for k in ("H", "K", "L", "BATCH"):
            f[k] = f[k].astype(int)
        return pd.DataFrame(f).drop_duplicates(subset=KEY)

    fr, fm = flat(ref), flat(mod)
    j = fr.merge(fm[KEY + ["I", "SIGI"]], on=KEY, how="inner", suffixes=("_ref", "_mod"))
    tot_ref += len(fr); tot_mod += len(fm); tot_keep += len(j)
    print(f"{s:10s} {len(fr):8,} {len(fm):8,} {len(j):8,} {len(j)/len(fr)*100:6.1f}%")

    for combo, (isrc, ssrc) in COMBOS.items():
        d = j.copy()
        d["I"] = d["I_ref"] if isrc == "ref" else d["I_mod"]
        d["SIGI"] = d["SIGI_ref"] if ssrc == "ref" else d["SIGI_mod"]
        d = d.drop(columns=[c for c in d.columns if c.endswith(("_ref", "_mod"))] + ["xr", "yr"])
        out = rs.DataSet(d, cell=cell, spacegroup=sg).set_index(["H", "K", "L"])
        out["I"] = out["I"].astype("Intensity")
        out["SIGI"] = out["SIGI"].astype("Stddev")
        out["SIGBG"] = out["SIGBG"].astype("Stddev")
        out["BATCH"] = out["BATCH"].astype("Batch")
        out["PARTIAL"] = out["PARTIAL"].astype(bool)
        for col in ("xcal", "ycal", "wavelength", "BG"):
            out[col] = out[col].astype("MTZReal")
        dest = OUT / f"swap_{combo}" / "gain_0,3_from_stills" / f"dials_files_{s}"
        out[ref.columns].write_mtz(str(dest / "integrated.mtz"))
        shutil.copy(REF / f"dials_files_{s}" / "rmsds.log", dest / "rmsds.log")

print(f"\ntotal  ref {tot_ref:,}  model {tot_mod:,}  matched {tot_keep:,} "
      f"({tot_keep/tot_ref*100:.1f}% of ref, {tot_keep/tot_mod*100:.1f}% of model)")
print("SWAP_TREES_BUILT")
