"""Compare weighted 200ns-off difference-map peak heights across arms.

CCsym is a half-dataset correlation, and profile fitting borrows information from
spatial neighbours that then feeds BOTH half-datasets -- the same mechanism this
project already found inflating CC-half at low signal. Difference-map peak heights
at the Fig 9C/D sidechains cannot be inflated that way: they are read from a single
merged map at positions fixed by the deposited model. So this is the borrowing-immune
endpoint, and it is the quantity the experiment actually cares about.

Heights are in units of the map's own rms, so arms are comparable despite different
overall map scales. Structure of the calculation follows pdz2_ref/scripts/wdm_peaks.py:
the map is in the primitive P1 cell while 5E1Y is C2, so the fractional transform is
chosen by requiring it preserve interatomic distances, then the extremum within
0.6 A of each atom is reported for both symmetry mates.
"""

import argparse
import itertools

import gemmi
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--arm", action="append", required=True,
                help="label=/path/to/wdm_rescut.mtz, repeatable")
ap.add_argument("--pdb", default="/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_ref/5e1y.pdb")
a = ap.parse_args()

st = gemmi.read_structure(a.pdb)
cellC = st.cell
op = gemmi.Op("a+b,a-b,-c")
Rm = np.array(op.rot, dtype=float) / op.DEN
cellP5 = cellC.changed_basis_forward(op, True)

cas = [at for ch in st[0] for res in ch for at in res if at.name == "CA"][:2]
f0, f1 = cellC.fractionalize(cas[0].pos), cellC.fractionalize(cas[1].pos)
dC = cas[0].pos.dist(cas[1].pos)
good = None
for name, Mx in [("R", Rm), ("Rinv", np.linalg.inv(Rm)), ("Rt", Rm.T),
                 ("RinvT", np.linalg.inv(Rm).T)]:
    q0 = cellP5.orthogonalize(gemmi.Fractional(*(Mx @ np.array([f0.x, f0.y, f0.z]))))
    q1 = cellP5.orthogonalize(gemmi.Fractional(*(Mx @ np.array([f1.x, f1.y, f1.z]))))
    if abs(q0.dist(q1) - dC) < 1e-3:
        good = (name, Mx)
assert good, "no distance-preserving transform found"
print(f"transform: {good[0]}")
M = good[1]

twofold = gemmi.Op("-x,y,-z")
targets = {368: ["OD1", "OD2"], 410: ["OG"], 415: ["OD1", "ND2"]}
shifts = list(itertools.product(*[np.linspace(-0.6, 0.6, 7)] * 3))

arms = []
for spec in a.arm:
    label, path = spec.split("=", 1)
    mtz = gemmi.read_mtz_file(path)
    grid = mtz.transform_f_phi_to_map("wDF", "Phi", sample_rate=3.0)
    arr = np.array(grid, copy=False)
    arms.append((label, grid, mtz.cell, float(arr.std())))
    print(f"{label:14s} grid {arr.shape}  rms {arr.std():.4f}")


def peak(grid, cellP, sig, fr):
    best = 0.0
    for d in shifts:
        off = cellP.fractionalize(gemmi.Position(*d))
        v = grid.interpolate_value(
            gemmi.Fractional(fr[0] + off.x, fr[1] + off.y, fr[2] + off.z)
        ) / sig
        if abs(v) > abs(best):
            best = v
    return best


sites, rows = [], []
for resnum, atoms in targets.items():
    for ch in st[0]:
        for res in ch:
            if res.seqid.num != resnum:
                continue
            for alt, conf in (("A", "1"), ("B", "2")):
                for an in atoms:
                    sel = [at for at in res if at.name == an and at.altloc in (alt, "", "\0")]
                    if not sel:
                        continue
                    fA = cellC.fractionalize(sel[0].pos)
                    fa = np.array([fA.x, fA.y, fA.z])
                    fb = np.array(twofold.apply_to_xyz(list(fa)))
                    for mate, fx in (("1", fa), ("2", fb)):
                        sites.append((f"{res.name}{resnum}", conf, an, mate))
                        rows.append([peak(g, cp, sg, M @ fx) for _, g, cp, sg in arms])

rows = np.array(rows)
hdr = "".join(f"{lab:>12s}" for lab, *_ in arms)
print(f"\n{'residue':10s} {'conf':>4s} {'atom':>5s} {'mate':>5s}{hdr}")
for (r, c, at, mate), vals in zip(sites, rows):
    print(f"{r:10s} {c:>4s} {at:>5s} {mate:>5s}" + "".join(f"{v:+12.1f}" for v in vals))

print(f"\n{'summary':27s}" + "".join(f"{lab:>12s}" for lab, *_ in arms))
print(f"{'mean |peak| (sigma)':27s}" + "".join(f"{v:12.2f}" for v in np.abs(rows).mean(0)))
print(f"{'max |peak| (sigma)':27s}" + "".join(f"{v:12.2f}" for v in np.abs(rows).max(0)))
print(f"{'n sites |peak| > 3 sigma':27s}" + "".join(f"{v:12d}" for v in (np.abs(rows) > 3).sum(0)))
ref = np.sign(rows[:, 0])
agree = [int((np.sign(rows[:, i]) == ref).sum()) for i in range(len(arms))]
print(f"{'sign agrees with ' + arms[0][0]:27s}" + "".join(f"{v:12d}" for v in agree))
print(f"\nsites: {len(rows)}")
print("WDM_PEAKS_DONE")
