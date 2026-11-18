"""Peak heights of the weighted 200ns-off difference map at the residues shown in
Fig 9C/D of the Laue-DIALS paper (Asp368, Ser410, Asn415), for both symmetry
mates. The map is in the primitive P1 cell while 5E1Y is in C2, so the fractional
transform between them is picked by requiring that it preserve interatomic
distances, then peaks are read within 0.6 A of each sidechain atom.
"""

import itertools

import gemmi
import numpy as np

R = "/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_ref"
mtz = gemmi.read_mtz_file(R + "/gain_0,3_from_stills/wdm_rescut.mtz")
grid = mtz.transform_f_phi_to_map("wDF", "Phi", sample_rate=3.0)
arr = np.array(grid, copy=False)
sig = arr.std()
print("map grid %s  rms %.4f  P1 cell %s" % (arr.shape, sig, mtz.cell))

st = gemmi.read_structure(R + "/5e1y.pdb")
cellC, cellP = st.cell, mtz.cell
op = gemmi.Op("a+b,a-b,-c")
Rm = np.array(op.rot, dtype=float) / op.DEN

# validate against 5E1Y's OWN primitive cell: our data cell is ~0.4% larger, so
# comparing there leaves a residual that is cell scale, not a wrong transform.
cellP5 = cellC.changed_basis_forward(op, True)
print("5E1Y primitive cell %s" % cellP5)
cas = [a for ch in st[0] for res in ch for a in res if a.name == "CA"][:2]
f0 = cellC.fractionalize(cas[0].pos)
f1 = cellC.fractionalize(cas[1].pos)
dC = cas[0].pos.dist(cas[1].pos)
good = None
for name, M in [("R", Rm), ("Rinv", np.linalg.inv(Rm)), ("Rt", Rm.T),
                ("RinvT", np.linalg.inv(Rm).T)]:
    g0 = M @ np.array([f0.x, f0.y, f0.z])
    g1 = M @ np.array([f1.x, f1.y, f1.z])
    q0 = cellP5.orthogonalize(gemmi.Fractional(*g0))
    q1 = cellP5.orthogonalize(gemmi.Fractional(*g1))
    d = q0.dist(q1)
    flag = "OK" if abs(d - dC) < 1e-3 else ""
    print("  candidate %-6s dist %.3f vs %.3f %s" % (name, d, dC, flag))
    if abs(d - dC) < 1e-3:
        good = (name, M)
assert good, "no distance-preserving transform found"
name, M = good
print("using transform: %s" % name)

twofold = gemmi.Op("-x,y,-z")
targets = {368: ["OD1", "OD2"], 410: ["OG"], 415: ["OD1", "ND2"]}
shifts = list(itertools.product(*[np.linspace(-0.6, 0.6, 7)] * 3))


def peak(fr):
    best = 0.0
    for d in shifts:
        off = cellP.fractionalize(gemmi.Position(*d))
        v = grid.interpolate_value(
            gemmi.Fractional(fr[0] + off.x, fr[1] + off.y, fr[2] + off.z)
        ) / sig
        if abs(v) > abs(best):
            best = v
    return best


print("\n%-12s %-5s %-5s %9s %9s" % ("residue", "conf", "atom", "mate 1", "mate 2"))
for resnum, atoms in targets.items():
    for ch in st[0]:
        for res in ch:
            if res.seqid.num != resnum:
                continue
            for alt, conf in (("A", "1"), ("B", "2")):
                for an in atoms:
                    sel = [a for a in res if a.name == an and a.altloc in (alt, "", "\0")]
                    if not sel:
                        continue
                    fA = cellC.fractionalize(sel[0].pos)
                    fa = np.array([fA.x, fA.y, fA.z])
                    fb = np.array(twofold.apply_to_xyz(list(fa)))
                    print("%-12s %-5s %-5s %+9.1f %+9.1f"
                          % (res.name + str(resnum), conf, an, peak(M @ fa), peak(M @ fb)))
