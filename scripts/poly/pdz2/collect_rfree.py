"""Collect PDZ2 off-state R factors across arms and careless replicates.

All ten refinements share one model, one protocol and the same 404 held-out reflections,
so arms differ only in their integrated intensities. Replicates are different careless
seeds on the same integration, which is the noise floor the arm comparison has to clear.
"""

import glob
import re

import numpy as np

B = "/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model/refine_off"
GROUPS = {
    "laue": ["laue", "laue_r1", "laue_r2"],
    "geom_both": ["geom_both", "geom_both_r1", "geom_both_r2"],
    "ml_clamp": ["ml_clamp", "ml_clamp_r1", "ml_clamp_r2"],
    "k7m_bfix": ["k7m_bfix"],
}


def rvals(arm):
    f = glob.glob(B + "/run_" + arm + "/refined*_001.log")
    if not f:
        return None
    t = open(f[0]).read()
    m = re.findall(r"^Final R-work = ([\d.]+), R-free = ([\d.]+)", t, re.M)
    return (float(m[-1][0]), float(m[-1][1])) if m else None


res = {}
print("%-14s %3s %21s %21s" % ("arm", "n", "R-work", "R-free"))
for g, arms in GROUPS.items():
    v = [x for x in (rvals(a) for a in arms) if x]
    if not v:
        print("%-14s  no logs" % g)
        continue
    rw = np.array([x[0] for x in v])
    rf = np.array([x[1] for x in v])
    res[g] = rf

    def se(x):
        return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")

    print("%-14s %3d %12.4f +-%6.4f %12.4f +-%6.4f"
          % (g, len(v), rw.mean(), se(rw), rf.mean(), se(rf)))

if "laue" in res and len(res["laue"]) > 1:
    L = res["laue"]
    print("\narm minus laue, R-free [lower is better]")
    for g, rf in res.items():
        if g == "laue" or len(rf) < 2:
            continue
        d = rf.mean() - L.mean()
        s = np.sqrt(rf.var(ddof=1) / len(rf) + L.var(ddof=1) / len(L))
        print("  %-14s %+8.4f  (%+.1f sigma)" % (g, d, d / s if s > 0 else 0.0))
    print("\nper-replicate R-free")
    for g, arms in GROUPS.items():
        v = [rvals(a) for a in arms]
        s = "  ".join("%.4f" % x[1] for x in v if x)
        print("  %-14s %s" % (g, s))
