"""Pool the 16 PDZ2 sweeps into one poly training set.

Concatenates counts/masks/metadata in a fixed sweep order, computes pooled
Anscombe and raw statistics over mask-valid pixels, and writes dataset.yaml.
sweep_ranges.yaml records each sweep's row range so predictions split back into
per-sweep MTZs for the reference pipeline's expt_concat.
"""

import numpy as np
import torch
import yaml
from pathlib import Path

from integrator.io import write_dataset_yaml

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--sbox", default="sbox")
ap.add_argument("--out", default="pooled")
ap.add_argument("--hw", type=int, default=21)
_a = ap.parse_args()

M = Path("/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model")
SBOX, OUT = M / _a.sbox, M / _a.out
OUT.mkdir(parents=True, exist_ok=True)
SWEEPS = [f"{p}_{t}" for p in "cdef" for t in ("off", "50ns", "100ns", "200ns")]

shapes = {}
for s in SWEEPS:
    c = np.load(SBOX / s / "counts.npy", mmap_mode="r")
    shapes[s] = c.shape
    assert c.shape[1:] == shapes[SWEEPS[0]][1:], (s, c.shape)
N = sum(v[0] for v in shapes.values())
npix = shapes[SWEEPS[0]][1]
print(f"{len(SWEEPS)} sweeps, {N:,} reflections, {npix} pixels each")

counts = np.lib.format.open_memmap(OUT / "counts.npy", mode="w+", dtype=np.uint16, shape=(N, npix))
masks = np.lib.format.open_memmap(OUT / "masks.npy", mode="w+", dtype=bool, shape=(N, npix))

ranges, metas, off = {}, [], 0
for s in SWEEPS:
    n = shapes[s][0]
    counts[off:off + n] = np.load(SBOX / s / "counts.npy", mmap_mode="r")
    masks[off:off + n] = np.load(SBOX / s / "masks.npy", mmap_mode="r")
    metas.append(torch.load(SBOX / s / "metadata.pt", weights_only=False))
    ranges[s] = [off, off + n]
    print(f"  {s:10s} rows {off:>7,}-{off + n:>7,}")
    off += n
assert off == N
counts.flush(); masks.flush()

keys = set(metas[0])
for m in metas[1:]:
    keys &= set(m)
meta = {k: torch.cat([torch.as_tensor(m[k]).reshape(-1) for m in metas]) for k in sorted(keys)}
# sweep index per reflection, so predictions can be split without re-deriving ranges
sweep_id = torch.cat([torch.full((shapes[s][0],), i, dtype=torch.long) for i, s in enumerate(SWEEPS)])
meta["sweep_id"] = sweep_id
torch.save(meta, OUT / "metadata.pt")
print(f"metadata: {len(meta)} keys, {len(meta['d']):,} rows")

# pooled stats over mask-valid pixels, streamed
acc = {"anscombe": [0.0, 0.0, 0], "raw": [0.0, 0.0, 0]}
CH = 50_000
for i in range(0, N, CH):
    c = counts[i:i + CH].astype(np.float64)
    m = masks[i:i + CH]
    v = c[m]
    a = 2.0 * np.sqrt(np.clip(v, 0, None) + 0.375)
    for key, arr in (("raw", v), ("anscombe", a)):
        acc[key][0] += arr.sum(); acc[key][1] += (arr ** 2).sum(); acc[key][2] += arr.size
stats = {}
for key, (s1, s2, n) in acc.items():
    mean = s1 / n
    stats[key] = [float(mean), float(s2 / n - mean ** 2)]
    print(f"  {key}: mean {stats[key][0]:.4f} var {stats[key][1]:.4f} over {n:,} valid pixels")

crystal = yaml.safe_load((SBOX / SWEEPS[0] / "crystal.yaml").read_text())
p = write_dataset_yaml(
    OUT,
    geometry={"d": 1, "h": _a.hw, "w": _a.hw},
    n_reflections=N,
    polychromatic=True,
    anscombe=True,
    files={"counts": "counts.npy", "masks": "masks.npy", "reference": "metadata.pt"},
    crystal=crystal,
    stats=stats,
)
(OUT / "sweep_ranges.yaml").write_text(yaml.safe_dump({"order": SWEEPS, "ranges": ranges}, sort_keys=False))
print(f"wrote {p} and sweep_ranges.yaml")
print("POOL_DONE")
