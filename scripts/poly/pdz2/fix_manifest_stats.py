"""Recompute a dataset manifest's normalization statistics from the counts on disk.

The data module transforms every network input as (anscombe(counts) - mean)/sd using
`stats` from dataset.yaml, so wrong values mis-scale and mis-centre all inputs. Three
Laue manifests disagree with their own counts (1118 hewl3 and hewl360 by 1.85x in sd,
1117 by 1.47x) while pdz2 and the monochromatic 9b7c agree, so this rewrites the wrong
ones.

Statistics are accumulated over EVERY mask-valid pixel rather than a sample, matching
how the pooling scripts compute them, and stats are stored as [mean, variance].

The previous values are preserved twice over: as a comment block in the file, because
checkpoints trained before the fix used them and need them for inference, and as a
timestamped backup copy of the whole manifest.
"""

import argparse
import re
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--chunk", type=int, default=40_000)
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

p = Path(args.data_dir)
dy = p / "dataset.yaml"
if not dy.exists():
    raise SystemExit(f"{dy} not found")
text = dy.read_text()
spec = yaml.safe_load(text) or {}
old = (spec.get("stats") or {})
if not old:
    raise SystemExit(f"{dy} has no stats block")

counts = np.load(p / "counts.npy", mmap_mode="r")
mf = p / "masks.npy"
masks = np.load(mf, mmap_mode="r") if mf.exists() else None
n = counts.shape[0]

acc = {"raw": [0.0, 0.0, 0], "anscombe": [0.0, 0.0, 0]}
for a in range(0, n, args.chunk):
    b = min(a + args.chunk, n)
    v = counts[a:b].astype(np.float64)
    if masks is not None:
        v = v[masks[a:b]]
    else:
        v = v.ravel()
    ans = 2.0 * np.sqrt(np.clip(v, 0, None) + 0.375)
    for key, arr in (("raw", v), ("anscombe", ans)):
        acc[key][0] += arr.sum()
        acc[key][1] += (arr ** 2).sum()
        acc[key][2] += arr.size

new = {}
for key, (s1, s2, cnt) in acc.items():
    mean = s1 / cnt
    new[key] = [float(mean), float(max(s2 / cnt - mean ** 2, 0.0))]

print(f"{p}\n  {acc['raw'][2]:,} mask-valid pixels over {n:,} reflections")
for key in ("raw", "anscombe"):
    o = old.get(key)
    if o and len(o) == 2:
        print(f"  {key:9s} old mean/sd {o[0]:.4f}/{np.sqrt(o[1]):.4f}   "
              f"new {new[key][0]:.4f}/{np.sqrt(new[key][1]):.4f}   "
              f"sd ratio {np.sqrt(new[key][1]) / max(np.sqrt(o[1]), 1e-12):.3f}")
    else:
        print(f"  {key:9s} (absent before) new mean/sd "
              f"{new[key][0]:.4f}/{np.sqrt(new[key][1]):.4f}")

if args.dry_run:
    print("  dry run; nothing written")
    raise SystemExit(0)

# replace only the stats block, leaving every other key byte-identical
m = re.search(r"^stats:\n(?:[ \t]+.*\n|[ \t]*-.*\n)*", text, flags=re.MULTILINE)
if m is None:
    raise SystemExit("could not locate the stats block to replace")

old_lines = "\n".join(f"#   {ln}" for ln in m.group(0).rstrip("\n").split("\n"))
block = (
    "stats:\n"
    f"  # Recomputed {date.today().isoformat()} from counts.npy over all mask-valid\n"
    "  # pixels. The previous values disagreed with the data, which mis-scaled and\n"
    "  # mis-centred every network input. Checkpoints trained before this date used\n"
    "  # the values below and need them for inference, so they are kept here.\n"
    f"{old_lines}\n"
    "  raw:\n"
    f"  - {new['raw'][0]!r}\n"
    f"  - {new['raw'][1]!r}\n"
    "  anscombe:\n"
    f"  - {new['anscombe'][0]!r}\n"
    f"  - {new['anscombe'][1]!r}\n"
)

backup = p / f"dataset.yaml.pre_statfix_{date.today().isoformat()}"
if not backup.exists():
    shutil.copy(dy, backup)
    print(f"  backed up to {backup.name}")
updated = text[: m.start()] + block + text[m.end():]
dy.write_text(updated)

# the file must still parse, and nothing but stats may have moved
check = yaml.safe_load(dy.read_text()) or {}
for k in set(spec) | set(check):
    if k == "stats":
        continue
    if spec.get(k) != check.get(k):
        raise SystemExit(f"key '{k}' changed during the rewrite; restore {backup}")
if check["stats"]["anscombe"] != new["anscombe"]:
    raise SystemExit("stats did not round-trip")
print(f"  wrote {dy.name}; all other keys verified unchanged")
print("MANIFEST_FIXED")
