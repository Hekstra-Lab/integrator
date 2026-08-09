"""
Plot completeness vs. resolution for cctbx, sqrt_squareplus, and asinh.

Data source: merging_refinement_results.txt
  - sqrt_squareplus: lines 383-407
  - asinh:           lines 800-824
  - cctbx:           lines 1222-1245

Completeness = [observed unique / total possible] per resolution shell.

NOTE: The Integrator is NOT 100% complete in all bins.
  - Low-to-mid resolution (bins 1-10, > 2.52 A): cctbx is near-complete
    (~90-100%); Integrator drops to ~55-85%.
  - High resolution (bins 12-20, < 2.44 A): cctbx drops sharply to 4%;
    Integrator maintains 33-58%.
  This is the key comparison — not a simple "Integrator = 100%" story.
"""

import matplotlib.pyplot as plt
import numpy as np

# ── Resolution shell upper bounds (d_min of each shell, Angstroms) ──────────
# Bins run from low resolution (bin 1, ~5.4 A) to high resolution (bin 20, 2.0 A).
# X-axis is inverted so high resolution appears on the right.
resolution = [
    5.4288,  # bin  1
    4.3089,  # bin  2
    3.7641,  # bin  3
    3.4200,  # bin  4
    3.1748,  # bin  5
    2.9876,  # bin  6
    2.8380,  # bin  7
    2.7144,  # bin  8
    2.6099,  # bin  9
    2.5198,  # bin 10
    2.4410,  # bin 11
    2.3713,  # bin 12
    2.3088,  # bin 13
    2.2525,  # bin 14
    2.2013,  # bin 15
    2.1544,  # bin 16
    2.1113,  # bin 17
    2.0715,  # bin 18
    2.0345,  # bin 19
    2.0000,  # bin 20
]

# ── Completeness (%) — verified from merging_refinement_results.txt ─────────

# sqrt_squareplus (lines 383-407)
sqrt_squareplus = [
    99.3,
    98.6,
    97.6,
    94.9,
    84.2,
    77.6,
    65.7,
    58.7,
    56.5,
    55.8,
    53.8,
    58.0,
    51.2,
    58.1,
    51.3,
    45.8,
    57.4,
    33.5,
    47.7,
    55.1,
]

# asinh (lines 800-824)
asinh = [
    99.3,
    98.5,
    97.6,
    94.6,
    85.5,
    76.4,
    67.5,
    55.4,
    52.5,
    55.6,
    51.2,
    50.0,
    56.5,
    54.2,
    52.5,
    52.2,
    54.5,
    49.6,
    57.2,
    47.3,
]

# cctbx baseline (lines 1222-1245)
cctbx = [
    99.8,
    99.7,
    99.6,
    99.4,
    99.1,
    98.7,
    98.0,
    95.3,
    93.5,
    90.6,
    81.9,
    71.0,
    52.1,
    44.9,
    32.9,
    23.4,
    15.6,
    5.8,
    9.2,
    4.1,
]

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))

ax.plot(
    resolution,
    cctbx,
    marker="o",
    linewidth=2.5,
    color="#2166ac",
    label="cctbx.xfel",
)

ax.plot(
    resolution,
    sqrt_squareplus,
    marker="s",
    linewidth=2.5,
    color="#d6604d",
    label="Integrator — sqrt-squareplus",
)

ax.plot(
    resolution,
    asinh,
    marker="^",
    linestyle="--",
    linewidth=2.5,
    color="#f4a582",
    label="Integrator — asinh",
)

# Shade the high-resolution region where the key difference is visible
ax.axvspan(
    2.0, 2.44, alpha=0.07, color="gray", label="High-res region (< 2.44 Å)"
)

ax.set_xlabel("Resolution (Å)", fontsize=13)
ax.set_ylabel("Completeness (%)", fontsize=13)
ax.set_title("Completeness vs. Resolution", fontsize=15)

ax.set_ylim(0, 105)
ax.invert_xaxis()  # crystallography convention: high resolution on the right

ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=10)

fig.tight_layout()

fig.savefig("completeness_vs_resolution.png", dpi=300, bbox_inches="tight")
fig.savefig("completeness_vs_resolution.pdf", bbox_inches="tight")

plt.show()

print("Saved: completeness_vs_resolution.png / .pdf")
print()
print("Key values at high resolution (< 2.44 A):")
print(f"{'Bin':>4}  {'d_min':>6}  {'cctbx':>8}  {'sqrt_sq':>10}  {'asinh':>8}")
print("-" * 50)
for i in range(11, 20):
    print(
        f"{i + 1:>4}  {resolution[i]:>6.4f}  "
        f"{cctbx[i]:>7.1f}%  "
        f"{sqrt_squareplus[i]:>9.1f}%  "
        f"{asinh[i]:>7.1f}%"
    )
