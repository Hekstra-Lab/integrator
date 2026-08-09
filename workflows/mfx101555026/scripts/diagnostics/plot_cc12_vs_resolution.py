"""
Plot CC1/2 vs resolution for cctbx, sqrt_squareplus, and asinh.

Data source: merging_refinement_results.txt
  CC1/2 values are in the scaling table (NOT the Intensity Statistics table).
  The scaling table column labeled "Completeness" shows [N/M] X% where
  X% = CC1/2 * 100. This was confirmed by cross-checking against the
  explicit "CC 1/2" list printed after each Intensity Statistics block.

  All three methods have ~100% crystallographic completeness in all bins
  (verified from Intensity Statistics tables, lines 261-284, 678-701,
  1100-1123). Do NOT make a "completeness vs resolution" plot.

Key observations:
  - cctbx CC1/2 decreases smoothly from 0.998 to 0.041 at 2.0 A
  - Integrator CC1/2 drops faster at mid-resolution then plateaus
    erratically at ~0.35-0.55 in the high-resolution tail
  - The non-monotonic Integrator behavior at high resolution requires
    further investigation — do not interpret as better signal
"""

import matplotlib.pyplot as plt

# ── Resolution shell upper bounds (d_min, Angstroms) ─────────────────────────
resolution = [
    5.4288,
    4.3089,
    3.7641,
    3.4200,
    3.1748,
    2.9876,
    2.8380,
    2.7144,
    2.6099,
    2.5198,
    2.4410,
    2.3713,
    2.3088,
    2.2525,
    2.2013,
    2.1544,
    2.1113,
    2.0715,
    2.0345,
    2.0000,
]

# ── CC1/2 values (0-1) verified from merging_refinement_results.txt ──────────
# sqrt_squareplus — scaling table lines 385-404, X% / 100
sqrt_squareplus = [
    0.993,
    0.986,
    0.976,
    0.949,
    0.842,
    0.776,
    0.657,
    0.587,
    0.565,
    0.558,
    0.538,
    0.580,
    0.512,
    0.581,
    0.513,
    0.458,
    0.574,
    0.335,
    0.477,
    0.551,
]

# asinh — scaling table lines 802-821, X% / 100
asinh = [
    0.993,
    0.985,
    0.976,
    0.946,
    0.856,
    0.763,
    0.675,
    0.554,
    0.525,
    0.556,
    0.512,
    0.500,
    0.565,
    0.542,
    0.525,
    0.522,
    0.545,
    0.496,
    0.572,
    0.473,
]

# cctbx baseline — scaling table lines 1224-1243, X% / 100
cctbx = [
    0.998,
    0.997,
    0.996,
    0.994,
    0.991,
    0.987,
    0.980,
    0.953,
    0.935,
    0.906,
    0.819,
    0.710,
    0.521,
    0.449,
    0.329,
    0.234,
    0.156,
    0.058,
    0.092,
    0.041,
]

# ── Plot ──────────────────────────────────────────────────────────────────────
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

ax.axhline(
    0.5,
    color="gray",
    linestyle=":",
    linewidth=1.0,
    label="CC½ = 0.5 reference",
)

ax.set_xlabel("Resolution (Å)", fontsize=13)
ax.set_ylabel("CC½", fontsize=13)
ax.set_title("CC½ vs. Resolution", fontsize=15)
ax.set_ylim(0, 1.05)
ax.invert_xaxis()  # crystallography convention: high resolution on the right
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=10)

fig.tight_layout()
fig.savefig("cc12_vs_resolution.png", dpi=300, bbox_inches="tight")
fig.savefig("cc12_vs_resolution.pdf", bbox_inches="tight")
plt.show()

print("Saved: cc12_vs_resolution.png / .pdf")
