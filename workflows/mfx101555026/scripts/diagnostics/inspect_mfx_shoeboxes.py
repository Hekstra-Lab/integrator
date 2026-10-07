import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_dir = Path("/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx")
data_dir = project_dir / "mfx_shoebox_r0269_012_rg058_with_d_1500"
out_dir = project_dir/"scripts/diagnostics/shoebox_inspection_1500_masked"


counts = np.load(data_dir / "counts.npy", mmap_mode="r")
masks = np.load(data_dir / "masks.npy", mmap_mode="r")
metadata = np.load(data_dir / "metadata.npy", allow_pickle=True).item()

intensity = metadata["intensity.sum.value"]

indices = [
    # 10%
    int(np.nanargmin(np.abs(intensity - np.nanpercentile(intensity, 10)))),

    # 50%
    int(np.nanargmin(np.abs(intensity - np.nanpercentile(intensity, 50)))),

    # 90%
    int(np.nanargmin(np.abs(intensity - np.nanpercentile(intensity, 90)))),
]


# Remove duplicate indices while keeping the same order
indices = list(dict.fromkeys(indices))

for idx in indices:
    img = counts[idx].reshape(25, 25)
    mask_img = masks[idx].reshape(25, 25)

    masked_img = np.ma.masked_where(~mask_img, img)

    plt.figure(figsize=(5, 5))
    plt.imshow(masked_img, vmin=0, vmax=500)
    plt.colorbar()
    plt.title(
        f"idx={idx}, intensity={intensity[idx]:.2f}, d={metadata['d'][idx]:.2f}",
        fontsize=8
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"shoebox_{idx}_masked.png", dpi=150, bbox_inches="tight")
    plt.close()

print("Wrote masked images to:", out_dir)


# Top 20 strongest reflections
# Make one grid image for the top 20 strongest reflections
top20_indices = np.argsort(intensity)[-20:][::-1]

#fig: the whole image, axes: the 5 x 4 collection

"""
fig = whole page

axes =
[
[plot1, plot2, plot3, plot4 ],
[plot5, plot6, plot7, plot8 ],
[plot9, plot10, plot11, plot12],
[plot13, plot14, plot15, plot16],
[plot17, plot18, plot19, plot20]
]
"""

fig, axes= plt.subplots(5, 4, figsize=(12,15))

#axes.flat: return plot1, plot2, plot3, ..., plot20
#top20_indices: return index1, index2, index3, ..., index20
#zip(axes.flat, top20_indices): pair (plot1, index1)

for ax, idx in zip(axes.flat, top20_indices):
    img = counts[idx].reshape(25, 25)
    mask_img = masks[idx].reshape(25, 25)

    masked_img = np.ma.masked_where(~mask_img, img)

    
    ax.imshow(masked_img, vmin=0, vmax=500)
    ax.set_title(
        f"idx={idx}\nI={intensity[idx]:.1f}, d={metadata['d'][idx]:.2f}",
        fontsize=7)
    ax.axis("off")

plt.tight_layout()
plt.savefig(out_dir / "top20_strongest_grid.png", dpi=150, bbox_inches="tight")
plt.close()
print("Wrote top 20 grid to:", out_dir / "top20_strongest_grid.png")
