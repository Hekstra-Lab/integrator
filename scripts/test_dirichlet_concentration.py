"""Test Dirichlet concentration fitting methods on real HEWL 9b7c data.

Compares:
  1. MOM bg-sub, resolution-only binning
  2. MOM bg-sub, adaptive 2D binning: resolution x azimuthal
     (azi sectors reduced per shell to maintain min_per_bin occupancy)

Reports middle slice (frame 1 of 3) for visual comparison.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# ── Data loading ──────────────────────────────────────────────────────────────

data_dir = Path("/Users/luis/from_harvard_rc/")
data = {x.stem: x for x in data_dir.glob("*9b7c/*.pt")}

print("Loading data...")
counts = torch.load(data["counts"], weights_only=True)
masks = torch.load(data["masks"], weights_only=True)
metadata = torch.load(data["metadata"], weights_only=False)

D, H, W = 3, 21, 21
n_pixels = D * H * W
n_pixels_per_frame = H * W
N = counts.shape[0]
print(f"Loaded {N} reflections, shoebox shape: {D}x{H}x{W}")

d = metadata["d"]

# ── Detector coordinates ─────────────────────────────────────────────────────

beam_x, beam_y = 2058.5, 2191.5
x_det = metadata["xyzcal.px.0"]
y_det = metadata["xyzcal.px.1"]

dx = x_det - beam_x
dy = y_det - beam_y
r_det = torch.sqrt(dx**2 + dy**2)
phi_det = torch.atan2(dy, dx)  # [-pi, pi]

print(f"Radial distance: min={r_det.min():.0f}, max={r_det.max():.0f} px")

# ── Binning schemes ──────────────────────────────────────────────────────────

n_res_bins = 30
max_azi_bins = 16
min_per_bin = 200

# Resolution bins via quantiles
res_quantiles = torch.linspace(0, 1, n_res_bins + 1)
res_edges = torch.quantile(d.float(), res_quantiles)
res_labels = torch.searchsorted(res_edges[1:-1], d).long()

# Adaptive azimuthal binning per resolution shell
azi_bins_per_shell = []
group_labels_2d = torch.zeros(N, dtype=torch.long)
offset = 0

for rb in range(n_res_bins):
    shell_mask = res_labels == rb
    phi_shell = phi_det[shell_mask]

    n_azi = max_azi_bins
    while n_azi > 1:
        azi_e = torch.linspace(-math.pi, math.pi, n_azi + 1)
        azi_lab = torch.searchsorted(azi_e[1:-1], phi_shell).long()
        occ = torch.bincount(azi_lab, minlength=n_azi)
        if occ.min().item() >= min_per_bin:
            break
        n_azi -= 1

    if n_azi > 1:
        azi_e = torch.linspace(-math.pi, math.pi, n_azi + 1)
        azi_lab = torch.searchsorted(azi_e[1:-1], phi_shell).long()
    else:
        azi_lab = torch.zeros(int(shell_mask.sum()), dtype=torch.long)

    group_labels_2d[shell_mask] = offset + azi_lab
    azi_bins_per_shell.append(n_azi)
    offset += n_azi

n_total_bins = offset

bin_counts_res = torch.bincount(res_labels, minlength=n_res_bins)
bin_counts_2d = torch.bincount(group_labels_2d, minlength=n_total_bins)
print(
    f"\nRes-only: {n_res_bins} bins, occupancy {bin_counts_res.min()}-{bin_counts_res.max()}"
)
print(
    f"Adaptive 2D: {n_total_bins} bins (max {max_azi_bins} azi, min_per_bin={min_per_bin})"
)
print(f"  Occupancy: min={bin_counts_2d.min()}, max={bin_counts_2d.max()}")
print(f"  Azi bins per shell: {azi_bins_per_shell}")

# ── Shared bg-subtraction ────────────────────────────────────────────────────

print("\nComputing bg-subtracted signal...")
counts_clean = counts.float().clamp(min=0)
masks_f = masks.float()
counts_masked = counts_clean * masks_f

counts_3d = counts_masked.reshape(N, D, n_pixels_per_frame)
masks_3d = masks_f.reshape(N, D, n_pixels_per_frame)

frame_counts = counts_3d.sum(dim=-1)
frame_n_pixels = masks_3d.sum(dim=-1)
min_frame_idx = frame_counts.argmin(dim=-1)
bg_frame_counts = frame_counts.gather(1, min_frame_idx.unsqueeze(-1)).squeeze(
    -1
)
bg_frame_n_pixels = frame_n_pixels.gather(
    1, min_frame_idx.unsqueeze(-1)
).squeeze(-1)
bg_per_pixel = bg_frame_counts / bg_frame_n_pixels.clamp(min=1)

signal = (counts_masked - bg_per_pixel.unsqueeze(-1) * masks_f).clamp(min=0)


# ── MOM fitting function ─────────────────────────────────────────────────────


def fit_mom_bgsub(signal, group_labels, n_bins, n_pix):
    """MOM on pre-computed bg-subtracted signal."""
    conc = torch.zeros(n_bins, n_pix)
    kappas = []
    for b in range(n_bins):
        sel = signal[group_labels == b]
        if len(sel) < 10:
            conc[b] = 1e-6
            kappas.append(0.0)
            continue
        totals = sel.sum(dim=1, keepdim=True).clamp(min=1)
        sel_norm = sel / totals
        p_bar = sel_norm.mean(dim=0)
        var_p = sel_norm.var(dim=0)
        valid = p_bar > 1e-6
        if valid.sum() > 0:
            ratio = (p_bar[valid] * (1 - p_bar[valid])) / var_p[valid].clamp(
                min=1e-12
            ) - 1
            kappa = ratio.median().clamp(min=1.0)
        else:
            kappa = torch.tensor(1.0)
        conc[b] = (kappa * p_bar).clamp(min=1e-6)
        kappas.append(kappa.item())
    return conc, kappas


# ── Fit both binning schemes ─────────────────────────────────────────────────

print("Fitting MOM bg-sub (resolution-only)...")
conc_res, kappa_res = fit_mom_bgsub(signal, res_labels, n_res_bins, n_pixels)

print("Fitting MOM bg-sub (2D: resolution x azimuthal)...")
conc_2d, kappa_2d = fit_mom_bgsub(
    signal, group_labels_2d, n_total_bins, n_pixels
)


# ── Extract middle slice ─────────────────────────────────────────────────────


def middle_slice(conc):
    return conc.reshape(-1, D, H, W)[:, D // 2, :, :]


conc_res_mid = middle_slice(conc_res)
conc_2d_mid = middle_slice(conc_2d)


# ── Plot 1: Resolution-only profiles ─────────────────────────────────────────

n_cols_res = min(n_res_bins, 10)
n_rows_res = math.ceil(n_res_bins / n_cols_res)
fig, axes = plt.subplots(
    n_rows_res, n_cols_res, figsize=(2.2 * n_cols_res, 2.5 * n_rows_res)
)
axes = np.atleast_2d(axes)
fig.suptitle("MOM bg-sub: Resolution-only binning (middle slice)", fontsize=13)

for b in range(n_res_bins):
    row, col = divmod(b, n_cols_res)
    d_lo, d_hi = res_edges[b].item(), res_edges[b + 1].item()
    img = conc_res_mid[b].numpy()
    axes[row, col].imshow(img, cmap="viridis")
    axes[row, col].set_title(
        f"d=[{d_lo:.2f},{d_hi:.2f}]\nκ={kappa_res[b]:.0f} Σα={img.sum():.0f}",
        fontsize=6,
    )
    axes[row, col].axis("off")

# Hide unused subplots
for b in range(n_res_bins, n_rows_res * n_cols_res):
    row, col = divmod(b, n_cols_res)
    axes[row, col].axis("off")

plt.tight_layout()
plt.savefig("/Users/luis/integrator/scripts/conc_res_only_bgsub.png", dpi=150)
plt.close()
print("\nSaved conc_res_only_bgsub.png")


# ── Plot 2: Azimuthal variation for selected resolution shells ────────────────

# Build per-shell offset lookup for indexing into conc_2d
shell_offsets = [0]
for n_azi in azi_bins_per_shell:
    shell_offsets.append(shell_offsets[-1] + n_azi)

# Pick 5 resolution bins with enough azi bins to be interesting
res_picks = [rb for rb in range(n_res_bins) if azi_bins_per_shell[rb] >= 4]
n_picks = min(6, len(res_picks))
res_picks = [
    res_picks[int(round(i * (len(res_picks) - 1) / (n_picks - 1)))]
    for i in range(n_picks)
]

max_azi_in_picks = max(azi_bins_per_shell[rb] for rb in res_picks)

fig, axes = plt.subplots(
    len(res_picks),
    max_azi_in_picks + 1,
    figsize=(1.4 * (max_azi_in_picks + 1), 2.2 * len(res_picks)),
)
fig.suptitle(
    "Azimuthal variation within resolution bins (adaptive, bg-sub MOM, middle slice)\n"
    "Last column = resolution-only (averaged over all azimuths)",
    fontsize=11,
)

for row, rb in enumerate(res_picks):
    d_lo, d_hi = res_edges[rb].item(), res_edges[rb + 1].item()
    n_azi_shell = azi_bins_per_shell[rb]
    base = shell_offsets[rb]

    # Compute vmin/vmax across this shell + res-only
    vmin = float("inf")
    vmax = 0
    for ab in range(n_azi_shell):
        img = conc_2d_mid[base + ab].numpy()
        vmin = min(vmin, img.min())
        vmax = max(vmax, img.max())
    img_res = conc_res_mid[rb].numpy()
    vmin = min(vmin, img_res.min())
    vmax = max(vmax, img_res.max())

    for ab in range(n_azi_shell):
        img = conc_2d_mid[base + ab].numpy()
        axes[row, ab].imshow(img, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[row, ab].axis("off")
        if row == 0:
            azi_lo = math.degrees(-math.pi + ab * 2 * math.pi / n_azi_shell)
            azi_hi = math.degrees(
                -math.pi + (ab + 1) * 2 * math.pi / n_azi_shell
            )
            axes[0, ab].set_title(f"{azi_lo:.0f}..{azi_hi:.0f}°", fontsize=5)

    # Last used column: resolution-only
    axes[row, n_azi_shell].imshow(
        img_res, cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[row, n_azi_shell].axis("off")
    if row == 0:
        axes[0, n_azi_shell].set_title("Res-only", fontsize=6)

    # Hide unused columns
    for ab in range(n_azi_shell + 1, max_azi_in_picks + 1):
        axes[row, ab].axis("off")

    axes[row, 0].set_ylabel(
        f"d=[{d_lo:.2f},{d_hi:.2f}]\n({n_azi_shell} azi)", fontsize=7
    )

plt.tight_layout()
plt.savefig("/Users/luis/integrator/scripts/conc_azi_variation.png", dpi=150)
plt.close()
print("Saved conc_azi_variation.png")


# ── Plot 3: kappa comparison ─────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Kappa histogram
valid_kappa_2d = [k for k in kappa_2d if k > 0]
axes[0].hist(
    kappa_res, bins=10, alpha=0.7, label=f"Res-only ({len(kappa_res)} bins)"
)
axes[0].hist(
    valid_kappa_2d,
    bins=20,
    alpha=0.7,
    label=f"2D ({len(valid_kappa_2d)} bins)",
)
axes[0].set_xlabel("kappa")
axes[0].set_title("Kappa distribution")
axes[0].legend()

# sum(alpha) histogram
sum_res = conc_res.sum(dim=1).numpy()
sum_2d = conc_2d.sum(dim=1).numpy()
axes[1].hist(sum_res[sum_res > 1], bins=10, alpha=0.7, label="Res-only")
axes[1].hist(sum_2d[sum_2d > 1], bins=20, alpha=0.7, label="2D (adaptive)")
axes[1].set_xlabel("sum(alpha)")
axes[1].set_title("Total concentration per bin")
axes[1].legend()

# Kappa vs resolution bin (2D shown as scatter per azi sector)
for rb in range(n_res_bins):
    n_azi_shell = azi_bins_per_shell[rb]
    base = shell_offsets[rb]
    kaps = [kappa_2d[base + ab] for ab in range(n_azi_shell)]
    axes[2].plot(
        [rb] * n_azi_shell, kaps, "o", alpha=0.4, markersize=3, color="C1"
    )
axes[2].plot(
    range(n_res_bins),
    kappa_res,
    "ks-",
    linewidth=2,
    markersize=4,
    label="Res-only",
)
axes[2].set_xlabel("Resolution bin")
axes[2].set_ylabel("kappa")
axes[2].set_title("Kappa vs res bin\n(dots = per azi sector)")
axes[2].legend()

plt.tight_layout()
plt.savefig("/Users/luis/integrator/scripts/conc_2d_kappa_stats.png", dpi=150)
plt.close()
print("Saved conc_2d_kappa_stats.png")


# ── Plot 4: Binning diagram on detector ─────────────────────────────────────

dx_arr = dx.numpy()
dy_arr = dy.numpy()
r_max = float(r_det.max()) * 1.05
azi_edges_np = np.linspace(-np.pi, np.pi, max_azi_bins + 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Left: color by profile group label
ax = axes[0]
ax.scatter(
    dx_arr,
    dy_arr,
    c=group_labels_2d.numpy(),
    cmap="nipy_spectral",
    s=0.3,
    alpha=0.4,
    rasterized=True,
)
for edge in azi_edges_np:
    ax.plot(
        [0, r_max * np.cos(edge)],
        [0, r_max * np.sin(edge)],
        "k-",
        linewidth=0.3,
        alpha=0.3,
    )
ax.plot(0, 0, "r+", markersize=10, markeredgewidth=2)
ax.set_aspect("equal")
ax.set_title(f"Profile bins ({n_total_bins} total)")
ax.set_xlabel("dx (px from beam center)")
ax.set_ylabel("dy (px from beam center)")

# Center: color by resolution bin
ax = axes[1]
sc = ax.scatter(
    dx_arr,
    dy_arr,
    c=res_labels.numpy(),
    cmap="viridis",
    s=0.3,
    alpha=0.4,
    rasterized=True,
)
for edge in azi_edges_np:
    ax.plot(
        [0, r_max * np.cos(edge)],
        [0, r_max * np.sin(edge)],
        "k-",
        linewidth=0.3,
        alpha=0.3,
    )
ax.plot(0, 0, "r+", markersize=10, markeredgewidth=2)
ax.set_aspect("equal")
ax.set_title(f"Resolution bins ({n_res_bins} shells)")
ax.set_xlabel("dx (px from beam center)")
fig.colorbar(sc, ax=ax, label="resolution bin")

# Right: azi bins per shell bar chart
ax = axes[2]
ax.bar(
    range(n_res_bins), azi_bins_per_shell, color="steelblue", edgecolor="white"
)
ax.axhline(
    max_azi_bins,
    color="red",
    linestyle="--",
    linewidth=1,
    label=f"max={max_azi_bins}",
)
ax.set_xlabel("Resolution shell")
ax.set_ylabel("Azimuthal bins")
ax.set_title("Adaptive azi bins per shell")
ax.legend()

fig.suptitle(
    f"Adaptive profile binning: {n_res_bins} res, max {max_azi_bins} azi, "
    f"{n_total_bins} total bins\n"
    f"beam center = ({beam_x:.1f}, {beam_y:.1f}) px, min_per_bin = {min_per_bin}",
    fontsize=11,
)
plt.tight_layout()
plt.savefig("/Users/luis/integrator/scripts/profile_binning.png", dpi=150)
plt.close()
print("Saved profile_binning.png")


# ── Summary ──────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\nResolution-only ({n_res_bins} bins):")
print(
    f"  kappa: min={min(kappa_res):.0f}, max={max(kappa_res):.0f}, "
    f"median={np.median(kappa_res):.0f}"
)
print(f"  sum(alpha): min={sum_res.min():.0f}, max={sum_res.max():.0f}")

print(
    f"\nAdaptive 2D ({n_total_bins} bins, max {max_azi_bins} azi, min_per_bin={min_per_bin}):"
)
print(
    f"  kappa: min={min(valid_kappa_2d):.0f}, max={max(valid_kappa_2d):.0f}, "
    f"median={np.median(valid_kappa_2d):.0f}"
)
print(
    f"  sum(alpha): min={sum_2d[sum_2d > 1].min():.0f}, max={sum_2d.max():.0f}"
)
print(f"  azi bins per shell: {azi_bins_per_shell}")
