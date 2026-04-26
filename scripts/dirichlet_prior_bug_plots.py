"""
Visualise the Dirichlet prior-scaling bug and its effect on the ELBO.

Produces three panels:
  1. Prior concentration heatmaps: buggy (sum=1) vs fixed (sum=K)
  2. KL(q || p) as a function of posterior alpha, for both scalings
  3. Effective loss breakdown: NLL + w * KL_prf, showing how downweighting masked the bug
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Dirichlet

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

DATA_DIR = (
    "/Users/luis/master/notebooks/integrator_notes/code/simulating_shoeboxes"
)
conc_raw = torch.load(f"{DATA_DIR}/concentration.pt", weights_only=False)

K = conc_raw.numel()  # 441 for 21x21
H = W = int(math.sqrt(K))

# ── Build buggy vs fixed priors ──────────────────────────────────────────
conc_buggy = conc_raw.clone()
conc_buggy[conc_buggy > 2] *= 40
conc_buggy = conc_buggy / conc_buggy.sum()  # sum = 1  (BUG)

conc_fixed = conc_raw.clone()
conc_fixed[conc_fixed > 2] *= 40
conc_fixed = conc_fixed / conc_fixed.sum() * K  # sum = K (FIXED)

# ── Panel 1: Heatmaps ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

im0 = axes[0].imshow(conc_buggy.reshape(H, W).numpy(), cmap="viridis")
axes[0].set_title(f"Buggy prior (sum={conc_buggy.sum():.2f})")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(conc_fixed.reshape(H, W).numpy(), cmap="viridis")
axes[1].set_title(f"Fixed prior (sum={conc_fixed.sum():.0f})")
plt.colorbar(im1, ax=axes[1], shrink=0.8)

# ratio
ratio = conc_fixed / conc_buggy
im2 = axes[2].imshow(ratio.reshape(H, W).numpy(), cmap="RdBu_r")
axes[2].set_title(f"Scale factor (fixed / buggy = {K}x)")
plt.colorbar(im2, ax=axes[2], shrink=0.8)

for ax in axes:
    ax.set_xlabel("pixel w")
    ax.set_ylabel("pixel h")

fig.suptitle(
    "Dirichlet prior concentration: buggy vs fixed scaling", fontweight="bold"
)
fig.tight_layout()
fig.savefig("scripts/dirichlet_prior_heatmaps.png", bbox_inches="tight")
print("Saved: scripts/dirichlet_prior_heatmaps.png")

# ── Panel 2: KL divergence vs posterior concentration ─────────────────────
# Simulate a posterior q = Dirichlet(alpha_q) where all alpha_q are uniform
# and sweep the total concentration alpha_0 = sum(alpha_q)
alpha0_values = np.logspace(-1, 3, 80)
kl_buggy_list = []
kl_fixed_list = []

for a0 in alpha0_values:
    alpha_q = torch.full(
        (K,), a0 / K
    )  # uniform posterior with total conc = a0
    q = Dirichlet(alpha_q)

    p_buggy = Dirichlet(conc_buggy)
    p_fixed = Dirichlet(conc_fixed)

    kl_b = torch.distributions.kl.kl_divergence(q, p_buggy).item()
    kl_f = torch.distributions.kl.kl_divergence(q, p_fixed).item()

    kl_buggy_list.append(kl_b)
    kl_fixed_list.append(kl_f)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(
    alpha0_values, kl_buggy_list, "r-", linewidth=2, label="Buggy (sum=1)"
)
ax2.plot(
    alpha0_values, kl_fixed_list, "b-", linewidth=2, label="Fixed (sum=K)"
)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Posterior total concentration α₀ = Σαᵢ")
ax2.set_ylabel("KL(q || p)  [nats]")
ax2.set_title("Dirichlet KL divergence: buggy vs fixed prior")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Annotate the typical operating range
ax2.axvspan(
    100, 1000, alpha=0.1, color="green", label="Typical training range"
)
ax2.annotate(
    "Typical\ntraining\nrange",
    xy=(300, 10),
    fontsize=9,
    color="green",
    ha="center",
)

fig2.tight_layout()
fig2.savefig("scripts/dirichlet_kl_comparison.png", bbox_inches="tight")
print("Saved: scripts/dirichlet_kl_comparison.png")

# ── Panel 3: Effective loss breakdown at different weights ────────────────
# Show how weight masks the bug
# Assume NLL ≈ 700, KL_i ≈ 3, KL_bg ≈ 2 (typical values from training)
nll_typical = 700
kl_i_typical = 3.0
kl_bg_typical = 2.0

# KL_prf at a typical posterior (alpha_0 ≈ 500, uniform)
alpha_q_typical = torch.full((K,), 500.0 / K)
q_typical = Dirichlet(alpha_q_typical)
kl_prf_buggy = torch.distributions.kl.kl_divergence(
    q_typical, Dirichlet(conc_buggy)
).item()
kl_prf_fixed = torch.distributions.kl.kl_divergence(
    q_typical, Dirichlet(conc_fixed)
).item()

weights = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
x_pos = np.arange(len(weights))
width = 0.35

loss_buggy = []
loss_fixed = []
kl_prf_buggy_weighted = []
kl_prf_fixed_weighted = []

for w in weights:
    kl_b = w * kl_prf_buggy + kl_i_typical + kl_bg_typical
    kl_f = w * kl_prf_fixed + kl_i_typical + kl_bg_typical
    loss_buggy.append(nll_typical + kl_b)
    loss_fixed.append(nll_typical + kl_f)
    kl_prf_buggy_weighted.append(w * kl_prf_buggy)
    kl_prf_fixed_weighted.append(w * kl_prf_fixed)

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

# Left: total ELBO loss
ax3a.bar(
    x_pos - width / 2, loss_buggy, width, label="Buggy prior", color="salmon"
)
ax3a.bar(
    x_pos + width / 2,
    loss_fixed,
    width,
    label="Fixed prior",
    color="steelblue",
)
ax3a.set_xticks(x_pos)
ax3a.set_xticklabels([str(w) for w in weights])
ax3a.set_xlabel("KL profile weight")
ax3a.set_ylabel("Total loss (NLL + KL)")
ax3a.set_title("Total ELBO loss vs prior weight")
ax3a.legend()
ax3a.grid(True, alpha=0.3, axis="y")

# Add NLL reference line
ax3a.axhline(
    y=nll_typical, color="gray", linestyle="--", alpha=0.5, label="NLL only"
)

# Right: weighted KL_prf only
ax3b.bar(
    x_pos - width / 2,
    kl_prf_buggy_weighted,
    width,
    label="Buggy prior",
    color="salmon",
)
ax3b.bar(
    x_pos + width / 2,
    kl_prf_fixed_weighted,
    width,
    label="Fixed prior",
    color="steelblue",
)
ax3b.set_xticks(x_pos)
ax3b.set_xticklabels([str(w) for w in weights])
ax3b.set_xlabel("KL profile weight")
ax3b.set_ylabel("w × KL(q_prf || p_prf)  [nats]")
ax3b.set_title("Weighted profile KL contribution")
ax3b.legend()
ax3b.grid(True, alpha=0.3, axis="y")

# Annotate the key insight
ax3b.annotate(
    f"Buggy: KL_prf = {kl_prf_buggy:.0f} nats\nFixed: KL_prf = {kl_prf_fixed:.1f} nats",
    xy=(0.02, 0.95),
    xycoords="axes fraction",
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
)

fig3.suptitle(
    f"How downweighting masked the Dirichlet scaling bug\n"
    f"(NLL≈{nll_typical}, KL_i≈{kl_i_typical}, KL_bg≈{kl_bg_typical})",
    fontweight="bold",
)
fig3.tight_layout()
fig3.savefig("scripts/dirichlet_loss_breakdown.png", bbox_inches="tight")
print("Saved: scripts/dirichlet_loss_breakdown.png")

# ── Print summary ────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(
    f"Buggy prior:  sum(alpha) = {conc_buggy.sum():.4f},  max = {conc_buggy.max():.4f}"
)
print(
    f"Fixed prior:  sum(alpha) = {conc_fixed.sum():.0f},  max = {conc_fixed.max():.1f}"
)
print(f"Scale factor: {K}x")
print("\nAt uniform posterior (α₀=500):")
print(f"  KL buggy  = {kl_prf_buggy:.1f} nats")
print(f"  KL fixed  = {kl_prf_fixed:.1f} nats")
print("\nAt weight=0.005:")
print(f"  w*KL buggy = {0.005 * kl_prf_buggy:.2f}  (hidden!)")
print(f"  w*KL fixed = {0.005 * kl_prf_fixed:.4f}")
print("\nAt weight=1.0:")
print(f"  w*KL buggy = {kl_prf_buggy:.1f}  (dominates loss!)")
print(f"  w*KL fixed = {kl_prf_fixed:.1f}")
