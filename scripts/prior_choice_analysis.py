"""
Analyze how prior choices on I and bg affect the ELBO landscape.

Key insight: tighten bg (well-constrained physically), leave I loose.
The profile p + tight bg prior is what separates signal from background.

Produces:
  1. Prior density plots: current vs recommended
  2. KL penalty landscape: how different (I, bg) posteriors are penalized
  3. Training comparison: loose bg vs tight bg priors
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import (
    Exponential,
    Gamma,
    HalfNormal,
    LogNormal,
)

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# ═══════════════════════════════════════════════════════════════════════════
#  Panel 1: Prior densities — current vs recommended
# ═══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Intensity priors ---
ax = axes[0]
x_I = np.linspace(0.1, 5000, 2000)
x_I_t = torch.tensor(x_I, dtype=torch.float32)

priors_I = {
    "Gamma(1, 0.5)\nmean=2 [current 2d]": Gamma(
        torch.tensor(1.0), torch.tensor(0.5)
    ),
    "Exp(0.001)\nmean=1000 [current sim]": Exponential(torch.tensor(0.001)),
    "Gamma(2, 0.002)\nmean=1000": Gamma(
        torch.tensor(2.0), torch.tensor(0.002)
    ),
    "LogNormal(5, 2)\nmedian=148": LogNormal(
        torch.tensor(5.0), torch.tensor(2.0)
    ),
    "LogNormal(6.9, 1.5)\nmedian=992": LogNormal(
        torch.tensor(6.9), torch.tensor(1.5)
    ),
}

for label, p in priors_I.items():
    log_prob = p.log_prob(x_I_t)
    prob = torch.exp(log_prob).numpy()
    prob = np.where(np.isfinite(prob), prob, 0)
    ax.plot(x_I, prob, linewidth=2, label=label)

ax.set_xlim(0, 5000)
ax.set_xlabel("Intensity I")
ax.set_ylabel("p(I)")
ax.set_title("Intensity priors")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Background priors ---
ax = axes[1]
x_bg = np.linspace(0.01, 30, 500)
x_bg_t = torch.tensor(x_bg, dtype=torch.float32)

priors_bg = {
    "Gamma(1, 0.5)\nmean=2 [current]": Gamma(
        torch.tensor(1.0), torch.tensor(0.5)
    ),
    "Exp(1.0)\nmean=1 [current sim]": Exponential(torch.tensor(1.0)),
    "Gamma(3, 0.5)\nmean=6, tight": Gamma(
        torch.tensor(3.0), torch.tensor(0.5)
    ),
    "Gamma(5, 1.0)\nmean=5, tighter": Gamma(
        torch.tensor(5.0), torch.tensor(1.0)
    ),
    "LogNormal(1, 0.5)\nmedian=2.7": LogNormal(
        torch.tensor(1.0), torch.tensor(0.5)
    ),
    "HalfNormal(3)\nmode=0, sd=3": HalfNormal(torch.tensor(3.0)),
}

for label, p in priors_bg.items():
    log_prob = p.log_prob(x_bg_t)
    prob = torch.exp(log_prob).numpy()
    prob = np.where(np.isfinite(prob), prob, 0)
    ax.plot(x_bg, prob, linewidth=2, label=label)

ax.set_xlim(0, 30)
ax.set_xlabel("Background bg (per pixel)")
ax.set_ylabel("p(bg)")
ax.set_title("Background priors")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle(
    "Prior density comparison: Intensity vs Background", fontweight="bold"
)
fig.tight_layout()
fig.savefig("scripts/prior_densities.png", bbox_inches="tight")
print("Saved: scripts/prior_densities.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Panel 2: KL penalty contours in (I_posterior_mean, bg_posterior_mean) space
# ═══════════════════════════════════════════════════════════════════════════


def kl_lognormal(mu_q, sig_q, mu_p, sig_p):
    """Analytic KL(LogNormal(mu_q, sig_q) || LogNormal(mu_p, sig_p))."""
    return (
        np.log(sig_p / sig_q)
        + (sig_q**2 + (mu_q - mu_p) ** 2) / (2 * sig_p**2)
        - 0.5
    )


def kl_gamma(k_q, r_q, k_p, r_p):
    """Analytic KL(Gamma(k_q, r_q) || Gamma(k_p, r_p))."""
    return (
        (k_q - k_p) * digamma(k_q)
        - gammaln(k_q)
        + gammaln(k_p)
        + k_p * (np.log(r_q) - np.log(r_p))
        + k_q * (r_p - r_q) / r_q
    )


# Grid of posterior means
I_means = np.linspace(10, 3000, 100)
bg_means = np.linspace(0.5, 20, 100)
I_grid, bg_grid = np.meshgrid(I_means, bg_means)

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

prior_configs = [
    {
        "title": "Current: Gamma(1,0.5) for both\n(identical priors!)",
        "kl_I": lambda I: kl_gamma(k_q=2, r_q=2 / I, k_p=1, r_p=0.5),
        "kl_bg": lambda bg: kl_gamma(k_q=2, r_q=2 / bg, k_p=1, r_p=0.5),
    },
    {
        "title": "Loose I + Tight bg:\nGamma(2,0.002) + Gamma(5,1)",
        "kl_I": lambda I: kl_gamma(k_q=2, r_q=2 / I, k_p=2, r_p=0.002),
        "kl_bg": lambda bg: kl_gamma(k_q=5, r_q=5 / bg, k_p=5, r_p=1.0),
    },
    {
        "title": "LogNormal priors:\nLN(6.9,1.5) + LN(1,0.5)",
        "kl_I": lambda I: kl_lognormal(
            mu_q=np.log(I) - 0.5 * 0.5**2, sig_q=0.5, mu_p=6.9, sig_p=1.5
        ),
        "kl_bg": lambda bg: kl_lognormal(
            mu_q=np.log(bg) - 0.5 * 0.3**2, sig_q=0.3, mu_p=1.0, sig_p=0.5
        ),
    },
]

for ax, cfg in zip(axes2, prior_configs, strict=False):
    kl_total = cfg["kl_I"](I_grid) + cfg["kl_bg"](bg_grid)
    kl_total = np.clip(kl_total, 0, 50)  # clip for visualization

    im = ax.contourf(I_grid, bg_grid, kl_total, levels=20, cmap="RdYlBu_r")
    plt.colorbar(im, ax=ax, label="KL(q||p) [nats]")
    ax.set_xlabel("Posterior mean intensity I")
    ax.set_ylabel("Posterior mean background bg")
    ax.set_title(cfg["title"], fontsize=10)

    # Mark the "sweet spot" — minimum KL
    idx = np.unravel_index(kl_total.argmin(), kl_total.shape)
    ax.plot(
        I_grid[idx], bg_grid[idx], "w*", markersize=15, markeredgecolor="k"
    )

fig2.suptitle(
    "KL penalty landscape: where does the prior push (I, bg)?",
    fontweight="bold",
)
fig2.tight_layout()
fig2.savefig("scripts/prior_kl_landscape.png", bbox_inches="tight")
print("Saved: scripts/prior_kl_landscape.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Panel 3: What happens at the observation level
# ═══════════════════════════════════════════════════════════════════════════

fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))

# Simulate a single shoebox observation
np.random.seed(42)
H, W = 21, 21
true_I = 500.0
true_bg = 3.0

# Make a peaked profile
y, x = np.mgrid[:H, :W]
cy, cx = H // 2, W // 2
profile = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 3**2))
profile = profile / profile.sum()

true_rate = true_I * profile + true_bg
counts = np.random.poisson(true_rate)

# Show true decomposition
ax = axes3[0]
ax.set_title("True decomposition")
ax.imshow(true_rate, cmap="viridis")
ax.text(
    1,
    1,
    f"I={true_I:.0f}, bg={true_bg:.1f}",
    color="white",
    fontsize=9,
    bbox=dict(facecolor="black", alpha=0.5),
)
ax.set_xlabel("pixel")
ax.set_ylabel("pixel")

# Show what "high bg, low I" looks like
ax = axes3[1]
wrong_I = 200.0
wrong_bg = (
    true_I * profile.max() + true_bg - wrong_I * profile.max()
)  # match peak
wrong_rate = wrong_I * profile + wrong_bg
ax.set_title(f"Degenerate: I={wrong_I:.0f}, bg={wrong_bg:.1f}")
ax.imshow(wrong_rate, cmap="viridis")
residual = np.abs(true_rate - wrong_rate).mean()
ax.text(
    1,
    1,
    f"Mean |residual| = {residual:.1f}",
    color="white",
    fontsize=9,
    bbox=dict(facecolor="black", alpha=0.5),
)
ax.set_xlabel("pixel")

# Show the difference
ax = axes3[2]
diff = wrong_rate - true_rate
im = ax.imshow(diff, cmap="RdBu_r", vmin=-5, vmax=5)
ax.set_title("Difference (degenerate - true)")
plt.colorbar(im, ax=ax, label="Δ rate")
ax.set_xlabel("pixel")
ax.text(
    1,
    1,
    "bg absorbs I → flat offset",
    color="black",
    fontsize=9,
    bbox=dict(facecolor="white", alpha=0.7),
)

fig3.suptitle(
    "The I/bg identifiability problem: rate = I·p + bg",
    fontweight="bold",
)
fig3.tight_layout()
fig3.savefig("scripts/prior_identifiability.png", bbox_inches="tight")
print("Saved: scripts/prior_identifiability.png")


# ═══════════════════════════════════════════════════════════════════════════
#  Summary table
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PRIOR RECOMMENDATIONS")
print(f"{'=' * 70}")

print("""
BACKGROUND (bg) — tighten this one:
  Physical constraint: per-pixel rate, typically 0-20 counts/pixel
  Current:     Gamma(1, 0.5) mean=2, var=4     — too loose, mode at 0
  Recommended: Gamma(5, 1.0) mean=5, var=5     — informative, mode=4
  Alternative: LogNormal(1, 0.5) median=2.7    — if using LogNormal surrogate
  Alternative: HalfNormal(3)                   — simple, mode=0 but tight

INTENSITY (I) — keep loose, match Wilson statistics:
  Physical: heavy-tailed, range 0 to ~10^6
  Current:     Gamma(1, 0.5) mean=2            — WAY too tight! kills signal
  Recommended: Gamma(2, 0.002) mean=1000       — broad, positive mode
  Alternative: LogNormal(6.9, 1.5) median=992  — if using LogNormal surrogate
  Alternative: Exponential(0.001) mean=1000    — simple, Wilson-like

PROFILE (p) — already good with the scaling fix:
  The Dirichlet with loaded concentration file + sum=K scaling is correct.

KEY INSIGHT:
  The asymmetry matters. bg is physically constrained (small, per-pixel).
  I spans orders of magnitude. Make the priors reflect this asymmetry.
  With tight bg and loose I, the model MUST learn the correct decomposition.
""")

# Quick KL comparison at typical operating point
print(f"{'=' * 70}")
print("KL at typical posterior (I≈500, bg≈3):")
print(f"{'=' * 70}")


configs = [
    (
        "Current: Gamma(1,0.5)/Gamma(1,0.5)",
        kl_gamma(2, 2 / 500, 1, 0.5),
        kl_gamma(2, 2 / 3, 1, 0.5),
    ),
    (
        "Loose I + Tight bg: Gamma(2,0.002)/Gamma(5,1)",
        kl_gamma(2, 2 / 500, 2, 0.002),
        kl_gamma(5, 5 / 3, 5, 1.0),
    ),
    (
        "LogNormal: LN(6.9,1.5)/LN(1,0.5)",
        kl_lognormal(np.log(500) - 0.125, 0.5, 6.9, 1.5),
        kl_lognormal(np.log(3) - 0.045, 0.3, 1.0, 0.5),
    ),
]

for label, kl_i, kl_bg in configs:
    print(f"  {label}")
    print(f"    KL_I  = {kl_i:>8.2f} nats")
    print(f"    KL_bg = {kl_bg:>8.2f} nats")
    print(f"    Total = {kl_i + kl_bg:>8.2f} nats")
    print()
