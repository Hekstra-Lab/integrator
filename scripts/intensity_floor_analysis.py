"""Analysis of the intensity floor: KL penalties and gradient vanishing.

Produces publication-quality plots showing:
1. Gamma prior PDF/log-PDF for different α values
2. KL cost as a function of posterior mean for different prior α
3. Digamma penalty breakdown
4. Poisson gradient SNR as a function of I/bg ratio
5. Gradient magnitude simulation with realistic shoeboxes

Usage:
    uv run python scripts/intensity_floor_analysis.py [--out plots/]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma, gammaln


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("."))
    p.add_argument("--tau", type=float, default=0.1, help="Prior rate (1/E[I])")
    p.add_argument("--bg", type=float, default=12.0, help="Background counts/pixel")
    return p.parse_args()


def gamma_kl(k_q, r_q, k_p, r_p):
    """KL(Gamma(k_q, r_q) || Gamma(k_p, r_p)) in natural parameterization."""
    return (
        (k_q - k_p) * digamma(k_q)
        - gammaln(k_q)
        + gammaln(k_p)
        + k_p * (np.log(r_q) - np.log(r_p))
        + k_q * (r_p / r_q - 1.0)
    )


def optimal_kl_for_mean(target_mean, alpha_prior, tau):
    """Find minimum KL for a Gamma posterior with a given mean, against Gamma(α, α·τ) prior.

    Scans over k_q; for each k_q, r_q = k_q / target_mean.
    """
    k_vals = np.geomspace(0.01, 200, 2000)
    r_vals = k_vals / target_mean
    r_p = alpha_prior * tau
    kls = gamma_kl(k_vals, r_vals, alpha_prior, r_p)
    valid = np.isfinite(kls)
    if not valid.any():
        return np.nan
    return np.min(kls[valid])


# ──────────────────────────────────────────────
# Plot 1: Prior PDF and log-PDF
# ──────────────────────────────────────────────
def plot_prior_densities(tau, out_dir):
    alphas = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    x = np.linspace(1e-3, 5.0 / tau, 1000)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for alpha in alphas:
        beta = alpha * tau
        log_pdf = (alpha - 1) * np.log(x) - beta * x + alpha * np.log(beta) - gammaln(alpha)
        pdf = np.exp(log_pdf)
        ax.plot(x, pdf, linewidth=1.5, label=f"α = {alpha}")
    ax.set_xlabel("Intensity I")
    ax.set_ylabel("p(I)")
    ax.set_title(f"Gamma(α, α·τ) prior PDF  (τ = {tau}, E[I] = {1/tau:.0f})")
    ax.set_xlim(0, 3.0 / tau)
    ax.set_ylim(0, None)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for alpha in alphas:
        beta = alpha * tau
        log_pdf = (alpha - 1) * np.log(x) - beta * x + alpha * np.log(beta) - gammaln(alpha)
        ax.plot(x, log_pdf, linewidth=1.5, label=f"α = {alpha}")
    ax.set_xlabel("Intensity I")
    ax.set_ylabel("log p(I)")
    ax.set_title("Log-density (note divergence at I->0 for α < 1)")
    ax.set_xlim(0, 3.0 / tau)
    ax.set_ylim(-15, None)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fname = out_dir / "prior_densities.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"wrote {fname}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Plot 2: KL cost vs posterior mean
# ──────────────────────────────────────────────
def plot_kl_vs_mean(tau, out_dir):
    alphas = [0.3, 0.5, 0.7, 1.0, 1.5]
    prior_mean = 1.0 / tau
    target_means = np.geomspace(0.1, prior_mean * 5, 200)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for alpha in alphas:
        kls = [optimal_kl_for_mean(m, alpha, tau) for m in target_means]
        axes[0].plot(target_means, kls, linewidth=1.5, label=f"α = {alpha}")
        axes[1].plot(
            target_means / prior_mean, kls, linewidth=1.5, label=f"α = {alpha}"
        )

    axes[0].set_xlabel("Posterior mean qi_mean")
    axes[0].set_ylabel("min KL(q || p)")
    axes[0].set_title(
        f"KL cost of placing posterior at a given mean\n"
        f"Prior: Gamma(α, α·τ), τ={tau}, E[I]={prior_mean:.0f}"
    )
    axes[0].set_xscale("log")
    axes[0].axvline(prior_mean, color="gray", linestyle="--", alpha=0.5, label=f"E[I] = {prior_mean:.0f}")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Posterior mean / Prior mean")
    axes[1].set_ylabel("min KL(q || p)")
    axes[1].set_title("Same, normalized by prior mean")
    axes[1].set_xscale("log")
    axes[1].axvline(1.0, color="gray", linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fname = out_dir / "kl_vs_posterior_mean.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"wrote {fname}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Plot 3: Digamma penalty term
# ──────────────────────────────────────────────
def plot_digamma_penalty(out_dir):
    k = np.geomspace(0.01, 10, 500)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(k, digamma(k), "b-", linewidth=2)
    ax.set_xlabel("k (posterior concentration)")
    ax.set_ylabel("ψ(k)")
    ax.set_title("Digamma function ψ(k)")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for alpha in [0.3, 0.5, 0.7, 1.0, 1.5]:
        penalty = (k - alpha) * digamma(k)
        ax.plot(k, penalty, linewidth=1.5, label=f"α = {alpha}")
    ax.set_xlabel("k (posterior concentration)")
    ax.set_ylabel("(k − α)·ψ(k)")
    ax.set_title(
        "Digamma penalty in KL\n"
        "Diverges as k->0; smaller α reduces the penalty"
    )
    ax.set_xscale("log")
    ax.set_ylim(-10, 10)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fname = out_dir / "digamma_penalty.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"wrote {fname}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Plot 4: Poisson gradient SNR
# ──────────────────────────────────────────────
def plot_gradient_snr(bg, out_dir):
    n_pixels = 625  # 25x25 shoebox
    profile = np.ones(n_pixels) / n_pixels  # uniform for simplicity

    I_values = np.geomspace(0.1, 1000, 200)

    grad_signal = np.zeros_like(I_values)
    grad_noise = np.zeros_like(I_values)

    for i, I in enumerate(I_values):
        rate = I * profile + bg
        # Expected gradient: sigma p_j * (1 - E[counts]/rate) = sigma p_j * (1 - 1) = 0
        # So the "signal" is the curvature times I:
        # ∂²NLL/∂I² = sigma p_j² * counts / rate²  ≈ sigma p_j² / rate
        fisher = np.sum(profile**2 / rate)
        grad_signal[i] = fisher * I

        # Gradient noise std: sqrt(sigma p_j² * counts / rate²) ≈ sqrt(sigma p_j² / rate)
        grad_noise[i] = np.sqrt(np.sum(profile**2 * rate / rate**2))

    snr = grad_signal / grad_noise

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.plot(I_values, grad_signal, "b-", linewidth=2, label="Signal (Fisher·I)")
    ax.plot(I_values, grad_noise, "r-", linewidth=2, label="Noise (√Fisher)")
    ax.set_xlabel("True intensity I")
    ax.set_ylabel("Gradient magnitude")
    ax.set_title(f"∂NLL/∂I: signal vs noise  (bg = {bg})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(I_values, snr, "k-", linewidth=2)
    ax.set_xlabel("True intensity I")
    ax.set_ylabel("Gradient SNR")
    ax.set_title("Gradient SNR ∝ I/bg at low I")
    ax.set_xscale("log")
    ax.axhline(1, color="red", linestyle="--", alpha=0.5, label="SNR = 1")
    ax.axvline(bg, color="gray", linestyle="--", alpha=0.5, label=f"I = bg = {bg}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(I_values / bg, snr, "k-", linewidth=2)
    ax.set_xlabel("I / bg")
    ax.set_ylabel("Gradient SNR")
    ax.set_title("Gradient SNR vs signal-to-background ratio")
    ax.set_xscale("log")
    ax.axhline(1, color="red", linestyle="--", alpha=0.5, label="SNR = 1")
    ax.axvline(1, color="gray", linestyle="--", alpha=0.5, label="I = bg")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fname = out_dir / "gradient_snr.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"wrote {fname}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Plot 5: Simulated gradient histograms
# ──────────────────────────────────────────────
def plot_gradient_simulation(bg, tau, out_dir):
    n_pixels = 625
    profile = np.exp(-0.5 * ((np.arange(n_pixels) - n_pixels // 2) / 5) ** 2)
    profile = profile / profile.sum()
    n_sims = 5000
    rng = np.random.default_rng(42)

    I_test = [0.1, 1.0, 10.0, 50.0, 200.0]
    prior_mean = 1.0 / tau

    fig, axes = plt.subplots(1, len(I_test), figsize=(4 * len(I_test), 4), sharey=True)

    for idx, I_true in enumerate(I_test):
        rate = I_true * profile + bg
        grads = np.zeros(n_sims)
        for s in range(n_sims):
            counts = rng.poisson(rate)
            grad = np.sum(profile * (1.0 - counts / rate))
            grads[s] = grad

        ax = axes[idx]
        ax.hist(grads, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="none")
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.axvline(grads.mean(), color="black", linestyle="-", linewidth=1.5,
                   label=f"mean={grads.mean():.4f}")
        ax.set_xlabel("∂NLL/∂I")
        ax.set_title(f"I = {I_true}\n(I/bg = {I_true/bg:.2f})")
        ax.legend(fontsize=7, loc="upper right")

    axes[0].set_ylabel("Density")
    fig.suptitle(
        f"Distribution of ∂NLL/∂I from Poisson sampling\n"
        f"bg = {bg}, E[I] = {prior_mean:.0f}, 25*25 shoebox",
        fontsize=12,
    )
    fig.tight_layout()
    fname = out_dir / "gradient_histograms.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"wrote {fname}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Plot 6: Combined ELBO gradient (NLL + KL)
# ──────────────────────────────────────────────
def plot_elbo_gradient(bg, tau, out_dir):
    """Show total ∂ELBO/∂(qi_mean) = ∂NLL/∂I + ∂KL/∂I for different α."""
    n_pixels = 625
    profile = np.ones(n_pixels) / n_pixels
    alphas = [0.3, 0.5, 0.7, 1.0]

    I_values = np.geomspace(0.1, 500, 300)
    prior_mean = 1.0 / tau

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for alpha in alphas:
        nll_grad = np.zeros_like(I_values)
        kl_grad = np.zeros_like(I_values)

        for i, I in enumerate(I_values):
            rate = I * profile + bg
            # NLL gradient (expected)
            nll_grad[i] = np.sum(profile * (1.0 - (I * profile + bg) / rate))
            # For Gamma(k,r) posterior with mean=I and k=1 (simplification):
            # KL gradient w.r.t. mean ≈ α·τ - α/I (from ∂KL/∂(1/r) · ∂(1/r)/∂mean)
            k_q = 1.0
            r_q = k_q / max(I, 1e-6)
            r_p = alpha * tau
            kl_grad[i] = alpha * tau - alpha / max(I, 1e-6)

        total = nll_grad + kl_grad
        axes[0].plot(I_values, kl_grad, linewidth=1.5, label=f"α = {alpha}")
        axes[1].plot(I_values, total, linewidth=1.5, label=f"α = {alpha}")

    axes[0].set_xlabel("qi_mean")
    axes[0].set_ylabel("∂KL/∂I")
    axes[0].set_title(f"KL gradient pushes toward prior mean = {prior_mean:.0f}")
    axes[0].set_xscale("log")
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[0].axvline(prior_mean, color="gray", linestyle=":", alpha=0.5, label=f"E[I]={prior_mean:.0f}")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("qi_mean")
    axes[1].set_ylabel("∂ELBO/∂I  (NLL + KL)")
    axes[1].set_title(
        f"Total gradient: equilibrium = intensity floor\n"
        f"bg = {bg}, smaller α -> floor closer to 0"
    )
    axes[1].set_xscale("log")
    axes[1].axhline(0, color="red", linestyle="--", alpha=0.5, label="equilibrium")
    axes[1].axvline(prior_mean, color="gray", linestyle=":", alpha=0.5)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fname = out_dir / "elbo_gradient.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"wrote {fname}")
    plt.close(fig)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    plot_prior_densities(args.tau, args.out)
    plot_kl_vs_mean(args.tau, args.out)
    plot_digamma_penalty(args.out)
    plot_gradient_snr(args.bg, args.out)
    plot_gradient_simulation(args.bg, args.tau, args.out)
    plot_elbo_gradient(args.bg, args.tau, args.out)


if __name__ == "__main__":
    main()
