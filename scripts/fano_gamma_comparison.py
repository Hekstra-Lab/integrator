"""
FanoGamma vs Gamma Comparison
=============================
Empirical test showing that FanoGamma (multiply by fano in rsample)
produces identical results to Gamma(k, rate=1/fano) (divide by rate).

Generates: fano_gamma_comparison.png
"""

# Import FanoGamma from the codebase
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Gamma, kl_divergence

sys.path.insert(0, "src")
from integrator.model.distributions.gamma import FanoGamma

torch.manual_seed(42)

K_MIN = 0.1
EPS = 1e-6


def _bound_k(raw_k, k_max, k_min):
    if k_max is not None:
        return k_min + (k_max - k_min) * torch.sigmoid(raw_k)
    return F.softplus(raw_k) + k_min


# ─── Test 1: Gradient scatter ────────────────────────────────────────────────


def gradient_scatter(ax_fano, ax_k):
    """Scatter plot: FanoGamma grads vs Gamma grads for many operating points."""
    ks = [0.5, 1.0, 5.0, 10.0, 50.0]
    fanos = [0.1, 0.5, 1.0, 5.0, 10.0]

    gamma_grads_fano = []
    fano_grads_fano = []
    gamma_grads_k = []
    fano_grads_k = []

    for k_val in ks:
        for f_val in fanos:
            for trial in range(200):
                seed = trial + int(k_val * 100) + int(f_val * 1000)

                # Gamma path
                k1 = torch.tensor(k_val, requires_grad=True)
                fano1 = torch.tensor(f_val, requires_grad=True)
                rate1 = 1.0 / fano1
                g1 = Gamma(k1, rate1)
                torch.manual_seed(seed)
                s1 = g1.rsample()
                s1.backward()

                # FanoGamma path
                k2 = torch.tensor(k_val, requires_grad=True)
                fano2 = torch.tensor(f_val, requires_grad=True)
                g2 = FanoGamma(k2, fano2)
                torch.manual_seed(seed)
                s2 = g2.rsample()
                s2.backward()

                if (
                    fano1.grad is not None
                    and fano2.grad is not None
                    and not torch.isnan(fano1.grad)
                    and not torch.isnan(fano2.grad)
                ):
                    gamma_grads_fano.append(fano1.grad.item())
                    fano_grads_fano.append(fano2.grad.item())

                if (
                    k1.grad is not None
                    and k2.grad is not None
                    and not torch.isnan(k1.grad)
                    and not torch.isnan(k2.grad)
                ):
                    gamma_grads_k.append(k1.grad.item())
                    fano_grads_k.append(k2.grad.item())

    gamma_grads_fano = np.array(gamma_grads_fano)
    fano_grads_fano = np.array(fano_grads_fano)
    gamma_grads_k = np.array(gamma_grads_k)
    fano_grads_k = np.array(fano_grads_k)

    # Fano gradient scatter
    ax_fano.scatter(gamma_grads_fano, fano_grads_fano, s=1, alpha=0.3, c="C0")
    lim = max(abs(gamma_grads_fano).max(), abs(fano_grads_fano).max()) * 1.1
    ax_fano.plot([-lim, lim], [-lim, lim], "k--", lw=0.5, label="y = x")
    ax_fano.set_xlabel("Gamma grad(fano)")
    ax_fano.set_ylabel("FanoGamma grad(fano)")
    ax_fano.set_title("Fano gradients")
    ax_fano.set_aspect("equal")
    max_diff = np.abs(gamma_grads_fano - fano_grads_fano).max()
    ax_fano.text(
        0.05,
        0.95,
        f"max |diff| = {max_diff:.2e}",
        transform=ax_fano.transAxes,
        va="top",
        fontsize=8,
    )

    # k gradient scatter
    ax_k.scatter(gamma_grads_k, fano_grads_k, s=1, alpha=0.3, c="C1")
    lim = max(abs(gamma_grads_k).max(), abs(fano_grads_k).max()) * 1.1
    ax_k.plot([-lim, lim], [-lim, lim], "k--", lw=0.5, label="y = x")
    ax_k.set_xlabel("Gamma grad(k)")
    ax_k.set_ylabel("FanoGamma grad(k)")
    ax_k.set_title("Concentration gradients")
    ax_k.set_aspect("equal")
    max_diff = np.abs(gamma_grads_k - fano_grads_k).max()
    ax_k.text(
        0.05,
        0.95,
        f"max |diff| = {max_diff:.2e}",
        transform=ax_k.transAxes,
        va="top",
        fontsize=8,
    )


# ─── Test 2: ELBO convergence overlay ────────────────────────────────────────


def convergence_overlay(
    ax_loss, ax_means, repam_name, repam_fn_gamma, repam_fn_fano
):
    """Run optimization with both Gamma and FanoGamma, overlay trajectories."""
    prior = Gamma(torch.tensor(1.0), torch.tensor(0.001))
    ys = torch.tensor([5.0, 50.0, 500.0])
    n_refl = len(ys)
    n_steps = 3000
    n_mc = 64

    results = {}

    for label, fn in [("Gamma", repam_fn_gamma), ("FanoGamma", repam_fn_fano)]:
        torch.manual_seed(42)
        raw_p1s = torch.zeros(n_refl, requires_grad=True)
        raw_p2s = torch.zeros(n_refl, requires_grad=True)
        optimizer = torch.optim.Adam([raw_p1s, raw_p2s], lr=0.01)

        loss_hist = []
        mean_hist = [[], [], []]

        for step in range(n_steps):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0)
            for i in range(n_refl):
                k, r, dist = fn(raw_p1s[i], raw_p2s[i])
                mean_hist[i].append((k / r).item())
                I_samples = dist.rsample((n_mc,))
                nll = -(
                    ys[i] * torch.log(I_samples + 1e-10) - I_samples
                ).mean()
                kl = kl_divergence(dist, prior)
                total_loss = total_loss + nll + kl
            total_loss.backward()
            if raw_p1s.grad is not None:
                raw_p1s.grad[torch.isnan(raw_p1s.grad)] = 0.0
            if raw_p2s.grad is not None:
                raw_p2s.grad[torch.isnan(raw_p2s.grad)] = 0.0
            optimizer.step()
            loss_hist.append(total_loss.item())

        results[label] = {"loss": loss_hist, "means": mean_hist}

    # Plot loss
    ax_loss.plot(results["Gamma"]["loss"], label="Gamma", lw=1.5, alpha=0.8)
    ax_loss.plot(
        results["FanoGamma"]["loss"],
        label="FanoGamma",
        ls="--",
        lw=1.5,
        alpha=0.8,
    )
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("ELBO loss")
    ax_loss.set_title(f"Repam {repam_name}: Loss")
    ax_loss.legend(fontsize=8)

    # Compute max loss difference
    loss_diff = np.abs(
        np.array(results["Gamma"]["loss"])
        - np.array(results["FanoGamma"]["loss"])
    )
    ax_loss.text(
        0.05,
        0.05,
        f"max |loss diff| = {loss_diff.max():.2e}",
        transform=ax_loss.transAxes,
        va="bottom",
        fontsize=8,
    )

    # Plot means
    colors = ["C0", "C1", "C2"]
    targets = [5, 50, 500]
    for i in range(3):
        ax_means.plot(
            results["Gamma"]["means"][i],
            color=colors[i],
            lw=1.5,
            alpha=0.8,
            label=f"Gamma y={targets[i]}",
        )
        ax_means.plot(
            results["FanoGamma"]["means"][i],
            color=colors[i],
            ls="--",
            lw=1.5,
            alpha=0.8,
            label=f"FanoGamma y={targets[i]}",
        )
    ax_means.axhline(5, color="k", ls=":", lw=0.5)
    ax_means.axhline(50, color="k", ls=":", lw=0.5)
    ax_means.axhline(500, color="k", ls=":", lw=0.5)
    ax_means.set_xlabel("Step")
    ax_means.set_ylabel("Posterior mean")
    ax_means.set_title(f"Repam {repam_name}: Means")
    ax_means.set_yscale("log")
    ax_means.legend(fontsize=6, ncol=2)


# ─── Repam functions that return (k, r, dist) ────────────────────────────────


def repam_d_gamma(raw_k, raw_fano):
    k = _bound_k(raw_k, None, K_MIN)
    fano = F.softplus(raw_fano) + EPS
    r = 1.0 / fano
    dist = Gamma(k, r)
    return k, r, dist


def repam_d_fano(raw_k, raw_fano):
    k = _bound_k(raw_k, None, K_MIN)
    fano = F.softplus(raw_fano) + EPS
    r = 1.0 / fano  # still need r for mean computation
    dist = FanoGamma(k, fano)
    return k, r, dist


def repam_b_gamma(raw_mu, raw_fano):
    mu = F.softplus(raw_mu) + EPS
    fano = F.softplus(raw_fano) + EPS
    r = 1.0 / fano
    k = (mu * r).clamp(min=K_MIN)
    dist = Gamma(k, r)
    return k, r, dist


def repam_b_fano(raw_mu, raw_fano):
    mu = F.softplus(raw_mu) + EPS
    fano = F.softplus(raw_fano) + EPS
    k = (mu / fano).clamp(min=K_MIN)
    r = 1.0 / fano
    dist = FanoGamma(k, fano)
    return k, r, dist


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))

    # Row 1: Gradient scatter
    print("Computing gradient scatter (5000 pairs)...")
    gradient_scatter(axes[0, 0], axes[0, 1])

    # Row 2: Repam D convergence overlay
    print("Running Repam D convergence (2 x 3000 steps)...")
    convergence_overlay(
        axes[1, 0], axes[1, 1], "D", repam_d_gamma, repam_d_fano
    )

    # Row 3: Repam B convergence overlay
    print("Running Repam B convergence (2 x 3000 steps)...")
    convergence_overlay(
        axes[2, 0], axes[2, 1], "B", repam_b_gamma, repam_b_fano
    )

    fig.suptitle("FanoGamma vs Gamma: Identical Results", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("fano_gamma_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: fano_gamma_comparison.png")
