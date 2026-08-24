"""Visualize the learned G(λ) spectrum and Wilson prior from a checkpoint.

Works for SpectralWilsonLoss (continuous) and PolyWilsonLoss (binned).

Usage (single checkpoint):
    uv run python scripts/visualize_spectrum.py <checkpoint.ckpt> [--out spectrum.png]
    uv run python scripts/visualize_spectrum.py <checkpoint.ckpt> --metadata /path/to/metadata.pt

Usage (epoch progression from checkpoint directory):
    uv run python scripts/visualize_spectrum.py --ckpt-dir /path/to/checkpoints/ [--out progression.png]
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize learned G(λ)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Path to .ckpt file")
    source.add_argument(
        "--ckpt-dir",
        type=Path,
        help="Directory of epoch*.ckpt files for progression plot",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Output image path"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to metadata.pt for Wilson prior τ(λ,d) plot",
    )
    return parser.parse_args()


def _chebyshev(x, degree):
    T = [torch.ones_like(x)]
    if degree >= 1:
        T.append(x)
    for k in range(2, degree + 1):
        T.append(2 * x * T[k - 1] - T[k - 2])
    return T


def _spectral_design_matrix(sd, lam):
    """Build Chebyshev design matrix. Returns (phi, degree)."""
    lam_mid = sd["loss.spectrum.lam_mid"]
    lam_scale = sd["loss.spectrum.lam_scale"]
    c = sd.get("loss.spectrum.c", sd.get("loss.spectrum.coeff_loc"))
    degree = c.shape[0] - 1
    x = (lam - lam_mid) / lam_scale
    phi = torch.stack(_chebyshev(x, degree), dim=-1)
    return phi, degree


def _get_lam_range(sd):
    """Return (lam_min, lam_max) for the x-axis."""
    if "loss.spectrum.lam_mid" in sd:
        lam_mid = float(sd["loss.spectrum.lam_mid"])
        lam_scale = float(sd["loss.spectrum.lam_scale"])
        return lam_mid - lam_scale, lam_mid + lam_scale
    elif "loss.wavelength_bin_edges" in sd:
        edges = sd["loss.wavelength_bin_edges"]
        return float(edges[0]), float(edges[-1])
    return 0.9, 1.3


def _compute_G_on_grid(sd, lam):
    """Compute G(λ) on a grid."""
    c = sd.get("loss.spectrum.c", sd.get("loss.spectrum.coeff_loc"))
    phi, _ = _spectral_design_matrix(sd, lam)
    log_G = phi @ c
    return torch.exp(log_G)


def plot_spectral(sd, ax_log, ax_G):
    """Plot Chebyshev spectrum (point estimate or variational)."""
    c = sd.get("loss.spectrum.c", sd.get("loss.spectrum.coeff_loc"))

    lam_min, lam_max = _get_lam_range(sd)
    lam = torch.linspace(lam_min, lam_max, 500)
    phi, degree = _spectral_design_matrix(sd, lam)

    log_G = phi @ c
    G = torch.exp(log_G)

    ax_log.plot(lam.numpy(), log_G.numpy(), "b-", linewidth=2)

    # Add uncertainty bands if variational (old checkpoints)
    if "loss.spectrum.coeff_log_scale" in sd:
        coeff_std = F.softplus(sd["loss.spectrum.coeff_log_scale"])
        log_G_std = (phi**2 @ coeff_std**2).sqrt()
        ax_log.fill_between(
            lam.numpy(),
            (log_G - log_G_std).numpy(),
            (log_G + log_G_std).numpy(),
            alpha=0.25,
            color="blue",
        )

    ax_log.set_ylabel("log G(λ)")
    ax_log.set_title(f"Learned spectrum (Chebyshev degree {degree})")

    ax_G.plot(lam.numpy(), G.numpy(), "r-", linewidth=2)
    ax_G.set_ylabel("G(λ)")
    ax_G.set_xlabel("Wavelength (Å)")
    ax_G.grid(alpha=0.3)


def plot_binned(sd, ax_log, ax_G):
    """Plot per-bin step function spectrum."""
    edges = sd["loss.wavelength_bin_edges"]
    loc = sd["loss.q_log_K_loc"]
    log_scale = sd["loss.q_log_K_log_scale"]
    std = F.softplus(log_scale)

    n_bins = loc.shape[0]

    log_G_mean = loc
    G_mean = torch.exp(loc)
    G_upper = torch.exp(loc + std)
    G_lower = torch.exp(loc - std)

    for i in range(n_bins):
        left, right = edges[i].item(), edges[i + 1].item()
        ax_log.plot([left, right], [log_G_mean[i]] * 2, "b-", linewidth=2)
        ax_log.fill_between(
            [left, right],
            (log_G_mean[i] - std[i]).item(),
            (log_G_mean[i] + std[i]).item(),
            alpha=0.2,
            color="blue",
        )

        ax_G.plot([left, right], [G_mean[i]] * 2, "r-", linewidth=2)
        ax_G.fill_between(
            [left, right],
            G_lower[i].item(),
            G_upper[i].item(),
            alpha=0.2,
            color="red",
        )

    for e in edges:
        ax_log.axvline(
            e.item(), color="gray", alpha=0.3, linewidth=0.5, linestyle="--"
        )
        ax_G.axvline(
            e.item(), color="gray", alpha=0.3, linewidth=0.5, linestyle="--"
        )

    ax_log.set_ylabel("log G(λ)")
    ax_log.set_title(f"Learned spectrum ({n_bins}-bin step function)")
    ax_G.set_ylabel("G(λ)")
    ax_G.set_xlabel("Wavelength (Å)")


def _get_wilson_ingredients(sd, metadata_path):
    """Extract B, G_grid, lam_grid, d_vals from checkpoint and metadata."""
    if "loss.raw_B" in sd:
        B = F.softplus(sd["loss.raw_B"]).item()
    else:
        B = F.softplus(sd["loss.q_log_B_loc"]).item()
    is_spectral = "loss.spectrum.c" in sd or "loss.spectrum.coeff_loc" in sd
    is_binned = "loss.wavelength_bin_edges" in sd

    meta = torch.load(metadata_path, map_location="cpu", weights_only=False)
    if not (isinstance(meta, dict) and "wavelength" in meta):
        return None
    wl_data = meta["wavelength"]
    d_data = meta.get("d", None)

    lam_min, lam_max = float(wl_data.min()), float(wl_data.max())
    lam_grid = torch.linspace(lam_min, lam_max, 200)

    if is_spectral:
        G_grid = _compute_G_on_grid(sd, lam_grid)
    elif is_binned:
        edges = sd["loss.wavelength_bin_edges"]
        loc = sd["loss.q_log_K_loc"]
        bins = torch.bucketize(lam_grid, edges, right=True) - 1
        bins = bins.clamp(0, loc.shape[0] - 1)
        G_grid = torch.exp(loc[bins])
    else:
        return None

    if d_data is not None:
        d_vals = [
            float(d_data.quantile(0.1)),
            float(d_data.quantile(0.5)),
            float(d_data.quantile(0.9)),
        ]
    else:
        d_vals = [2.0, 5.0, 15.0]

    return B, G_grid, lam_grid, d_vals


def plot_wilson_prior(sd, ax_tau, ax_EI, metadata_path):
    """Plot Wilson prior τ and E[I] = 1/τ."""
    result = _get_wilson_ingredients(sd, metadata_path)
    if result is None:
        print("metadata.pt must contain 'wavelength' key")
        return
    B, G_grid, lam_grid, d_vals = result

    for d in d_vals:
        s_sq = 1.0 / (4.0 * max(d, 0.01) ** 2)
        exp_term = torch.exp(torch.tensor(2.0 * B * s_sq))

        tau = (1.0 / G_grid) * exp_term
        ax_tau.plot(
            lam_grid.numpy(), tau.numpy(), linewidth=2, label=f"d = {d:.1f} Å"
        )

        EI = 1.0 / tau
        ax_EI.plot(
            lam_grid.numpy(), EI.numpy(), linewidth=2, label=f"d = {d:.1f} Å"
        )

    ax_tau.set_ylabel("τ (prior rate)")
    ax_tau.set_title(f"Wilson prior τ = G⁻¹·exp(2Bs²),  B = {B:.1f}")
    ax_tau.legend(fontsize=8)
    ax_tau.set_yscale("log")
    ax_tau.grid(alpha=0.3)

    ax_EI.set_ylabel("E[I] = 1/τ")
    ax_EI.set_xlabel("Wavelength (Å)")
    ax_EI.set_title("Prior expected intensity")
    ax_EI.legend(fontsize=8)
    ax_EI.set_yscale("log")
    ax_EI.grid(alpha=0.3)


def _get_B(sd):
    if "loss.raw_B" in sd:
        return F.softplus(sd["loss.raw_B"]).item()
    return F.softplus(sd["loss.q_log_B_loc"]).item()


def _extract_epoch(ckpt):
    """Get epoch number from checkpoint dict or filename."""
    return ckpt.get("epoch", 0)


def _default_plot_dir(ckpt_dir):
    """Derive plots/ directory as sibling of checkpoints/."""
    plots_dir = ckpt_dir.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    return plots_dir


def _save_spectrum_csv(csv_path, lam_np, epochs, G_curves):
    """Save spectrum data: wavelength + one G(λ) column per epoch."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["wavelength"] + [f"epoch_{int(e)}" for e in epochs]
        writer.writerow(header)
        for j in range(len(lam_np)):
            row = [f"{lam_np[j]:.6f}"] + [f"{G[j]:.6f}" for G in G_curves]
            writer.writerow(row)
    print(f"wrote {csv_path}")


def _save_bfactor_csv(csv_path, epochs, B_values):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "B_factor"])
        for ep, b in zip(epochs, B_values):
            writer.writerow([int(ep), f"{b:.4f}"])
    print(f"wrote {csv_path}")


def _extract_bg_prior_curves(sd):
    """Extract bg prior rate and concentration curves from state dict."""
    c_rate = sd["loss.bg_prior.c_rate"]
    c_alpha = sd["loss.bg_prior.c_alpha"]
    r_mid = sd["loss.bg_prior.r_mid"].item()
    r_scale = sd["loss.bg_prior.r_scale"].item()
    degree = c_rate.shape[0] - 1

    r = torch.linspace(r_mid - r_scale, r_mid + r_scale, 300)
    x = ((r - r_mid) / r_scale).clamp(-1.0, 1.0)
    phi = torch.stack(_chebyshev(x, degree), dim=-1)

    bg_rate = torch.exp(phi @ c_rate).numpy()
    bg_conc = F.softplus(phi @ c_alpha).numpy()
    return r.numpy(), bg_rate, bg_conc


def plot_epoch_progression(ckpt_dir, out_dir, metadata_path=None):
    """Plot G(λ) curves and B-factor across all epoch checkpoints."""
    ckpt_paths = sorted(ckpt_dir.glob("epoch*.ckpt"))
    if not ckpt_paths:
        print(f"No epoch*.ckpt found in {ckpt_dir}")
        return

    epochs = []
    B_values = []
    G_curves = []
    tau_pol_values = []
    bg_rate_curves = []
    bg_conc_curves = []
    r_grid = None
    r_grid_prf = None
    lam = None
    has_bg = False
    has_pol = False
    has_prf_prior = False
    has_absorption = False
    has_conc_fn = False
    prf_sigma_curves = []
    absorption_curves = []
    conc_curves = []
    s_sq_grid = None
    d_grid = None

    for path in ckpt_paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        epoch = _extract_epoch(ckpt)
        epochs.append(epoch)
        B_values.append(_get_B(sd))

        if lam is None:
            lam_min, lam_max = _get_lam_range(sd)
            lam = torch.linspace(lam_min, lam_max, 500)
            has_bg = _has_bg_prior(sd)
            has_pol = "loss.tau_pol" in sd or "loss.raw_tau_pol" in sd
            has_prf_prior = _has_profile_prior(sd)
            has_absorption = "loss.absorption.mu_coeffs" in sd
            has_conc_fn = _has_concentration_fn(sd)

        is_spectral = (
            "loss.spectrum.c" in sd or "loss.spectrum.coeff_loc" in sd
        )
        if is_spectral:
            G_curves.append(_compute_G_on_grid(sd, lam).numpy())
        elif "loss.wavelength_bin_edges" in sd:
            edges = sd["loss.wavelength_bin_edges"]
            loc = sd["loss.q_log_K_loc"]
            bins = torch.bucketize(lam, edges, right=True) - 1
            bins = bins.clamp(0, loc.shape[0] - 1)
            G_curves.append(torch.exp(loc[bins]).numpy())

        if has_bg:
            r_np, rate, conc = _extract_bg_prior_curves(sd)
            if r_grid is None:
                r_grid = r_np
            bg_rate_curves.append(rate)
            bg_conc_curves.append(conc)

        if has_prf_prior:
            r_np_prf, s_mean, s_min, s_max = _extract_profile_prior_curves(sd)
            if r_grid_prf is None:
                r_grid_prf = r_np_prf
            prf_sigma_curves.append((s_mean, s_min, s_max))

        if has_absorption:
            absorption_curves.append({
                "mu_coeffs": sd["loss.absorption.mu_coeffs"].clone(),
                "path_coeffs": sd["loss.absorption.path_coeffs"].clone(),
                "r_mid": sd["loss.absorption.r_mid"].clone(),
                "r_scale": sd["loss.absorption.r_scale"].clone(),
            })

        if has_conc_fn:
            s_sq_np, d_np, alpha_np = _extract_concentration_curves(sd)
            if s_sq_grid is None:
                s_sq_grid = s_sq_np
                d_grid = d_np
            conc_curves.append(alpha_np)

        if has_pol:
            if "loss.tau_pol" in sd:
                tau_pol_values.append(sd["loss.tau_pol"].item())
            else:
                tau_pol_values.append(torch.tanh(sd["loss.raw_tau_pol"]).item())

    lam_np = lam.numpy()
    epochs = np.array(epochs)
    B_values = np.array(B_values)
    cmap = sns.cubehelix_palette(
        start=0.5,
        rot=-0.55,
        dark=0,
        light=0.8,
        as_cmap=True,
    )
    norm = Normalize(vmin=epochs.min(), vmax=epochs.max())
    run_name = ckpt_dir.parent.parent.name

    # --- Spectrum plot ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    ax = axes[0]
    for ep, G in zip(epochs, G_curves):
        ax.plot(
            lam_np, np.log(G), color=cmap(norm(ep)), alpha=0.7, linewidth=1
        )
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Epoch")
    ax.set_ylabel("log G(λ)")
    ax.set_xlabel("Wavelength (Å)")
    ax.set_title("Learned log-spectrum over training")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for ep, G in zip(epochs, G_curves):
        ax.plot(lam_np, G, color=cmap(norm(ep)), alpha=0.7, linewidth=1)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Epoch")
    ax.set_ylabel("G(λ)")
    ax.set_xlabel("Wavelength (Å)")
    ax.set_title("Learned spectrum G(λ) over training")
    ax.grid(alpha=0.3)

    fig.suptitle(f"{run_name}  ({len(epochs)} checkpoints)", fontsize=12)
    fig.tight_layout()
    spectrum_path = out_dir / "spectrum_progression.png"
    fig.savefig(spectrum_path, dpi=150, bbox_inches="tight")
    print(f"wrote {spectrum_path}")
    plt.close(fig)

    _save_spectrum_csv(
        out_dir / "spectrum_progression.csv", lam_np, epochs, G_curves
    )

    # --- Wilson prior E[I] progression ---
    if metadata_path is not None:
        meta = torch.load(metadata_path, map_location="cpu", weights_only=False)
        if isinstance(meta, dict) and "d" in meta:
            d_data = meta["d"]
            d_vals = [
                float(d_data.quantile(0.1)),
                float(d_data.quantile(0.5)),
                float(d_data.quantile(0.9)),
            ]
            fig, axes_ei = plt.subplots(
                len(d_vals), 1, figsize=(10, 4 * len(d_vals)), sharex=True
            )
            for di, d in enumerate(d_vals):
                ax = axes_ei[di]
                s_sq = 1.0 / (4.0 * max(d, 0.01) ** 2)
                for ep, (G, B) in enumerate(zip(G_curves, B_values)):
                    G_t = torch.from_numpy(G).float()
                    exp_term = np.exp(2.0 * B * s_sq)
                    tau = (1.0 / G_t.numpy()) * exp_term
                    EI = 1.0 / tau
                    ax.plot(
                        lam_np,
                        EI,
                        color=cmap(norm(epochs[ep])),
                        alpha=0.7,
                        linewidth=1,
                    )
                fig.colorbar(
                    ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Epoch"
                )
                ax.set_ylabel("E[I] = 1/τ")
                ax.set_title(f"Prior expected intensity at d = {d:.1f} Å")
                ax.set_yscale("log")
                ax.grid(alpha=0.3)
            axes_ei[-1].set_xlabel("Wavelength (Å)")
            fig.suptitle(
                f"Wilson prior E[I] over training - {run_name}", fontsize=12
            )
            fig.tight_layout()
            ei_path = out_dir / "wilson_EI_progression.png"
            fig.savefig(ei_path, dpi=150, bbox_inches="tight")
            print(f"wrote {ei_path}")
            plt.close(fig)

            # --- E[I] vs λ at final epoch, colored by resolution ---
            G_last = G_curves[-1]
            B_last = B_values[-1]
            d_min = float(d_data.min().clamp(min=0.5))
            d_max = float(d_data.quantile(0.95))
            d_range = np.linspace(d_min, d_max, 40)
            d_cmap = plt.cm.viridis
            d_norm = Normalize(vmin=d_min, vmax=d_max)

            for yscale, suffix in [("log", "log"), ("linear", "linear")]:
                fig, ax = plt.subplots(figsize=(10, 5))
                for d in d_range:
                    s_sq = 1.0 / (4.0 * d**2)
                    EI = G_last * np.exp(-2.0 * B_last * s_sq)
                    ax.plot(
                        lam_np, EI,
                        color=d_cmap(d_norm(d)),
                        alpha=0.8, linewidth=1,
                    )
                fig.colorbar(
                    ScalarMappable(norm=d_norm, cmap=d_cmap),
                    ax=ax, label="d-spacing (Å)",
                )
                ax.set_xlabel("Wavelength (Å)")
                ax.set_ylabel("E[I] = 1/τ")
                ax.set_yscale(yscale)
                ax.set_title(
                    f"Prior E[I] at epoch {int(epochs[-1])}  |  "
                    f"B = {B_last:.1f}  ({yscale} scale)"
                )
                ax.grid(alpha=0.3)
                fig.tight_layout()
                fname = out_dir / f"wilson_EI_by_resolution_{suffix}.png"
                fig.savefig(fname, dpi=150, bbox_inches="tight")
                print(f"wrote {fname}")
                plt.close(fig)

            # --- E[I] vs resolution (1/d²) at final epoch, colored by λ ---
            wl_min, wl_max = float(lam.min()), float(lam.max())
            wl_range = np.linspace(wl_min, wl_max, 30)
            wl_cmap = plt.cm.plasma
            wl_norm = Normalize(vmin=wl_min, vmax=wl_max)
            s_sq_grid = np.linspace(
                1.0 / (4.0 * d_max**2), 1.0 / (4.0 * d_min**2), 300
            )

            fig, ax = plt.subplots(figsize=(10, 5))
            for wl in wl_range:
                wl_t = torch.tensor([wl], dtype=torch.float32)
                G_wl = float(_compute_G_on_grid(sd, wl_t))
                EI = G_wl * np.exp(-2.0 * B_last * s_sq_grid)
                ax.plot(
                    s_sq_grid, EI,
                    color=wl_cmap(wl_norm(wl)),
                    alpha=0.8, linewidth=1,
                )
            fig.colorbar(
                ScalarMappable(norm=wl_norm, cmap=wl_cmap),
                ax=ax, label="Wavelength (Å)",
            )
            ax.set_xlabel("s² = 1/(4d²)  (Å⁻²)")
            ax.set_ylabel("E[I] = 1/τ")
            ax.set_yscale("log")
            ax.set_title(
                f"Prior E[I] vs resolution at epoch {int(epochs[-1])}  |  "
                f"B = {B_last:.1f}"
            )
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fname = out_dir / "wilson_EI_by_wavelength.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"wrote {fname}")
            plt.close(fig)

    # --- B-factor plot (+ tau_pol if present) ---
    n_scalar = 1 + (1 if has_pol else 0)
    fig, axes_scalar = plt.subplots(
        n_scalar, 1, figsize=(8, 4 * n_scalar), squeeze=False
    )
    ax = axes_scalar[0, 0]
    ax.plot(epochs, B_values, "o-", color="tab:red", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("B factor")
    ax.set_title(f"B-factor over training - {run_name}")
    ax.grid(alpha=0.3)

    if has_pol:
        ax = axes_scalar[1, 0]
        ax.plot(epochs, tau_pol_values, "o-", color="tab:purple", markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("τ_pol (polarization ratio)")
        ax.set_title(f"Polarization ratio over training - {run_name}")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    bfactor_path = out_dir / "bfactor_progression.png"
    fig.savefig(bfactor_path, dpi=150, bbox_inches="tight")
    print(f"wrote {bfactor_path}")
    plt.close(fig)

    _save_bfactor_csv(out_dir / "bfactor_progression.csv", epochs, B_values)

    # --- Background prior progression ---
    if has_bg:
        fig, axes_bg = plt.subplots(3, 1, figsize=(10, 10))

        ax = axes_bg[0]
        for ep, rate in zip(epochs, bg_rate_curves):
            ax.plot(r_grid, rate, color=cmap(norm(ep)), alpha=0.7, linewidth=1)
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Epoch"
        )
        ax.set_ylabel("τ_bg (prior rate)")
        ax.set_title("Background prior rate τ_bg(r) over training")
        ax.grid(alpha=0.3)

        ax = axes_bg[1]
        for ep, conc in zip(epochs, bg_conc_curves):
            ax.plot(r_grid, conc, color=cmap(norm(ep)), alpha=0.7, linewidth=1)
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Epoch"
        )
        ax.set_ylabel("α_bg (concentration)")
        ax.set_title("Background prior concentration α_bg(r) over training")
        ax.grid(alpha=0.3)

        ax = axes_bg[2]
        for ep, rate in zip(epochs, bg_rate_curves):
            ax.plot(
                r_grid,
                1.0 / rate,
                color=cmap(norm(ep)),
                alpha=0.7,
                linewidth=1,
            )
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Epoch"
        )
        ax.set_ylabel("E[bg] = 1/τ_bg")
        ax.set_xlabel("Radius from beam center (px)")
        ax.set_title("Background prior expected value E[bg](r) over training")
        ax.grid(alpha=0.3)

        fig.suptitle(f"{run_name}  ({len(epochs)} checkpoints)", fontsize=12)
        fig.tight_layout()
        bg_path = out_dir / "bg_prior_progression.png"
        fig.savefig(bg_path, dpi=150, bbox_inches="tight")
        print(f"wrote {bg_path}")
        plt.close(fig)

    # --- Profile prior progression ---
    if has_prf_prior:
        fig, ax = plt.subplots(figsize=(10, 5))
        for ep, (s_mean, s_min, s_max) in zip(epochs, prf_sigma_curves):
            c_val = cmap(norm(ep))
            ax.plot(r_grid_prf, s_mean, color=c_val, alpha=0.8, linewidth=1.5)
            ax.fill_between(
                r_grid_prf, s_min, s_max,
                color=c_val, alpha=0.15,
            )
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Epoch"
        )
        ax.set_ylabel("sigma_prior (profile prior scale)")
        ax.set_xlabel("Radius from beam center (px)")
        ax.set_title(
            "Profile prior scale sigma_prior(r, θ) - line: mean over θ, band: min/max"
        )
        ax.grid(alpha=0.3)
        fig.suptitle(f"{run_name}  ({len(epochs)} checkpoints)", fontsize=12)
        fig.tight_layout()
        prf_path = out_dir / "profile_prior_progression.png"
        fig.savefig(prf_path, dpi=150, bbox_inches="tight")
        print(f"wrote {prf_path}")
        plt.close(fig)

    # --- Absorption correction progression ---
    if has_absorption:
        ab0 = absorption_curves[0]
        r_mid = ab0["r_mid"].item()
        r_scale = ab0["r_scale"].item()
        r_ab = torch.linspace(r_mid - r_scale, r_mid + r_scale, 200)
        rn = ((r_ab - r_mid) / r_scale).clamp(-1.0, 1.0)
        lam_min_f, lam_max_f = float(lam.min()), float(lam.max())
        wl_test = [lam_min_f, (lam_min_f + lam_max_f) / 2, lam_max_f]

        fig, axes_ab = plt.subplots(1, len(wl_test), figsize=(6 * len(wl_test), 5))
        for wi, wl_val in enumerate(wl_test):
            ax = axes_ab[wi]
            for ei, ab in enumerate(absorption_curves):
                mu = F.softplus(ab["mu_coeffs"][0] + ab["mu_coeffs"][1] * wl_val**3)
                path_degree = ab["path_coeffs"].shape[0] - 1
                phi_r = torch.stack(_chebyshev(rn, path_degree), dim=-1)
                path_len = F.softplus(phi_r @ ab["path_coeffs"])
                f_A = torch.exp(-mu * path_len).numpy()
                ax.plot(
                    r_ab.numpy(), f_A,
                    color=cmap(norm(epochs[ei])),
                    alpha=0.7, linewidth=1,
                )
            fig.colorbar(
                ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Epoch"
            )
            ax.set_ylabel("f_A (absorption factor)")
            ax.set_xlabel("Radius from beam center (px)")
            ax.set_title(f"λ = {wl_val:.3f} Å")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)

        fig.suptitle(
            f"Absorption correction f_A(λ, r) over training - {run_name}",
            fontsize=12,
        )
        fig.tight_layout()
        ab_path = out_dir / "absorption_progression.png"
        fig.savefig(ab_path, dpi=150, bbox_inches="tight")
        print(f"wrote {ab_path}")
        plt.close(fig)

    # --- Concentration α(s²) progression ---
    if has_conc_fn:
        fig, axes_conc = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes_conc[0]
        for ep, alpha in zip(epochs, conc_curves):
            ax.plot(
                s_sq_grid, alpha,
                color=cmap(norm(ep)), alpha=0.7, linewidth=1,
            )
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Epoch"
        )
        ax.set_xlabel("s² = 1/(4d²)  (Å⁻²)")
        ax.set_ylabel("α (concentration)")
        ax.set_title("Intensity prior concentration α(s²) over training")
        ax.grid(alpha=0.3)

        ax = axes_conc[1]
        for ep, alpha in zip(epochs, conc_curves):
            ax.plot(
                d_grid, alpha,
                color=cmap(norm(ep)), alpha=0.7, linewidth=1,
            )
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Epoch"
        )
        ax.set_xlabel("d-spacing (Å)")
        ax.set_ylabel("α (concentration)")
        ax.set_title("Intensity prior concentration α(d) over training")
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.grid(alpha=0.3)

        fig.suptitle(f"{run_name}  ({len(epochs)} checkpoints)", fontsize=12)
        fig.tight_layout()
        conc_path = out_dir / "concentration_progression.png"
        fig.savefig(conc_path, dpi=150, bbox_inches="tight")
        print(f"wrote {conc_path}")
        plt.close(fig)


def _has_profile_prior(sd):
    return "loss.profile_prior.c" in sd


def _extract_profile_prior_curves(sd):
    """Extract sigma_prior(r, θ) from state dict.

    Returns r_grid, sigma_mean, sigma_min, sigma_max evaluated over angles.
    """
    c = sd["loss.profile_prior.c"]
    r_mid = sd["loss.profile_prior.r_mid"].item()
    r_scale = sd["loss.profile_prior.r_scale"].item()

    n_r = 300
    r = torch.linspace(r_mid - r_scale, r_mid + r_scale, n_r)
    rn = ((r - r_mid) / r_scale).clamp(-1.0, 1.0)

    if c.dim() == 1:
        # Radial only (no angular terms)
        degree = c.shape[0] - 1
        phi = torch.stack(_chebyshev(rn, degree), dim=-1)
        sigma = F.softplus(phi @ c).numpy()
        return r.numpy(), sigma, sigma, sigma

    # Angular: c is (n_angular, n_radial)
    n_angular, n_radial = c.shape
    degree = n_radial - 1
    fourier_order = (n_angular - 1) // 2
    phi_r = torch.stack(_chebyshev(rn, degree), dim=-1)  # (n_r, n_radial)

    n_theta = 72
    thetas = torch.linspace(0, 2 * torch.pi, n_theta)
    # Evaluate at all (r, θ) pairs
    all_sigma = torch.zeros(n_r, n_theta)
    for ti, theta in enumerate(thetas):
        out = phi_r @ c[0]  # DC radial term
        for m in range(1, fourier_order + 1):
            out = out + (phi_r @ c[2 * m - 1]) * torch.cos(m * theta)
            out = out + (phi_r @ c[2 * m]) * torch.sin(m * theta)
        all_sigma[:, ti] = F.softplus(out)

    sigma_np = all_sigma.numpy()
    return (
        r.numpy(),
        sigma_np.mean(axis=1),
        sigma_np.min(axis=1),
        sigma_np.max(axis=1),
    )


def _has_concentration_fn(sd):
    return "loss.concentration_fn.c" in sd


def _extract_concentration_curves(sd):
    """Extract α(s²) from state dict. Returns (s_sq, d_spacing, alpha)."""
    c = sd["loss.concentration_fn.c"]
    s_sq_mid = sd["loss.concentration_fn.s_sq_mid"].item()
    s_sq_scale = sd["loss.concentration_fn.s_sq_scale"].item()
    degree = c.shape[0] - 1

    s_sq = torch.linspace(s_sq_mid - s_sq_scale, s_sq_mid + s_sq_scale, 300)
    s_sq = s_sq.clamp(min=1e-8)
    xn = ((s_sq - s_sq_mid) / s_sq_scale).clamp(-1.0, 1.0)
    phi = torch.stack(_chebyshev(xn, degree), dim=-1)
    alpha = F.softplus(phi @ c).numpy()
    d_spacing = 1.0 / (2.0 * torch.sqrt(s_sq)).numpy()

    return s_sq.numpy(), d_spacing, alpha


def _has_bg_prior(sd):
    return "loss.bg_prior.c_rate" in sd


def plot_bg_prior(sd, ax_rate, ax_conc):
    """Plot learned background prior rate and concentration as functions of r."""
    c_rate = sd["loss.bg_prior.c_rate"]
    c_alpha = sd["loss.bg_prior.c_alpha"]
    r_mid = sd["loss.bg_prior.r_mid"].item()
    r_scale = sd["loss.bg_prior.r_scale"].item()
    degree = c_rate.shape[0] - 1

    r = torch.linspace(r_mid - r_scale, r_mid + r_scale, 500)
    x = ((r - r_mid) / r_scale).clamp(-1.0, 1.0)
    phi = torch.stack(_chebyshev(x, degree), dim=-1)

    bg_rate = torch.exp(phi @ c_rate)
    bg_conc = F.softplus(phi @ c_alpha)

    r_np = r.numpy()

    ax_rate.plot(r_np, bg_rate.numpy(), "b-", linewidth=2)
    ax_rate.set_ylabel("τ_bg (prior rate)")
    ax_rate.set_title(f"Background prior rate τ_bg(r)  (degree {degree})")
    ax_rate.grid(alpha=0.3)

    ax_conc.plot(r_np, bg_conc.numpy(), "r-", linewidth=2)
    ax_conc.set_ylabel("α_bg (concentration)")
    ax_conc.set_xlabel("Radius from beam center (px)")
    ax_conc.set_title(
        f"Background prior concentration α_bg(r)  (degree {degree})"
    )
    ax_conc.grid(alpha=0.3)


def plot_single_checkpoint(args):
    """Original single-checkpoint visualization."""
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    is_spectral = "loss.spectrum.c" in sd or "loss.spectrum.coeff_loc" in sd
    is_binned = "loss.wavelength_bin_edges" in sd

    if not is_spectral and not is_binned:
        print("Checkpoint does not contain a polychromatic loss.")
        return

    if "loss.raw_B" in sd:
        B_mean = F.softplus(sd["loss.raw_B"])
    else:
        B_mean = F.softplus(sd["loss.q_log_B_loc"])
    epoch = ckpt.get("epoch", "?")
    kind = "spectral" if is_spectral else "binned"

    has_alpha = "loss.log_alpha_per_group" in sd
    has_bg = _has_bg_prior(sd)
    has_prf_prior = _has_profile_prior(sd)
    has_conc_fn = _has_concentration_fn(sd)
    has_pol = "loss.tau_pol" in sd or "loss.raw_tau_pol" in sd
    n_panels = 2
    if args.metadata:
        n_panels += 2
    if has_alpha:
        n_panels += 1
    if has_conc_fn:
        n_panels += 1
    if has_bg:
        n_panels += 2
    if has_prf_prior:
        n_panels += 1

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(8, 3 * n_panels), sharex=False
    )

    if is_spectral:
        plot_spectral(sd, axes[0], axes[1])
    else:
        plot_binned(sd, axes[0], axes[1])

    panel_idx = 2
    if args.metadata:
        plot_wilson_prior(
            sd, axes[panel_idx], axes[panel_idx + 1], args.metadata
        )
        panel_idx += 2

    if has_alpha:
        ax_alpha = axes[panel_idx]
        alpha = F.softplus(sd["loss.log_alpha_per_group"]).numpy()
        n_bins = len(alpha)
        ax_alpha.bar(range(n_bins), alpha, color="steelblue", edgecolor="none")
        ax_alpha.set_xlabel("Resolution bin")
        ax_alpha.set_ylabel("α (learned concentration)")
        ax_alpha.set_title(
            f"Learned prior concentration per bin ({n_bins} bins)"
        )
        ax_alpha.grid(alpha=0.3, axis="y")
        for i, v in enumerate(alpha):
            ax_alpha.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=6)
        panel_idx += 1

    if has_conc_fn:
        s_sq_np, d_np, alpha_curve = _extract_concentration_curves(sd)
        ax_conc = axes[panel_idx]
        ax_conc.plot(d_np, alpha_curve, "m-", linewidth=2)
        ax_conc.set_xlabel("d-spacing (Å)")
        ax_conc.set_ylabel("α (concentration)")
        ax_conc.set_title("Intensity prior concentration α(d)")
        ax_conc.set_xlim(ax_conc.get_xlim()[::-1])
        ax_conc.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="α = 1 (Exponential)")
        ax_conc.legend(fontsize=8)
        ax_conc.grid(alpha=0.3)
        panel_idx += 1

    if has_bg:
        plot_bg_prior(sd, axes[panel_idx], axes[panel_idx + 1])
        panel_idx += 2

    if has_prf_prior:
        r_np, s_mean, s_min, s_max = _extract_profile_prior_curves(sd)
        ax_prf = axes[panel_idx]
        ax_prf.plot(r_np, s_mean, "g-", linewidth=2, label="mean over θ")
        ax_prf.fill_between(r_np, s_min, s_max, color="green", alpha=0.2, label="min/max over θ")
        ax_prf.set_ylabel("sigma_prior (profile prior scale)")
        ax_prf.set_xlabel("Radius from beam center (px)")
        ax_prf.set_title("Profile prior scale sigma_prior(r, θ)")
        ax_prf.legend(fontsize=8)
        ax_prf.grid(alpha=0.3)
        panel_idx += 1

    # Build subtitle with all learned params
    subtitle = f"Epoch {epoch}  |  B = {B_mean.item():.2f}  |  {kind}"
    if has_pol:
        if "loss.tau_pol" in sd:
            tau_pol = sd["loss.tau_pol"].item()
        else:
            tau_pol = torch.tanh(sd["loss.raw_tau_pol"]).item()
        subtitle += f"  |  τ_pol = {tau_pol:.3f}"

    fig.suptitle(subtitle, fontsize=11)
    plt.tight_layout()

    out = args.out or args.checkpoint.with_suffix(".png").name
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


def main():
    args = parse_args()

    if args.ckpt_dir is not None:
        if args.out is not None:
            out_dir = Path(args.out)
        else:
            out_dir = _default_plot_dir(args.ckpt_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_epoch_progression(args.ckpt_dir, out_dir, metadata_path=args.metadata)
    else:
        plot_single_checkpoint(args)


if __name__ == "__main__":
    main()
