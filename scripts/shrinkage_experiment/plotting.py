"""Publication plots for the shrinkage bias experiment.

Layout: 2 columns (strong prior | weak prior) x 3 rows = 6 panels.
Each panel overlays Case 1 prediction, Case 1 actual, Case 2 direct,
and Case 2 encoder results.
"""

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _bin_data(
    I_true: torch.Tensor,
    values: torch.Tensor,
    n_bins: int = 12,
    log_spaced: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin values by I_true and compute mean +/- SEM per bin.

    Returns: (bin_centers, bin_means, bin_sems, bin_counts)
    """
    I_np = I_true.numpy()
    v_np = values.numpy()

    if log_spaced:
        bins = np.logspace(
            np.log10(max(I_np.min(), 1.0)), np.log10(I_np.max()), n_bins + 1
        )
    else:
        bins = np.linspace(I_np.min(), I_np.max(), n_bins + 1)

    centers, means, sems, counts = [], [], [], []
    for i in range(n_bins):
        mask = (I_np >= bins[i]) & (I_np < bins[i + 1])
        if i == n_bins - 1:
            mask = (I_np >= bins[i]) & (I_np <= bins[i + 1])
        n = mask.sum()
        if n >= 3:
            centers.append(
                np.sqrt(bins[i] * bins[i + 1])
                if log_spaced
                else (bins[i] + bins[i + 1]) / 2
            )
            vals = v_np[mask]
            means.append(vals.mean())
            sems.append(vals.std() / np.sqrt(n))
            counts.append(n)

    return np.array(centers), np.array(means), np.array(sems), np.array(counts)


def make_plots(
    prior_results: list[dict],
    output_dir: str,
    n_bins: int = 12,
) -> None:
    """Generate the 3x2 panel figure.

    Args:
        prior_results: list of dicts, one per prior setting. Each dict has:
            - label: str
            - alpha0, beta0: prior params
            - I_true: (N,) tensor
            - case1: dict from run_case1
            - case2_direct: dict from run_case2_direct
            - case2_encoder: dict from run_case2_encoder
            - quadrature_case1: (N,) log evidence
            - quadrature_case2: (N,) log evidence
        output_dir: where to save plots
        n_bins: number of I_true bins
    """
    n_priors = len(prior_results)
    fig, axes = plt.subplots(3, n_priors, figsize=(7 * n_priors, 14))
    if n_priors == 1:
        axes = axes[:, None]

    colors = {
        "predicted": "#999999",
        "case1": "#2196F3",
        "direct": "#FF9800",
        "encoder": "#E91E63",
        "full": "#4CAF50",
    }

    for col, pr in enumerate(prior_results):
        alpha0 = pr["alpha0"]
        beta0 = pr["beta0"]
        mu0 = alpha0 / beta0
        w = beta0 / (beta0 + 1.0)
        I_true = pr["I_true"]
        c1 = pr["case1"]
        c2d = pr["case2_direct"]
        c2e = pr["case2_encoder"]

        # ── Row 0: Bias vs I_true ──
        ax = axes[0, col]
        I_sorted = torch.sort(I_true).values
        pred_line = w * (mu0 - I_sorted)

        ax.plot(
            I_sorted.numpy(),
            pred_line.numpy(),
            "-",
            color=colors["predicted"],
            lw=2,
            label=f"Predicted: w({chr(956)}₀ - I)",
            zorder=5,
        )

        bias_series = [
            (c1["bias"], "case1", colors["case1"], "Case 1 (B=0, exact)"),
            (c2d["bias"], "direct", colors["direct"], "Case 2 direct (B>0)"),
            (
                c2e["bias"],
                "encoder",
                colors["encoder"],
                "Case 2 encoder (B>0)",
            ),
        ]
        c3 = pr.get("case3_full")
        if c3 is not None:
            bias_series.append(
                (
                    c3["bias"],
                    "full",
                    colors["full"],
                    "Case 3 full (all unknown)",
                )
            )

        for data, key, color, label in bias_series:
            # Per-reflection scatter
            ax.scatter(
                I_true.numpy(),
                data.numpy(),
                s=8,
                color=color,
                alpha=0.15,
                zorder=1,
                rasterized=True,
            )
            centers, means, sems, _ = _bin_data(I_true, data, n_bins)
            ax.errorbar(
                centers,
                means,
                yerr=sems,
                fmt="o-",
                color=color,
                markersize=4,
                capsize=3,
                label=label,
                alpha=0.8,
                zorder=3,
            )

        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.axvline(
            mu0,
            color="gray",
            ls=":",
            lw=0.5,
            label=f"I = {chr(956)}₀ = {mu0:.0f}",
        )
        ax.set_xlabel("I_true")
        ax.set_ylabel("Bias = E[I] - I_true")
        ax.set_title(f"Prior: Gamma({alpha0}, {beta0}), w={w:.4f}")
        ax.set_xscale("log")
        ax.legend(fontsize=7, loc="best")

        # ── Row 1: KL gap vs I_true ──
        ax = axes[1, col]
        q_c1 = pr.get("quadrature_case1")
        q_c2 = pr.get("quadrature_case2")

        if q_c1 is not None:
            kl_gap_c1 = q_c1 - c1["elbo"]
            centers, means, sems, _ = _bin_data(I_true, kl_gap_c1, n_bins)
            ax.errorbar(
                centers,
                means,
                yerr=sems,
                fmt="s-",
                color=colors["case1"],
                markersize=4,
                capsize=3,
                label="Case 1 (should be ~0)",
                alpha=0.8,
            )

        if q_c2 is not None:
            kl_gap_c2 = q_c2 - c2d["elbo"]
            centers, means, sems, _ = _bin_data(I_true, kl_gap_c2, n_bins)
            ax.errorbar(
                centers,
                means,
                yerr=sems,
                fmt="o-",
                color=colors["direct"],
                markersize=4,
                capsize=3,
                label="Case 2 direct",
                alpha=0.8,
            )

        if q_c2 is not None and "elbo" in c2e:
            kl_gap_enc = q_c2 - c2e["elbo"]
            centers, means, sems, _ = _bin_data(I_true, kl_gap_enc, n_bins)
            ax.errorbar(
                centers,
                means,
                yerr=sems,
                fmt="^-",
                color=colors["encoder"],
                markersize=4,
                capsize=3,
                label="Case 2 encoder",
                alpha=0.8,
            )

        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.set_xlabel("I_true")
        ax.set_ylabel("KL gap = log p(X) - ELBO (nats)")
        ax.set_title("Approximation Error")
        ax.set_xscale("log")
        ax.set_ylim(bottom=-0.5)
        ax.legend(fontsize=7, loc="best")

        # ── Row 2: Effective shrinkage weight ──
        ax = axes[2, col]

        shrinkage_series = [
            (c2d["mu"], c2d["S"], colors["direct"], "Case 2 direct"),
            (c2e["mu"], c2d["S"], colors["encoder"], "Case 2 encoder"),
        ]
        if c3 is not None:
            shrinkage_series.append(
                (c3["mu"], c2d["S"], colors["full"], "Case 3 full")
            )

        for data_mu, data_S, color, label in shrinkage_series:
            denom = mu0 - data_S
            valid = denom.abs() > max(50.0, 0.05 * mu0)
            w_eff = torch.where(
                valid, (data_mu - data_S) / denom, torch.tensor(float("nan"))
            )
            valid_I = I_true[valid]
            valid_w = w_eff[valid]
            if valid_I.numel() > 0:
                centers, means, sems, _ = _bin_data(valid_I, valid_w, n_bins)
                ax.errorbar(
                    centers,
                    means,
                    yerr=sems,
                    fmt="o-",
                    color=color,
                    markersize=4,
                    capsize=3,
                    label=label,
                    alpha=0.8,
                )

        ax.axhline(
            w,
            color=colors["case1"],
            ls="--",
            lw=2,
            label=f"Case 1: w = {w:.4f}",
        )
        ax.set_xlabel("I_true")
        ax.set_ylabel("Effective shrinkage weight")
        ax.set_title("Effective Shrinkage")
        ax.set_xscale("log")
        ax.legend(fontsize=7, loc="best")

    plt.tight_layout()
    out_path = f"{output_dir}/shrinkage_bias_experiment.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def make_linearity_plot(
    prior_results: list[dict],
    output_dir: str,
) -> None:
    """Shrinkage linearity test: (μ_i - S_i) vs (μ₀ - S_i).

    If shrinkage is a constant weight w, all points lie on y = w·x.
    Deviation from the line reveals non-constant (intensity-dependent) shrinkage.
    Points colored by background-to-signal ratio 441·B/I.
    """
    n_priors = len(prior_results)
    has_c3 = any(pr.get("case3_full") is not None for pr in prior_results)
    n_cases = 4 if has_c3 else 3

    fig, axes = plt.subplots(
        n_priors,
        n_cases,
        figsize=(5.5 * n_cases, 5 * n_priors),
        squeeze=False,
    )

    for row, pr in enumerate(prior_results):
        alpha0 = pr["alpha0"]
        beta0 = pr["beta0"]
        mu0 = alpha0 / beta0
        w = beta0 / (beta0 + 1.0)
        I_true = pr["I_true"]
        B_true = pr["B_true"]
        c1 = pr["case1"]
        c2d = pr["case2_direct"]
        c2e = pr["case2_encoder"]
        c3 = pr.get("case3_full")

        # Background-to-signal ratio: 441*B / I  (total bg counts / total signal)
        bg_signal = (441.0 * B_true / I_true.clamp(min=1.0)).numpy()
        log_bg_signal = np.log10(np.clip(bg_signal, 1e-3, None))

        cases = [
            (c1, "Case 1 (B=0, exact)", c1["S"]),
            (c2d, "Case 2 direct (B>0)", c2d["S"]),
            (c2e, "Case 2 encoder (B>0)", c2e["S"]),
        ]
        if c3 is not None:
            cases.append((c3, "Case 3 full (all unknown)", c3["S"]))

        # Shared color limits across columns
        vmin, vmax = log_bg_signal.min(), log_bg_signal.max()

        for col_idx, (case_data, case_label, S) in enumerate(cases):
            ax = axes[row, col_idx]

            x = (mu0 - S).numpy()  # μ₀ - S_i
            y = (case_data["mu"] - S).numpy()  # μ_i - S_i

            # Reference line: y = w · x
            pad = 0.05 * (x.max() - x.min())
            x_line = np.array([x.min() - pad, x.max() + pad])
            ax.plot(
                x_line,
                w * x_line,
                "--",
                color="gray",
                lw=2,
                zorder=5,
                label=f"slope w = {w:.4f}",
            )

            # Scatter colored by bg/signal ratio
            sc = ax.scatter(
                x,
                y,
                c=log_bg_signal,
                s=14,
                alpha=0.65,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
                zorder=2,
                edgecolors="none",
            )

            ax.axhline(0, color="gray", ls=":", lw=0.5, alpha=0.5)
            ax.axvline(0, color="gray", ls=":", lw=0.5, alpha=0.5)

            ax.set_xlabel(f"{chr(956)}{chr(8320)} {chr(8722)} S_i")
            ax.set_ylabel(f"{chr(956)}_i {chr(8722)} S_i")
            ax.set_title(
                f"{case_label}\nGamma({alpha0}, {beta0}), w={w:.4f}",
                fontsize=10,
            )
            ax.legend(fontsize=8, loc="upper left")

            cb = plt.colorbar(sc, ax=ax, pad=0.02)
            cb.set_label("log\u2081\u2080(441\u00b7B/I)", fontsize=8)

    plt.tight_layout()
    out_path = f"{output_dir}/shrinkage_linearity.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()
