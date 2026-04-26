"""Diagnostic for per-bin background priors.

Bins reflections by resolution (d), then per bin:
  * Extracts a crude per-reflection bg value (quiet-frame bg for 3D,
    border-pixel mean for 2D) — the same logic priors/loss uses.
  * Reads the DIALS bg value from metadata["background.mean"].
  * Fits a Gamma MLE AND an Exponential MLE to each; reports loglik.
  * Plots histograms + Gamma + Exponential overlays, linear and log-x.
  * Also plots a side-by-side crude vs DIALS comparison per bin.

Outputs figures + CSVs of per-bin fit params.

Usage:
    uv run python scripts/bg_prior_diagnostic.py \\
        --data-dir /path/to/pytorch_data \\
        --n-bins 30 \\
        --out-dir ./bg_prior_diagnostic
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats


@dataclass
class BinFit:
    values: np.ndarray
    gamma_alpha: float
    gamma_rate: float
    gamma_ll: float
    exp_rate: float
    exp_ll: float


def load_data(data_dir: Path):
    counts = torch.load(
        data_dir / "counts.pt", weights_only=False, map_location="cpu"
    )
    masks = torch.load(
        data_dir / "masks.pt", weights_only=False, map_location="cpu"
    )
    metadata = torch.load(
        data_dir / "metadata.pt", weights_only=False, map_location="cpu"
    )
    return counts, masks, metadata


def compute_crude_bg_per_refl(
    counts: torch.Tensor, masks: torch.Tensor, D: int, H: int, W: int
) -> torch.Tensor:
    """Per-reflection bg estimate (same logic as _bg_subtract_signal)."""
    n_pixels_per_frame = H * W
    N = counts.shape[0]

    counts_clean = counts.float().clamp(min=0)
    masks_f = masks.float()
    counts_masked = counts_clean * masks_f

    counts_3d = counts_masked.reshape(N, D, n_pixels_per_frame)
    masks_3d = masks_f.reshape(N, D, n_pixels_per_frame)

    if D > 1:
        frame_counts = counts_3d.sum(dim=-1)
        frame_n_pixels = masks_3d.sum(dim=-1)
        min_frame_idx = frame_counts.argmin(dim=-1)
        bg_frame_counts = frame_counts.gather(
            1, min_frame_idx.unsqueeze(-1)
        ).squeeze(-1)
        bg_frame_n_pixels = frame_n_pixels.gather(
            1, min_frame_idx.unsqueeze(-1)
        ).squeeze(-1)
        bg = bg_frame_counts / bg_frame_n_pixels.clamp(min=1)
    else:
        frame = counts_3d[:, 0, :]
        frame_2d = frame.reshape(N, H, W)
        border_mask = torch.ones(H, W, dtype=torch.bool)
        border_mask[2:-2, 2:-2] = False
        border_vals = frame_2d[:, border_mask]
        bg = border_vals.mean(dim=-1)

    return bg


def bin_by_resolution(
    d: torch.Tensor, n_bins: int, min_per_bin: int = 50
) -> tuple[torch.Tensor, torch.Tensor, int]:
    while n_bins > 1:
        quantiles = torch.linspace(0, 1, n_bins + 1)
        bin_edges = torch.quantile(d.float(), quantiles)
        labels = torch.searchsorted(bin_edges[1:-1], d).long()
        if torch.bincount(labels, minlength=n_bins).min() >= min_per_bin:
            return labels, bin_edges, n_bins
        n_bins -= 1
    labels = torch.zeros(len(d), dtype=torch.long)
    bin_edges = torch.tensor([d.min(), d.max()])
    return labels, bin_edges, 1


def fit_bin(values: np.ndarray) -> BinFit:
    """Fit Gamma + Exponential MLE. Values must be > 0."""
    alpha, _, scale = stats.gamma.fit(values, floc=0.0)
    rate = 1.0 / scale
    ll_gamma = stats.gamma.logpdf(values, a=alpha, scale=scale).mean()

    # Exponential MLE: rate = 1/mean.
    exp_rate = 1.0 / values.mean()
    ll_exp = stats.expon.logpdf(values, scale=1.0 / exp_rate).mean()

    return BinFit(
        values=values,
        gamma_alpha=alpha,
        gamma_rate=rate,
        gamma_ll=ll_gamma,
        exp_rate=exp_rate,
        exp_ll=ll_exp,
    )


def _plot_one_fit(
    ax, fit: BinFit, log_x: bool, color: str, label_prefix: str, hist: bool
):
    """Draw histogram (optional) + Gamma + Exp overlays on ax."""
    v = fit.values
    if log_x:
        v_pos = v[v > 0]
        log_v = np.log10(v_pos)
        lo, hi = log_v.min(), log_v.max()
        if hist:
            ax.hist(
                log_v,
                bins=50,
                density=True,
                alpha=0.45,
                color=color,
                label=f"{label_prefix} data (n={v_pos.size})",
            )
        y = np.linspace(lo, hi, 400)
        x = 10.0**y
        g_pdf = stats.gamma.pdf(
            x, a=fit.gamma_alpha, scale=1.0 / fit.gamma_rate
        )
        e_pdf = stats.expon.pdf(x, scale=1.0 / fit.exp_rate)
        ax.plot(
            y,
            np.log(10) * x * g_pdf,
            color=color,
            ls="-",
            lw=1.4,
            label=f"{label_prefix} Γ ll={fit.gamma_ll:.2f}",
        )
        ax.plot(
            y,
            np.log(10) * x * e_pdf,
            color=color,
            ls="--",
            lw=1.1,
            label=f"{label_prefix} Exp ll={fit.exp_ll:.2f}",
        )
        ax.set_xlabel("log10(bg)")
    else:
        lo = np.quantile(v, 0.001)
        hi = np.quantile(v, 0.999)
        if hist:
            ax.hist(
                v,
                bins=50,
                range=(lo, hi),
                density=True,
                alpha=0.45,
                color=color,
                label=f"{label_prefix} data (n={v.size})",
            )
        x = np.linspace(max(lo, 1e-8), hi, 400)
        g_pdf = stats.gamma.pdf(
            x, a=fit.gamma_alpha, scale=1.0 / fit.gamma_rate
        )
        e_pdf = stats.expon.pdf(x, scale=1.0 / fit.exp_rate)
        ax.plot(
            x,
            g_pdf,
            color=color,
            ls="-",
            lw=1.4,
            label=f"{label_prefix} Γ ll={fit.gamma_ll:.2f}",
        )
        ax.plot(
            x,
            e_pdf,
            color=color,
            ls="--",
            lw=1.1,
            label=f"{label_prefix} Exp ll={fit.exp_ll:.2f}",
        )
        ax.set_xlabel("bg")


def plot_bin_grid(
    bin_fits: list[BinFit | None],
    bin_edges: torch.Tensor,
    title: str,
    log_x: bool,
    out_path: Path,
):
    n_bins = len(bin_fits)
    ncols = 6
    nrows = math.ceil(n_bins / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.6))
    axes = np.atleast_2d(axes).flatten()

    for i in range(n_bins):
        ax = axes[i]
        fit = bin_fits[i]
        if fit is None or fit.values.size == 0:
            ax.set_axis_off()
            continue

        _plot_one_fit(
            ax, fit, log_x, color="steelblue", label_prefix="", hist=True
        )

        d_lo = float(bin_edges[i])
        d_hi = float(bin_edges[i + 1])
        ax.set_title(
            f"bin {i}: d=[{d_lo:.2f},{d_hi:.2f}]\n"
            f"α={fit.gamma_alpha:.2f}, β={fit.gamma_rate:.3g}",
            fontsize=8,
        )
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="best")

    for j in range(n_bins, len(axes)):
        axes[j].set_axis_off()

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_comparison_grid(
    crude_fits: list[BinFit | None],
    dials_fits: list[BinFit | None],
    bin_edges: torch.Tensor,
    title: str,
    log_x: bool,
    out_path: Path,
):
    n_bins = len(crude_fits)
    ncols = 6
    nrows = math.ceil(n_bins / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.4, nrows * 2.8))
    axes = np.atleast_2d(axes).flatten()

    for i in range(n_bins):
        ax = axes[i]
        c = crude_fits[i]
        dfit = dials_fits[i]
        if (c is None or c.values.size == 0) and (
            dfit is None or dfit.values.size == 0
        ):
            ax.set_axis_off()
            continue
        if c is not None and c.values.size > 0:
            _plot_one_fit(
                ax,
                c,
                log_x,
                color="steelblue",
                label_prefix="crude",
                hist=True,
            )
        if dfit is not None and dfit.values.size > 0:
            _plot_one_fit(
                ax,
                dfit,
                log_x,
                color="darkorange",
                label_prefix="dials",
                hist=True,
            )

        d_lo = float(bin_edges[i])
        d_hi = float(bin_edges[i + 1])
        ax.set_title(f"bin {i}: d=[{d_lo:.2f},{d_hi:.2f}]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=5.5, loc="best")

    for j in range(n_bins, len(axes)):
        axes[j].set_axis_off()

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_params_vs_bin(
    crude_fits: list[BinFit | None],
    dials_fits: list[BinFit | None],
    bin_edges: torch.Tensor,
    out_path: Path,
):
    """Per-bin α, β, and Δloglik (Γ - Exp) for crude vs DIALS."""
    bin_mid = 0.5 * (bin_edges[:-1] + bin_edges[1:]).numpy()

    def extract(fits, attr):
        return np.array(
            [getattr(f, attr) if f is not None else np.nan for f in fits]
        )

    c_alpha = extract(crude_fits, "gamma_alpha")
    d_alpha = extract(dials_fits, "gamma_alpha")
    c_rate = extract(crude_fits, "gamma_rate")
    d_rate = extract(dials_fits, "gamma_rate")
    c_gll = extract(crude_fits, "gamma_ll")
    c_ell = extract(crude_fits, "exp_ll")
    d_gll = extract(dials_fits, "gamma_ll")
    d_ell = extract(dials_fits, "exp_ll")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(bin_mid, c_alpha, "o-", color="steelblue", label="crude")
    ax.plot(bin_mid, d_alpha, "s-", color="darkorange", label="dials")
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="Exp (α=1)")
    ax.set_xlabel("bin midpoint d (Å)")
    ax.set_ylabel("Gamma α")
    ax.set_title("Concentration α per bin")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(bin_mid, c_rate, "o-", color="steelblue", label="crude")
    ax.plot(bin_mid, d_rate, "s-", color="darkorange", label="dials")
    ax.set_yscale("log")
    ax.set_xlabel("bin midpoint d (Å)")
    ax.set_ylabel("Gamma rate β (log)")
    ax.set_title("Rate β per bin")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(bin_mid, c_gll - c_ell, "o-", color="steelblue", label="crude")
    ax.plot(bin_mid, d_gll - d_ell, "s-", color="darkorange", label="dials")
    ax.axhline(0.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("bin midpoint d (Å)")
    ax.set_ylabel("loglik(Γ) − loglik(Exp) per sample")
    ax.set_title("Gamma advantage over Exp (higher = Γ better)")
    ax.legend(fontsize=8)

    # crude vs DIALS: α and rate on identity lines
    ax = axes[1, 1]
    ax.scatter(d_alpha, c_alpha, c="C2", label="α")
    lo = np.nanmin([d_alpha, c_alpha])
    hi = np.nanmax([d_alpha, c_alpha])
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="identity")
    ax.set_xlabel("DIALS α")
    ax.set_ylabel("crude α")
    ax.set_title("crude α vs DIALS α per bin")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_crude_vs_dials_scatter(
    crude_bg: torch.Tensor,
    dials_bg: torch.Tensor,
    labels: torch.Tensor,
    out_path: Path,
    n_sample: int = 50_000,
):
    """Per-reflection scatter: crude bg vs DIALS bg, colored by resolution bin."""
    N = crude_bg.numel()
    if N > n_sample:
        idx = torch.randperm(N)[:n_sample]
    else:
        idx = torch.arange(N)
    c = crude_bg[idx].numpy()
    d = dials_bg[idx].numpy()
    lbl = labels[idx].numpy()

    pos = (c > 0) & (d > 0)
    c, d, lbl = c[pos], d[pos], lbl[pos]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    ax = axes[0]
    sc = ax.scatter(d, c, c=lbl, cmap="viridis", s=3, alpha=0.3)
    lo = min(c.min(), d.min())
    hi = max(c.max(), d.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="identity")
    ax.set_xlabel("DIALS bg (per pixel)")
    ax.set_ylabel("crude bg (per pixel)")
    ax.set_title("Per-reflection: crude vs DIALS (linear)")
    ax.legend()
    plt.colorbar(sc, ax=ax, label="resolution bin")

    ax = axes[1]
    sc = ax.scatter(d, c, c=lbl, cmap="viridis", s=3, alpha=0.3)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="identity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("DIALS bg (log)")
    ax.set_ylabel("crude bg (log)")
    ax.set_title("Per-reflection: crude vs DIALS (log-log)")
    ax.legend()
    plt.colorbar(sc, ax=ax, label="resolution bin")

    corr = float(np.corrcoef(c, d)[0, 1])
    fig.suptitle(
        f"crude vs DIALS per-reflection (n={c.size} sampled, Pearson r={corr:.3f})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def fit_source(
    bg_values: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int,
    name: str,
) -> list[BinFit | None]:
    fits: list[BinFit | None] = []
    dropped = 0
    for b in range(int(n_bins)):
        v = bg_values[labels == b].numpy()
        v_pos = v[v > 0]
        dropped += v.size - v_pos.size
        if v_pos.size < 10:
            fits.append(None)
            continue
        fits.append(fit_bin(v_pos))
    print(
        f"[{name}] dropped {dropped}/{bg_values.numel()} non-positive values"
    )
    return fits


def write_csv(
    fits: list[BinFit | None], bin_edges: torch.Tensor, path: Path, name: str
):
    with open(path, "w") as f:
        f.write(
            "bin,d_lo,d_hi,n,gamma_alpha,gamma_rate,gamma_ll,"
            "exp_rate,exp_ll,gamma_minus_exp_ll\n"
        )
        for b, fit in enumerate(fits):
            d_lo = float(bin_edges[b])
            d_hi = float(bin_edges[b + 1])
            if fit is None:
                f.write(f"{b},{d_lo:.4f},{d_hi:.4f},0,,,,,,\n")
                continue
            advantage = fit.gamma_ll - fit.exp_ll
            f.write(
                f"{b},{d_lo:.4f},{d_hi:.4f},{fit.values.size},"
                f"{fit.gamma_alpha:.4f},{fit.gamma_rate:.6g},{fit.gamma_ll:.4f},"
                f"{fit.exp_rate:.6g},{fit.exp_ll:.4f},{advantage:.4f}\n"
            )
    print(f"[{name}] wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--n-bins", type=int, default=30)
    p.add_argument("--out-dir", type=Path, default=Path("bg_prior_diagnostic"))
    p.add_argument("--D", type=int, default=3)
    p.add_argument("--H", type=int, default=21)
    p.add_argument("--W", type=int, default=21)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    counts, masks, metadata = load_data(args.data_dir)
    d = metadata["d"].float()
    dials_bg = metadata["background.mean"].float()

    print(
        f"Loaded {counts.shape[0]} reflections; shoebox {args.D}x{args.H}x{args.W}"
    )
    print(f"d range: [{d.min():.2f}, {d.max():.2f}] Å")

    crude_bg = compute_crude_bg_per_refl(counts, masks, args.D, args.H, args.W)
    print(
        f"crude bg  : min={crude_bg.min():.3f} median={crude_bg.median():.3f} "
        f"max={crude_bg.max():.3f}"
    )
    print(
        f"DIALS bg  : min={dials_bg.min():.3f} median={dials_bg.median():.3f} "
        f"max={dials_bg.max():.3f}"
    )

    labels, bin_edges, n_bins_actual = bin_by_resolution(d, args.n_bins)
    if n_bins_actual != args.n_bins:
        print(f"n_bins reduced from {args.n_bins} to {n_bins_actual}")

    crude_fits = fit_source(crude_bg, labels, n_bins_actual, "crude")
    dials_fits = fit_source(dials_bg, labels, n_bins_actual, "dials")

    for name, fits in (("crude", crude_fits), ("dials", dials_fits)):
        plot_bin_grid(
            fits,
            bin_edges,
            f"{name} bg — linear",
            log_x=False,
            out_path=args.out_dir / f"bg_prior_{name}_linear.png",
        )
        plot_bin_grid(
            fits,
            bin_edges,
            f"{name} bg — log-x",
            log_x=True,
            out_path=args.out_dir / f"bg_prior_{name}_log.png",
        )
        write_csv(
            fits, bin_edges, args.out_dir / f"bg_prior_{name}_fits.csv", name
        )

    plot_comparison_grid(
        crude_fits,
        dials_fits,
        bin_edges,
        title="crude vs DIALS per bin — linear",
        log_x=False,
        out_path=args.out_dir / "bg_prior_comparison_linear.png",
    )
    plot_comparison_grid(
        crude_fits,
        dials_fits,
        bin_edges,
        title="crude vs DIALS per bin — log-x",
        log_x=True,
        out_path=args.out_dir / "bg_prior_comparison_log.png",
    )

    plot_params_vs_bin(
        crude_fits,
        dials_fits,
        bin_edges,
        out_path=args.out_dir / "bg_prior_params_per_bin.png",
    )
    plot_crude_vs_dials_scatter(
        crude_bg,
        dials_bg,
        labels,
        out_path=args.out_dir / "bg_prior_crude_vs_dials_scatter.png",
    )

    print(f"Done. Figures + CSVs in {args.out_dir}")


if __name__ == "__main__":
    main()
