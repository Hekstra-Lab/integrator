"""Diagnostic for per-bin intensity priors.

Mirrors bg_prior_diagnostic.py, but for reflection intensities.

Bins reflections by resolution d, then per bin fits:
  * Gamma MLE on I  (shape α, rate β — free both)
  * Exponential MLE (α=1, rate = 1/mean)  — the Wilson acentric reference

The question: does Wilson's α=1 hold empirically, or is α drifting
(indicating twinning, anomalous signal, solvent contribution, etc.)?

Interpretation of α:
  * α = 1  → acentric Wilson (Exponential)
  * α = ½  → centric Wilson
  * α > 1  → narrower than Exp: intensities are more concentrated near
             the mean (ordered structure effect)
  * α < 1  → broader than Exp: heavier tails (possible outlier/ice/bad)

E[I²]/E[I]² = (α+1)/α = 1 + 1/α is the Wilson moment ratio.

Uses DIALS `intensity.prf.value` from metadata.pt — no model required.

Usage:
    uv run python scripts/intensity_prior_diagnostic.py \\
        --data-dir /path/to/pytorch_data \\
        --n-bins 30 \\
        --out-dir ./intensity_prior_diagnostic
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
    moment_ratio: float  # E[I²]/E[I]²


def load_metadata(data_dir: Path) -> dict:
    return torch.load(
        data_dir / "metadata.pt", weights_only=False, map_location="cpu"
    )


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
    """Fit Gamma and Exp MLE. values must be > 0."""
    alpha, _, scale = stats.gamma.fit(values, floc=0.0)
    rate = 1.0 / scale
    ll_gamma = stats.gamma.logpdf(values, a=alpha, scale=scale).mean()

    exp_rate = 1.0 / values.mean()
    ll_exp = stats.expon.logpdf(values, scale=1.0 / exp_rate).mean()

    mean_v = values.mean()
    e_i2 = (values**2).mean()
    moment_ratio = float(e_i2 / mean_v**2)

    return BinFit(
        values=values,
        gamma_alpha=alpha,
        gamma_rate=rate,
        gamma_ll=ll_gamma,
        exp_rate=exp_rate,
        exp_ll=ll_exp,
        moment_ratio=moment_ratio,
    )


def fit_source(
    i_values: torch.Tensor, labels: torch.Tensor, n_bins: int, name: str
) -> list[BinFit | None]:
    fits: list[BinFit | None] = []
    dropped = 0
    for b in range(int(n_bins)):
        v = i_values[labels == b].numpy()
        v_pos = v[v > 0]
        dropped += v.size - v_pos.size
        if v_pos.size < 10:
            fits.append(None)
            continue
        fits.append(fit_bin(v_pos))
    print(
        f"[{name}] dropped {dropped}/{i_values.numel()} non-positive values "
        f"(incl. NaNs filtered upstream)"
    )
    return fits


def plot_bin_grid(
    fits: list[BinFit | None],
    bin_edges: torch.Tensor,
    title: str,
    log_x: bool,
    out_path: Path,
    normalize: bool = True,
):
    """Per-bin histogram + Γ and Exp fits. If normalize=True, plot I/⟨I⟩
    so all bins share a common x-axis (Wilson-normalized view).
    """
    n_bins = len(fits)
    ncols = 6
    nrows = math.ceil(n_bins / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.6))
    axes = np.atleast_2d(axes).flatten()

    for i in range(n_bins):
        ax = axes[i]
        fit = fits[i]
        if fit is None or fit.values.size == 0:
            ax.set_axis_off()
            continue

        v = fit.values
        if normalize:
            mean_v = v.mean()
            v_plot = v / mean_v
            # α unchanged under scaling; rates rescale
            g_rate = fit.gamma_rate * mean_v
            e_rate = fit.exp_rate * mean_v
        else:
            v_plot = v
            g_rate = fit.gamma_rate
            e_rate = fit.exp_rate

        if log_x:
            v_pos = v_plot[v_plot > 0]
            log_v = np.log10(v_pos)
            lo, hi = log_v.min(), log_v.max()
            ax.hist(
                log_v,
                bins=50,
                density=True,
                alpha=0.45,
                color="teal",
                label=f"data (n={v_pos.size})",
            )
            y = np.linspace(lo, hi, 400)
            x = 10.0**y
            g_pdf = stats.gamma.pdf(x, a=fit.gamma_alpha, scale=1.0 / g_rate)
            e_pdf = stats.expon.pdf(x, scale=1.0 / e_rate)
            ax.plot(
                y,
                np.log(10) * x * g_pdf,
                color="C3",
                ls="-",
                lw=1.4,
                label=f"Γ α={fit.gamma_alpha:.2f} ll={fit.gamma_ll:.2f}",
            )
            ax.plot(
                y,
                np.log(10) * x * e_pdf,
                color="C3",
                ls="--",
                lw=1.1,
                label=f"Exp ll={fit.exp_ll:.2f}",
            )
            ax.set_xlabel("log10(I/⟨I⟩)" if normalize else "log10(I)")
        else:
            lo = np.quantile(v_plot, 0.001)
            hi = np.quantile(v_plot, 0.995)
            ax.hist(
                v_plot,
                bins=50,
                range=(lo, hi),
                density=True,
                alpha=0.45,
                color="teal",
                label=f"data (n={v_plot.size})",
            )
            x = np.linspace(max(lo, 1e-8), hi, 400)
            g_pdf = stats.gamma.pdf(x, a=fit.gamma_alpha, scale=1.0 / g_rate)
            e_pdf = stats.expon.pdf(x, scale=1.0 / e_rate)
            ax.plot(
                x,
                g_pdf,
                color="C3",
                ls="-",
                lw=1.4,
                label=f"Γ α={fit.gamma_alpha:.2f} ll={fit.gamma_ll:.2f}",
            )
            ax.plot(
                x,
                e_pdf,
                color="C3",
                ls="--",
                lw=1.1,
                label=f"Exp ll={fit.exp_ll:.2f}",
            )
            ax.set_xlabel("I/⟨I⟩" if normalize else "I")

        d_lo = float(bin_edges[i])
        d_hi = float(bin_edges[i + 1])
        ax.set_title(
            f"bin {i}: d=[{d_lo:.2f},{d_hi:.2f}]\n"
            f"⟨I²⟩/⟨I⟩²={fit.moment_ratio:.2f}",
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


def plot_summary(
    fits: list[BinFit | None], bin_edges: torch.Tensor, out_path: Path
):
    bin_mid = 0.5 * (bin_edges[:-1] + bin_edges[1:]).numpy()

    def ex(attr):
        return np.array(
            [getattr(f, attr) if f is not None else np.nan for f in fits]
        )

    alpha = ex("gamma_alpha")
    mrat = ex("moment_ratio")
    g_ll = ex("gamma_ll")
    e_ll = ex("exp_ll")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(bin_mid, alpha, "o-", color="teal", label="fitted α")
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="Wilson acentric (α=1)")
    ax.axhline(0.5, color="r", ls=":", lw=0.8, label="Wilson centric (α=0.5)")
    ax.set_xlabel("bin midpoint d (Å)")
    ax.set_ylabel("Gamma α")
    ax.set_title("Fitted α vs Wilson theory")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(bin_mid, mrat, "o-", color="teal", label="empirical ⟨I²⟩/⟨I⟩²")
    ax.plot(
        bin_mid,
        1 + 1 / alpha,
        "s--",
        color="C3",
        alpha=0.7,
        label="Γ prediction (1 + 1/α)",
    )
    ax.axhline(2.0, color="k", ls="--", lw=0.8, label="Wilson acentric = 2")
    ax.axhline(3.0, color="r", ls=":", lw=0.8, label="Wilson centric = 3")
    ax.set_xlabel("bin midpoint d (Å)")
    ax.set_ylabel("Wilson moment ratio")
    ax.set_title("Second moment vs Wilson reference")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(bin_mid, g_ll - e_ll, "o-", color="teal")
    ax.axhline(0.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("bin midpoint d (Å)")
    ax.set_ylabel("loglik(Γ) − loglik(Exp) per sample")
    ax.set_title("Gamma advantage over Exp (Wilson default)")

    ax = axes[1, 1]
    # cumulative advantage (nats per whole dataset)
    counts = np.array([f.values.size if f is not None else 0 for f in fits])
    ax.plot(bin_mid, (g_ll - e_ll) * counts, "o-", color="teal")
    ax.axhline(0.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("bin midpoint d (Å)")
    ax.set_ylabel("Γ − Exp loglik summed across bin")
    ax.set_title("Cumulative Γ advantage (nats)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_csv(
    fits: list[BinFit | None], bin_edges: torch.Tensor, path: Path, name: str
):
    with open(path, "w") as f:
        f.write(
            "bin,d_lo,d_hi,n,gamma_alpha,gamma_rate,gamma_ll,"
            "exp_rate,exp_ll,gamma_minus_exp_ll,moment_ratio\n"
        )
        for b, fit in enumerate(fits):
            d_lo = float(bin_edges[b])
            d_hi = float(bin_edges[b + 1])
            if fit is None:
                f.write(f"{b},{d_lo:.4f},{d_hi:.4f},0,,,,,,,\n")
                continue
            adv = fit.gamma_ll - fit.exp_ll
            f.write(
                f"{b},{d_lo:.4f},{d_hi:.4f},{fit.values.size},"
                f"{fit.gamma_alpha:.4f},{fit.gamma_rate:.6g},"
                f"{fit.gamma_ll:.4f},{fit.exp_rate:.6g},{fit.exp_ll:.4f},"
                f"{adv:.4f},{fit.moment_ratio:.4f}\n"
            )
    print(f"[{name}] wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--n-bins", type=int, default=30)
    p.add_argument(
        "--out-dir", type=Path, default=Path("intensity_prior_diagnostic")
    )
    p.add_argument(
        "--key",
        type=str,
        default="intensity.prf.value",
        help="Metadata key for the intensity estimate",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.data_dir)
    if args.key not in metadata:
        raise SystemExit(
            f"metadata missing {args.key!r}; available: {list(metadata.keys())[:20]}"
        )

    d = metadata["d"].float()
    i_prf = metadata[args.key].float()

    # Filter NaNs before binning/fitting
    finite = torch.isfinite(d) & torch.isfinite(i_prf)
    d = d[finite]
    i_prf = i_prf[finite]

    print(f"Loaded {d.numel()} reflections")
    print(f"d range: [{d.min():.2f}, {d.max():.2f}] Å")
    pos = (i_prf > 0).sum().item()
    print(
        f"{args.key}: min={i_prf.min():.2f} median={i_prf.median():.2f} "
        f"max={i_prf.max():.2f}  (positive: {pos}/{i_prf.numel()})"
    )

    labels, bin_edges, n_bins_actual = bin_by_resolution(d, args.n_bins)
    if n_bins_actual != args.n_bins:
        print(f"n_bins reduced from {args.n_bins} to {n_bins_actual}")

    fits = fit_source(i_prf, labels, n_bins_actual, "prf")

    plot_bin_grid(
        fits,
        bin_edges,
        f"{args.key} per bin — linear (normalized I/⟨I⟩)",
        log_x=False,
        out_path=args.out_dir / "intensity_linear_normalized.png",
        normalize=True,
    )
    plot_bin_grid(
        fits,
        bin_edges,
        f"{args.key} per bin — log-x (normalized I/⟨I⟩)",
        log_x=True,
        out_path=args.out_dir / "intensity_log_normalized.png",
        normalize=True,
    )
    plot_bin_grid(
        fits,
        bin_edges,
        f"{args.key} per bin — linear (raw I)",
        log_x=False,
        out_path=args.out_dir / "intensity_linear_raw.png",
        normalize=False,
    )
    plot_summary(fits, bin_edges, args.out_dir / "intensity_summary.png")
    write_csv(
        fits, bin_edges, args.out_dir / "intensity_prior_fits.csv", "prf"
    )

    print(f"Done. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
