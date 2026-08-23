"""Model-checking figures computed from a single full pass over the data."""

from __future__ import annotations

import numpy as np

from .figure_style import paper_style


def plot_ppc_zscores(
    z: np.ndarray, rate: np.ndarray | None = None, n_bins: int = 12
):
    """Pixel-level posterior predictive check.

    Left: the Pearson residual `(counts − rate)/√rate` against a standard
    normal, which is what a correct Poisson rate model produces. Right:
    the same residual's spread as a function of the fitted rate, which
    separates a global misfit from one confined to weak or bright pixels.
    """
    import matplotlib.pyplot as plt

    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.4))
        lo, hi = np.percentile(z, [0.5, 99.5])
        axes[0].hist(
            z, bins=80, range=(lo, hi), density=True, color="#0173B2",
            alpha=0.75,
        )
        xs = np.linspace(lo, hi, 300)
        axes[0].plot(
            xs,
            np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi),
            color="0.2",
            linestyle="--",
            label="N(0, 1)",
        )
        axes[0].set_xlabel("Pearson residual")
        axes[0].set_ylabel("density")
        axes[0].set_title(
            f"mean {z.mean():.2f}, sd {z.std():.2f}", fontsize=8
        )
        axes[0].legend()

        if rate is not None:
            rate = np.asarray(rate, dtype=float).ravel()
            good = np.isfinite(rate) & (rate > 0)
            edges = np.quantile(rate[good], np.linspace(0, 1, n_bins + 1))
            centers, spreads = [], []
            zz = np.asarray(z).ravel()
            for lo_e, hi_e in zip(edges[:-1], edges[1:], strict=False):
                sel = good & (rate >= lo_e) & (rate < hi_e)
                if sel.sum() < 20:
                    continue
                centers.append(float(np.median(rate[sel])))
                spreads.append(float(np.std(zz[sel])))
            axes[1].plot(centers, spreads, marker="o", color="#DE8F05")
            axes[1].axhline(
                1.0, color="0.3", linestyle="--", linewidth=0.8
            )
            axes[1].set_xscale("log")
            axes[1].set_xlabel("fitted rate (counts/px)")
            axes[1].set_ylabel("sd of residual")
            axes[1].set_title("dispersion vs rate", fontsize=8)
        else:
            axes[1].axis("off")
        fig.suptitle("posterior predictive check", fontsize=9)
    return fig


def plot_model_vs_dials(
    model_i: np.ndarray,
    dials_i: np.ndarray,
    dials_sigma: np.ndarray | None = None,
):
    """Model intensity against DIALS, with the weak tail called out.

    The interesting region is not the diagonal at high intensity, where
    any method agrees, but the weak end where the prior moves estimates.
    """
    import matplotlib.pyplot as plt

    model_i = np.asarray(model_i, dtype=float)
    dials_i = np.asarray(dials_i, dtype=float)
    good = np.isfinite(model_i) & np.isfinite(dials_i)
    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
        positive = good & (model_i > 0) & (dials_i > 0)
        hb = axes[0].hexbin(
            dials_i[positive],
            model_i[positive],
            xscale="log",
            yscale="log",
            gridsize=60,
            cmap="magma",
            bins="log",
            mincnt=1,
        )
        lim = [
            float(np.percentile(dials_i[positive], 0.1)),
            float(np.percentile(dials_i[positive], 99.9)),
        ]
        axes[0].plot(lim, lim, color="white", linestyle="--", linewidth=0.9)
        fig.colorbar(hb, ax=axes[0], fraction=0.046, pad=0.02, label="log N")
        axes[0].set_xlabel("DIALS intensity")
        axes[0].set_ylabel("model posterior intensity")

        if dials_sigma is not None:
            sigma = np.asarray(dials_sigma, dtype=float)
            snr = np.where(sigma > 0, dials_i / sigma, np.nan)
            sel = good & np.isfinite(snr)
            edges = np.quantile(snr[sel], np.linspace(0, 1, 15))
            centers, ratios = [], []
            for lo, hi in zip(edges[:-1], edges[1:], strict=False):
                bin_sel = sel & (snr >= lo) & (snr < hi)
                if bin_sel.sum() < 20:
                    continue
                centers.append(float(np.median(snr[bin_sel])))
                ratios.append(
                    float(
                        np.median(model_i[bin_sel])
                        / max(np.median(dials_i[bin_sel]), 1e-6)
                    )
                )
            axes[1].plot(centers, ratios, marker="o", color="#0173B2")
            axes[1].axhline(
                1.0, color="0.3", linestyle="--", linewidth=0.8
            )
            axes[1].set_xscale("log")
            axes[1].set_xlabel("DIALS I/σ")
            axes[1].set_ylabel("median model / DIALS")
            axes[1].set_title("shrinkage in the weak tail", fontsize=8)
        else:
            axes[1].axis("off")
        fig.suptitle("model intensities against DIALS", fontsize=9)
    return fig
