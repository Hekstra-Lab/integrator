"""Fit the Wilson scale G and B factor from a DIALS reflection file.

Answers "what is the real intensity scale of this dataset?", so that `init_G` /
`init_B` on a `monochromatic_wilson` loss can be set from -- or checked against
-- a real crystal instead of the neutral `G = 1` default.

The fit is the same estimator the model runs on its own raw shoebox counts
(`WilsonLoss.wilson_fit`), so the G printed here is directly comparable to the
`wilson_G` logged at epoch 1.

No DIALS install needed: the `.refl` msgpack is read via reciprocalspaceship.

    uv run python scripts/wilson_fit_refl.py /path/to/scaled.refl

LP convention (this is the easy thing to get wrong -- see --lp):
    DIALS stores lp = L/P and applies it multiplicatively, `I_dials = I_raw * lp`.
    Our generative model is the inverse: the Poisson likelihood is on RAW counts,
    so the loss divides by lp. That means

        lp_correction: false  ->  G is on RAW intensity      ->  fit I_dials / lp
        lp_correction: true   ->  G is on LP-CORRECTED I     ->  fit I_dials

    Both are printed. Match the row to the `lp_correction` in your config.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from integrator.model.loss.wilson_loss import WilsonLoss

INTENSITY_COLS = {
    "sum": "intensity.sum.value",
    "prf": "intensity.prf.value",
    "scale": "intensity.scale.value",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit Wilson G and B from a DIALS .refl file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("refl", type=Path, help="DIALS reflection file (scaled.refl)")
    p.add_argument(
        "--intensity",
        choices=sorted(INTENSITY_COLS),
        default="sum",
        help="intensity column; 'sum' is closest to what the model integrates",
    )
    p.add_argument(
        "--lp",
        choices=["both", "raw", "corrected"],
        default="both",
        help="frame to report: raw (lp_correction:false), corrected (true), or both",
    )
    p.add_argument("--bins", type=int, default=20, help="resolution bins for the fit")
    p.add_argument("--dmin", type=float, default=None, help="high-resolution cutoff (A)")
    p.add_argument("--dmax", type=float, default=None, help="low-resolution cutoff (A)")
    p.add_argument(
        "--cell",
        type=str,
        default=None,
        help="a,b,c,al,be,ga -- only needed if the file has no 'd' column",
    )
    p.add_argument(
        "--spacegroup", type=str, default=None, help="e.g. P43212 (with --cell)"
    )
    p.add_argument("--plot", type=Path, default=None, help="save a Wilson plot here")
    return p.parse_args()


def load_refl(args):
    """Read the .refl into a DataFrame with the columns we need."""
    import reciprocalspaceship.io as rs_io

    want = [
        INTENSITY_COLS[args.intensity],
        "intensity.sum.variance",
        "lp",
        "d",
        "partiality",
        "flags",
    ]
    ds = rs_io.read_dials_stills(
        str(args.refl),
        extra_cols=want,
        unitcell=args.cell,
        spacegroup=args.spacegroup,
        numjobs=1,
    )
    return ds


def resolution(ds):
    """d-spacing per reflection, from the column or computed from the cell."""
    if "d" in ds:
        return np.asarray(ds["d"], dtype=np.float64)
    if ds.cell is None or ds.spacegroup is None:
        raise SystemExit(
            "This file has no 'd' column, so d-spacings must be computed from "
            "the crystal: pass --cell a,b,c,al,be,ga and --spacegroup."
        )
    return np.asarray(ds.compute_dHKL()["dHKL"], dtype=np.float64)


def fit_frame(i, s_sq, n_bins):
    """(G, B) via the model's own estimator, so numbers match training."""
    return WilsonLoss.wilson_fit(
        torch.as_tensor(i, dtype=torch.float64),
        torch.as_tensor(s_sq, dtype=torch.float64),
        n_bins,
    )


def binned(i, s_sq, n_bins):
    """Equal-count bins in s^2 -> (mean s^2, mean I) per bin.

    Mirrors the binning inside `WilsonLoss.wilson_fit` so the R^2 below
    describes the same fit the model would make.
    """
    keep = i > 0
    x, y = s_sq[keep], i[keep]
    order = np.argsort(x)
    x, y = x[order], y[order]
    idx = np.linspace(0, len(x), n_bins + 1).astype(int)
    edges = [(a, b) for a, b in zip(idx[:-1], idx[1:], strict=True) if b - a > 2]
    xb = np.array([x[a:b].mean() for a, b in edges])
    yb = np.array([y[a:b].mean() for a, b in edges])
    return xb, yb


def r_squared(i, s_sq, G, B, n_bins):
    """How well Wilson's straight line describes this frame."""
    xb, yb = binned(i, s_sq, n_bins)
    obs = np.log(yb)
    pred = np.log(G) - 2.0 * B * xb
    ss_res = ((obs - pred) ** 2).sum()
    ss_tot = ((obs - obs.mean()) ** 2).sum()
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def wilson_plot(path, frames, s_sq, n_bins):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, i, (G, B) in frames:
        keep = i > 0
        x, y = s_sq[keep], i[keep]
        order = np.argsort(x)
        x, y = x[order], y[order]
        idx = np.linspace(0, len(x), n_bins + 1).astype(int)
        edges = list(zip(idx[:-1], idx[1:], strict=True))
        xb = np.array([x[a:b].mean() for a, b in edges if b - a > 2])
        yb = np.array([y[a:b].mean() for a, b in edges if b - a > 2])
        (line,) = ax.plot(xb, np.log(yb), "o", ms=4, label=f"{label} (data)")
        ax.plot(
            xb,
            np.log(G) - 2 * B * xb,
            "-",
            color=line.get_color(),
            label=f"{label}: G={G:.4g}, B={B:.1f}",
        )
    ax.set_xlabel(r"$s^2 = 1/(4d^2)$  [$\AA^{-2}$]")
    ax.set_ylabel(r"$\log \langle I \rangle$")
    ax.set_title("Wilson plot")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nwrote {path}")


def main():
    args = parse_args()
    if not args.refl.exists():
        raise SystemExit(f"no such file: {args.refl}")

    ds = load_refl(args)
    icol = INTENSITY_COLS[args.intensity]
    if icol not in ds:
        raise SystemExit(
            f"'{icol}' not in {args.refl.name}. Present intensity columns: "
            f"{sorted(c for c in ds if c.startswith('intensity'))}"
        )

    i_all = np.asarray(ds[icol], dtype=np.float64)
    d = resolution(ds)
    n_read = len(i_all)

    keep = np.isfinite(i_all) & np.isfinite(d) & (d > 0) & (i_all > 0)
    if args.dmin is not None:
        keep &= d >= args.dmin
    if args.dmax is not None:
        keep &= d <= args.dmax
    i_all, d = i_all[keep], d[keep]
    if len(i_all) < 3 * args.bins:
        raise SystemExit(
            f"only {len(i_all)} usable reflections for {args.bins} bins; "
            "loosen the cutoffs or lower --bins"
        )
    s_sq = 1.0 / (4.0 * d**2)

    has_lp = "lp" in ds
    lp = np.asarray(ds["lp"], dtype=np.float64)[keep] if has_lp else None

    print(f"file          {args.refl}")
    print(f"intensity     {icol}")
    print(f"reflections   {n_read:,} read -> {len(i_all):,} used (I>0, finite d)")
    print(f"resolution    {d.max():.2f} - {d.min():.2f} A   ({args.bins} bins)")
    if has_lp:
        print(f"lp            mean {lp.mean():.4f}, range {lp.min():.4f}-{lp.max():.4f}")
    else:
        print("lp            column absent -- only the as-stored frame is reported")

    frames = []
    want_raw = args.lp in ("both", "raw") and has_lp
    want_corr = args.lp in ("both", "corrected") or not has_lp
    if want_raw:
        i_raw = i_all / np.clip(lp, 1e-8, None)
        frames.append(("raw (lp_correction: false)", i_raw, fit_frame(i_raw, s_sq, args.bins)))
    if want_corr:
        frames.append(
            ("as-stored (lp_correction: true)", i_all, fit_frame(i_all, s_sq, args.bins))
        )

    print(f"\n{'frame':34s} {'G':>12s} {'B (A^2)':>10s} {'R^2':>8s}")
    for label, i, (G, B) in frames:
        print(
            f"{label:34s} {G:12.4g} {B:10.2f} "
            f"{r_squared(i, s_sq, G, B, args.bins):8.5f}"
        )

    if has_lp and len(frames) > 1:
        # If lp trends with resolution, ignoring it biases B as well as G; if it
        # is flat in s^2, the choice only rescales G.
        rho = np.corrcoef(np.log(np.clip(lp, 1e-8, None)), s_sq)[0, 1]
        dB = abs(frames[0][2][1] - frames[1][2][1])
        print(f"\nlp vs resolution:  corr(log lp, s^2) = {rho:+.3f}")
        print(
            f"  the two frames differ by {frames[1][2][0] / frames[0][2][0]:.3f}x in G "
            f"and {dB:.2f} A^2 in B"
        )

    # one recommendation per frame. Printing only the first invites pasting
    # the raw numbers into an lp_correction: true config, which mis-inits G by
    # the mean lp -- a factor of several, and the classic route to a collapsed
    # G/B.
    for label, _, (G, B) in frames:
        setting = "true" if label.startswith("as-stored") else "false"
        print(f"\nfor lp_correction: {setting}   ({label})")
        print(f"    init_G: {G:.4g}")
        print(f"    init_B: {B:.1f}")
    B = frames[0][2][1]
    if B <= 0:
        print(
            "\nNOTE: B <= 0 means <I> rises with resolution -- suspicious. Check the "
            "intensity column and any resolution cutoff before trusting init_B."
        )

    if args.plot:
        wilson_plot(args.plot, frames, s_sq, args.bins)


if __name__ == "__main__":
    main()
