"""Compare two per-observation DIALS `.refl` files head-to-head (model A vs B).

Both files are expected to be the same reflections with model-written
intensities (e.g. two integrators, or an integrator vs an amortized model's
per-observation scaled intensity), so observations are matched on the stable
geometric key (Miller index + calculated centroid xyzcal.px). Reports, overall
and per resolution shell:

  - CC of per-observation intensities (linear, log, Spearman).
  - ratio I_A / I_B (geometric mean, median, MAD) -- systematic scale/offset.
  - I/sigma for each model -- relative confidence (does one over/under-state it).
  - overlap of the two observation sets.

Reads via reciprocalspaceship's stills reader (no DIALS runtime needed).

Usage:
    uv run python scripts/compare_refl.py \
        --refl1 amortized/observations.refl --label1 amortized \
        --refl2 integrator/integrated.refl --label2 integrator \
        [--col intensity.prf.value] [--bins 10] [--out-dir refl_compare]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import reciprocalspaceship as rs
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compare_refl")

_EXTRA_COLS = [
    "intensity.prf.value", "intensity.prf.variance",
    "intensity.sum.value", "intensity.sum.variance",
    "xyzcal.px", "d",
]


def _pos_cols(ds: rs.DataSet) -> list[str] | None:
    """The three xyzcal.px component column names, however the reader split them."""
    for triple in (["xyzcal.px.0", "xyzcal.px.1", "xyzcal.px.2"],
                   ["xyzcal.px_0", "xyzcal.px_1", "xyzcal.px_2"]):
        if all(c in ds.columns for c in triple):
            return triple
    return None


def _load(path: Path, col: str) -> dict:
    """Read a stills `.refl` into HKL+position-keyed intensity / sigma / d arrays."""
    ds = rs.io.read_dials_stills(str(path), extra_cols=_EXTRA_COLS)
    if col not in ds.columns:
        alt = "intensity.sum.value"
        if alt not in ds.columns:
            raise KeyError(f"{col} not in {list(ds.columns)}")
        logger.warning("%s missing %s; using %s", path.name, col, alt)
        col = alt
    var_col = col.replace(".value", ".variance")
    var = ds[var_col].to_numpy(np.float64) if var_col in ds.columns else None

    pos = _pos_cols(ds)
    if pos is None:
        raise KeyError(f"No xyzcal.px columns in {path.name}: {list(ds.columns)}")
    xyz = np.stack([ds[c].to_numpy(np.float64) for c in pos], axis=1)
    hkl = ds.get_hkls().astype(np.int64)
    d = ds["d"].to_numpy(np.float64) if "d" in ds.columns else \
        ds.compute_dHKL()["dHKL"].to_numpy(np.float64)
    out = {
        "I": ds[col].to_numpy(np.float64),
        "SIG": np.sqrt(np.clip(var, 0, None)) if var is not None else None,
        "d": d,
        # per-observation key: HKL + centroid rounded to 0.1 px (bitwise-equal
        # across files derived from the same template, so rounding is just slop).
        "key": np.array(["%d,%d,%d,%.1f,%.1f,%.1f" % (h, k, l, x, y, z)
                         for (h, k, l), (x, y, z) in zip(hkl, xyz)]),
        "n": len(ds),
        "col": col,
    }
    logger.info("%s: %d observations (intensity=%s)", path.name, out["n"], col)
    return out


def _match(a: dict, b: dict) -> dict:
    """Inner-join two observation sets on the geometric key."""
    idx_b = {k: j for j, k in enumerate(b["key"])}
    ia, ib = [], []
    for j, k in enumerate(a["key"]):
        m = idx_b.get(k)
        if m is not None:
            ia.append(j)
            ib.append(m)
    ia, ib = np.array(ia, np.int64), np.array(ib, np.int64)
    out = {"n_common": len(ia)}
    out["I1"], out["I2"] = a["I"][ia], b["I"][ib]
    out["d1"], out["d2"] = a["d"][ia], b["d"][ib]
    out["S1"] = a["SIG"][ia] if a["SIG"] is not None else None
    out["S2"] = b["SIG"][ib] if b["SIG"] is not None else None
    return out


def _cc(x, y) -> tuple[float, int]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), int(ok.sum())
    return float(pearsonr(x[ok], y[ok])[0]), int(ok.sum())


def _log_cc(x, y) -> float:
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return float(pearsonr(np.log(x[ok]), np.log(y[ok]))[0]) if ok.sum() >= 3 \
        else float("nan")


def _spearman(x, y) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok]).correlation) if ok.sum() >= 3 \
        else float("nan")


def _ratio_stats(x, y) -> tuple[float, float, float]:
    """Geometric-mean, median, and MAD of the ratio x/y over positive pairs."""
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    r = x[ok] / y[ok]
    gmean = float(np.exp(np.mean(np.log(r)))) if r.size else float("nan")
    med = float(np.median(r)) if r.size else float("nan")
    mad = float(np.median(np.abs(r - med))) if r.size else float("nan")
    return gmean, med, mad


def _isig(I, S) -> float:
    if S is None:
        return float("nan")
    ok = np.isfinite(I) & np.isfinite(S) & (S > 0)
    return float(np.median(I[ok] / S[ok])) if ok.sum() else float("nan")


def _bins(d: np.ndarray, n_bins: int) -> list[tuple[float, float]]:
    """Shells as (d_hi, d_lo), d_hi > d_lo, low-resolution first (see compare_mtz)."""
    dd = d[np.isfinite(d)]
    if dd.size < n_bins * 2:
        return [(float(np.nanmax(d)), float(np.nanmin(d)))] if dd.size else []
    inv = np.quantile((1.0 / dd) ** 3, np.linspace(0, 1, n_bins + 1))
    e = inv ** (-1.0 / 3.0)  # -> d, decreasing (low-res to high-res)
    return [(float(e[i]), float(e[i + 1])) for i in range(n_bins)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refl1", type=Path, required=True)
    p.add_argument("--refl2", type=Path, required=True)
    p.add_argument("--label1", default="refl1")
    p.add_argument("--label2", default="refl2")
    p.add_argument("--col", default="intensity.prf.value",
                   help="intensity column to compare")
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=Path("refl_compare"))
    args = p.parse_args()

    a, b = _load(args.refl1, args.col), _load(args.refl2, args.col)
    m = _match(a, b)
    if m["n_common"] == 0:
        raise SystemExit(
            "No shared observations (HKL+centroid). The two .refl files may not "
            "be derived from the same predicted reflections."
        )
    frac = m["n_common"] / max(min(a["n"], b["n"]), 1)
    if frac < 0.5:
        logger.warning(
            "Only %.0f%% of the smaller set matched -- the two .refl files may "
            "cover different predicted reflections (test split?).", 100 * frac,
        )

    cc, n_cc = _cc(m["I1"], m["I2"])
    logcc = _log_cc(m["I1"], m["I2"])
    sp = _spearman(m["I1"], m["I2"])
    gmean, med, mad = _ratio_stats(m["I1"], m["I2"])
    isig1, isig2 = _isig(m["I1"], m["S1"]), _isig(m["I2"], m["S2"])

    # sigma(I) / error-model calibration. Median I/sigma is scale-invariant (both
    # I and sigma scale with the intensity gauge), so it compares confidence
    # directly. The sigma ratio is scale-matched by the intensity ratio gmean
    # (I1 ~ gmean*I2 => sigma1 ~ gmean*sigma2 if the error models agree).
    has_sig = m["S1"] is not None and m["S2"] is not None
    if has_sig:
        s_cc, n_scc = _cc(m["S1"], m["S2"])
        s_gmean, s_med, _ = _ratio_stats(m["S1"], gmean * m["S2"])
    else:
        s_cc, n_scc, s_gmean, s_med = float("nan"), 0, float("nan"), float("nan")

    L1, L2 = args.label1, args.label2
    lines = [
        "=" * 64,
        f"PER-OBSERVATION REFL COMPARISON   {L1}  vs  {L2}",
        "=" * 64,
        f"  observations:  {L1}={a['n']:>8d}   {L2}={b['n']:>8d}   "
        f"common={m['n_common']:>8d}",
        f"  overlap:       {100 * m['n_common'] / max(min(a['n'], b['n']), 1):5.1f}%"
        " of the smaller set",
        f"  intensity col: {a['col']}",
        "",
        "  -- intensity agreement --",
        f"  CC (linear):   {cc:6.3f}   (n={n_cc})",
        f"  CC (log):      {logcc:6.3f}",
        f"  CC (Spearman): {sp:6.3f}",
        f"  ratio {L1}/{L2}:  gmean={gmean:.4g}  median={med:.4g}  MAD={mad:.3g}",
        "",
        "  -- sigma(I) / error-model calibration --",
        f"  median I/sigma:  {L1}={isig1:6.2f}   {L2}={isig2:6.2f}",
    ]
    if has_sig:
        lines += [
            f"  sigma CC (linear):  {s_cc:6.3f}   (n={n_scc})",
            f"  sigma ratio {L1}/{L2} (scale-matched):  gmean={s_gmean:.4g}  "
            f"median={s_med:.4g}",
        ]
    else:
        lines.append("  (no intensity.*.variance column -> sigma comparison skipped)")
    lines.append("")

    d_all = np.where(np.isfinite(m["d1"]), m["d1"], m["d2"])
    shells = _bins(d_all, args.bins)
    bc, bcc, bratio = [], [], []
    if shells:
        lines.append("  -- per resolution shell (low-res -> high-res) --")
        lines.append(f"  {'d_hi':>6} {'d_lo':>6} {'n':>7} {'CC':>6} {'med I1/I2':>10}")
        for d_hi, d_lo in shells:
            sel = (d_all <= d_hi) & (d_all > d_lo)
            if sel.sum() < 3:
                continue
            scc, _ = _cc(m["I1"][sel], m["I2"][sel])
            _, smed, _ = _ratio_stats(m["I1"][sel], m["I2"][sel])
            lines.append(f"  {d_hi:6.2f} {d_lo:6.2f} {int(sel.sum()):7d} "
                         f"{scc:6.3f} {smed:10.4g}")
            bc.append(0.5 * (d_hi + d_lo)); bcc.append(scc); bratio.append(smed)
    lines.append("=" * 64)
    report = "\n".join(lines)
    print(report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.txt").write_text(report + "\n")
    _plots(m, args.out_dir, L1, L2, cc, bc, bcc, bratio, gmean)
    logger.info("Wrote report + plots to %s", args.out_dir)


def _plots(m, out_dir, L1, L2, cc, bc, bcc, bratio, gmean) -> None:
    ok = np.isfinite(m["I1"]) & np.isfinite(m["I2"]) & (m["I1"] > 0) & (m["I2"] > 0)
    if ok.sum() >= 10:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(m["I1"][ok], m["I2"][ok], s=3, alpha=0.2, edgecolors="none")
        lo = float(min(m["I1"][ok].min(), m["I2"][ok].min()))
        hi = float(max(m["I1"][ok].max(), m["I2"][ok].max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"{L1}  intensity"); ax.set_ylabel(f"{L2}  intensity")
        ax.set_title(f"per-obs intensity   CC={cc:.3f}  n={int(ok.sum())}")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / "intensity_scatter.png", dpi=130)
        plt.close(fig)

        r = m["I1"][ok] / m["I2"][ok]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(np.log10(r), bins=80, color="steelblue", alpha=0.8)
        ax.axvline(0, color="r", ls="--", lw=1, label="ratio = 1")
        ax.set_xlabel(f"log10( {L1} / {L2} )"); ax.set_ylabel("observations")
        ax.set_title("intensity ratio"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / "ratio_hist.png", dpi=130)
        plt.close(fig)

    # sigma(I) scatter (log-log), scale-matched by the intensity ratio so it
    # isolates error-model calibration from any overall intensity-scale gap.
    if m["S1"] is not None and m["S2"] is not None:
        s2 = gmean * m["S2"]
        ok = np.isfinite(m["S1"]) & np.isfinite(s2) & (m["S1"] > 0) & (s2 > 0)
        if ok.sum() >= 10:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(m["S1"][ok], s2[ok], s=3, alpha=0.2, edgecolors="none")
            lo = float(min(m["S1"][ok].min(), s2[ok].min()))
            hi = float(max(m["S1"][ok].max(), s2[ok].max()))
            ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(f"{L1}  $\\sigma$(I)")
            ax.set_ylabel(f"{L2}  gmean$\\cdot\\sigma$(I)")
            ax.set_title(f"sigma(I), scale-matched   n={int(ok.sum())}")
            ax.legend(loc="upper left", fontsize=8)
            fig.tight_layout(); fig.savefig(out_dir / "sigma_scatter.png", dpi=130)
            plt.close(fig)

    if bc:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        ax[0].plot(bc, bcc, "o-"); ax[0].invert_xaxis()
        ax[0].set_xlabel("d (A)"); ax[0].set_ylabel("CC"); ax[0].grid(alpha=0.3)
        ax[0].set_title("CC vs resolution")
        ax[1].plot(bc, bratio, "s-", color="darkorange"); ax[1].invert_xaxis()
        ax[1].axhline(1.0, color="r", ls="--", lw=1)
        ax[1].set_xlabel("d (A)"); ax[1].set_ylabel(f"median {L1}/{L2}")
        ax[1].grid(alpha=0.3); ax[1].set_title("ratio vs resolution")
        fig.tight_layout(); fig.savefig(out_dir / "vs_resolution.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
