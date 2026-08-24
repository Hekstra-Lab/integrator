"""Compare two merged MTZ files head-to-head (model A vs model B).

Built to compare the best amortized-merging model against the best
integrator-then-DIALS model, but works for any two merged MTZs. Matches on
Miller index and reports, both overall and per resolution shell:

  - CC of merged intensities (linear, log, Spearman) -- overall agreement.
  - R_iso = sum|k*I_A - I_B| / sum(0.5(k*I_A + I_B)) after a least-squares
    scale k -- a scale-free magnitude-of-disagreement number.
  - CCanom = CC of the anomalous differences DANO = I(+) - I(-) between the two
    datasets -- the headline for anomalous work: do the two merges agree on the
    Bijvoet signal, not just the bulk intensity.
  - Completeness / overlap of the two HKL sets.

Robust to column naming: merged intensity is read from IMEAN, else the mean of
I(+)/I(-), else I; anomalous from I(+)/I(-) (falls back to F(+)/F(-)). Reads via
reciprocalspaceship (no DIALS/gemmi runtime needed beyond rs).

Usage:
    uv run python scripts/compare_mtz.py \
        --mtz1 amortized/merged.mtz --label1 amortized \
        --mtz2 integrator/dials_merged.mtz --label2 integrator \
        [--bins 10] [--out-dir mtz_compare]
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
logger = logging.getLogger("compare_mtz")


def _col(ds: rs.DataSet, *names: str) -> np.ndarray | None:
    """First present column among `names`, as a float64 numpy array (else None)."""
    for n in names:
        if n in ds.columns:
            return ds[n].to_numpy(dtype=np.float64)
    return None


def _merged_intensity(ds: rs.DataSet) -> tuple[np.ndarray, np.ndarray | None]:
    """Per-HKL merged intensity (and sigma) from whatever columns exist.

    Order of preference: IMEAN (true merged mean), else the mean of the Friedel
    mates I(+)/I(-), else a plain I column. Sigma follows the same source.
    """
    imean = _col(ds, "IMEAN")
    if imean is not None:
        return imean, _col(ds, "SIGIMEAN", "SIGI")
    ip, im = _col(ds, "I(+)"), _col(ds, "I(-)")
    if ip is not None and im is not None:
        i = np.nanmean(np.stack([ip, im]), axis=0)
        sp, sm = _col(ds, "SIGI(+)"), _col(ds, "SIGI(-)")
        sig = None
        if sp is not None and sm is not None:
            sig = 0.5 * np.sqrt(np.nan_to_num(sp) ** 2 + np.nan_to_num(sm) ** 2)
        return i, sig
    i = _col(ds, "I", "IOBS", "F")  # last resort: amplitude if no intensity
    if i is None:
        raise KeyError(f"No intensity column found in {list(ds.columns)}")
    return i, _col(ds, "SIGI", "SIGF")


def _anomalous_diff(ds: rs.DataSet) -> np.ndarray | None:
    """DANO = I(+) - I(-) per base HKL (falls back to F(+) - F(-)); None if absent."""
    ip, im = _col(ds, "I(+)"), _col(ds, "I(-)")
    if ip is not None and im is not None:
        return ip - im
    fp, fm = _col(ds, "F(+)"), _col(ds, "F(-)")
    if fp is not None and fm is not None:
        return fp - fm
    return None


def _load(path: Path) -> dict:
    """Read an MTZ into HKL-keyed arrays: intensity, sigma, DANO, resolution."""
    ds = rs.read_mtz(str(path))
    try:
        d = ds.compute_dHKL()["dHKL"].to_numpy(dtype=np.float64)
    except Exception:
        d = np.full(len(ds), np.nan)
    i, sig = _merged_intensity(ds)
    out = {
        "hkl": ds.get_hkls().astype(np.int64),
        "I": i,
        "SIG": sig if sig is not None else np.full_like(i, np.nan),
        "DANO": _anomalous_diff(ds),
        "d": d,
        "columns": list(ds.columns),
        "n": len(ds),
    }
    logger.info("%s: %d reflections, columns=%s", path.name, out["n"],
                out["columns"])
    return out


def _hkl_keys(hkl: np.ndarray) -> np.ndarray:
    """1-D structured/string key per HKL row for fast set intersection."""
    return np.array(["%d,%d,%d" % (h, k, l) for h, k, l in hkl])


def _match(a: dict, b: dict) -> dict:
    """Inner-join two loaded MTZs on Miller index; carry aligned arrays."""
    ka, kb = _hkl_keys(a["hkl"]), _hkl_keys(b["hkl"])
    idx_b = {k: j for j, k in enumerate(kb)}
    ia, ib = [], []
    for j, k in enumerate(ka):
        m = idx_b.get(k)
        if m is not None:
            ia.append(j)
            ib.append(m)
    ia, ib = np.array(ia, dtype=np.int64), np.array(ib, dtype=np.int64)
    out = {"n_common": len(ia)}
    for key in ("I", "SIG", "d"):
        out[key + "1"] = a[key][ia]
        out[key + "2"] = b[key][ib]
    for src, dst in ((a, "DANO1"), (b, "DANO2")):
        out[dst] = src["DANO"][ia if src is a else ib] \
            if src["DANO"] is not None else None
    return out


def _cc(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Pearson CC over finite pairs; returns (cc, n_used)."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), int(ok.sum())
    return float(pearsonr(x[ok], y[ok])[0]), int(ok.sum())


def _log_cc(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 3:
        return float("nan")
    return float(pearsonr(np.log(x[ok]), np.log(y[ok]))[0])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).correlation)


def _ratio(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Geometric-mean and median of x/y over positive finite pairs."""
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    r = x[ok] / y[ok]
    if r.size == 0:
        return float("nan"), float("nan")
    return float(np.exp(np.mean(np.log(r)))), float(np.median(r))


def _isig(I: np.ndarray, S: np.ndarray) -> float:
    """Median I/sigma over finite, positive-sigma pairs."""
    ok = np.isfinite(I) & np.isfinite(S) & (S > 0)
    return float(np.median(I[ok] / S[ok])) if ok.sum() else float("nan")


def _scale_and_riso(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares scale k (y ~ k*x) and R_iso between the two, scale-matched."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    denom = float((x * x).sum())
    k = float((x * y).sum() / denom) if denom > 0 else float("nan")
    num = np.abs(k * x - y).sum()
    den = (0.5 * np.abs(k * x + y)).sum()
    r_iso = float(num / den) if den > 0 else float("nan")
    return k, r_iso


def _resolution_bins(d: np.ndarray, n_bins: int) -> list[tuple[float, float]]:
    """Shells as (d_hi, d_lo) with d_hi > d_lo, low-resolution (high d) first.

    Edges are equal-volume-ish: quantiles of d^-3 over the finite d-spacings.
    """
    dd = d[np.isfinite(d)]
    if dd.size < n_bins * 2:
        return [(float(np.nanmax(d)), float(np.nanmin(d)))] if dd.size else []
    inv = np.quantile((1.0 / dd) ** 3, np.linspace(0, 1, n_bins + 1))
    edges = inv ** (-1.0 / 3.0)  # -> d, DECREASING (low-res to high-res)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(n_bins)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mtz1", type=Path, required=True)
    p.add_argument("--mtz2", type=Path, required=True)
    p.add_argument("--label1", default="mtz1")
    p.add_argument("--label2", default="mtz2")
    p.add_argument("--bins", type=int, default=10, help="resolution shells")
    p.add_argument("--out-dir", type=Path, default=Path("mtz_compare"))
    args = p.parse_args()

    a, b = _load(args.mtz1), _load(args.mtz2)
    m = _match(a, b)
    n_common = m["n_common"]
    if n_common == 0:
        raise SystemExit(
            "No shared Miller indices. The two MTZs may be in different ASUs; "
            "re-index both to a common setting before comparing."
        )
    frac = n_common / max(min(a["n"], b["n"]), 1)
    if frac < 0.5:
        logger.warning(
            "Only %.0f%% of the smaller set matched -- the two MTZs may use a "
            "different ASU/indexing convention; treat the numbers with care.",
            100 * frac,
        )

    cc, n_cc = _cc(m["I1"], m["I2"])
    logcc = _log_cc(m["I1"], m["I2"])
    sp = _spearman(m["I1"], m["I2"])
    k, r_iso = _scale_and_riso(m["I1"], m["I2"])

    ccanom = float("nan")
    n_anom = 0
    if m["DANO1"] is not None and m["DANO2"] is not None:
        ccanom, n_anom = _cc(m["DANO1"], m["DANO2"])

    # sigma(I): error-model calibration. SIG is scaled with I, so compare on the
    # SAME scale k (sigma_1 -> k*sigma_1) before the ratio, else a pure intensity
    # scale difference masquerades as a sigma miscalibration.
    has_sig = np.isfinite(m["SIG1"]).any() and np.isfinite(m["SIG2"]).any()
    s_cc, n_scc = _cc(m["SIG1"], m["SIG2"]) if has_sig else (float("nan"), 0)
    s_gmean, s_med = _ratio(k * m["SIG1"], m["SIG2"]) if has_sig else \
        (float("nan"), float("nan"))
    isig1, isig2 = _isig(m["I1"], m["SIG1"]), _isig(m["I2"], m["SIG2"])

    L1, L2 = args.label1, args.label2
    lines = [
        "=" * 66,
        f"MERGED MTZ COMPARISON   {L1}  vs  {L2}",
        "=" * 66,
        f"  reflections:   {L1}={a['n']:>8d}   {L2}={b['n']:>8d}   "
        f"common={n_common:>8d}",
        f"  overlap:       {100 * n_common / max(min(a['n'], b['n']), 1):5.1f}% "
        "of the smaller set",
        "",
        "  -- merged intensity agreement --",
        f"  CC (linear):   {cc:6.3f}   (n={n_cc})",
        f"  CC (log):      {logcc:6.3f}",
        f"  CC (Spearman): {sp:6.3f}",
        f"  scale k ({L1}->{L2}): {k:.4g}    R_iso={r_iso:6.3f}",
        "",
        "  -- sigma(I) / error-model calibration --",
    ]
    if has_sig:
        lines += [
            f"  CC (linear):   {s_cc:6.3f}   (n={n_scc})",
            f"  ratio {L1}/{L2} (scale-matched):  gmean={s_gmean:.4g}  "
            f"median={s_med:.4g}",
            f"  median I/sigma:  {L1}={isig1:6.2f}   {L2}={isig2:6.2f}",
        ]
    else:
        lines.append("  n/a (one or both MTZs lack a SIGI/SIGF column)")
    lines += ["", "  -- anomalous agreement (the headline) --"]
    if n_anom:
        lines.append(f"  CCanom:        {ccanom:6.3f}   (n={n_anom} Bijvoet pairs)")
    else:
        lines.append("  CCanom:        n/a (one or both MTZs lack I(+)/I(-))")
    lines.append("")

    # Per-resolution-shell table.
    d_all = np.where(np.isfinite(m["d1"]), m["d1"], m["d2"])
    shells = _resolution_bins(d_all, args.bins)
    if shells:
        lines.append("  -- per resolution shell (low-res -> high-res) --")
        lines.append(f"  {'d_hi':>6} {'d_lo':>6} {'n':>6} {'CC':>6} "
                     f"{'CCanom':>7} {'R_iso':>6}")
        bin_centers, bin_cc, bin_ccanom = [], [], []
        for d_hi, d_lo in shells:
            sel = (d_all <= d_hi) & (d_all > d_lo)
            if sel.sum() < 3:
                continue
            bcc, _ = _cc(m["I1"][sel], m["I2"][sel])
            _, briso = _scale_and_riso(m["I1"][sel], m["I2"][sel])
            banom = float("nan")
            if m["DANO1"] is not None and m["DANO2"] is not None:
                banom, _ = _cc(m["DANO1"][sel], m["DANO2"][sel])
            lines.append(f"  {d_hi:6.2f} {d_lo:6.2f} {int(sel.sum()):6d} "
                         f"{bcc:6.3f} {banom:7.3f} {briso:6.3f}")
            bin_centers.append(0.5 * (d_hi + d_lo))
            bin_cc.append(bcc)
            bin_ccanom.append(banom)
    else:
        bin_centers = bin_cc = bin_ccanom = []
    lines.append("=" * 66)
    report = "\n".join(lines)
    print(report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.txt").write_text(report + "\n")
    _plots(m, args.out_dir, L1, L2, bin_centers, bin_cc, bin_ccanom, cc, ccanom, k)
    logger.info("Wrote report + plots to %s", args.out_dir)


def _plots(m, out_dir, L1, L2, bc, bcc, banom, cc, ccanom, k) -> None:
    # 1. Merged-I scatter (log-log).
    ok = np.isfinite(m["I1"]) & np.isfinite(m["I2"]) & (m["I1"] > 0) & (m["I2"] > 0)
    if ok.sum() >= 10:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(m["I1"][ok], m["I2"][ok], s=3, alpha=0.25, edgecolors="none")
        lo = float(min(m["I1"][ok].min(), m["I2"][ok].min()))
        hi = float(max(m["I1"][ok].max(), m["I2"][ok].max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"{L1}  merged I"); ax.set_ylabel(f"{L2}  merged I")
        ax.set_title(f"merged intensity   CC={cc:.3f}  n={int(ok.sum())}")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / "intensity_scatter.png", dpi=130)
        plt.close(fig)

    # 2. sigma(I) scatter (log-log), scale-matched so it isolates calibration.
    s1 = k * m["SIG1"]
    ok = np.isfinite(s1) & np.isfinite(m["SIG2"]) & (s1 > 0) & (m["SIG2"] > 0)
    if ok.sum() >= 10:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(s1[ok], m["SIG2"][ok], s=3, alpha=0.25, edgecolors="none")
        lo = float(min(s1[ok].min(), m["SIG2"][ok].min()))
        hi = float(max(s1[ok].max(), m["SIG2"][ok].max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"{L1}  k$\\cdot\\sigma$(I)"); ax.set_ylabel(f"{L2}  $\\sigma$(I)")
        ax.set_title(f"sigma(I), scale-matched   n={int(ok.sum())}")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir / "sigma_scatter.png", dpi=130)
        plt.close(fig)

    # 3. Anomalous-difference scatter (the CCanom plot), linear and signed.
    if m["DANO1"] is not None and m["DANO2"] is not None:
        ok = np.isfinite(m["DANO1"]) & np.isfinite(m["DANO2"])
        if ok.sum() >= 10:
            x, y = m["DANO1"][ok], m["DANO2"][ok]
            lim = float(np.percentile(np.abs(np.concatenate([x, y])), 99))
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(x, y, s=3, alpha=0.25, edgecolors="none")
            ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="y = x")
            ax.axhline(0, color="0.8", lw=0.5); ax.axvline(0, color="0.8", lw=0.5)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_xlabel(f"{L1}  DANO = I(+)-I(-)")
            ax.set_ylabel(f"{L2}  DANO")
            ax.set_title(f"anomalous difference   CCanom={ccanom:.3f}")
            ax.legend(loc="upper left", fontsize=8)
            fig.tight_layout(); fig.savefig(out_dir / "anomalous_scatter.png", dpi=130)
            plt.close(fig)

    # 3. Per-shell CC / CCanom vs resolution.
    if bc:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(bc, bcc, "o-", label="CC (intensity)")
        if np.isfinite(banom).any():
            ax.plot(bc, banom, "s-", label="CCanom")
        ax.invert_xaxis()  # high resolution to the right
        ax.set_xlabel("resolution  d (A)"); ax.set_ylabel("CC")
        ax.set_ylim(min(0.0, np.nanmin(banom) if banom else 0.0) - 0.05, 1.02)
        ax.set_title(f"{L1} vs {L2}  by resolution")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "cc_vs_resolution.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
