"""Decompose observed Bijvoet differences into signal vs noise components.

Reads a phenix `refine_001.mtz` (which has both observed F(+)/F(-) and
model-predicted F(+)/F(-) from anomalous scattering) and compares them:

    dF_obs   = F-obs-filtered(+) - F-obs-filtered(-)
    dF_model = F-model(+)        - F-model(-)        # real anomalous signal
    sigma_DANO = sqrt(SIGF(+)² + SIGF(-)²)            # statistical sigma

Then:
    pearson(dF_obs, dF_model)  - how much of |dF_obs| is real signal
    residual = dF_obs - dF_model
    |residual| / sigma_DANO    ~1 if residual is purely statistical
                               >1 if there's systematic excess noise

Interpretation:
  - pearson high (>0.5), |residual|/sigma ~1:  model captures real signal,
    noise is statistical -> peaks should be detectable
  - pearson low, |residual|/sigma >>1:         model isn't tracking signal
    AND has systematic excess noise -> architectural fix needed
  - pearson high, |residual|/sigma >>1:        signal IS there but sigmas
    are underestimated -> post-hoc sigma inflation will help

Also breaks down per resolution shell since anomalous signal concentrates
at low-to-mid resolution while noise is uniform.

Usage:
    uv run python scripts/analyze_anomalous_signal.py <variant_dir>/refine_001.mtz
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import reciprocalspaceship as rs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


COL_NAMES = {
    "F_obs_p":    "F-obs-filtered(+)",
    "F_obs_m":    "F-obs-filtered(-)",
    "sig_obs_p":  "SIGF-obs-filtered(+)",
    "sig_obs_m":  "SIGF-obs-filtered(-)",
    "F_model_p":  "F-model(+)",
    "F_model_m":  "F-model(-)",
}


def _safe(arr):
    return np.asarray(arr, dtype=np.float64)


def print_section(name, **stats):
    print(f"\n{name}")
    print("-" * len(name))
    for k, v in stats.items():
        if isinstance(v, (int, np.integer)):
            print(f"  {k:32s} = {v}")
        else:
            print(f"  {k:32s} = {v:.4f}")


def analyze(mtz_path: Path, n_bins: int = 10) -> None:
    logger.info("Reading %s", mtz_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = rs.read_mtz(str(mtz_path))

    missing = [v for v in COL_NAMES.values() if v not in ds.columns]
    if missing:
        raise KeyError(
            f"refine_001.mtz missing columns: {missing}. "
            "Make sure phenix.refine completed and the MTZ has F-obs-filtered and F-model."
        )

    F_obs_p   = _safe(ds[COL_NAMES["F_obs_p"]])
    F_obs_m   = _safe(ds[COL_NAMES["F_obs_m"]])
    sig_p     = _safe(ds[COL_NAMES["sig_obs_p"]])
    sig_m     = _safe(ds[COL_NAMES["sig_obs_m"]])
    F_model_p = _safe(ds[COL_NAMES["F_model_p"]])
    F_model_m = _safe(ds[COL_NAMES["F_model_m"]])

    # Resolution per reflection. rs versions differ on what
    # compute_dHKL returns - handle both.
    if "dHKL" in ds.columns:
        d = _safe(ds["dHKL"])
    else:
        result = ds.compute_dHKL()
        if isinstance(result, rs.DataSet) and "dHKL" in result.columns:
            d = _safe(result["dHKL"])
        else:
            d = _safe(result)

    mask = (
        np.isfinite(F_obs_p) & np.isfinite(F_obs_m)
        & np.isfinite(F_model_p) & np.isfinite(F_model_m)
        & np.isfinite(sig_p) & np.isfinite(sig_m)
        & (sig_p > 0) & (sig_m > 0)
        & (F_obs_p > 0) & (F_obs_m > 0)
    )

    n = int(mask.sum())
    if n < 100:
        raise RuntimeError(f"only {n} complete Friedel pairs - nothing to analyze")

    F_obs_p, F_obs_m = F_obs_p[mask], F_obs_m[mask]
    sig_p, sig_m     = sig_p[mask], sig_m[mask]
    F_model_p, F_model_m = F_model_p[mask], F_model_m[mask]
    d = d[mask]

    dF_obs   = F_obs_p - F_obs_m
    dF_model = F_model_p - F_model_m
    sigma_DANO = np.sqrt(sig_p ** 2 + sig_m ** 2)
    F_mean = 0.5 * (F_obs_p + F_obs_m)
    residual = dF_obs - dF_model

    print(f"\n=== Anomalous signal analysis: {mtz_path.name} ===\n")
    print(f"N Friedel pairs                 = {n}")
    print(f"Resolution range                = {d.min():.2f} – {d.max():.2f} Å")

    # ---- overall magnitudes ----
    print_section(
        "Magnitudes (mean over all pairs)",
        **{
            "<F>":                F_mean.mean(),
            "<|dF_obs|>":         np.abs(dF_obs).mean(),
            "<|dF_model|>":       np.abs(dF_model).mean(),
            "<sigma_DANO>":       sigma_DANO.mean(),
            "<|residual|>":       np.abs(residual).mean(),
        },
    )

    # ---- as fractions of F ----
    print_section(
        "Relative magnitudes (median per-pair |x|/F)",
        **{
            "median |dF_obs|/F":     float(np.median(np.abs(dF_obs) / F_mean)),
            "median |dF_model|/F":   float(np.median(np.abs(dF_model) / F_mean)),
            "median sigma_DANO/F":   float(np.median(sigma_DANO / F_mean)),
            "median |residual|/F":   float(np.median(np.abs(residual) / F_mean)),
        },
    )

    # ---- correlations ----
    pear_signed = float(np.corrcoef(dF_obs, dF_model)[0, 1])
    pear_abs    = float(np.corrcoef(np.abs(dF_obs), np.abs(dF_model))[0, 1])
    print_section(
        "Signal correlations (higher = more of |dF_obs| is real signal)",
        **{
            "pearson(dF_obs, dF_model) signed":   pear_signed,
            "pearson(|dF_obs|, |dF_model|)":      pear_abs,
        },
    )

    # ---- residual interpretation ----
    res_over_sigma = np.abs(residual) / sigma_DANO
    print_section(
        "Residual = dF_obs - dF_model  (the 'unexplained' Bijvoet noise)",
        **{
            "median |residual| / sigma_DANO":     float(np.median(res_over_sigma)),
            "mean   |residual| / sigma_DANO":     float(np.mean(res_over_sigma)),
            "  interpretation":                   0.0,  # placeholder, see below
        },
    )
    med = float(np.median(res_over_sigma))
    print(f"\n  median |residual|/sigma_DANO interpretation:")
    if 0.7 <= med <= 1.3:
        print(f"    {med:.2f} ≈ 1  -> noise is STATISTICAL (matches predicted sigmas)")
    elif med < 0.7:
        print(f"    {med:.2f} < 1  -> sigmas overestimate noise (conservative)")
    else:
        print(f"    {med:.2f} > 1  -> sigmas UNDERestimate noise by ~{med:.1f}* - SYSTEMATIC excess")
        print(f"                   anomalous map noise floor is inflated by ~{med:.1f}*")

    # ---- per-resolution bin ----
    print(f"\nPer-resolution shell ({n_bins} bins, equal pair count):")
    print(f"  {'d_lo':>6}  {'d_hi':>6}  {'N':>6}  {'<F>':>8}  "
          f"{'|dF_obs|/F':>10}  {'|dF_model|/F':>12}  "
          f"{'sigma/F':>8}  {'|res|/sig':>10}  {'r(dFo,dFm)':>11}")
    order = np.argsort(-d)  # high d first (low resolution)
    bin_edges = np.linspace(0, n, n_bins + 1, dtype=int)
    for b in range(n_bins):
        idx = order[bin_edges[b]:bin_edges[b + 1]]
        if len(idx) < 10:
            continue
        di = d[idx]
        dFo = dF_obs[idx]
        dFm = dF_model[idx]
        Fm  = F_mean[idx]
        sg  = sigma_DANO[idx]
        rs_ = residual[idx]
        r_bin = float(np.corrcoef(dFo, dFm)[0, 1]) if np.std(dFm) > 0 else float("nan")
        print(
            f"  {di.max():6.2f}  {di.min():6.2f}  {len(idx):6d}  {Fm.mean():8.2f}  "
            f"{float(np.median(np.abs(dFo)/Fm)):10.4f}  "
            f"{float(np.median(np.abs(dFm)/Fm)):12.4f}  "
            f"{float(np.median(sg/Fm)):8.4f}  "
            f"{float(np.median(np.abs(rs_)/sg)):10.2f}  "
            f"{r_bin:11.3f}"
        )

    print("\nWhat to look for:")
    print("  - |dF_obs|/F   :  what you actually measure per HKL")
    print("  - |dF_model|/F :  what the refined structure predicts (real signal)")
    print("  - sigma/F      :  statistical-only floor (from SIGF)")
    print("  - |res|/sig    :  ~1 = noise calibrated, >>1 = systematic excess")
    print("  - r(dFo,dFm)   :  per-bin correlation of obs and model Bijvoet")
    print("                    high at low-res = signal captured")
    print("                    low everywhere = noise dominates\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mtz_path", type=Path, help="phenix refine_001.mtz")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    if not args.mtz_path.exists():
        raise FileNotFoundError(args.mtz_path)
    analyze(args.mtz_path, n_bins=args.n_bins)


if __name__ == "__main__":
    main()
