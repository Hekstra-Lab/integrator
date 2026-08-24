"""Write the canonical merging_stats.csv for a careless-scaled Laue run.

Both arms of the project emit the same table next to their merged MTZ, so
one figure can put DIALS-scaled monochromatic runs and careless-scaled Laue
runs on the same axes:

    bin, d_max, d_min, n_obs, n_unique, cc_half, cc_anom, r_pim, i_over_sigma

CC1/2, CCanom and I/sigma come from careless's own tools, which know how the
half-datasets were split. R-pim needs multiplicity and so is computed here
from the unmerged input, the same quantity DIALS reports.

Usage:
    python scripts/poly/emit_merging_stats.py \
        --scaling-dir <predictions>/epoch_0039/scaling --config 4 \
        --unmerged <predictions>/epoch_0039/preds_epoch_0039.mtz
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

COLUMNS = [
    "bin",
    "d_max",
    "d_min",
    "n_obs",
    "n_unique",
    "cc_half",
    "cc_anom",
    "r_pim",
    "i_over_sigma",
]


def parse_args():
    p = argparse.ArgumentParser(description="Canonical merging stats (Laue)")
    p.add_argument("--scaling-dir", required=True, type=Path)
    p.add_argument("--config", default=4, type=int)
    p.add_argument(
        "--unmerged",
        type=Path,
        default=None,
        help="Unmerged MTZ from integrator.predict; needed for R-pim",
    )
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--careless-env",
        default="crls",
        help="micromamba env holding careless.* (empty to use PATH)",
    )
    return p.parse_args()


def run_careless_tool(tool: str, mtz: Path, env: str, bins: int):
    """Run a careless.* statistic and return its CSV as a DataFrame.

    The console scripts in that environment predate the storage migration
    and carry dead shebangs, so they are invoked through python.
    """
    out = Path(tempfile.mkdtemp()) / f"{tool}.csv"
    cmd = (
        f'python "$(command -v careless.{tool})" {mtz} '
        f"-b {bins} -o {out}"
    )
    if env:
        hook = (
            "/n/lab_storage/hekstra_lab/people/aldama/micromamba/"
            "etc/profile.d/mamba.sh"
        )
        cmd = f"source {hook} && micromamba activate {env} && {cmd}"
    proc = subprocess.run(
        ["bash", "-lc", cmd], capture_output=True, text=True, check=False
    )
    if proc.returncode or not out.exists():
        print(f"  careless.{tool} unavailable: {proc.stderr.strip()[:160]}")
        return None
    return pl.read_csv(out)


def _resolution_edges(d: np.ndarray, bins: int) -> np.ndarray:
    """Equal-volume-ish shells: quantiles of 1/d^2."""
    s = 1.0 / np.clip(d, 1e-6, None) ** 2
    return np.quantile(s, np.linspace(0, 1, bins + 1))


def unmerged_stats(path: Path, bins: int) -> pl.DataFrame:
    """Per-shell n_obs, n_unique, R-pim and I/sigma from the unmerged MTZ."""
    import reciprocalspaceship as rs

    ds = rs.read_mtz(str(path))
    ds = ds.compute_dHKL().hkl_to_asu()
    d = ds["dHKL"].to_numpy().astype(float)
    intensity = ds["I"].to_numpy().astype(float)
    sigma = ds["SIGI"].to_numpy().astype(float)
    # one integer id per unique ASU reflection
    hkl = np.ascontiguousarray(ds.get_hkls().astype(np.int64))
    _, group = np.unique(hkl, axis=0, return_inverse=True)

    edges = _resolution_edges(d, bins)
    s = 1.0 / np.clip(d, 1e-6, None) ** 2
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (s >= lo) & (s <= hi if i == bins - 1 else s < hi)
        if sel.sum() == 0:
            continue
        g = group[sel]
        obs = intensity[sel]
        order = np.argsort(g)
        g_sorted, obs_sorted = g[order], obs[order]
        _, starts, counts = np.unique(
            g_sorted, return_index=True, return_counts=True
        )
        sums = np.add.reduceat(obs_sorted, starts)
        means = sums / counts
        # R-pim = sum_h sqrt(1/(n_h-1)) sum_i |I_hi - <I_h>| / sum_h sum_i I_hi
        expanded = np.repeat(means, counts)
        abs_dev = np.abs(obs_sorted - expanded)
        dev_per_group = np.add.reduceat(abs_dev, starts)
        multi = np.where(counts > 1, np.sqrt(1.0 / (counts - 1)), 0.0)
        denom = np.abs(obs_sorted).sum()
        r_pim = float((multi * dev_per_group).sum() / denom) if denom else None
        with np.errstate(divide="ignore", invalid="ignore"):
            isigi = float(
                np.nanmean(obs / np.where(sigma[sel] > 0, sigma[sel], np.nan))
            )
        rows.append(
            {
                "bin": i,
                "d_max": float(1.0 / np.sqrt(max(lo, 1e-12))),
                "d_min": float(1.0 / np.sqrt(max(hi, 1e-12))),
                "n_obs": int(sel.sum()),
                "n_unique": int(len(counts)),
                "r_pim": r_pim,
                "i_over_sigma": isigi,
            }
        )
    return pl.DataFrame(rows)


def _careless_column(frame: pl.DataFrame, wanted: str) -> pl.Series | None:
    for name in frame.columns:
        if name.lower().replace(" ", "").replace("_", "") == wanted:
            return frame[name]
    return None


def main():
    args = parse_args()
    scaling = args.scaling_dir
    base = f"config{args.config}"
    out = args.out or scaling / "merging_stats.csv"

    xval = scaling / f"{base}_xval_0.mtz"
    merged = scaling / f"{base}_0.mtz"
    if not xval.exists():
        raise SystemExit(f"no cross-validation MTZ at {xval}")

    table = None
    if args.unmerged and args.unmerged.exists():
        table = unmerged_stats(args.unmerged, args.bins)
        print(f"  unmerged: {len(table)} shells from {args.unmerged.name}")
    else:
        print("  no --unmerged MTZ: n_obs, n_unique, r_pim, i_over_sigma blank")

    for tool, column in (("cchalf", "cc_half"), ("ccanom", "cc_anom")):
        frame = run_careless_tool(tool, xval, args.careless_env, args.bins)
        if frame is None:
            continue
        values = _careless_column(frame, column.replace("_", ""))
        if values is None or table is None or len(values) != len(table):
            print(f"  {column}: shape mismatch, left blank")
            continue
        table = table.with_columns(values.alias(column))

    if table is None:
        raise SystemExit("nothing to write: no unmerged MTZ and no careless stats")

    for column in COLUMNS:
        if column not in table.columns:
            table = table.with_columns(pl.lit(None).alias(column))
    table = table.select(COLUMNS).sort("bin")
    table.write_csv(out)
    print(f"wrote {out} ({len(table)} shells)  [merged: {merged.name}]")


if __name__ == "__main__":
    main()
