#!/usr/bin/env python
"""Merging statistics for a model that merges internally.

`scripts/poly/emit_merging_stats.py` reads a careless scaling directory and an
unmerged MTZ, which is the right shape for the integrate-then-scale pipeline.
A merging model has neither: it consumes observations and emits merged
structure factors in one pass, so the only unmerged quantity it leaves behind
is its own per-observation posterior.

That is enough. Symmetry mates share an `asu_id`, so splitting the
observations of each unique reflection into two halves gives the usual
half-set statistics, computed on exactly the intensities the model believes:

    CC(1/2)   correlation between the two half-set means
    R-split   the half-set disagreement careless reports
    R-pim     precision-indicating merging R, which rewards multiplicity
    <I/sigma> from the model's own posterior

Reported per resolution shell, because a merging model can get the overall
scale right and the resolution dependence wrong -- which is exactly what the
mis-centred Wilson prior did on 9b7c, where the ratio to DIALS ran 2.34 to
0.32 across the shells while the median sat near 0.78.

The split is deterministic given `--seed`, so two runs are comparable.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("pred_dir", type=Path, help="directory of prediction parquets")
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--min-multiplicity", type=int, default=2)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def half_split(group_id: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Assign each observation to half 0 or 1, balanced within its group.

    Vectorized: a random key shuffles observations within a group, then the
    rank inside the group alternates the label, so each group splits as evenly
    as its multiplicity allows. A loop over the half-million unique
    reflections of a 9b7c dataset is far too slow to run per shell.
    """
    key = rng.random(len(group_id))
    order = np.lexsort((key, group_id))
    g = group_id[order]
    # rank of each observation within its group
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    group_start = np.repeat(starts, np.diff(np.r_[starts, len(g)]))
    rank = np.arange(len(g)) - group_start
    half = np.empty(len(group_id), dtype=np.int8)
    half[order] = (rank % 2).astype(np.int8)
    return half


def shell_stats(i: np.ndarray, gid: np.ndarray, half: np.ndarray, min_mult: int):
    """CC(1/2), R-split and R-pim from the two half-set means."""
    uniq, inv = np.unique(gid, return_inverse=True)
    n = len(uniq)
    sums = np.zeros((2, n))
    cnts = np.zeros((2, n))
    np.add.at(sums, (half, inv), i)
    np.add.at(cnts, (half, inv), 1.0)
    ok = (cnts[0] > 0) & (cnts[1] > 0) & (cnts.sum(0) >= min_mult)
    if ok.sum() < 10:
        return None
    m0 = sums[0][ok] / cnts[0][ok]
    m1 = sums[1][ok] / cnts[1][ok]

    cc = float(np.corrcoef(m0, m1)[0, 1])
    denom = np.abs(m0 + m1).sum()
    r_split = float((1 / np.sqrt(2)) * np.abs(m0 - m1).sum() / max(denom, 1e-12))

    # R-pim over all observations, not just the half means
    tot = np.zeros(n)
    cnt = np.zeros(n)
    np.add.at(tot, inv, i)
    np.add.at(cnt, inv, 1.0)
    mean_all = tot / np.maximum(cnt, 1)
    sel = cnt[inv] >= min_mult
    num = np.abs(i[sel] - mean_all[inv][sel])
    w = np.sqrt(1.0 / np.maximum(cnt[inv][sel] - 1, 1))
    r_pim = float((w * num).sum() / max(np.abs(i[sel]).sum(), 1e-12))
    return cc, r_split, r_pim, int(ok.sum()), float(cnt.mean())


def main() -> None:
    a = parse_args()
    import polars as pl

    files = sorted(glob.glob(f"{a.pred_dir}/**/*.parquet", recursive=True))
    if not files:
        raise SystemExit(f"no parquets under {a.pred_dir}")

    # Read only what the statistics need. A prediction set carries the full
    # per-observation profile as well (`qp_mean` is 625 floats a row), so the
    # files run to ~9 GB for a 9b7c dataset while these four columns are a
    # few tens of MB.
    have = set(pl.scan_parquet(files[0]).collect_schema().names())
    need = {"asu_id", "d", "qi_mean"}
    missing = need - have
    if missing:
        raise SystemExit(f"predictions lack {sorted(missing)}; have {sorted(have)}")
    cols = sorted(need | ({"qi_var"} & have))
    df = pl.read_parquet(files, columns=cols)

    gid = df["asu_id"].to_numpy()
    i = df["qi_mean"].to_numpy().astype(float)
    d = df["d"].to_numpy().astype(float)
    sig = (
        np.sqrt(np.clip(df["qi_var"].to_numpy().astype(float), 1e-12, None))
        if "qi_var" in df.columns
        else None
    )

    rng = np.random.default_rng(a.seed)
    half = half_split(gid, rng)
    print(f"  {len(df):,} observations, {len(np.unique(gid)):,} unique reflections")
    print(f"  mean multiplicity {len(df) / max(len(np.unique(gid)), 1):.2f}\n")

    s2 = 1.0 / np.clip(d, 1e-6, None) ** 2
    edges = np.quantile(s2, np.linspace(0, 1, a.bins + 1))
    rows = []
    hdr = f"  {'d (A)':>15s} {'n_uniq':>8s} {'mult':>6s} {'CC1/2':>7s} {'Rsplit':>7s} {'Rpim':>7s}"
    if sig is not None:
        hdr += f" {'<I/sig>':>8s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for k in range(a.bins):
        m = (s2 >= edges[k]) & (s2 <= edges[k + 1])
        if m.sum() < 50:
            continue
        st = shell_stats(i[m], gid[m], half[m], a.min_multiplicity)
        if st is None:
            continue
        cc, rs, rp, nu, mult = st
        lo, hi = 1 / np.sqrt(edges[k + 1]), 1 / np.sqrt(max(edges[k], 1e-12))
        line = f"  {hi:7.2f}-{lo:6.2f} {nu:8,d} {mult:6.2f} {cc:7.3f} {rs:7.3f} {rp:7.3f}"
        if sig is not None:
            line += f" {np.mean(i[m] / sig[m]):8.2f}"
        print(line)
        rows.append(dict(d_max=hi, d_min=lo, n_unique=nu, multiplicity=mult,
                         cc_half=cc, r_split=rs, r_pim=rp))

    st = shell_stats(i, gid, half, a.min_multiplicity)
    if st:
        cc, rs, rp, nu, mult = st
        line = f"\n  {'overall':>15s} {nu:8,d} {mult:6.2f} {cc:7.3f} {rs:7.3f} {rp:7.3f}"
        if sig is not None:
            line += f" {np.mean(i / sig):8.2f}"
        print(line)

    if a.out and rows:
        import polars as pl

        pl.DataFrame(rows).write_csv(a.out)
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
