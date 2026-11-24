"""Merged structure factors scored against the simulation's true F(+) and F(-), per resolution shell.

The truth is datsimx's `ground_truth.mtz` (`--writeMtz`): F(+) and F(-) calculated
from the model at the beam's mean wavelength, with anomalous scattering. Both
files are mapped to the reciprocal-space asymmetric unit of the truth's space
group; a centric reflection's truth F(-) is missing in that file and is taken as
its F(+). Per shell, on the reflections present in both:

    log k, b = least squares of log(F / F_true) = log k + b s^2, s^2 = 1/(4 d^2), whole domain
    F'       = F / (k exp(b s^2))                    the merged F with that scale and B removed
    cc_F     = corr(F, F_true)                       Friedel mates as separate rows
    cc_F_kB  = corr(F', F_true)
    r_F      = sum |F' - F_true| / sum F_true

The (k, b) fit is needed because careless gives its scale model dHKL: a factor
exp(-2 b s^2) passes between the per-observation scale and F^2 without changing
any prediction, and the laue-dials reference merge of mos010 carries
2b = 34.5 A^2 against the truth (cc_F over the whole domain 0.867 before the fit,
0.926 after). `dB_I` = 2b is written with k.
    cc_I     = corr(I, F_true^2)                     when the file carries I(+)/I(-)
    cc_dano  = corr(F(+) - F(-), F_true(+) - F_true(-))   acentric, both mates present
    cc_dano_crossed = corr(F(+) - F(-), F_true(-) - F_true(+))  must be the negative of cc_dano;
                      a positive value is a flipped (+)/(-) convention (docs/exporting.md)
    completeness = merged reflections / truth reflections in the shell (Friedel mates separate)

Shells are equal-count in the truth's reflections between --dmin and the
truth's lowest resolution; the row with shell = "all" is the whole domain.

`--predictions` reads careless's `<base>_predictions_0.mtz` and scores its
per-observation `Scale` column, g_hat, against the true scale
g = I_true / |F_true(+ or -)|^2 from the truth table (`--truth-table`), matched
through the unmerged MTZ careless read (`--unmerged`). careless's F is on its own
scale, so log(g_hat / g) is compared after removing its mean: the sd of that
deviation and the correlation of log g_hat with log g, overall, per 0.01 A of
wavelength and per decade of I_true. g_hat is first multiplied by k^2 exp(2 b s^2),
the merged-F fit above, so the gauge is removed from both. `mean_log_ipred` and
`sd_log_ipred` are the mean and sd of log(Ipred / I_true), careless's
prediction of each observation against its truth, which no gauge reaches; its
mean carries the integrator's overall intensity scale.

usage: python score_merged_vs_truth.py --merged config4_0.mtz --truth ground_truth.mtz
           --method laue_integrate --dataset mos010_pc_default --dmin 1.78 --out scores_merged.csv
           [--predictions config4_predictions_0.mtz --unmerged integrated.mtz
            --truth-table truth_reflections.csv --scale-out scores_scale.csv]
Appends to --out (and --scale-out) when they exist. `--joined-out` writes the
per-reflection F, F' and F_true of this one method as parquet (overwritten).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import reciprocalspaceship as rs


def friedel_long(ds: rs.DataSet, cols: dict[str, tuple[str, str]], sg) -> pl.DataFrame:
    """Rows (H, K, L in the ASU, plus) with one value column per entry of `cols`, from a merged anomalous file."""
    ds = ds.reset_index()
    hkl = ds[["H", "K", "L"]].to_numpy().astype(int)
    asu, _ = rs.utils.hkl_to_asu(hkl, sg)
    centric = rs.utils.is_centric(asu, sg)
    out = []
    for plus in (True, False):
        d = {"H": asu[:, 0], "K": asu[:, 1], "L": asu[:, 2], "plus": np.full(len(ds), plus), "centric": centric}
        for name, (cp, cm) in cols.items():
            d[name] = ds[cp if plus else cm].to_numpy().astype(float)
        out.append(pl.DataFrame(d))
    return pl.concat(out)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")


def shell_rows(j: pl.DataFrame, edges: np.ndarray, has_I: bool):
    """One dict per shell and one for the whole domain."""
    j = j.with_columns(shell=pl.Series(np.digitize(j["d"].to_numpy(), edges)))
    groups = [(str(s), j.filter(pl.col("shell") == s)) for s in range(1, len(edges))] + [("all", j)]
    for name, g in groups:
        if g.height == 0:
            continue
        both = g.filter(pl.col("F").is_not_nan())
        r = dict(shell=name, d_max=float(g["d"].max()), d_min=float(g["d"].min()),
                 n_true=int(g.height), n=int(both.height), completeness=both.height / g.height)
        F, T = both["F"].to_numpy(), both["F_true"].to_numpy()
        Fc = both["F_kB"].to_numpy()
        r.update(cc_F=corr(F, T), cc_F_kB=corr(Fc, T), r_F=float(np.abs(Fc - T).sum() / T.sum()))
        if has_I:
            bi = g.filter(pl.col("I").is_not_nan())
            r["cc_I"] = corr(bi["I"].to_numpy(), bi["F_true"].to_numpy() ** 2)
        a = (g.filter(~pl.col("centric")).pivot(on="plus", index=["H", "K", "L"], values=["F", "F_true"])
             .drop_nans().drop_nulls())
        if a.height > 2:
            dF = (a["F_true"] - a["F_false"]).to_numpy()
            dT = (a["F_true_true"] - a["F_true_false"]).to_numpy()
            r.update(n_dano=int(a.height), cc_dano=corr(dF, dT), cc_dano_crossed=corr(dF, -dT))
        yield r


def score_scale(pred_path: Path, unmerged_path: Path, truth_table: Path, F_true: pl.DataFrame, sg,
                method: str, dataset: str, out: Path, logk: float, b: float, cell) -> None:
    """careless's per-observation `Scale` against I_true / |F_true|^2.

    The predictions file is not in the input's row order, its indices are a
    symmetry mate of the observed ones and its `image_id` is 0-based, so each
    row is matched to the unmerged input on (image, ASU index, Friedel sign,
    Iobs equal to the input I to 1e-4); centric rows match without the sign. The
    image offset is the one of 0 and 1 that matches more rows. Rows that match
    nothing are counted and dropped. The input's observed index and BATCH then
    join the truth table.
    """
    p = rs.read_mtz(str(pred_path)).reset_index()
    u = rs.read_mtz(str(unmerged_path)).reset_index()

    def c(d, k, t=int):
        return d[k].to_numpy().astype(t)

    au, iu = rs.utils.hkl_to_asu(u[["H", "K", "L"]].to_numpy().astype(int), sg)
    ap, ip = rs.utils.hkl_to_asu(p[["H", "K", "L"]].to_numpy().astype(int), sg)
    U = pl.DataFrame({"img": c(u, "BATCH"), "H": au[:, 0], "K": au[:, 1], "L": au[:, 2], "sign": iu % 2,
                      "I": c(u, "I", float), "lam": c(u, "wavelength", float), "ui": np.arange(len(u)),
                      "h": c(u, "H"), "k": c(u, "K"), "l": c(u, "L")})
    centric = rs.utils.is_centric(ap, sg)
    best = None
    for off in (0, 1):
        Pp = pl.DataFrame({"img": c(p, "image_id") + off, "H": ap[:, 0], "K": ap[:, 1], "L": ap[:, 2],
                           "sign": ip % 2, "centric": centric, "Iobs": c(p, "Iobs", float), "ipred": c(p, "Ipred", float),
                           "scale": c(p, "Scale", float), "test": c(p, "test"), "pi": np.arange(len(p))})
        same = (pl.col("Iobs") - pl.col("I")).abs() <= 1e-4 * pl.col("I").abs() + 1e-3
        acen = Pp.filter(~pl.col("centric")).join(U, on=["img", "H", "K", "L", "sign"]).filter(same)
        cen = Pp.filter(pl.col("centric")).join(U.drop("sign"), on=["img", "H", "K", "L"]).filter(same)
        j = pl.concat([acen, cen], how="diagonal_relaxed").unique("pi", keep="first")
        j = j.unique("ui", keep="first")
        if best is None or j.height > best[1].height:
            best = (off, j)
    off, j = best
    print(f"predictions: {len(p)} rows, {j.height} matched one-to-one to the unmerged input (image_id + {off}), "
          f"{len(p) - j.height} dropped")
    t = (pl.read_csv(truth_table, columns=["shot", "h", "k", "l", "i_true", "lam_mean"])
         .with_columns(img=pl.col("shot").str.split("_").list.last().cast(int)))
    j = j.join(t, on=["img", "h", "k", "l"], how="inner")
    print(f"  {j.height} joined the truth table; median |lambda_input - lambda_true| "
          f"{float((j['lam'] - j['lam_mean']).abs().median()):.2g} A")
    plus = F_true.filter(pl.col("plus")).select("H", "K", "L", Fp="F_true")
    minus = F_true.filter(~pl.col("plus")).select("H", "K", "L", Fm="F_true")
    j = (j.join(plus, on=["H", "K", "L"], how="left").join(minus, on=["H", "K", "L"], how="left")
         .with_columns(F=pl.when((pl.col("sign") == 1) | pl.col("Fm").is_null()).then("Fp").otherwise("Fm"))
         .filter((pl.col("i_true") > 0) & (pl.col("F") > 0) & (pl.col("scale") > 0))
         .with_columns(lg=(pl.col("i_true") / pl.col("F") ** 2).log()))
    j = j.with_columns(d=pl.Series(cell.calculate_d_array(j.select("H", "K", "L").to_numpy())))
    j = j.with_columns(lgh=pl.col("scale").log() + 2 * logk + 2 * b / (4 * pl.col("d") ** 2))
    j = j.with_columns(dev=(pl.col("lgh") - pl.col("lg")) - (pl.col("lgh") - pl.col("lg")).mean())
    j = j.filter(pl.col("ipred") > 0).with_columns(lip=(pl.col("ipred") / pl.col("i_true")).log())
    print(f"  {j.height} with I_true > 0 and F_true > 0")

    rows = []

    def add(axis, name, g):
        if g.height < 30:
            return
        rows.append(dict(axis=axis, bin=name, n=g.height, lam_mean=float(g["lam"].mean()),
                         sd_log_ratio=float(g["dev"].std()), mean_log_ratio=float(g["dev"].mean()),
                         mean_log_ipred=float(g["lip"].mean()), sd_log_ipred=float(g["lip"].std()),
                         cc_log=corr(g["lgh"].to_numpy(), g["lg"].to_numpy())))

    add("all", "all", j)
    for lo in np.arange(np.floor(j["lam"].min() * 100) / 100, j["lam"].max(), 0.01):
        add("wavelength", f"{lo:.2f}", j.filter((pl.col("lam") >= lo) & (pl.col("lam") < lo + 0.01)))
    for dec in range(-1, 5):
        add("i_true_decade", str(dec), j.filter(pl.col("i_true").log10().floor() == dec))
    res = pl.DataFrame(rows).with_columns(method=pl.lit(method), dataset=pl.lit(dataset))
    with pl.Config(tbl_rows=40, float_precision=4):
        print(res.drop("method", "dataset"))
    if out.exists():
        res = pl.concat([pl.read_csv(out, infer_schema_length=None), res], how="diagonal_relaxed")
    res.write_csv(out)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dmin", type=float, required=True)
    ap.add_argument("--shells", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--f-cols", default="F(+),F(-)")
    ap.add_argument("--i-cols", default="I(+),I(-)")
    ap.add_argument("--predictions", type=Path)
    ap.add_argument("--truth-table", type=Path)
    ap.add_argument("--unmerged", type=Path, help="the unmerged MTZ careless read (BATCH, wavelength, I)")
    ap.add_argument("--scale-out", type=Path)
    ap.add_argument("--joined-out", type=Path, help="per-reflection F, F after the (k, b) fit and F_true (parquet)")
    a = ap.parse_args()

    tr = rs.read_mtz(str(a.truth))
    sg = tr.spacegroup
    t = friedel_long(tr, {"F_true": ("F(+)", "F(-)")}, sg)
    t = t.with_columns(F_true=pl.when(pl.col("centric") & pl.col("F_true").is_nan())
                       .then(pl.col("F_true").first().over("H", "K", "L")).otherwise(pl.col("F_true")))
    t = t.filter(pl.col("F_true").is_not_nan() & (pl.col("F_true") > 0))
    t_all = t
    t = t.filter(~(pl.col("centric") & ~pl.col("plus")))  # one row per centric reflection
    cell = tr.cell
    t = t.with_columns(d=pl.Series(cell.calculate_d_array(t.select("H", "K", "L").to_numpy())))
    t = t.filter(pl.col("d") >= a.dmin)

    m = rs.read_mtz(str(a.merged))
    fp, fm = a.f_cols.split(",")
    cols = {"F": (fp, fm)}
    ip, im = a.i_cols.split(",")
    has_I = ip in m.columns and im in m.columns
    if has_I:
        cols["I"] = (ip, im)
    mm = friedel_long(m, cols, sg).drop("centric")
    mm = mm.unique(["H", "K", "L", "plus"], keep="first")
    j = t.join(mm, on=["H", "K", "L", "plus"], how="left").with_columns(
        pl.col("F").fill_null(float("nan")), *( [pl.col("I").fill_null(float("nan"))] if has_I else [] ))
    print(f"truth reflections (Friedel mates separate) to {a.dmin} A: {t.height}; merged present: "
          f"{j.filter(pl.col('F').is_not_nan()).height}; merged rows outside the truth: "
          f"{mm.join(t, on=['H', 'K', 'L', 'plus'], how='anti').filter(pl.col('F').is_not_nan()).height}")

    both = j.filter(pl.col("F").is_not_nan())
    pos = both.filter((pl.col("F") > 0) & (pl.col("F_true") > 0))
    s2 = (1 / (4 * pos["d"] ** 2)).to_numpy()
    A = np.vstack([np.ones_like(s2), s2]).T
    logk, b = np.linalg.lstsq(A, np.log((pos["F"] / pos["F_true"]).to_numpy()), rcond=None)[0]
    k = float(np.exp(logk))
    j = j.with_columns(F_kB=pl.col("F") / (k * (b / (4 * pl.col("d") ** 2)).exp()))
    edges = np.quantile(t["d"].to_numpy(), np.linspace(1, 0, a.shells + 1))
    edges[0], edges[-1] = np.inf, 0
    edges = edges[::-1]  # increasing for digitize: shell 1 is the highest resolution
    rows = list(shell_rows(j, edges, has_I))
    if a.joined_out is not None:
        (j.filter(pl.col("F").is_not_nan())
         .select("H", "K", "L", "plus", "centric", "d", "F", "F_kB", "F_true")
         .with_columns(method=pl.lit(a.method), dataset=pl.lit(a.dataset))
         .write_parquet(a.joined_out))
        print(f"wrote {a.joined_out}")
    out = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        method=pl.lit(a.method), dataset=pl.lit(a.dataset), k=pl.lit(k), dB_I=pl.lit(2 * b), merged=pl.lit(str(a.merged)))
    if a.out.exists():
        out = pl.concat([pl.read_csv(a.out, infer_schema_length=None), out], how="diagonal_relaxed")
    out.write_csv(a.out)
    with pl.Config(tbl_rows=30, tbl_cols=20, tbl_width_chars=200, float_precision=4):
        print(pl.DataFrame(rows, infer_schema_length=None))
    print(f"log(F / F_true) = {logk:.4f} + {b:.3f} s^2 (k = {k:.5g}, dB on I = {2 * b:.2f} A^2); wrote {a.out}")

    if a.predictions is not None:
        if a.truth_table is None or a.scale_out is None or a.unmerged is None:
            raise SystemExit("--predictions needs --unmerged, --truth-table and --scale-out")
        score_scale(a.predictions, a.unmerged, a.truth_table, t_all, sg, a.method, a.dataset, a.scale_out,
                    float(logk), float(b), tr.cell)


if __name__ == "__main__":
    main()
