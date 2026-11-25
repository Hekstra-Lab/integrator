"""Prior checks of trained runs' Wilson intensity prior: against the simulation's true intensities, and prior-predictively.

Prior against truth (possible only on a simulation). For every shoebox
reflection with a truth row, joined on (image_num = shot number, h, k, l) and
with n_harmonics == 1, and for each run, the run's own prior Gamma(c_j, r_j)
from `intensity_prior_params` gives

    u_j = F_prior(I_true,j)          uniform if the prior is the distribution the
                                     true intensities were drawn from
    ratio = mean(I_true) / mean(c_j / r_j)   per bin; 1 if the prior mean is right

reported per bin of d, of wavelength, and of resolution-by-centricity. The
prior is the acentric Gamma(1, tau) for every reflection unless the metadata
carries `centric`; the centric zones of P 21 21 21 (h = 0, k = 0 or l = 0) are
split out here because their true intensities are chi-squared with one degree
of freedom, 12 var(u) above 1 under an exponential prior. 12 var(u) > 1 in a
bin means the true intensities are more dispersed than the prior allows (a
prior mean that is wrong within the bin does this too); < 1, less dispersed.

Prior predictive (the check available on real data, Box 1980; Gabry et al.
2019). Each `--sim` directory is a `sbc_simulate.py` prior draw on the same
geometry; per resolution bin, the 5..95 % quantiles of three shoebox
statistics -- the total count over valid pixels, the maximum pixel, and the
median pixel -- are written for the real shoeboxes and for the draw.

Writes in --out: prior_check_summary.csv (method, dataset, binning, group, lo,
hi, n, ratio, mean_minus_half, var12, ks), prior_check_ecdf.csv (ECDF(z) - z of
u on z = 0..1 per row of the summary) and prior_predictive.csv (method,
dataset, statistic, lo, hi, q, real, sim).

usage: python prior_check.py --run NAME=CKPT:CONFIG [--run ...]   (CKPT "none": the prior from the config alone) --data REAL_DIR
           --truth truth_reflections.csv --dataset NAME --out DIR [--sim NAME=SBC_DIR ...]
           [--reference metadata_friedel.npy] [--n-bins 30]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml
from scipy import special

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sbc_ranks import D_EDGES, Z, ecdf_diff, summarise  # noqa: E402

LAM_EDGES = list(np.round(np.arange(1.02, 1.181, 0.02), 3))
QS = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])


def truth_for(meta: dict, truth_csv: Path) -> np.ndarray:
    """I_true per metadata row (NaN where no truth row or a harmonic)."""
    t = pl.read_csv(truth_csv).with_columns(
        shot_n=pl.col("shot").str.split("_").list.last().cast(pl.Int64))
    md = pl.DataFrame({"row": np.arange(len(meta["d"])),
                       "shot_n": np.asarray(meta["image_num"]).astype(np.int64),
                       "h": np.asarray(meta["H"]).astype(np.int64), "k": np.asarray(meta["K"]).astype(np.int64),
                       "l": np.asarray(meta["L"]).astype(np.int64),
                       "nh": np.asarray(meta.get("n_harmonics", np.ones(len(meta["d"])))).astype(np.int64)})
    j = md.join(t.select("shot_n", "h", "k", "l", "i_true"), on=["shot_n", "h", "k", "l"], how="left")
    j = j.unique(subset="row", keep="first").sort("row")
    out = j["i_true"].to_numpy().astype(np.float64)
    out[j["nh"].to_numpy() != 1] = np.nan
    return out


def prior_params(ckpt: Path, config: Path, meta: dict, batch: int = 50000):
    from integrator.utils.factory_utils import construct_integrator
    cfg = yaml.safe_load(config.read_text())
    la = cfg["loss"]["args"]
    for key in ("spectrum_init_from", "eb_init_from"):
        if la.get(key) and not Path(la[key]).exists():
            if str(ckpt) == "none":
                raise SystemExit(f"{key} {la[key]} is not on this machine; pass a config whose path exists")
            la[key] = None  # the checkpoint carries the prior
    model = construct_integrator(cfg)
    if str(ckpt) != "none":  # "none": the prior as the config alone builds it (a fixed prior from files)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    loss = model.loss.eval()
    N = len(meta["d"])
    conc, rate = np.empty(N), np.empty(N)
    with torch.no_grad():
        for i in range(0, N, batch):
            md = {k: torch.as_tensor(np.asarray(v[i:i + batch], dtype=np.float32)) for k, v in meta.items()}
            if "group_label" in md:
                md["group_label"] = md["group_label"].long()
            if "centric" in md:
                md["centric"] = md["centric"].bool()
            c, r = loss.intensity_prior_params(md, torch.device("cpu"))
            conc[i:i + batch] = c.expand_as(r).numpy()
            rate[i:i + batch] = r.numpy()
    return conc, rate


def shoebox_stats(counts, masks, chunk: int = 20000) -> dict[str, np.ndarray]:
    N = len(counts)
    tot, mx, med = np.empty(N), np.empty(N), np.empty(N)
    for i in range(0, N, chunk):
        c = np.asarray(counts[i:i + chunk], dtype=np.float64)
        m = np.asarray(masks[i:i + chunk]).reshape(c.shape).astype(bool)
        tot[i:i + chunk] = (c * m).sum(1)
        mx[i:i + chunk] = np.where(m, c, -np.inf).max(1)
        med[i:i + chunk] = np.nanmedian(np.where(m, c, np.nan), 1)
    return {"total": tot, "max": mx, "median": med}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True, help="NAME=CKPT:CONFIG")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sim", action="append", default=[], help="NAME=SBC_DIR, a prior draw on the same geometry")
    ap.add_argument("--reference", default="metadata.npy", help="the metadata file in --data (e.g. metadata_friedel.npy, with `centric`)")
    ap.add_argument("--n-bins", type=int, default=30, help="group_labels_<n>.npy, the resolution bins a binned prior reads")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    from integrator.io import load_data
    meta = load_data(a.data / a.reference)
    meta = {k: np.asarray(v) for k, v in meta.items()}
    gl = a.data / f"group_labels_{a.n_bins}.npy"
    if gl.exists():
        meta["group_label"] = np.asarray(load_data(gl))
    I = truth_for(meta, a.truth)
    d, lam = meta["d"].astype(np.float64), meta["wavelength"].astype(np.float64)
    h, k, l = (meta[c].astype(np.int64) for c in ("H", "K", "L"))
    centric = (h == 0) | (k == 0) | (l == 0)
    ok = np.isfinite(I)
    print(f"{ok.sum()} of {len(I)} reflections have a truth row and one harmonic; {centric[ok].mean():.3f} centric")

    summ, ecdf = [], []
    for spec in a.run:
        name, rest = spec.split("=", 1)
        ckpt, cfgp = rest.split(":", 1)
        conc, rate = prior_params(Path(ckpt), Path(cfgp), meta)
        u = special.gammainc(conc, rate * np.where(ok, I, 0.0))
        pm = conc / rate
        for binning, x, edges in (("all", np.zeros_like(d), [-1, 1]), ("d", d, D_EDGES), ("wavelength", lam, LAM_EDGES)):
            idx = np.digitize(x, edges) - 1
            for group, gsel in (("all", np.ones_like(ok)), ("acentric", ~centric), ("centric", centric)):
                if binning == "wavelength" and group != "all":
                    continue
                for b in range(len(edges) - 1):
                    s = ok & gsel & (idx == b)
                    key = dict(method=name, dataset=a.dataset, binning=binning, group=group,
                               lo=float(edges[b]), hi=float(edges[b + 1]))
                    sm = summarise(u[s])
                    ratio = float(I[s].mean() / pm[s].mean()) if s.any() else np.nan
                    summ.append(key | dict(ratio=ratio) | sm)
                    ecdf += [key | dict(n=sm["n"], z=float(z), ecdf_minus_z=float(e)) for z, e in zip(Z, ecdf_diff(u[s]))]
        allrow = [r for r in summ if r["method"] == name and r["binning"] == "all" and r["group"] == "all"][0]
        print(f"{name}: ratio {allrow['ratio']:.3f}, 12 var(u) {allrow['var12']:.3f}, mean(u) - 1/2 {allrow['mean_minus_half']:+.3f}")
    pl.DataFrame(summ).write_csv(a.out / "prior_check_summary.csv")
    pl.DataFrame(ecdf).write_csv(a.out / "prior_check_ecdf.csv")

    if a.sim:
        real = shoebox_stats(np.load(a.data / "counts.npy", mmap_mode="r"), np.load(a.data / "masks.npy", mmap_mode="r"))
        rows = []
        idx = np.digitize(d, D_EDGES) - 1
        for spec in a.sim:
            name, sdir = spec.split("=", 1)
            sdir = Path(sdir)
            sim = shoebox_stats(np.load(sdir / "counts.npy", mmap_mode="r"), np.load(sdir / "masks.npy", mmap_mode="r"))
            for stat in real:
                for b in range(len(D_EDGES) - 1):
                    s = idx == b
                    if s.sum() < 50:
                        continue
                    qr, qs = np.quantile(real[stat][s], QS), np.quantile(sim[stat][s], QS)
                    rows += [dict(method=name, dataset=a.dataset, statistic=stat, lo=float(D_EDGES[b]),
                                  hi=float(D_EDGES[b + 1]), q=float(q), real=float(x), sim=float(y))
                             for q, x, y in zip(QS, qr, qs)]
        pl.DataFrame(rows).write_csv(a.out / "prior_predictive.csv")
        with pl.Config(tbl_rows=200, tbl_cols=10, float_precision=1):
            print(pl.DataFrame(rows).filter(pl.col("q").is_in([0.5, 0.95])).pivot(
                on="q", index=["method", "statistic", "lo"], values=["real", "sim"]))
    with pl.Config(tbl_rows=200, tbl_cols=12, float_precision=3):
        print(pl.DataFrame(summ).filter(pl.col("binning") == "d").select(
            "method", "group", "lo", "hi", "n", "ratio", "mean_minus_half", "var12"))


if __name__ == "__main__":
    main()
