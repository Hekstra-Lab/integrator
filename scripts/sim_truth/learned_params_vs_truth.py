"""Learned Wilson spectrum G(lambda) and learned profile parameters of trained runs, against the simulation's truth.

Wilson prior of `polychromatic_wilson`: E[I | lambda, d] = G(lambda) exp(-B / (2 d^2)),
log G a Chebyshev series in lambda. G and B trade against each other through the
correlation of lambda with d, and the truth's fall with resolution is not one
exponential in 1/d^2 (a free fit of B_true gives 29.4 on d < 3 A, 34.2 on d < 4 A
and 38.6 on every d of mos010), so G is compared at each run's own B:

    G_true(lambda | B) = mean over shoebox reflections in the lambda bin of I_true exp(B / (2 d^2))

the maximum-likelihood G of an exponential prior with that B, on the reflections
the run trained on and every d, with no free constant (`G_truth_given_B`, one curve
per run). The free fits, by weighted least squares of the log cell mean of I_true
on cells of (lambda, d), are written to wilson_B.csv for d < `--wilson-dmax` and
for every d. The source spectrum (the datsimx `--specFile`, wavelength and weight
columns) and the `--enSteps` points datsimx simulated (linear in energy, weights
interpolated linearly in energy, as `mx_simulate.py` does) are written beside it on
their own scale; the figure scales them by one constant. `lambda_count` is the
number of shoebox reflections per 0.002 A.

For `streak_kmix_profile` and `streak_kmix_v2_profile` runs the metadata prior
mean theta0 + f(m) is evaluated on every reflection and unpacked into
along and across standard deviations (px) and weights per component, reduced to
medians and quartiles in bins of detector radius and of wavelength. For runs with
a trained residual basis W (`decoder.weight`, 625 x latent) the basis is dumped.

usage: python learned_params_vs_truth.py --run NAME=CKPT:CONFIG_LOG [--run ...]
           --metadata shoeboxes/metadata.npy --truth truth/truth_reflections.csv
           --spec e080_2.lam --en-steps 322 --dataset mos010_pc_default --out-dir DIR
Writes DIR/spectrum_vs_truth.csv, DIR/wilson_B.csv, DIR/kmix_params.csv and
DIR/residual_basis.npz.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import yaml

from integrator.model.distributions.profile_geometry import detector_polar
from integrator.registry import REGISTRY

ENERGY_CONV = 12398.419843320025  # eV A, simtbx.diffBragg.utils.ENERGY_CONV
KMIX = ("streak_kmix_profile", "streak_kmix_v2_profile")


def chebyshev_log_G(sd: dict, lam: np.ndarray) -> np.ndarray:
    """log G(lambda) of `ChebyshevSpectrum`, with its clamp to the fitted domain."""
    c = sd["loss.spectrum.c"].double()
    x = ((torch.as_tensor(lam, dtype=torch.float64) - sd["loss.spectrum.lam_mid"].double())
         / sd["loss.spectrum.lam_scale"].double()).clamp(-1, 1)
    T = [torch.ones_like(x), x]
    for _ in range(2, c.shape[0]):
        T.append(2 * x * T[-1] - T[-2])
    return (torch.stack(T[: c.shape[0]], -1) @ c).numpy()


def fit_truth_G(lam, d, i_true, lam_edges, n_d=12, min_n=30):
    """Weighted least squares of log cell-mean I_true on (one level per lambda bin, -1/(2 d^2))."""
    df = pl.DataFrame({"lam": lam, "d": d, "y": i_true})
    d_edges = np.quantile(d, np.linspace(0, 1, n_d + 1))
    df = df.with_columns(
        lb=pl.Series(np.digitize(lam, lam_edges) - 1),
        db=pl.Series(np.clip(np.digitize(d, d_edges) - 1, 0, n_d - 1)),
    ).filter((pl.col("lb") >= 0) & (pl.col("lb") < len(lam_edges) - 1))
    cells = (df.group_by("lb", "db").agg(m=pl.col("y").mean(), n=pl.len(), x=(1 / (2 * pl.col("d") ** 2)).mean(),
                                         lam=pl.col("lam").mean())
             .filter((pl.col("n") >= min_n) & (pl.col("m") > 0)))
    lbs = np.sort(cells["lb"].unique().to_numpy())
    col = {b: i for i, b in enumerate(lbs)}
    A = np.zeros((cells.height, len(lbs) + 1))
    A[np.arange(cells.height), [col[b] for b in cells["lb"].to_list()]] = 1
    A[:, -1] = -cells["x"].to_numpy()
    w = np.sqrt(cells["n"].to_numpy())
    coef = np.linalg.lstsq(A * w[:, None], np.log(cells["m"].to_numpy()) * w, rcond=None)[0]
    per = df.group_by("lb").agg(lam=pl.col("lam").mean(), n=pl.len()).filter(pl.col("lb").is_in(lbs)).sort("lb")
    return per["lam"].to_numpy(), coef[:-1], float(coef[-1]), per["n"].to_numpy()


def build_qp(cfg: dict, sd: dict):
    q = cfg["surrogates"]["qp"]
    mod = REGISTRY["surrogates"][q["name"]](**q["args"])
    sub = {k[len("surrogates.qp."):]: v for k, v in sd.items() if k.startswith("surrogates.qp.")}
    mod.load_state_dict(sub, strict=True)
    return mod.eval(), q["name"]


def binned(values: np.ndarray, x: np.ndarray, edges: np.ndarray):
    idx = np.digitize(x, edges) - 1
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() < 50:
            continue
        q = np.quantile(values[sel], [0.25, 0.5, 0.75])
        yield b, float(x[sel].mean()), int(sel.sum()), q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True, help="NAME=CKPT:CONFIG_LOG")
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--en-steps", type=int, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--lam-step", type=float, default=0.005)
    ap.add_argument("--wilson-dmax", type=float, default=3.0, help="fit G_true and B_true on d below this (A)")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    m = np.load(a.metadata, allow_pickle=True).item()
    md = pl.DataFrame({"img": m["image_num"].astype(int), "h": m["H"].astype(int), "k": m["K"].astype(int),
                       "l": m["L"].astype(int), "row": np.arange(len(m["d"])), "d": m["d"],
                       "lam": m["wavelength"]})
    t = (pl.read_csv(a.truth, columns=["shot", "h", "k", "l", "i_true"])
         .with_columns(img=pl.col("shot").str.split("_").list.last().cast(int)))
    j = md.join(t, on=["img", "h", "k", "l"])
    print(f"{j.height} of {md.height} shoeboxes matched a truth row")

    rows = []
    lam_s, w_s = np.loadtxt(a.spec, delimiter=",", skiprows=1).T
    rows += [dict(kind="spectrum", method="truth", wavelength=x, value=y) for x, y in zip(lam_s, w_s)]
    E = ENERGY_CONV / lam_s
    Eg = np.linspace(E.min() + 1e-6, E.max() - 1e-6, a.en_steps)
    order = np.argsort(E)
    wg = np.interp(Eg, E[order], w_s[order])
    rows += [dict(kind="spectrum_sample", method="truth", wavelength=ENERGY_CONV / x, value=y) for x, y in zip(Eg, wg)]

    edges = np.arange(np.floor(md["lam"].min() / a.lam_step) * a.lam_step, md["lam"].max() + a.lam_step, a.lam_step)
    jw = j.filter(pl.col("d") < a.wilson_dmax)
    _, _, B_cut, _ = fit_truth_G(jw["lam"].to_numpy(), jw["d"].to_numpy(), jw["i_true"].to_numpy(), edges)
    _, _, B_all, _ = fit_truth_G(j["lam"].to_numpy(), j["d"].to_numpy(), j["i_true"].to_numpy(), edges)
    B_rows = [dict(method=f"truth_fit_d_lt_{a.wilson_dmax:g}", B=B_cut), dict(method="truth_fit_all_d", B=B_all)]
    print(f"free truth fit: B = {B_cut:.2f} on d < {a.wilson_dmax}, {B_all:.2f} on every d")
    rows += [dict(kind="lambda_count", method="truth", wavelength=x, value=float(n))
             for x, n in zip(*np.unique(np.round(md["lam"].to_numpy() / 0.002) * 0.002, return_counts=True))]
    jl = j.with_columns(lb=pl.Series(np.digitize(j["lam"].to_numpy(), edges) - 1))

    grid = np.linspace(edges[0], edges[-1], 400)
    kmix_rows, bases = [], {}
    meta_t = {k: torch.as_tensor(np.asarray(v, dtype=np.float32)) for k, v in m.items()}
    for spec in a.run:
        name, rest = spec.split("=", 1)
        ckpt, cfg_path = rest.split(":", 1)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        B = float(F.softplus(sd["loss.raw_B"])) + float(cfg["loss"]["args"].get("b_min", 1.0))
        B_rows.append(dict(method=name, B=B))
        gt = (jl.group_by("lb").agg(lam=pl.col("lam").mean(), n=pl.len(),
                                    G=(pl.col("i_true") * (B / (2 * pl.col("d") ** 2)).exp()).mean())
              .filter(pl.col("n") >= 50).sort("lam"))
        rows += [dict(kind="G_truth_given_B", method=name, wavelength=x, value=y, n=n)
                 for x, y, n in gt.select("lam", "G", "n").iter_rows()]
        rows += [dict(kind="G_learned", method=name, wavelength=x, value=float(np.exp(y)))
                 for x, y in zip(grid, chebyshev_log_G(sd, grid))]
        qp, qname = build_qp(cfg, sd)
        W = sd.get("surrogates.qp.decoder.weight")
        if qname in KMIX and getattr(qp, "residual", "") == "none":
            W = None  # residual: none freezes the basis and cuts it out of the forward
        if W is not None:
            side = int(round(W.shape[0] ** 0.5))
            bases[name] = W.T.reshape(-1, side, side).numpy()
        if qname not in KMIX:
            print(f"{name}: B = {B:.2f}, {qname}, basis {None if W is None else tuple(W.shape)}")
            continue
        with torch.no_grad():
            _, th, _ = qp.prior_mean(meta_t, torch.float32, "cpu")
            al, ac, w, _ = (v.numpy() for v in qp.unpack(th))
            r_px, _, _ = detector_polar(meta_t, qp.beam_center)
        r_px = r_px.numpy().reshape(-1)
        lam = m["wavelength"]
        if hasattr(qp, "eta_sd"):
            with torch.no_grad():
                for i, s in enumerate(torch.as_tensor(qp.eta_sd).reshape(-1).numpy()):
                    kmix_rows.append(dict(method=name, axis="index", param="eta_sd", component=i, x=i, n=0,
                                          q25=float(s), median=float(s), q75=float(s)))
        for axis, x, xe in (("radius_px", r_px, np.quantile(r_px, np.linspace(0, 1, 21))),
                            ("wavelength", lam, edges)):
            for param, arr in (("along_sd", al), ("across_sd", ac), ("weight", w)):
                for k in range(arr.shape[1]):
                    for _, xc, n, q in binned(arr[:, k], x, xe):
                        kmix_rows.append(dict(method=name, axis=axis, param=param, component=k, x=xc, n=n,
                                              q25=q[0], median=q[1], q75=q[2]))
        print(f"{name}: B = {B:.2f}, {qname}; median along {np.median(al, 0).round(2)}, "
              f"across {np.median(ac, 0).round(2)}, weight {np.median(w, 0).round(3)}")

    ds = pl.lit(a.dataset).alias("dataset")
    pl.DataFrame(rows, infer_schema_length=None).with_columns(ds).write_csv(a.out_dir / "spectrum_vs_truth.csv")
    pl.DataFrame(B_rows).with_columns(ds).write_csv(a.out_dir / "wilson_B.csv")
    if kmix_rows:
        pl.DataFrame(kmix_rows).with_columns(ds).write_csv(a.out_dir / "kmix_params.csv")
    np.savez_compressed(a.out_dir / "residual_basis.npz", **bases)
    print(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()
