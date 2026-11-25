"""PIT values of the true latents of an `sbc_simulate.py` dataset under a trained run's posteriors.

The trained run is applied to the simulated shoeboxes (the data module, binning
labels and checkpoint exactly as `integrator.cli.predict` loads them). For each
reflection j and test quantity T, with the truth from `sbc_truth.npz`:

    I, b      u = F_q(x_true), q the run's Gamma q(I), q(b) (shape m^2/v, rate m/v
              from qi_mean/qi_var, qbg_mean/qbg_var; exact for the svae family)
    eta_k     u = Phi((eta_true,k - loc_k) / sd_k), q(eta) the amortized Normal of
              streak_kmix_v2_profile, one row per component k of the shape vector
    loglik    T = sum_k mask (c log(I p_k(eta) + b) - (I p_k(eta) + b)), the joint
              log-likelihood of the shoebox's own counts; u = (#{T_s < T_true} + U)/(S + 1)
              over S draws (I, b, eta)_s from the product q(I) q(b) q(eta)
    n_eff     T = 1 / sum p(eta)^2, the same rank form
    i_rb      u = Phi((I_true - i_rb) / sqrt(var_rb)); a point estimate and its variance,
              not a posterior, scored on the same footing

If inference were exact, every u of a posterior quantity would be Uniform(0, 1)
over all reflections and in any subset selected by the data or by covariates the
inference conditions on (d, the metadata) -- NOT in a subset selected by the
true value: an exact posterior shrinks towards the prior, so within a bin of
I_true the faint truths sit low and the bright ones high. Posterior quantities
are therefore binned by the posterior mean of I (`qi_mean`, a function of the
counts); `i_rb`, a point estimate whose calibration given the truth is the
question, is binned by I_true. The joint log-likelihood
is the quantity that detects a posterior ignoring the data (Modrak et al. 2023;
Saeilynoja et al. 2025, arXiv:2502.03279).

The reflections share the global parameters the run learned, so they are not
independent replicates in the strict SBC sense, and q(I) q(b) q(eta) is the
product of the run's marginals, whatever dependence the integrator builds
between them.

Outputs, in --out, each row keyed by method, dataset, quantity and a bin of
qi_mean (`qi_mean`), of I_true (`i_true`) or of d (`d`), plus the whole set (`all`):
    sbc_summary.csv  n, mean(u) - 1/2, 12 var(u) (1 if calibrated, > 1 posterior
                     too narrow), sqrt(n) max|ECDF - u| (Kolmogorov)
    sbc_ecdf.csv     ECDF(z) - z on z = 0, 0.01, ..., 1, which fig_sim_sbc.py draws
                     with simultaneous 95 % bands computed there from n
    sbc_pit.parquet  the per-reflection u values

`--posterior` scores a `sbc_simulate.py --mode posterior` dataset: the truth was
drawn from q(. | y_obs) of the real shoebox, so the reference is the posterior
given both shoeboxes. For I and b the conjugate update is repeated with
q(. | y_obs) in place of the prior (the integrator's `_intensity_prior` and
`_background_prior` return the saved q(. | y_obs) parameters of each
reflection), which is exact for the conjugate q given the gate. For eta,
whose amortized q is not conjugate, the two Gaussian posteriors are combined
and the shared prior N(eta_mean, sigma^2) divided out once:
precision 1/s_obs^2 + 1/s_sim^2 - 1/sigma^2 (floored at 1/s_sim^2 where it is
not positive; the count is printed).

usage: python sbc_ranks.py --ckpt CKPT --config CONFIG_LOG --data SBC_DIR
           --method NAME --dataset NAME --out DIR [--draws 64] [--device cuda]
           [--integrator-args rb_profile=mean] [--posterior]
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
from scipy import special, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sbc_simulate import eta_prior_mean, predict_batches, profile_at, q_eta  # noqa: E402

from integrator.model.posterior.collapsed import collapsed_mixture, laplace_collapsed_batch  # noqa: E402

I_EDGES = [0, 10, 30, 100, 300, 1000, 3000, math.inf]
D_EDGES = [1.7, 1.8, 1.9, 2.0, 2.2, 2.5, 3.0, 4.0, math.inf]
Z = np.linspace(0, 1, 101)
KEYS = ["refl_ids", "qi_mean", "qi_var", "qbg_mean", "qbg_var", "i_rb", "var_rb", "d"]


def gamma_pit(x, mean, var):
    return special.gammainc(mean * mean / var, x * mean / var)


def summarise(u: np.ndarray) -> dict:
    u = np.sort(u[np.isfinite(u)])
    n = len(u)
    if n == 0:
        return dict(n=0, mean_minus_half=np.nan, var12=np.nan, ks=np.nan)
    D = max(np.max(np.arange(1, n + 1) / n - u), np.max(u - np.arange(n) / n))
    return dict(n=n, mean_minus_half=float(u.mean() - 0.5), var12=float(12 * u.var()), ks=float(math.sqrt(n) * D))


def ecdf_diff(u: np.ndarray) -> np.ndarray:
    u = np.sort(u[np.isfinite(u)])
    return np.searchsorted(u, Z, side="right") / max(len(u), 1) - Z


def loglik(counts, mask, I, b, p):
    lam = I[..., None] * p + b[..., None]
    return (mask * (counts * torch.log(lam.clamp_min(1e-30)) - lam)).sum(-1)


def rank_pit(T_true, T_draws, gen):
    below = (T_draws < T_true[None]).sum(0).double()
    U = torch.rand(below.shape, generator=gen, dtype=torch.float64, device=below.device)
    return (below + U) / (T_draws.shape[0] + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True, help="the trained run's config_log.yaml")
    ap.add_argument("--data", type=Path, required=True, help="the sbc_simulate.py output directory")
    ap.add_argument("--method", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--integrator-args", default="", help="k=v,k=v added to integrator.args (e.g. rb_profile=mean)")
    ap.add_argument("--posterior", action="store_true", help="score a --mode posterior dataset")
    ap.add_argument("--construction", choices=("model", "laplace_collapsed", "model_collapsed"), default="model",
                    help="model: the run's own q (default); laplace_collapsed: the full-covariance Laplace q(eta) "
                         "with (I, b) | eta collapsed (integrator.model.posterior.collapsed), built from the run's networks; "
                         "model_collapsed: the run's own full-covariance q(eta) (svae_collapsed: eta_loc, eta_chol) "
                         "with (I, b) | eta collapsed")
    ap.add_argument("--lc-eta-samples", type=int, default=16)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    from integrator.utils import construct_integrator

    cfg = yaml.safe_load(a.config.read_text())
    cfg["data_loader"]["args"]["data_dir"] = str(a.data)
    cfg["data_loader"]["args"]["shoebox_file_names"]["data_dir"] = str(a.data)
    for kv in filter(None, a.integrator_args.split(",")):
        k, v = kv.split("=", 1)
        cfg["integrator"]["args"][k] = yaml.safe_load(v)
    cfg["integrator"]["args"]["predict_keys"] = KEYS + (["eta_loc", "eta_chol"] if a.construction == "model_collapsed" else [])
    model = construct_integrator(cfg)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    model = model.eval().to(a.device)
    qp = model.surrogates["qp"]

    t = np.load(a.data / "sbc_truth.npz")
    row_of = {int(r): i for i, r in enumerate(t["refl_ids"])}
    tI, tb, teta = (torch.as_tensor(t[k]) for k in ("I", "b", "eta"))
    n_floor = 0
    if a.posterior:
        obs = {k: torch.as_tensor(t[k]).to(a.device) for k in ("qI_conc", "qI_rate", "qb_conc", "qb_rate", "eta_loc", "eta_sdq")}

        def lookup(md):
            return torch.as_tensor([row_of[int(r)] for r in md["refl_ids"].reshape(-1).cpu()], device=a.device)

        model._intensity_prior = lambda md, dev: (obs["qI_conc"][lookup(md)], obs["qI_rate"][lookup(md)])
        model._background_prior = lambda md, dev: (obs["qb_conc"][lookup(md)], obs["qb_rate"][lookup(md)])

    gen = torch.Generator(device=a.device).manual_seed(a.seed)
    qp64 = None
    cols: dict[str, list] = {}
    with torch.no_grad():
        for counts, mask, md, out, raw_delta, raw_scale in predict_batches(model, cfg, a.device):
            rows = torch.as_tensor([row_of[int(r)] for r in out["refl_ids"].cpu()])
            I0, b0, eta0 = (x[rows].to(a.device) for x in (tI, tb, teta))
            c = counts.reshape(counts.shape[0], -1).float()
            m = mask.reshape(mask.shape[0], -1).float()

            phi, fxy, eta_mean = eta_prior_mean(qp, md, c)
            if raw_delta is None:
                if a.construction != "model_collapsed":
                    raise SystemExit("this model exposes no amortized q(eta) heads; use --construction model_collapsed")
                loc, sd = out["eta_loc"], out["eta_chol"].pow(2).sum(-1).sqrt()
            else:
                loc, sd = q_eta(qp, eta_mean, raw_delta, raw_scale)
            if a.posterior:
                r = rows.to(a.device)
                lo, so, sig = obs["eta_loc"][r], obs["eta_sdq"][r], qp.eta_sd.to(loc)
                prec = 1 / sd**2 + 1 / so**2 - 1 / sig**2
                bad = prec <= 1 / sd**2 * 1e-3
                n_floor += int(bad.any(-1).sum())
                prec = torch.where(bad, 1 / sd**2, prec)
                loc = torch.where(bad, loc, (loc / sd**2 + lo / so**2 - eta_mean / sig**2) / prec)
                sd = prec.rsqrt()
            u_eta = 0.5 * (1 + torch.erf((eta0 - loc) / (sd * math.sqrt(2))))

            S = a.draws
            qI = torch.distributions.Gamma(out["qi_mean"] ** 2 / out["qi_var"], out["qi_mean"] / out["qi_var"])
            qb = torch.distributions.Gamma(out["qbg_mean"] ** 2 / out["qbg_var"], out["qbg_mean"] / out["qbg_var"])
            Is, bs = qI.sample((S,)), qb.sample((S,))
            etas = loc + sd * torch.randn((S,) + loc.shape, generator=gen, device=a.device)
            T_ll, T_ne = [], []
            for s in range(S):  # one draw at a time keeps memory at B x P x K
                ps = profile_at(qp, phi, fxy, etas[s], m)
                T_ll.append(loglik(c, m, Is[s], bs[s], ps))
                T_ne.append(1 / ps.pow(2).sum(-1))
            p0 = profile_at(qp, phi, fxy, eta0, m)

            if a.construction in ("laplace_collapsed", "model_collapsed"):
                if a.posterior:
                    raise SystemExit(f"--construction {a.construction} is implemented for prior SBC only")
                if qp64 is None:
                    import copy
                    qp64 = copy.deepcopy(qp).double().eval()
                with torch.inference_mode(False):
                    a_I, r_I = model.loss.intensity_prior_params(md, a.device)
                    a_b, r_b = model._background_prior(md, a.device)
                    mdc = {k: v.clone() if torch.is_tensor(v) else v for k, v in md.items()}
                    if a.construction == "laplace_collapsed":
                        lc = laplace_collapsed_batch(qp64, mdc, counts.clone(), mask.clone(), raw_delta.clone(), raw_scale.clone(),
                                                     a_I, r_I, torch.as_tensor(a_b), torch.as_tensor(r_b),
                                                     out["qi_mean"].clone(), out["qbg_mean"].clone(),
                                                     n_eta=a.lc_eta_samples, generator=gen, keep=True)
                    else:
                        md64 = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in mdc.items()}
                        phi64, fxy64, _ = qp64.eta_prior(md64, m.double())
                        Lc = out["eta_chol"].double()
                        lc = collapsed_mixture(qp64, phi64, fxy64, c.double(), m.double(), out["eta_loc"].double(),
                                               Lc @ Lc.transpose(1, 2), a_I, r_I, torch.as_tensor(a_b), torch.as_tensor(r_b),
                                               n_eta=a.lc_eta_samples, generator=gen, keep=True)
                with torch.no_grad():
                    I0d, b0d = I0.double(), b0.double()
                    uI = torch.stack([q.cdf_I(I0d) for _, q in lc["draws"]]).mean(0)
                    ub = torch.stack([q.cdf_b(b0d) for _, q in lc["draws"]]).mean(0)
                    per = max(1, a.draws // len(lc["draws"]))
                    T_ll, T_ne = [], []
                    for ps, q in lc["draws"]:
                        Is_, bs_ = q.sample(per, generator=gen)
                        T_ll += [loglik(c.double(), m.double(), Is_[:, k], bs_[:, k], ps) for k in range(per)]
                        T_ne.append(1 / ps.pow(2).sum(-1))
                    p0d = qp64.profile_from_eta(lc["phi"], lc["fxy"], eta0.double(), m.double())
                    sdl = torch.diagonal(lc["eta_cov"], dim1=1, dim2=2).sqrt()
                    u_eta = 0.5 * (1 + torch.erf((eta0.double() - lc["eta_mode"]) / (sdl * math.sqrt(2))))
                    out = dict(out, qi_mean=lc["i_mean"].float(), qi_var=lc["i_var"].float(),
                               qbg_mean=lc["b_mean"].float(), qbg_var=lc["b_var"].float())
                add = {
                    "refl_ids": out["refl_ids"], "I_true": I0, "d": out["d"], "qi_mean": out["qi_mean"],
                    "u_I": uI, "u_b": ub,
                    "u_loglik": rank_pit(loglik(c.double(), m.double(), I0d, b0d, p0d), torch.stack(T_ll), gen),
                    "u_n_eff": rank_pit(1 / p0d.pow(2).sum(-1), torch.stack(T_ne), gen),
                    "u_i_rb": torch.as_tensor(stats.norm.cdf(((I0 - out["i_rb"]) / out["var_rb"].sqrt()).cpu().numpy())),
                }
                for k in range(u_eta.shape[-1]):
                    add[f"u_eta_{k}"] = u_eta[:, k]
                for k, v in add.items():
                    cols.setdefault(k, []).append(torch.as_tensor(v).reshape(-1).double().cpu().numpy())
                continue

            add = {
                "refl_ids": out["refl_ids"], "I_true": I0, "d": out["d"], "qi_mean": out["qi_mean"],
                "u_I": torch.as_tensor(gamma_pit(I0.cpu().numpy(), out["qi_mean"].cpu().numpy(), out["qi_var"].cpu().numpy())),
                "u_b": torch.as_tensor(gamma_pit(b0.cpu().numpy(), out["qbg_mean"].cpu().numpy(), out["qbg_var"].cpu().numpy())),
                "u_loglik": rank_pit(loglik(c, m, I0, b0, p0), torch.stack(T_ll), gen),
                "u_n_eff": rank_pit(1 / p0.pow(2).sum(-1), torch.stack(T_ne), gen),
                "u_i_rb": torch.as_tensor(stats.norm.cdf(((I0 - out["i_rb"]) / out["var_rb"].sqrt()).cpu().numpy())),
            }
            for k in range(u_eta.shape[-1]):
                add[f"u_eta_{k}"] = u_eta[:, k]
            for k, v in add.items():
                cols.setdefault(k, []).append(torch.as_tensor(v).reshape(-1).double().cpu().numpy())

    if a.posterior:
        print(f"eta: combined precision floored for {n_floor} reflections")
    pit = pl.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    pit.write_parquet(a.out / "sbc_pit.parquet")
    I, d, qm = pit["I_true"].to_numpy(), pit["d"].to_numpy(), pit["qi_mean"].to_numpy()
    summ, ecdf = [], []
    for q in [c for c in pit.columns if c.startswith("u_")]:
        u = pit[q].to_numpy()
        for binning, x, edges in (("all", np.zeros_like(I), [-1, 1]), ("qi_mean", qm, I_EDGES), ("i_true", I, I_EDGES),
                                  ("d", d, D_EDGES)):
            idx = np.digitize(x, edges) - 1
            for k in range(len(edges) - 1):
                s = idx == k
                key = dict(method=a.method, dataset=a.dataset, quantity=q[2:], binning=binning,
                           lo=float(edges[k]), hi=float(edges[k + 1]))
                sm = summarise(u[s])
                summ.append(key | sm)
                ecdf += [key | dict(n=sm["n"], z=float(z), ecdf_minus_z=float(e)) for z, e in zip(Z, ecdf_diff(u[s]))]
    pl.DataFrame(summ).write_csv(a.out / "sbc_summary.csv")
    pl.DataFrame(ecdf).write_csv(a.out / "sbc_ecdf.csv")
    with pl.Config(tbl_rows=300, tbl_cols=12):
        print(pl.DataFrame(summ).filter(pl.col("binning").is_in(["all", "qi_mean"])).select(
            "quantity", "binning", "lo", "hi", "n", "mean_minus_half", "var12", "ks"))


if __name__ == "__main__":
    main()
