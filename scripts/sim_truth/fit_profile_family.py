"""How well the `streak_kmix_v2_profile` family can represent the TRUE spot profiles, per number of components K.

For reflections sampled by `dump_profile_truth_examples.py --sample N` (their
noise-free true profiles p, `bragg * own` normalised, on the shoebox window)
and each K in --ks, the family's shape vector eta (3K + 3 numbers: K along and
K across widths, K weight logits, a centre shift, a tilt) is fitted per
reflection, with no prior, by minimising the cross-entropy -sum_k p_k log q_k(eta)
(Adam, from the class's own initial grid and, for K = 3, also from a trained
checkpoint's metadata prior mean; the better fit is kept). Per reflection:

    KL     = sum p log(p / q)             what the family cannot represent (nats)
    neff   = n_eff(q) / n_eff(p)           n_eff = 1 / sum x^2; < 1 the best fit is narrower
    kappa  = sum q p / sum q^2             the expected ratio of a profile fit with profile q
                                           and the true background to the true intensity,
                                           when the counts follow p: the bias the family
                                           itself imposes on the intensity

reported by bins of i_true and of d. A family that can represent the truth has
KL ~ 0, neff ~ 1, kappa ~ 1.

With --prior NAME=CKPT:CONFIG (repeatable) it also evaluates each trained model's
PRIOR profile on the same reflections, the mean over 16 draws eta ~ N(theta0 +
f(m), sigma^2) with no data: the n_eff ratio and kappa that a reflection too
faint to move its shape would get (profile_prior.csv). --ks "" skips the family fit.

usage: python fit_profile_family.py --npz PROFILES.npz --metadata metadata.npy --config CONFIG
           [--ckpt CKPT] [--ks 3,5,7] [--steps 400] [--prior NAME=CKPT:CONFIG ...] --out DIR
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml


def fit(qp, md, mask, p_true, eta0, steps, lr=0.05):
    eta = eta0.clone().requires_grad_(True)
    phi, fxy, _ = qp.eta_prior(md, mask)
    opt = torch.optim.Adam([eta], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        q = qp.profile_from_eta(phi, fxy, eta, mask)
        loss = -(p_true * torch.log(q.clamp_min(1e-30))).sum(-1)
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        q = qp.profile_from_eta(phi, fxy, eta, mask)
        ce = -(p_true * torch.log(q.clamp_min(1e-30))).sum(-1)
    return eta.detach(), q, ce


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--ks", default="3,5,7")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prior", action="append", default=[], help="NAME=CKPT:CONFIG, a trained model's prior profile")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    from integrator.io import load_data
    from integrator.utils.factory_utils import construct_integrator

    z = np.load(a.npz, allow_pickle=True)
    meta = {k: np.asarray(v) for k, v in load_data(a.metadata).items()}
    row_of = {int(r): i for i, r in enumerate(meta["refl_ids"])}
    rows = np.array([row_of[int(r)] for r in z["refl_ids"]])
    md = {k: torch.as_tensor(v[rows].astype(np.float64)) for k, v in meta.items()}
    P = torch.as_tensor(z["true_profile"].reshape(len(rows), -1), dtype=torch.float64)
    mask = torch.ones_like(P)
    ent = -(P * torch.log(P.clamp_min(1e-300))).sum(-1)
    n_eff_p = 1 / (P**2).sum(-1)
    cfg0 = yaml.safe_load(a.config.read_text())
    rows_out = []
    prior_rows = []
    for spec in a.prior:
        name, rest = spec.split("=", 1)
        ckp, cfgp = rest.split(":", 1)
        pc = yaml.safe_load(Path(cfgp).read_text())
        pc["integrator"]["args"].pop("init_from_checkpoint", None)
        qpp = construct_integrator(pc).surrogates["qp"]
        sdp = torch.load(ckp, map_location="cpu", weights_only=False)["state_dict"]
        qpp.load_state_dict({k[len("surrogates.qp."):]: v for k, v in sdp.items() if k.startswith("surrogates.qp.")})
        qpp = qpp.double().eval()
        torch.manual_seed(0)
        with torch.no_grad():
            phi, fxy, em = qpp.eta_prior(md, mask)
            sig = qpp.eta_sd.double()
            qs = torch.stack([qpp.profile_from_eta(phi, fxy, em + sig * torch.randn_like(em), mask) for _ in range(16)]).mean(0)
        ne = (1 / (qs**2).sum(-1) / n_eff_p).numpy()
        kp = ((qs * P).sum(-1) / (qs**2).sum(-1)).numpy()
        for i in range(len(rows)):
            prior_rows.append(dict(prior=name, refl_ids=int(z["refl_ids"][i]), i_true=float(z["i_true"][i]),
                                   d=float(z["d"][i]), neff_ratio=float(ne[i]), kappa=float(kp[i])))
    if prior_rows:
        dfp = pl.DataFrame(prior_rows)
        dfp.write_csv(a.out / "profile_prior.csv")
        with pl.Config(tbl_rows=80, tbl_cols=10, float_precision=3):
            for b, col, edges in (("ibin", "i_true", [10, 30, 100, 300, 1000, 3000]), ("dbin", "d", [1.8, 2.0, 2.5, 3.0, 4.0])):
                print(dfp.with_columns(pl.col(col).cut(edges).alias(b)).group_by("prior", b).agg(
                    pl.len(), pl.col("neff_ratio").median(), pl.col("kappa").mean()).sort("prior", b))
    for K in (int(k) for k in a.ks.split(",") if k):
        cfg = copy.deepcopy(cfg0)
        cfg["surrogates"]["qp"]["args"]["n_components"] = K
        cfg["integrator"]["args"].pop("init_from_checkpoint", None)
        torch.manual_seed(0)
        qp = construct_integrator(cfg).surrogates["qp"].double().eval()
        for prm in qp.parameters():
            prm.requires_grad_(False)
        with torch.no_grad():
            _, _, eta_grid = qp.eta_prior(md, mask)
            eta_grid = qp.theta0.new_zeros(eta_grid.shape)
            eta_grid[:, :-1] = qp.theta0  # the class's initial grid, f(m) = 0
        starts = [eta_grid]
        if K == 3 and a.ckpt is not None:
            sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"]
            q3 = copy.deepcopy(qp)
            q3.load_state_dict({k[len("surrogates.qp."):]: v for k, v in sd.items() if k.startswith("surrogates.qp.")})
            with torch.no_grad():
                starts.append(q3.double().eta_prior(md, mask)[2])
        best_ce, best_q = None, None
        for s0 in starts:
            _, q, ce = fit(qp, md, mask, P, s0, a.steps)
            if best_ce is None:
                best_ce, best_q = ce, q
            else:
                better = ce < best_ce
                best_ce = torch.where(better, ce, best_ce)
                best_q = torch.where(better[:, None], q, best_q)
        kl = (best_ce - ent).numpy()
        neff = (1 / (best_q**2).sum(-1) / n_eff_p).numpy()
        kappa = ((best_q * P).sum(-1) / (best_q**2).sum(-1)).numpy()
        for i in range(len(rows)):
            rows_out.append(dict(K=K, refl_ids=int(z["refl_ids"][i]), i_true=float(z["i_true"][i]), d=float(z["d"][i]),
                                 kl=float(kl[i]), neff_ratio=float(neff[i]), kappa=float(kappa[i])))
        print(f"K={K}: median KL {np.median(kl):.4f}, median neff ratio {np.median(neff):.3f}, "
              f"median kappa {np.median(kappa):.4f}, mean kappa {np.mean(kappa):.4f}", flush=True)
    if not rows_out:
        return
    df = pl.DataFrame(rows_out)
    df.write_csv(a.out / "profile_family_fit.csv")
    with pl.Config(tbl_rows=60, tbl_cols=10, float_precision=4):
        b = df.with_columns(ibin=pl.col("i_true").cut([10, 30, 100, 300, 1000, 3000]))
        print(b.group_by("K", "ibin").agg(pl.len(), pl.col("kl").median(), pl.col("neff_ratio").median(),
                                          pl.col("kappa").mean()).sort("K", "ibin"))
        b = df.with_columns(dbin=pl.col("d").cut([1.8, 2.0, 2.5, 3.0, 4.0]))
        print(b.group_by("K", "dbin").agg(pl.len(), pl.col("kl").median(), pl.col("neff_ratio").median(),
                                          pl.col("kappa").mean()).sort("K", "dbin"))


if __name__ == "__main__":
    main()
