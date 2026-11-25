"""Fit the profile prior of `streak_kmix_v2_profile` once, from reflections whose own pixels determine their shape (two-stage empirical Bayes).

The profile prior is eta ~ N(theta0 + f(m), diag sigma^2), f an MLP of the nine
geometry features. Learned jointly with everything else, it is set mostly by
reflections whose pixels cannot determine a shape: on mos010 photon counting the
learned prior profile of faint and high-resolution reflections was 1.4-1.7x too
wide in n_eff under the collapsed bound (kappa 1.13-1.19, the faint bias it
causes) and too narrow under the conjugate svae (kappa 0.88-0.95), while the
family fitted freely to the true profiles gives kappa 0.996-1.009
(scripts/sim_truth/fit_profile_family.py).

Stage 1, per reflection with a prior-free intensity estimate `--column` (default
i_rb, from an export of the same checkpoint) at or above `--min-i` photons: the
Laplace mode eta*_j of (eta, I, b) with the profile prior widened by
`--eta-prior-scale` (100: removed) and the eta block S_j of its inverse Hessian
(`integrator.model.posterior.collapsed.laplace_eta`), from the checkpoint's own
amortized q(eta) location.

Stage 2, alternated `--rounds` times: theta0 and f by minimising
sum_j sum_k (eta*_jk - theta0_k - f_k(m_j))^2 / (S_j,kk + sigma_k^2) (Adam, from the
checkpoint's values), then sigma_k^2 = max(mean_j r_jk^2 - mean_j S_j,kk, floor)
over the reflections with S_j,kk below the median (the method-of-moments
population variance). The tilt offset (the last eta component) has prior mean 0
under `tilt: none`; only its sigma is fitted.

Writes a checkpoint identical to `--ckpt` except for `surrogates.qp.theta0`,
`surrogates.qp.shape_head.*` and `surrogates.qp.eta_sd_raw`, and a CSV of the
selected reflections' modes and the fitted prior means. Hold the prior fixed
afterwards (optimizer.frozen_params) so the result stays a fixed prior.

usage: python fit_profile_prior.py --ckpt CKPT --config CONFIG --pred PRED.parquet --out-ckpt OUT.ckpt
           [--column i_rb] [--min-i 300] [--eta-prior-scale 100] [--rounds 3] [--steps 2000] [--device cuda]
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim_truth"))
from sbc_simulate import predict_batches  # noqa: E402

from integrator.model.distributions.profile_geometry import geometry_features  # noqa: E402
from integrator.model.posterior.collapsed import laplace_collapsed_batch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True, help="export of the same checkpoint with --column")
    ap.add_argument("--out-ckpt", type=Path, required=True)
    ap.add_argument("--column", default="i_rb")
    ap.add_argument("--min-i", type=float, default=300.0)
    ap.add_argument("--eta-prior-scale", type=float, default=100.0)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--sigma-floor", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.out_ckpt.exists():
        raise SystemExit(f"{a.out_ckpt} exists; choose a new name")

    from integrator.utils.factory_utils import construct_integrator

    cfg = yaml.safe_load(a.config.read_text())
    cfg["integrator"]["args"]["predict_keys"] = ["refl_ids", "qi_mean", "qbg_mean"]
    model = construct_integrator(cfg)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"], strict=True)
    model = model.eval().to(a.device)
    qp = model.surrogates["qp"]
    qp64 = copy.deepcopy(qp).double().eval()
    for prm in qp64.parameters():
        prm.requires_grad_(False)

    sel_ids = set(pl.read_parquet(a.pred).filter(pl.col(a.column) >= a.min_i)["refl_ids"].cast(pl.Int64).to_list())
    print(f"{len(sel_ids)} reflections with {a.column} >= {a.min_i}")
    modes, covd, feats_l, ds, ids = [], [], [], [], []
    for counts, mask, md, out, raw_delta, raw_scale in predict_batches(model, cfg, a.device):
        keep = torch.as_tensor([int(r) in sel_ids for r in out["refl_ids"].cpu()], device=counts.device)
        if not keep.any():
            continue
        mdk = {k: (v[keep] if torch.is_tensor(v) and v.shape[:1] == keep.shape else v) for k, v in md.items()}
        with torch.inference_mode(False):
            a_I, r_I = model.loss.intensity_prior_params(mdk, counts.device)
            a_b, r_b = model._background_prior(mdk, counts.device)
            lc = laplace_collapsed_batch(
                qp64, {k: v.clone() if torch.is_tensor(v) else v for k, v in mdk.items()},
                counts[keep].clone(), mask[keep].clone(), raw_delta[keep].clone(), raw_scale[keep].clone(),
                a_I.clone(), r_I.clone(), torch.as_tensor(a_b).clone(), torch.as_tensor(r_b).clone(),
                out["qi_mean"][keep].clone(), out["qbg_mean"][keep].clone(), n_eta=1,
                eta_prior_scale=a.eta_prior_scale)
        f, _ = geometry_features({k: v.double() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in mdk.items()},
                                 qp.beam_center, qp.lambda_min, qp.lambda_max)
        modes.append(lc["eta_mode"].cpu())
        covd.append(torch.diagonal(lc["eta_cov"], dim1=1, dim2=2).cpu())
        feats_l.append(f.cpu())
        ds.append(mdk["d"].reshape(-1).double().cpu())
        ids.append(out["refl_ids"][keep].cpu())
    E, V, Fm = torch.cat(modes).double(), torch.cat(covd).double(), torch.cat(feats_l).double()
    print(f"stage 1: {len(E)} Laplace modes; median diag var of eta {V.median(0).values.numpy().round(4).tolist()}")

    NZ = qp.NZ
    head = copy.deepcopy(qp.shape_head).double().to("cpu")
    theta0 = qp.theta0.detach().double().cpu().clone()
    sig2 = (qp.eta_sd.detach().double().cpu() ** 2).clone()
    for rnd in range(a.rounds):
        th = theta0.clone().requires_grad_(True)
        for prm in head.parameters():
            prm.requires_grad_(True)
        opt = torch.optim.Adam([th, *head.parameters()], lr=1e-3)
        w = 1 / (V[:, :NZ] + sig2[None, :NZ])
        loss = torch.zeros(())
        for _ in range(a.steps):
            opt.zero_grad()
            r = E[:, :NZ] - th - head(Fm)
            loss = (w * r**2).sum(1).mean()
            loss.backward()
            opt.step()
        theta0 = th.detach()
        with torch.no_grad():
            r = torch.cat([E[:, :NZ] - theta0 - head(Fm), E[:, NZ:]], 1)  # tilt: prior mean 0
            good = V <= V.median(0).values
            num = torch.where(good, r**2, torch.zeros_like(r)).sum(0) / good.sum(0).clamp_min(1)
            meas = torch.where(good, V, torch.zeros_like(V)).sum(0) / good.sum(0).clamp_min(1)
            sig2 = (num - meas).clamp_min(a.sigma_floor**2)
        print(f"round {rnd}: weighted loss {float(loss.detach()):.4f}, sigma {sig2.sqrt().numpy().round(3).tolist()}", flush=True)

    sd = ck["state_dict"]
    sd["surrogates.qp.theta0"] = theta0.to(sd["surrogates.qp.theta0"].dtype)
    for k, v in head.state_dict().items():
        key = f"surrogates.qp.shape_head.{k}"
        sd[key] = v.detach().to(sd[key].dtype)
    smax_t = torch.as_tensor(qp.eta_sd_max, dtype=torch.float64).cpu().expand_as(sig2)
    frac = (sig2.sqrt() / smax_t).clamp(1e-4, 1 - 1e-4)
    sd["surrogates.qp.eta_sd_raw"] = torch.logit(frac).to(sd["surrogates.qp.eta_sd_raw"].dtype)
    clipped = int((sig2.sqrt() >= smax_t * (1 - 1e-4)).sum())
    for k in ("optimizer_states", "lr_schedulers", "callbacks"):
        ck.pop(k, None)
    ck["fit_profile_prior"] = {"source": str(a.ckpt), "pred": str(a.pred), "column": a.column, "min_i": a.min_i,
                               "n": len(E), "eta_prior_scale": a.eta_prior_scale}
    a.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, a.out_ckpt)
    with torch.no_grad():
        fit = torch.cat([theta0 + head(Fm), torch.zeros(len(E), E.shape[1] - NZ, dtype=E.dtype)], 1)
    cols = {"refl_ids": torch.cat(ids).numpy(), "d": torch.cat(ds).numpy()}
    for k in range(E.shape[1]):
        cols[f"mode_{k}"], cols[f"var_{k}"], cols[f"fit_{k}"] = E[:, k].numpy(), V[:, k].numpy(), fit[:, k].numpy()
    pl.DataFrame(cols).write_csv(a.out_ckpt.with_suffix(".csv"))
    print(f"wrote {a.out_ckpt} ({clipped} sigma components at the bound); selected per d bin:",
          np.histogram(torch.cat(ds).numpy(), bins=[0, 1.8, 2.0, 2.5, 3.0, 4.0, 99])[0].tolist())


if __name__ == "__main__":
    main()
