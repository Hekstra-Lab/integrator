"""Shoeboxes drawn from a trained run's own generative model, for simulation-based calibration (SBC).

Prior SBC (Talts et al. 2018; Saeilynoja et al. 2025, arXiv:2502.03279): every
latent of every reflection is drawn from the model's prior, the counts from the
model's likelihood, and the inference is then trained on these counts alone.
If inference is exact, the PIT value of each true latent under its posterior,
u = F_q(theta_true), is uniform; `sbc_ranks.py` computes it.

For reflection j, with the geometry (d, lambda, position, window, mask) of a real
dataset and the global parameters of a trained checkpoint held fixed:

    I_j   ~ Gamma(c_j, r_j)                     the run's Wilson prior, intensity_prior_params
    eta_j ~ N(theta0 + f(m_j), diag sigma^2)     the run's profile-shape prior (streak_kmix_v2_profile,
                                                 shape_posterior amortized, residual none)
    p_j   = masked_softmax(log(g(eta_j) + floor)) the same map from eta to profile as the forward
    b_j   ~ Gamma(alpha_g, beta_g)               per resolution bin g, see below
    c_jk  ~ Poisson(I_j p_jk + b_j)              on the valid pixels; 0 on masked pixels

The background prior is empirical, per resolution bin of `group_labels_<n>.npy`:
the median of each real shoebox's valid pixels, m_j, gives the bin mean
mu_g = mean(m_j) and the bin variance v_g = var(m_j) - mean(pi/2 m_j / n_valid_j)
(the second term is the sampling variance of a median of n Poisson pixels);
alpha_g = mu_g^2 / v_g, beta_g = alpha_g / mu_g. The training config must use
the same (alpha_g, beta_g), so they are written to `bg_prior.yaml` as the
`bg_concentration` and `bg_rate` lists the loss takes.

Writes, in `--out`: counts.npy (float32, integer-valued), masks.npy,
metadata.npy, group_labels_<n>.npy and the other files of the source directory
except counts; dataset.yaml with the stats recomputed from the new counts; and
sbc_truth.npz holding, per reflection, I, b, eta, the prior mean of eta, sigma,
the prior parameters of I and b, and the profile's n_eff = 1 / sum p^2.

usage: python sbc_simulate.py --ckpt CKPT --config CONFIG_LOG --data DIR --out DIR
           [--n-bins 30] [--seed 0] [--batch 8192] [--device cpu]
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from integrator.io import load_data
from integrator.model.distributions.profile_surrogates import masked_softmax
from integrator.utils.factory_utils import construct_integrator


def empirical_bg_prior(counts: np.ndarray, masks: np.ndarray, groups: np.ndarray, n_bins: int, chunk: int = 20000):
    """Per-bin (alpha, beta) from the medians of each shoebox's valid pixels."""
    med = np.empty(len(counts))
    nv = masks.reshape(len(masks), -1).sum(1)
    for i in range(0, len(counts), chunk):
        c = np.asarray(counts[i:i + chunk], dtype=np.float64).reshape(-1, counts.shape[-1])
        m = np.asarray(masks[i:i + chunk]).reshape(c.shape).astype(bool)
        med[i:i + chunk] = np.nanmedian(np.where(m, c, np.nan), axis=1)
    alpha, beta, mu_out = np.empty(n_bins), np.empty(n_bins), np.empty(n_bins)
    for g in range(n_bins):
        s = (groups == g) & (nv > 0)
        mu = med[s].mean()
        v = med[s].var() - (math.pi / 2 * med[s] / nv[s]).mean()
        if v <= 0:
            raise ValueError(f"bin {g}: shoebox medians vary less than their sampling noise")
        alpha[g] = mu * mu / v
        beta[g] = alpha[g] / mu
        mu_out[g] = mu
    return alpha, beta, mu_out


def prior_profile(qp, md: dict, mask: torch.Tensor, gen: torch.Generator):
    """eta ~ N(eta_mean, sigma^2) and its profile, by the steps of StreakKMixV2ProfileSurrogate.forward."""
    phi, mu, _ = qp.prior_mean(md, mask.dtype, mask.device)
    _, hh, ww = qp.profile_shape
    fxy = torch.stack([
        (md["xyzcal.px.0"].reshape(-1) - md["bbox.0"].reshape(-1)).to(mask) - 0.5 - (ww - 1) / 2,
        (md["xyzcal.px.1"].reshape(-1) - md["bbox.2"].reshape(-1)).to(mask) - 0.5 - (hh - 1) / 2,
    ], -1)
    B = mask.shape[0]
    if qp.tilt == "metadata":
        from integrator.model.distributions.profile_geometry import geometry_features
        feats, _ = geometry_features(md, qp.beam_center, qp.lambda_min, qp.lambda_max)
        psi = (math.pi / 4) * torch.tanh(qp.tilt_head(feats.to(mask))[:, 0])
    else:
        psi = mask.new_zeros(B)
    eta_mean = torch.cat([mu, psi[:, None]], -1)
    sig = qp.eta_sd.to(mask)
    eta = eta_mean + sig * torch.randn(eta_mean.shape, generator=gen, dtype=mask.dtype, device=mask.device)
    u, v = qp.frame(phi, eta[:, -1], eta[:, :-1], fxy)
    g = qp.base_uv(u, v, eta[:, :-1], mask)
    floor = qp.base_floor / qp.decoder.weight.shape[0]
    return eta, eta_mean, sig, masked_softmax(torch.log(g + floor), mask)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True, help="the run's config_log.yaml")
    ap.add_argument("--data", type=Path, required=True, help="the real shoebox directory whose geometry is reused")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-bins", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; choose a new directory")

    cfg = yaml.safe_load(a.config.read_text())
    qcfg = cfg["surrogates"]["qp"]
    if qcfg["name"] != "streak_kmix_v2_profile" or qcfg["args"].get("residual", "detector") != "none" \
            or qcfg["args"].get("shape_posterior", "none") != "amortized":
        raise SystemExit("implemented for streak_kmix_v2_profile with shape_posterior amortized, residual none")
    model = construct_integrator(cfg)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=True)
    model = model.eval().to(a.device)
    qp, loss = model.surrogates["qp"], model.loss

    counts_real = np.load(a.data / "counts.npy", mmap_mode="r")
    masks = np.load(a.data / "masks.npy", mmap_mode="r")
    groups = np.asarray(load_data(a.data / f"group_labels_{a.n_bins}.npy")).astype(np.int64)
    meta = load_data(a.data / "metadata.npy")
    N, P = counts_real.shape
    alpha, beta, mu_bg = empirical_bg_prior(counts_real, masks, groups, a.n_bins)

    gen = torch.Generator(device=a.device).manual_seed(a.seed)
    out_counts = np.zeros((N, P), dtype=np.float32)
    sig = qp.eta_sd
    tr = {k: [] for k in ("I", "b", "eta", "eta_mean", "i_conc", "i_rate", "n_eff")}
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=a.device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=a.device)
    with torch.no_grad():
        for i in range(0, N, a.batch):
            sl = slice(i, min(i + a.batch, N))
            md = {k: torch.as_tensor(np.asarray(v[sl], dtype=np.float32), device=a.device) for k, v in meta.items()}
            mask = torch.as_tensor(np.asarray(masks[sl]), device=a.device).float()
            conc, rate = loss.intensity_prior_params(md, a.device)
            conc = conc.expand_as(rate)
            I = torch._standard_gamma(conc.double(), generator=gen).float() / rate
            g = torch.as_tensor(groups[sl], device=a.device)
            b = torch._standard_gamma(alpha_t[g].double(), generator=gen).float() / beta_t[g]
            eta, eta_mean, sig, p = prior_profile(qp, md, mask, gen)
            lam = (I[:, None] * p + b[:, None]) * mask
            out_counts[sl] = torch.poisson(lam, generator=gen).cpu().numpy()
            for k, v in (("I", I), ("b", b), ("eta", eta), ("eta_mean", eta_mean), ("i_conc", conc),
                         ("i_rate", rate), ("n_eff", 1.0 / p.pow(2).sum(-1))):
                tr[k].append(v.cpu().numpy())

    a.out.mkdir(parents=True)
    np.save(a.out / "counts.npy", out_counts)
    for f in a.data.iterdir():
        if f.name not in ("counts.npy", "dataset.yaml") and f.is_file():
            shutil.copy2(f, a.out / f.name)
    from integrator.cli.make_shoeboxes import _stats_from_memmap
    spec = yaml.safe_load((a.data / "dataset.yaml").read_text())
    spec["stats"] = _stats_from_memmap(a.out / "counts.npy", a.out / "masks.npy")
    spec["sbc"] = {"source_data": str(a.data), "ckpt": str(a.ckpt), "config": str(a.config),
                   "seed": a.seed, "mode": "prior"}
    (a.out / "dataset.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    (a.out / "bg_prior.yaml").write_text(yaml.safe_dump({
        "n_bins": a.n_bins, "bg_concentration": [float(x) for x in alpha],
        "bg_rate": [float(x) for x in beta], "bg_mean": [float(x) for x in mu_bg]}, sort_keys=False))
    cat = {k: np.concatenate(v) for k, v in tr.items()}
    np.savez(a.out / "sbc_truth.npz", **cat, b_conc=alpha[groups], b_rate=beta[groups],
             eta_sd=sig.cpu().numpy(), refl_ids=np.asarray(meta["refl_ids"]))
    print(f"wrote {N} shoeboxes to {a.out}; bg mean {mu_bg.min():.1f}-{mu_bg.max():.1f} /px, "
          f"alpha {alpha.min():.1f}-{alpha.max():.1f}; median I {np.median(cat['I']):.1f}, "
          f"median n_eff {np.median(cat['n_eff']):.1f}")


if __name__ == "__main__":
    main()
