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

Posterior SBC (`--mode posterior`, Saeilynoja et al. 2025): the latents are
drawn instead from the run's posterior given each REAL shoebox y_obs,

    I'_j ~ q(I | y_obs,j),  b'_j ~ q(b | y_obs,j),  eta'_j ~ q(eta | y_obs,j)

(the Gamma q(I), q(b) moment-matched from qi_mean/qi_var, qbg_mean/qbg_var, and
the amortized Normal q(eta)), and a second shoebox y_sim,j ~ Poisson(I' p(eta') + b')
is simulated from them. The run's own config and priors are kept; no background
prior is fitted. The reference posterior for the rank is the one given BOTH
shoeboxes, q(theta | y_obs, y_sim), which `sbc_ranks.py --posterior` builds
from the parameters of q(. | y_obs) saved here (`qI_conc`, `qI_rate`, `qb_conc`,
`qb_rate`, `eta_loc`, `eta_sdq`). Rows the data module drops (dead or
under-filled windows) keep zero counts and are dropped again at scoring.

usage: python sbc_simulate.py --ckpt CKPT --config CONFIG_LOG --data DIR --out DIR
           [--mode prior|posterior] [--n-bins 30] [--seed 0] [--batch 8192] [--device cpu]
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


def eta_prior_mean(qp, md: dict, like: torch.Tensor):
    """phi, the window-centred centroid fxy and the prior mean of eta, as the forward builds them."""
    return qp.eta_prior(md, like)


def profile_at(qp, phi, fxy, eta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The profile of shape vector eta, (..., B, NE) -> (..., B, P), by the steps of the forward."""
    return qp.profile_from_eta(phi, fxy, eta, mask)


def q_eta(qp, eta_mean: torch.Tensor, raw_delta: torch.Tensor, raw_scale: torch.Tensor):
    """Location and sd of the amortized q(eta) from the raw outputs of eta_delta, eta_scale."""
    return qp.q_eta_from_raw(eta_mean, raw_delta, raw_scale)


def predict_batches(model, cfg: dict, device):
    """(counts, mask, metadata, predict outputs, raw_delta, raw_scale) per batch of the config's data,
    loaded as `integrator.cli.predict` loads it; the raw q(eta) head outputs are captured from the forward."""
    from integrator.utils import construct_data_loader, inject_binning_labels
    dl = construct_data_loader(cfg)
    dl.setup()
    inject_binning_labels(dl, cfg)
    qp = model.surrogates["qp"]
    with torch.no_grad():
        for bi, (counts, shoebox, mask, md) in enumerate(dl.predict_dataloader()):
            counts, shoebox, mask = counts.to(device), shoebox.to(device), mask.to(device)
            md = {k: v.to(device) if torch.is_tensor(v) else v for k, v in md.items()}
            qp._refine_capture = []
            try:
                out = model.predict_step((counts, shoebox, mask, md), bi)
                # a class that does not call the surrogate's forward (svae_collapsed) captures nothing
                raw_delta, raw_scale = qp._refine_capture if len(qp._refine_capture) == 2 else (None, None)
            finally:
                qp._refine_capture = None
            yield counts, mask, md, out, raw_delta, raw_scale


def prior_profile(qp, md: dict, mask: torch.Tensor, gen: torch.Generator):
    """eta ~ N(eta_mean, sigma^2) and its profile."""
    phi, fxy, eta_mean = eta_prior_mean(qp, md, mask)
    sig = qp.eta_sd.to(mask)
    eta = eta_mean + sig * torch.randn(eta_mean.shape, generator=gen, dtype=mask.dtype, device=mask.device)
    return eta, eta_mean, sig, profile_at(qp, phi, fxy, eta, mask)


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
    ap.add_argument("--mode", choices=("prior", "posterior"), default="prior")
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
    gen = torch.Generator(device=a.device).manual_seed(a.seed)
    out_counts = np.zeros((N, P), dtype=np.float32)
    sig = qp.eta_sd
    if a.mode == "posterior":
        return simulate_posterior(a, cfg, model, meta, masks, out_counts, gen)
    alpha, beta, mu_bg = empirical_bg_prior(counts_real, masks, groups, a.n_bins)
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

    write_dataset(a, out_counts)
    (a.out / "bg_prior.yaml").write_text(yaml.safe_dump({
        "n_bins": a.n_bins, "bg_concentration": [float(x) for x in alpha],
        "bg_rate": [float(x) for x in beta], "bg_mean": [float(x) for x in mu_bg]}, sort_keys=False))
    cat = {k: np.concatenate(v) for k, v in tr.items()}
    np.savez(a.out / "sbc_truth.npz", **cat, b_conc=alpha[groups], b_rate=beta[groups],
             eta_sd=sig.cpu().numpy(), refl_ids=np.asarray(meta["refl_ids"]))
    print(f"wrote {N} shoeboxes to {a.out}; bg mean {mu_bg.min():.1f}-{mu_bg.max():.1f} /px, "
          f"alpha {alpha.min():.1f}-{alpha.max():.1f}; median I {np.median(cat['I']):.1f}, "
          f"median n_eff {np.median(cat['n_eff']):.1f}")


def write_dataset(a, out_counts: np.ndarray) -> None:
    """counts.npy, the source directory's other files, and dataset.yaml with the stats of the new counts."""
    a.out.mkdir(parents=True)
    np.save(a.out / "counts.npy", out_counts)
    for f in a.data.iterdir():
        if f.name not in ("counts.npy", "dataset.yaml") and f.is_file():
            shutil.copy2(f, a.out / f.name)
    from integrator.cli.make_shoeboxes import _stats_from_memmap
    spec = yaml.safe_load((a.data / "dataset.yaml").read_text())
    spec["stats"] = _stats_from_memmap(a.out / "counts.npy", a.out / "masks.npy")
    spec["sbc"] = {"source_data": str(a.data), "ckpt": str(a.ckpt), "config": str(a.config),
                   "seed": a.seed, "mode": a.mode}
    (a.out / "dataset.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))


def simulate_posterior(a, cfg, model, meta, masks, out_counts, gen) -> None:
    """Latents from q(. | y_obs) of every real shoebox; a second shoebox simulated from them."""
    qp = model.surrogates["qp"]
    model.predict_keys = ["refl_ids", "qi_mean", "qi_var", "qbg_mean", "qbg_var"]
    for k in ("data_dir",):
        cfg["data_loader"]["args"][k] = str(a.data)
        cfg["data_loader"]["args"]["shoebox_file_names"][k] = str(a.data)
    row_of = {int(r): i for i, r in enumerate(np.asarray(meta["refl_ids"]))}
    keys = ("rows", "I", "b", "eta", "qI_conc", "qI_rate", "qb_conc", "qb_rate", "eta_loc", "eta_sdq", "eta_mean", "n_eff")
    tr = {k: [] for k in keys}
    for counts, mask, md, out, raw_delta, raw_scale in predict_batches(model, cfg, a.device):
        rows = torch.as_tensor([row_of[int(r)] for r in out["refl_ids"].cpu()])
        m = mask.reshape(mask.shape[0], -1).float()
        qIc, qIr = out["qi_mean"] ** 2 / out["qi_var"], out["qi_mean"] / out["qi_var"]
        qbc, qbr = out["qbg_mean"] ** 2 / out["qbg_var"], out["qbg_mean"] / out["qbg_var"]
        I = torch._standard_gamma(qIc.double(), generator=gen).float() / qIr
        b = torch._standard_gamma(qbc.double(), generator=gen).float() / qbr
        phi, fxy, eta_mean = eta_prior_mean(qp, md, m)
        loc, sdq = q_eta(qp, eta_mean, raw_delta, raw_scale)
        eta = loc + sdq * torch.randn(loc.shape, generator=gen, device=loc.device, dtype=loc.dtype)
        p = profile_at(qp, phi, fxy, eta, m)
        out_counts[rows.numpy()] = torch.poisson((I[:, None] * p + b[:, None]) * m, generator=gen).cpu().numpy()
        for k, v in zip(keys, (rows, I, b, eta, qIc, qIr, qbc, qbr, loc, sdq, eta_mean, 1.0 / p.pow(2).sum(-1))):
            tr[k].append(v.cpu().numpy())
    write_dataset(a, out_counts)
    cat = {k: np.concatenate(v) for k, v in tr.items()}
    rows = cat.pop("rows")
    np.savez(a.out / "sbc_truth.npz", **cat, eta_sd=qp.eta_sd.detach().cpu().numpy(),
             refl_ids=np.asarray(meta["refl_ids"])[rows])
    print(f"posterior mode: wrote {len(rows)} simulated shoeboxes to {a.out}; "
          f"median I' {np.median(cat['I']):.1f}, median b' {np.median(cat['b']):.2f}")


if __name__ == "__main__":
    main()
