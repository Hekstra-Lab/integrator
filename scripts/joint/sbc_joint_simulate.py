"""Simulation-based calibration of the joint model, pixels to scaled and merged intensities, with every prior frozen.

Draws the whole generative model, then infers with `integrator.model.joint.marginal`:

    reflection h:   s_h^2 ~ U(s2_min, s2_max), centric with probability --centric-frac,
                    J_h ~ Wilson(Sigma(s_h)), Sigma = G0 exp(-2 B s^2)  (acentric Exp, centric Gamma(1/2))
                    multiplicity n_h = 1 + NegBin(r = 4, mean --mult-mean - 1)  (mos010 Friedel-separate:
                    median 5, mean 5.7, 4.6% at 1)
    observation i:  lambda_i ~ U(lam_min, lam_max), log g_i = sum_k theta_k T_k(lambda~_i) (Chebyshev,
                    lambda~ in [-1, 1]); streak azimuth phi_i ~ U(0, pi)
                    eta_i = (log sd_along, log sd_across) ~ N(mu, diag sd_eta^2), mu depending on lambda~
                    b_i ~ Gamma(a_b, r_b)
                    c_ik ~ Poisson(g_i J_h p_ik(eta_i) + b_i) on a P x P window, p an oriented Gaussian

Inference, priors at their generating values:

    1. per observation, a Laplace fit of (eta, I, b) with no intensity prior gives the importance
       proposal q(eta) (its eta block of the inverse Hessian, sd x --q-inflate);
    2. l_i(I) on the observation's own I grid (`obs_grid`), eta integrated by importance
       sampling with p/q weights (`obs_loglik_is`, --n-eta draws), b by Laplace;
    3. the scale: theta by maximising sum_h log int p(J) prod_i l_i(g_i J) dJ (L-BFGS), the
       scale gradient stopped for reflections of multiplicity 1; Sigma is frozen at the truth,
       so the overall scale is identified (docs/joint_model_closed_forms.md, gauge section);
    4. the posterior of J_h on its grid at the TRUE theta and at the fitted theta; PIT
       u_h = F(J_h true). The J test is the true-theta PIT. Every reflection of one dataset
       shares the same theta error, so the u_h are not independent given theta_hat; the
       fitted-theta PIT shows the effect of that error (largest for bright reflections, whose
       relative sd is below the overall level's), and mixing over q(theta) moves all u_h
       toward 1/2 together, which is why it is not used as a calibration test.
    5. the Laplace sd of theta (inverse Hessian of the negative log marginal at the fit, by
       central differences of the gradient) and z = (theta_hat - theta_true) / sd; across
       independent seeds z should be standard normal, the test of the scale fit.

Writes --out (parquet, one row per reflection: J_true, s2, centric, n_obs, the posterior mean
and sd of J and u at the true and fitted theta) and --out.with_suffix('.json') (theta true and
fitted, its Laplace sd and z, the sd of log(g_hat / g) over observations, and the PIT summary: 12 var(u), mean(u) - 1/2
and sqrt(n) KS distance, overall and by multiplicity and by posterior-mean bins; never by J_true).

usage: python sbc_joint_simulate.py --out OUT.parquet [--n-refl 2000] [--seed 0] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import torch

from integrator.model.joint.marginal import (
    interp_loglik,
    j_grid,
    log_prior_J,
    ls_intensity,
    obs_grid,
    obs_loglik_is,
    reflection_marginal,
)
from integrator.model.posterior.collapsed import laplace_eta, log_lik

D = torch.float64


def cheb(x: torch.Tensor, K: int) -> torch.Tensor:
    T = [torch.ones_like(x), x]
    for _ in range(2, K):
        T.append(2 * x * T[-1] - T[-2])
    return torch.stack(T[:K], -1)


def make_profile(P: int, phi: torch.Tensor):
    """Oriented-Gaussian profile on a P x P window, centred: eta (B, 2) -> p (B, P*P), for fixed azimuths phi (B,)."""
    ax = torch.arange(P, dtype=D, device=phi.device) - (P - 1) / 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    xx, yy = xx.reshape(-1), yy.reshape(-1)
    u = torch.cos(phi)[:, None] * xx + torch.sin(phi)[:, None] * yy
    v = -torch.sin(phi)[:, None] * xx + torch.cos(phi)[:, None] * yy

    def profile(eta: torch.Tensor) -> torch.Tensor:
        sa, sc = torch.exp(eta[:, 0:1]), torch.exp(eta[:, 1:2])
        q = -0.5 * ((u / sa) ** 2 + (v / sc) ** 2)
        return torch.softmax(q, -1)

    return profile


def pit_summary(u: np.ndarray) -> dict:
    u = np.sort(u[np.isfinite(u)])
    n = len(u)
    if n == 0:
        return {"n": 0}
    ks = np.max(np.maximum(np.arange(1, n + 1) / n - u, u - np.arange(n) / n))
    return {"n": int(n), "12var": float(12 * u.var()), "mean_minus_half": float(u.mean() - 0.5), "sqrtn_ks": float(np.sqrt(n) * ks)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-refl", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--P", type=int, default=15)
    ap.add_argument("--G0", type=float, default=400.0)
    ap.add_argument("--B", type=float, default=10.0)
    ap.add_argument("--s2-range", default="0.01,0.08")
    ap.add_argument("--centric-frac", type=float, default=0.1)
    ap.add_argument("--mult-mean", type=float, default=5.7)
    ap.add_argument("--lam-range", default="1.0,1.2")
    ap.add_argument("--theta", default="0.0,0.8,-0.3,0.1", help="true Chebyshev coefficients of log g in lambda")
    ap.add_argument("--a-b", type=float, default=6.0)
    ap.add_argument("--r-b", type=float, default=4.0)
    ap.add_argument("--eta-mu", default="0.9,0.0", help="prior mean of (log sd_along, log sd_across) at lambda~ = 0")
    ap.add_argument("--eta-slope", default="0.2,0.0", help="change of the prior mean per unit lambda~")
    ap.add_argument("--eta-sd", default="0.15,0.15")
    ap.add_argument("--n-eta", type=int, default=16)
    ap.add_argument("--q-inflate", type=float, default=1.3)
    ap.add_argument("--obs-grid", type=int, default=256)
    ap.add_argument("--j-grid", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device
    g = torch.Generator(device=dev).manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    # ---- the generative model
    H = a.n_refl
    s2lo, s2hi = (float(x) for x in a.s2_range.split(","))
    s2 = torch.as_tensor(rng.uniform(s2lo, s2hi, H), dtype=D, device=dev)
    centric = torch.as_tensor(rng.uniform(size=H) < a.centric_frac, device=dev)
    sigma = a.G0 * torch.exp(-2 * a.B * s2)
    k = torch.where(centric, 0.5, 1.0).to(D)
    J = torch.as_tensor(rng.gamma(k.cpu().numpy(), (sigma / k).cpu().numpy()), dtype=D, device=dev)
    r = 4.0
    mu_n = a.mult_mean - 1
    n_obs = 1 + rng.negative_binomial(r, r / (r + mu_n), H)
    refl = torch.as_tensor(np.repeat(np.arange(H), n_obs), device=dev)
    N = len(refl)
    lamlo, lamhi = (float(x) for x in a.lam_range.split(","))
    lam = torch.as_tensor(rng.uniform(lamlo, lamhi, N), dtype=D, device=dev)
    lt = 2 * (lam - lamlo) / (lamhi - lamlo) - 1
    theta_true = torch.as_tensor([float(x) for x in a.theta.split(",")], dtype=D, device=dev)
    K = len(theta_true)
    Tl = cheb(lt, K)
    g_true = torch.exp(Tl @ theta_true)
    phi = torch.as_tensor(rng.uniform(0, math.pi, N), dtype=D, device=dev)
    eta_mu = (torch.as_tensor([float(x) for x in a.eta_mu.split(",")], dtype=D, device=dev)[None]
              + lt[:, None] * torch.as_tensor([float(x) for x in a.eta_slope.split(",")], dtype=D, device=dev)[None])
    eta_sd = torch.as_tensor([float(x) for x in a.eta_sd.split(",")], dtype=D, device=dev)
    eta = eta_mu + eta_sd * torch.as_tensor(rng.standard_normal((N, 2)), dtype=D, device=dev)
    b = torch.as_tensor(rng.gamma(a.a_b, 1 / a.r_b, N), dtype=D, device=dev)
    profile_all = make_profile(a.P, phi)
    p_true = profile_all(eta)
    rate = g_true[:, None] * J[refl][:, None] * p_true + b[:, None]
    c = torch.as_tensor(rng.poisson(rate.cpu().numpy()), dtype=D, device=dev)
    m = torch.ones_like(c)
    I_true = g_true * J[refl]
    print(f"{H} reflections, {N} observations; I_true median {I_true.median():.1f}, "
          f"10/50/90% {np.percentile(I_true.cpu().numpy(), [10, 50, 90]).round(1).tolist()}; "
          f"background mean {b.mean():.2f} per pixel over {a.P * a.P} pixels", flush=True)

    # ---- per observation: Laplace proposal for eta, then l_i(I) on its own grid
    nodes_l, ll_l, ihat_l, sd_l = [], [], [], []
    for s in range(0, N, a.chunk):
        sl = slice(s, min(s + a.chunk, N))
        cs, ms, prof = c[sl], m[sl], make_profile(a.P, phi[sl])
        p0 = prof(eta_mu[sl])
        i0, b0, sd0 = ls_intensity(cs, ms, p0)
        mu_s, sd_e = eta_mu[sl], eta_sd

        def nlp(th, cs=cs, ms=ms, prof=prof, mu_s=mu_s):
            e, I, bb = th[:, :2], th[:, 2], th[:, 3]
            return (-log_lik(cs, ms, I, bb, prof(e)) - (a.a_b - 1) * torch.log(bb) + a.r_b * bb
                    + 0.5 * (((e - mu_s) / sd_e) ** 2).sum(-1))

        mode, cov, _ = laplace_eta(nlp, torch.cat([mu_s, i0.clamp_min(0.5)[:, None], b0[:, None]], 1), steps=30,
                                   positive=(2, 3))
        q_cov = cov[:, :2, :2] * a.q_inflate**2
        i_hat, _, sd = ls_intensity(cs, ms, prof(mode[:, :2]))
        nodes = obs_grid(i_hat, sd, grid=a.obs_grid)
        with torch.no_grad():
            ll = obs_loglik_is(cs, ms, prof, mode[:, :2], q_cov, mu_s, sd_e, nodes, a.a_b, a.r_b,
                               n_eta=a.n_eta, generator=g)
        nodes_l.append(nodes)
        ll_l.append(ll)
        ihat_l.append(i_hat)
        sd_l.append(sd)
        print(f"  observations {sl.stop}/{N}", flush=True)
    nodes, ll, i_hat, sd = (torch.cat(x) for x in (nodes_l, ll_l, ihat_l, sd_l))
    multi = torch.as_tensor(n_obs > 1, device=dev)[refl]

    def log_marginal(theta: torch.Tensor, grad_mask: torch.Tensor | None = None):
        gg = torch.exp(Tl @ theta)
        if grad_mask is not None:  # stop the scale gradient of multiplicity-1 reflections
            gg = torch.where(grad_mask, gg, gg.detach())
        xs, dx = j_grid(i_hat, sd, gg, refl, H, grid=a.j_grid)
        oll = interp_loglik(nodes, ll, gg[:, None] * xs[refl])
        lm, w = reflection_marginal(oll, refl, H, log_prior_J(xs, "wilson", sigma, centric), dx)
        return lm, w, xs, dx

    # ---- the scale: type-II maximum likelihood of theta, Sigma frozen
    theta = torch.zeros(K, dtype=D, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([theta], lr=1, max_iter=200, line_search_fn="strong_wolfe", tolerance_grad=1e-9)

    def closure():
        opt.zero_grad()
        loss = -log_marginal(theta, multi)[0].sum()
        loss.backward()
        return loss

    # repeated L-BFGS passes until theta stops moving: a single pass with the default tolerances
    # stopped short on 20,000 reflections (the loss is O(1e6)), which shrank theta toward its
    # start of 0 and gave errors smaller than the Laplace sd
    for it in range(50):
        before = theta.detach().clone()
        opt.step(closure)
        if (theta.detach() - before).abs().max() < 1e-7:
            break
    gnorm = float(torch.autograd.grad(-log_marginal(theta, multi)[0].sum(), theta)[0].norm())
    print(f"L-BFGS passes {it + 1}; gradient norm at the fit {gnorm:.3g}", flush=True)
    theta_hat = theta.detach()
    dlog = (Tl @ (theta_hat - theta_true)).cpu().numpy()
    print(f"theta true {theta_true.cpu().numpy().round(4).tolist()}  fitted {theta_hat.cpu().numpy().round(4).tolist()}; "
          f"sd of log(g_hat / g) {dlog.std():.4f}, mean {dlog.mean():+.4f}", flush=True)

    # ---- Laplace posterior of theta: the inverse Hessian of -log marginal at theta_hat, by central
    # differences of the autograd gradient (the interpolated l_i is piecewise linear in I, so its
    # analytic second derivative misses the curvature between nodes)
    def grad_at(t):
        t = t.clone().requires_grad_(True)
        gr, = torch.autograd.grad(-log_marginal(t, multi)[0].sum(), t)
        return gr

    hstep = 1e-3
    Hm = torch.stack([(grad_at(theta_hat + hstep * e) - grad_at(theta_hat - hstep * e)) / (2 * hstep)
                      for e in torch.eye(K, dtype=D, device=dev)])
    Hm = 0.5 * (Hm + Hm.T)
    theta_cov = torch.linalg.inv(Hm)
    theta_sd = theta_cov.diagonal().clamp_min(0).sqrt()
    print(f"Laplace sd of theta {theta_sd.cpu().numpy().round(5).tolist()}; "
          f"(theta_hat - theta_true) / sd {((theta_hat - theta_true) / theta_sd).cpu().numpy().round(2).tolist()}", flush=True)

    # ---- posterior of J and PIT at the true and the fitted scale
    rows = {"J_true": J.cpu().numpy(), "s2": s2.cpu().numpy(), "centric": centric.cpu().numpy(), "n_obs": n_obs}
    summ = {"theta_true": theta_true.cpu().tolist(), "theta_hat": theta_hat.cpu().tolist(),
            "sd_log_g_err": float(dlog.std()), "mean_log_g_err": float(dlog.mean()),
            "n_refl": H, "n_obs": N, "args": vars(a) | {"out": str(a.out)}}
    summ["theta_sd"] = theta_sd.cpu().tolist()
    summ["lbfgs_passes"], summ["grad_norm"] = it + 1, gnorm
    summ["theta_z"] = ((theta_hat - theta_true) / theta_sd).cpu().tolist()
    for tag, ths in (("true", theta_true[None]), ("fit", theta_hat[None])):
        ms, m2, us = 0, 0, 0
        for th in ths:  # equal-weight mixture over the theta draws (one draw for true and fit)
            with torch.no_grad():
                _, w, xs, dx = log_marginal(th)
            frac = ((J[:, None] - (xs - dx[:, None] / 2)) / dx[:, None]).clamp(0, 1)
            ms = ms + (w * xs).sum(1) / len(ths)
            m2 = m2 + ((w * xs**2).sum(1) + dx**2 / 12) / len(ths)
            us = us + (w * frac).sum(1) / len(ths)
        mean, var, u = ms, m2 - ms**2, us.cpu().numpy()
        rows[f"mean_{tag}"], rows[f"sd_{tag}"], rows[f"u_{tag}"] = mean.cpu().numpy(), var.sqrt().cpu().numpy(), u
        pm = mean.cpu().numpy()
        by = {"all": pit_summary(u)}
        for lo, hi in ((1, 2), (2, 4), (4, 8), (8, 1000)):
            by[f"mult_{lo}-{hi - 1}"] = pit_summary(u[(n_obs >= lo) & (n_obs < hi)])
        edges = np.quantile(pm, [0, 0.25, 0.5, 0.75, 1.0])
        for q in range(4):
            sel = (pm >= edges[q]) & (pm <= edges[q + 1])
            by[f"postmean_q{q + 1}_{edges[q]:.0f}-{edges[q + 1]:.0f}"] = pit_summary(u[sel])
        by["centric"] = pit_summary(u[centric.cpu().numpy()])
        summ[f"pit_{tag}"] = by
        print(f"PIT at the {tag} theta:")
        for kk, v in by.items():
            if v.get("n"):
                print(f"  {kk:28s} n {v['n']:6d}  12var {v['12var']:.3f}  mean-1/2 {v['mean_minus_half']:+.3f}  sqrt(n)KS {v['sqrtn_ks']:.2f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(a.out)
    a.out.with_suffix(".json").write_text(json.dumps(summ, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
