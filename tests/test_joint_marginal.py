"""The joint model's per-reflection marginal against brute-force quadrature."""

import math

import numpy as np
import torch
from scipy import integrate, special

from integrator.model.joint.marginal import (
    j_grid,
    log_prior_J,
    ls_intensity,
    obs_loglik,
    obs_loglik_is,
    reflection_marginal,
)

D = torch.float64


def gauss_profile(width, P=7):
    x = torch.arange(P, dtype=D) - (P - 1) / 2
    w = torch.as_tensor(width, dtype=D).reshape(-1, 1)
    q = torch.exp(-0.5 * (x[None] / w) ** 2)
    return q / q.sum(-1, keepdim=True)


def brute_obs_loglik(c, p, I, a_b, r_b):
    def f(b):
        lam = I * p + b
        return np.exp(np.sum(c * np.log(lam) - lam) + (a_b - 1) * np.log(b) - r_b * b + a_b * np.log(r_b) - special.gammaln(a_b))
    return math.log(integrate.quad(f, 1e-12, 200, limit=400, points=[0.1, 1, 5, 20])[0])


def make(seed=0, n_obs=(3, 1), a_b=4.0, r_b=2.0, J=(40.0, 15.0)):
    rng = np.random.default_rng(seed)
    refl, c, g, p = [], [], [], []
    for h, n in enumerate(n_obs):
        for _ in range(n):
            gi = rng.uniform(0.5, 2.0)
            pi = gauss_profile(rng.uniform(0.8, 1.5))[0].numpy()
            bi = rng.gamma(a_b, 1 / r_b)
            c.append(rng.poisson(gi * J[h] * pi + bi))
            refl.append(h)
            g.append(gi)
            p.append(pi)
    return (torch.as_tensor(np.array(c), dtype=D), torch.as_tensor(np.array(p), dtype=D),
            torch.as_tensor(np.array(g), dtype=D), torch.as_tensor(refl), a_b, r_b)


def test_obs_loglik_matches_quadrature_over_b():
    c, p, g, refl, a_b, r_b = make()
    m = torch.ones_like(c)
    I = torch.tensor([[0.5, 10.0, 40.0, 90.0]], dtype=D).expand(c.shape[0], -1)
    got = obs_loglik(c, m, p, I, a_b, r_b)
    for i in range(c.shape[0]):
        for k in range(I.shape[1]):
            ref = brute_obs_loglik(c[i].numpy(), p[i].numpy(), float(I[i, k]), a_b, r_b)
            assert abs(float(got[i, k]) - ref) < 0.02, (i, k, float(got[i, k]), ref)


def test_reflection_marginal_matches_quadrature_and_gradient():
    c, p, g, refl, a_b, r_b = make()
    m = torch.ones_like(c)
    H = 2
    sigma = torch.tensor([30.0, 30.0], dtype=D)
    centric = torch.tensor([False, True])
    i_hat, _, sd = ls_intensity(c, m, p)
    xs, dx = j_grid(i_hat, sd, g, refl, H, grid=256)

    def log_m(gv):
        ll = obs_loglik(c, m, p, gv[:, None] * xs[refl], a_b, r_b)
        return reflection_marginal(ll, refl, H, log_prior_J(xs, "wilson", sigma, centric), dx)[0]

    got = log_m(g)
    for h in range(H):
        idx = (refl == h).nonzero()[:, 0].tolist()
        k = 0.5 if bool(centric[h]) else 1.0

        def f(J, idx=idx, k=k):
            lp = k * math.log(k / 30.0) - special.gammaln(k) + (k - 1) * math.log(J) - k / 30.0 * J
            return math.exp(lp + sum(brute_obs_loglik(c[i].numpy(), p[i].numpy(), float(g[i]) * J, a_b, r_b) for i in idx))
        ref = math.log(integrate.quad(f, 1e-9, float(xs[h, -1]) * 1.5, limit=200, points=[1.0, 10.0, 40.0])[0])
        assert abs(float(got[h]) - ref) < 0.05, (h, float(got[h]), ref)
    gv = g.clone().requires_grad_(True)
    log_m(gv).sum().backward()
    eps = 1e-5
    for i in range(len(g)):
        e = torch.zeros_like(g)
        e[i] = eps
        fd = float((log_m(g + e).sum() - log_m(g - e).sum()) / (2 * eps))
        assert abs(float(gv.grad[i]) - fd) < 1e-3 * max(1.0, abs(fd)), (i, float(gv.grad[i]), fd)


def test_scale_gauge_is_exact_with_the_prior_scale():
    """g -> c g with Sigma -> Sigma / c leaves every reflection's marginal unchanged (J -> J / c)."""
    c, p, g, refl, a_b, r_b = make(seed=3)
    m = torch.ones_like(c)
    H = 2
    sigma = torch.tensor([30.0, 30.0], dtype=D)
    out = []
    for f in (1.0, 2.5):
        i_hat, _, sd = ls_intensity(c, m, p)
        xs, dx = j_grid(i_hat, sd, g * f, refl, H, grid=512)
        ll = obs_loglik(c, m, p, (g * f)[:, None] * xs[refl], a_b, r_b)
        out.append(reflection_marginal(ll, refl, H, log_prior_J(xs, "wilson", sigma / f), dx)[0])
    assert torch.allclose(out[0], out[1], atol=1e-3)


def test_importance_weights_give_the_marginal_over_eta():
    """l(I) = int l(I | eta) p(eta) deta: importance sampling from a shifted q agrees with a 1-D quadrature in eta."""
    c, p, g, refl, a_b, r_b = make(seed=5, n_obs=(1,))
    m = torch.ones_like(c)
    I = torch.tensor([[5.0, 20.0, 60.0]], dtype=D)
    mu, sd = torch.tensor([[0.0]], dtype=D), torch.tensor([0.3], dtype=D)

    def profile(eta):
        return gauss_profile(torch.exp(eta[:, 0]))

    gq = torch.Generator().manual_seed(0)
    got = obs_loglik_is(c, m, profile, torch.tensor([[0.15]], dtype=D), torch.tensor([[[0.12]]], dtype=D),
                        mu, sd, I, a_b, r_b, n_eta=4000, generator=gq)
    etas = torch.linspace(-2.0, 2.0, 801, dtype=D)
    ll = torch.stack([obs_loglik(c, m, profile(e.reshape(1, 1)), I, a_b, r_b)[0] for e in etas])
    lp = -0.5 * (etas / 0.3) ** 2 - math.log(0.3 * math.sqrt(2 * math.pi))
    ref = torch.logsumexp(ll + lp[:, None], 0) + math.log(float(etas[1] - etas[0]))
    assert torch.allclose(got[0], ref, atol=0.02), (got, ref)
    # equal weights under q estimate a different quantity
    unweighted = torch.logsumexp(torch.stack([obs_loglik(c, m, profile(torch.tensor([[0.15]], dtype=D) + 0.12**0.5 * torch.randn(1, 1, dtype=D, generator=gq)), I, a_b, r_b)[0]
                                              for _ in range(2000)]), 0) - math.log(2000)
    assert (unweighted - ref).abs().max() > 0.05


def test_interpolated_loglik_matches_direct_and_its_gradient():
    from integrator.model.joint.marginal import interp_loglik, obs_grid

    c, p, g, refl, a_b, r_b = make(seed=7)
    m = torch.ones_like(c)
    i_hat, _, sd = ls_intensity(c, m, p)
    nodes = obs_grid(i_hat, sd, grid=256)
    ll = obs_loglik(c, m, p, nodes, a_b, r_b)
    I = torch.stack([torch.linspace(float(nodes[i, 0]) + 0.1, float(nodes[i, -1]) - 0.1, 37, dtype=D) for i in range(len(g))])
    direct = obs_loglik(c, m, p, I, a_b, r_b)
    got = interp_loglik(nodes, ll, I)
    assert (got - direct).abs().max() < 0.02
    Ig = I.clone().requires_grad_(True)
    interp_loglik(nodes, ll, Ig).sum().backward()
    Id = I.clone().requires_grad_(True)
    obs_loglik(c, m, p, Id, a_b, r_b).sum().backward()
    assert (Ig.grad - Id.grad).abs().max() < 0.02


def test_hermite_interpolation_matches_direct_and_is_smooth():
    from integrator.model.joint.marginal import interp_loglik_hermite, obs_grid

    c, p, g, refl, a_b, r_b = make(seed=7)
    m = torch.ones_like(c)
    i_hat, _, sd = ls_intensity(c, m, p)
    nodes = obs_grid(i_hat, sd, grid=256)
    ll = obs_loglik(c, m, p, nodes, a_b, r_b)
    I = torch.stack([torch.linspace(float(nodes[i, 0]) + 0.1, float(nodes[i, -1]) - 0.1, 401, dtype=D) for i in range(len(g))])
    direct = obs_loglik(c, m, p, I, a_b, r_b)
    got = interp_loglik_hermite(nodes, ll, I)
    assert (got - direct).abs().max() < 2e-3
    Ig = I.clone().requires_grad_(True)
    interp_loglik_hermite(nodes, ll, Ig).sum().backward()
    Id = I.clone().requires_grad_(True)
    obs_loglik(c, m, p, Id, a_b, r_b).sum().backward()
    Il = I.clone().requires_grad_(True)
    from integrator.model.joint.marginal import interp_loglik
    interp_loglik(nodes, ll, Il).sum().backward()
    err_h, err_l = (Ig.grad - Id.grad).abs().max(), (Il.grad - Id.grad).abs().max()
    assert err_h < 0.25 * err_l, (float(err_h), float(err_l))
