"""`integrator.model.posterior.collapsed`: the (I, b) posterior given a profile, and the Laplace step."""

from __future__ import annotations

import math

import torch

from integrator.model.posterior.collapsed import collapsed, laplace_eta, log_lik


def _profile(P=225, side=15, sx=2.0, sy=1.0):
    y, x = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    g = torch.exp(-0.5 * (((x - 7.0) / sx) ** 2 + ((y - 7.0) / sy) ** 2)).reshape(-1).double()
    return g / g.sum()


def _draw(n, gen, I_rate=0.01, a_b=40.0, r_b=2.0):
    p = _profile()
    I = torch.distributions.Gamma(torch.ones(n, dtype=torch.float64), torch.full((n,), I_rate, dtype=torch.float64)).sample()
    b = torch.distributions.Gamma(torch.full((n,), a_b, dtype=torch.float64), torch.full((n,), r_b, dtype=torch.float64)).sample()
    c = torch.poisson(I[:, None] * p + b[:, None], generator=gen)
    return c, torch.ones_like(c), p.expand(n, -1), I, b


def test_moments_and_evidence_match_a_brute_force_grid():
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    c, m, p, I0, b0 = _draw(6, gen)
    one = torch.ones(6, dtype=torch.float64)
    q = collapsed(c, m, p, one, 0.01 * one, 40 * one, 2 * one, grid=256)
    mI, vI = q.mean_var_I()
    mb, vb = q.mean_var_b()
    for j in range(6):
        Ig = torch.linspace(1e-3, float(mI[j] + 12 * vI[j].sqrt()), 1500, dtype=torch.float64)
        bg = torch.linspace(float(mb[j] - 12 * vb[j].sqrt()), float(mb[j] + 12 * vb[j].sqrt()), 600, dtype=torch.float64).clamp_min(1e-3)
        II, BB = torch.meshgrid(Ig, bg, indexing="ij")
        lp = (log_lik(c[j], m[j], II.reshape(-1), BB.reshape(-1), p[j]).reshape(II.shape)
              + torch.distributions.Gamma(1.0, 0.01).log_prob(II) + torch.distributions.Gamma(40.0, 2.0).log_prob(BB))
        mx = lp.max()
        w = torch.exp(lp - mx)
        dA = (Ig[1] - Ig[0]) * (bg[1] - bg[0])
        Z = mx + torch.log(w.sum() * dA)
        w = w / w.sum()
        assert abs(float((w * II).sum() - mI[j])) < 0.02 * float(vI[j].sqrt()) + 0.05
        assert abs(float((w * II**2).sum() - (w * II).sum() ** 2) / float(vI[j]) - 1) < 0.03
        assert abs(float((w * BB).sum() - mb[j])) < 0.02 * float(vb[j].sqrt())
        assert abs(float(Z - q.log_z[j])) < 0.02


def test_pit_is_uniform_on_data_drawn_from_the_model():
    gen = torch.Generator().manual_seed(1)
    c, m, p, I0, b0 = _draw(4000, gen)
    one = torch.ones(4000, dtype=torch.float64)
    q = collapsed(c, m, p, one, 0.01 * one, 40 * one, 2 * one)
    for u in (q.cdf_I(I0), q.cdf_b(b0)):
        u = torch.sort(u).values
        n = len(u)
        D = max(float((torch.arange(1, n + 1) / n - u).max()), float((u - torch.arange(n) / n).max()))
        assert math.sqrt(n) * D < 1.63  # 1 % critical value
        assert abs(float(12 * u.var()) - 1) < 0.08


def test_laplace_recovers_a_gaussian():
    torch.manual_seed(2)
    A = torch.randn(3, 4, 4, dtype=torch.float64)
    P = A @ A.transpose(1, 2) + 4 * torch.eye(4, dtype=torch.float64)  # precision
    mu = torch.randn(3, 4, dtype=torch.float64)

    def nlp(th):
        d = th - mu
        return 0.5 * torch.einsum("bi,bij,bj->b", d, P, d)

    mode, cov, npd = laplace_eta(nlp, torch.zeros(3, 4, dtype=torch.float64), steps=20)
    torch.testing.assert_close(mode, mu, rtol=0, atol=1e-8)
    torch.testing.assert_close(cov, torch.linalg.inv(P), rtol=1e-6, atol=1e-10)
    assert not npd.any()
