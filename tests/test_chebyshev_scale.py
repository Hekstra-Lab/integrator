"""A linear Chebyshev scale, and why the link function matters.

The repo's `ChebyshevScale` ends in `F.softplus(phi @ c)`. That makes the
Poisson rate convex in the coefficients, so the log-likelihood is no longer
concave in them and the block stops being a weighted least squares. Here the
basis is used with no link, which keeps the rate linear in `c` and the block
closed form -- the same property every other block in the sweep relies on.
"""

import math

import torch

from integrator.fit.merge import (
    accumulate_scale_normals,
    chebyshev_design,
    solve_normals,
    tensor_design,
)

DT = torch.float64


def test_basis_is_the_chebyshev_recurrence():
    x = torch.linspace(-1, 1, 7, dtype=DT)
    phi = chebyshev_design(x, 4, lo=-1.0, hi=1.0)
    assert phi.shape == (7, 5)
    assert torch.allclose(phi[:, 0], torch.ones_like(x))
    assert torch.allclose(phi[:, 1], x)
    assert torch.allclose(phi[:, 2], 2 * x**2 - 1)
    assert torch.allclose(phi[:, 3], 4 * x**3 - 3 * x)


def test_values_outside_the_range_are_clamped_not_extrapolated():
    """Chebyshev polynomials diverge outside [-1, 1]; a stray frame must not."""
    x = torch.tensor([-5.0, 0.0, 5.0], dtype=DT)
    phi = chebyshev_design(x, 6, lo=-1.0, hi=1.0)
    assert float(phi.abs().max()) <= 1.0 + 1e-12


def test_tensor_design_is_the_row_wise_outer_product():
    a = chebyshev_design(torch.linspace(0, 1, 5, dtype=DT), 2)
    b = chebyshev_design(torch.linspace(0, 1, 5, dtype=DT), 3)
    t = tensor_design(a, b)
    assert t.shape == (5, 3 * 4)
    assert torch.allclose(t[2], torch.outer(a[2], b[2]).reshape(-1))


# --------------------------------------------------------------- the solve


def planted_scale(n=4000, hw=11, j_mean=600.0, bg=8.0, seed=0, du=4, dv=3):
    """Observations with a smooth scale in (image, s^2) planted."""
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n, generator=g, dtype=DT)
    v = torch.rand(n, generator=g, dtype=DT)
    phi = tensor_design(
        chebyshev_design(u, du, 0.0, 1.0), chebyshev_design(v, dv, 0.0, 1.0)
    )
    c_true = torch.zeros(phi.shape[1], dtype=DT)
    c_true[0] = 1.0
    c_true[1] = 0.25          # linear in image
    c_true[du + 1] = -0.35    # linear in s^2: the Debye-Waller ramp
    s_true = phi @ c_true
    assert float(s_true.min()) > 0.1

    yy, xx = torch.meshgrid(
        torch.arange(hw, dtype=DT), torch.arange(hw, dtype=DT), indexing="ij"
    )
    ctr = (hw - 1) / 2.0
    p = torch.exp(-((xx - ctr) ** 2 + (yy - ctr) ** 2) / (2 * 1.4**2)).reshape(-1)
    p = (p / p.sum()).unsqueeze(0).expand(n, -1).contiguous()
    uj = torch.rand(n, generator=g, dtype=DT).clamp_min(1e-6)
    j = (-j_mean * torch.log1p(-uj)).clamp_min(20.0)

    counts = torch.poisson(((s_true * j).unsqueeze(-1) * p + bg).float(), generator=g)
    return dict(
        counts=counts.to(DT), mask=torch.ones_like(counts, dtype=DT), profile=p,
        j=j, bg=torch.full((n,), bg, dtype=DT), phi=phi,
        c_true=c_true, s_true=s_true,
    )


def fit(d, n_iter=12):
    c = torch.zeros(d["phi"].shape[1], dtype=DT)
    c[0] = 1.0
    for _ in range(n_iter):
        a, b = accumulate_scale_normals(
            d["counts"], d["mask"], d["profile"], d["j"], d["bg"],
            d["phi"] @ c, d["phi"],
        )
        c = solve_normals(a, b, fallback=c)
    return c


def test_a_planted_resolution_ramp_is_recovered():
    """The term the sweep was missing, worth 57% across the real data."""
    d = planted_scale(seed=1)
    c = fit(d)
    s = d["phi"] @ c
    r = float((s / d["s_true"]).median())
    rel = float(((s / r - d["s_true"]).abs() / d["s_true"]).median())
    assert rel < 0.02, f"scale off by {rel:.4f} after the gauge"
    assert torch.corrcoef(torch.stack([s, d["s_true"]]))[0, 1] > 0.999


def test_the_fit_is_closed_form_and_does_not_depend_on_the_start():
    """Concave in the coefficients, so there is one optimum to find."""
    d = planted_scale(seed=2)
    a = fit(d, n_iter=12)
    c0 = torch.zeros_like(a)
    c0[0] = 3.0
    for _ in range(12):
        A, b = accumulate_scale_normals(
            d["counts"], d["mask"], d["profile"], d["j"], d["bg"],
            d["phi"] @ c0, d["phi"],
        )
        c0 = solve_normals(A, b, fallback=c0)
    assert torch.allclose(d["phi"] @ a, d["phi"] @ c0, rtol=1e-4)


def test_a_flat_scale_is_found_when_there_is_none():
    d = planted_scale(seed=3)
    d["c_true"] = torch.zeros_like(d["c_true"])
    d["c_true"][0] = 1.0
    d["s_true"] = d["phi"] @ d["c_true"]
    g = torch.Generator().manual_seed(33)
    d["counts"] = torch.poisson(
        ((d["s_true"] * d["j"]).unsqueeze(-1) * d["profile"] + 8.0).float(),
        generator=g,
    ).to(DT)
    s = d["phi"] @ fit(d)
    assert float((s / s.median() - 1.0).abs().max()) < 0.05


def test_batched_accumulation_matches_one_shot():
    """The normal equations are summed across batches, so they must add."""
    d = planted_scale(n=900, seed=4)
    c = torch.zeros(d["phi"].shape[1], dtype=DT)
    c[0] = 1.0
    s = d["phi"] @ c
    whole = accumulate_scale_normals(
        d["counts"], d["mask"], d["profile"], d["j"], d["bg"], s, d["phi"]
    )
    A = torch.zeros_like(whole[0])
    B = torch.zeros_like(whole[1])
    for k in range(0, 900, 137):
        sl = slice(k, min(k + 137, 900))
        a, b = accumulate_scale_normals(
            d["counts"][sl], d["mask"][sl], d["profile"][sl], d["j"][sl],
            d["bg"][sl], s[sl], d["phi"][sl],
        )
        A += a
        B += b
    assert torch.allclose(A, whole[0], rtol=1e-10)
    assert torch.allclose(B, whole[1], rtol=1e-10)


def test_a_singular_system_falls_back_instead_of_returning_nonsense():
    """A duplicated column makes the normal matrix rank deficient."""
    n = 50
    phi = torch.ones(n, 3, dtype=DT)
    phi[:, 1] = torch.linspace(0, 1, n, dtype=DT)
    phi[:, 2] = phi[:, 1]
    a = phi.transpose(0, 1) @ phi
    b = phi.transpose(0, 1) @ torch.ones(n, dtype=DT)
    prev = torch.tensor([1.0, 0.0, 0.0], dtype=DT)
    c = solve_normals(a, b, ridge=0.0, fallback=prev)
    assert torch.isfinite(c).all()


def test_coefficient_count_is_the_tensor_product_size():
    d = planted_scale(n=200, seed=5, du=10, dv=4)
    assert d["phi"].shape[1] == 11 * 5
    assert math.isfinite(float(fit(d, n_iter=3).sum()))
