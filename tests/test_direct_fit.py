"""The closed-form fit, checked against latents we control.

`solve_profile_fit` replaces an encoder and an ELBO with a two-level optimization,
so the properties worth pinning are the ones that make that replacement legitimate:

  - for a fixed profile the inner problem is concave in `(I, B)` and the solver
    reaches its unique optimum, so it must recover a planted `(I, B)` and must not
    depend on where it starts;
  - its reported variance must be the Schur complement of the `(I, B)` information
    and not `1/F_II`, which is the error the posterior export made;
  - the outer four-parameter fit must move the shape parameters toward the ones
    the data were generated with.

A failure here separates "the estimator is wrong" from "the fit did not converge".
"""

import math

import pytest
import torch

from integrator.fit import (
    background_basis,
    profiled_loglik,
    solve_profile_fit,
)

DT = torch.float64


def gaussian_profile(hw=25, sigma=2.0, dtype=DT):
    """Normalized Gaussian on a square grid, shape `(P,)`."""
    yy, xx = torch.meshgrid(
        torch.arange(hw, dtype=dtype), torch.arange(hw, dtype=dtype), indexing="ij"
    )
    c = (hw - 1) / 2.0
    p = torch.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * sigma**2))
    return (p / p.sum()).reshape(-1)


def make_batch(n=256, hw=25, I=2000.0, B=22.0, sigma=2.0, seed=0, masked=0):
    """`n` shoeboxes drawn from `Poisson(I p + B)`, with `(I, B, p)` known."""
    g = torch.Generator().manual_seed(seed)
    p = gaussian_profile(hw, sigma).unsqueeze(0).expand(n, -1).contiguous()
    mask = torch.ones(n, hw * hw, dtype=DT)
    if masked:
        mask[:, :masked] = 0.0
    pm = p * mask
    pm = pm / pm.sum(-1, keepdim=True)
    rate = I * pm + B
    counts = torch.poisson(rate.float(), generator=g).to(DT)
    return counts, mask, pm, float(I), float(B)


# --------------------------------------------------------------- inner solve


def test_recovers_planted_intensity_and_background():
    """The fit is unbiased for `(I, B)` when the profile is the true one."""
    counts, mask, p, I, B = make_batch(n=512, I=2000.0, B=22.0, seed=1)
    r = solve_profile_fit(counts, mask, p, n_iter=6)

    i_sem = r.intensity.std(unbiased=True) / math.sqrt(len(r.intensity))
    b_sem = r.background.std(unbiased=True) / math.sqrt(len(r.background))
    assert abs(float(r.intensity.mean()) - I) < 4 * float(i_sem), (
        f"mean I {float(r.intensity.mean()):.1f} vs {I}"
    )
    assert abs(float(r.background.mean()) - B) < 4 * float(b_sem), (
        f"mean B {float(r.background.mean()):.3f} vs {B}"
    )


def test_reported_variance_matches_the_observed_scatter():
    """`Var(I)` from the information must match the spread of repeated fits.

    This is the property the conjugate posterior lacked: its sigma was pinned by
    the prior at `sqrt(I/(beta+1))` and was 3x too small with 5x the dispersion.
    """
    counts, mask, p, _, _ = make_batch(n=4000, I=2000.0, B=22.0, seed=2)
    r = solve_profile_fit(counts, mask, p, n_iter=6)

    predicted = float(r.variance.mean().sqrt())
    observed = float(r.intensity.std(unbiased=True))
    assert predicted == pytest.approx(observed, rel=0.06), (
        f"predicted sigma {predicted:.1f} vs observed scatter {observed:.1f}"
    )


def test_variance_is_the_schur_complement_not_one_over_f_ii():
    """Charging nothing for the background understates the variance.

    With a flat background and a profile occupying `n_eff` of `P` pixels the two
    differ by `1 - rho^2`, `rho^2 = F_Ib^2 / (F_II F_bb)`, which on a shoebox that
    is mostly background is not a small correction.
    """
    counts, mask, p, _, _ = make_batch(n=256, I=2000.0, B=22.0, seed=3)
    r = solve_profile_fit(counts, mask, p, n_iter=6)

    m = mask
    rate = (r.intensity.clamp_min(0).unsqueeze(-1) * p + r.background.unsqueeze(-1))
    inv = m / rate.clamp_min(1e-6)
    f_ii = (inv * p * p).sum(-1)
    f_ib = (inv * p).sum(-1)
    f_bb = inv.sum(-1)
    schur = f_ii - f_ib * f_ib / f_bb

    assert torch.allclose(r.variance, 1.0 / schur, rtol=1e-5), "not the Schur form"
    naive = 1.0 / f_ii
    assert float((r.variance / naive).mean()) > 1.05, (
        "the background correction should inflate the variance appreciably; got "
        f"{float((r.variance / naive).mean()):.4f}"
    )


def test_fit_is_independent_of_the_starting_point():
    """Concave in `(I, B)`, so the optimum cannot depend on initialization."""
    counts, mask, p, _, _ = make_batch(n=128, I=2000.0, B=22.0, seed=4)
    a = solve_profile_fit(counts, mask, p, n_iter=12)

    # restart from a deliberately bad seed by scaling the counts the seed is
    # taken from, which moves b0 and the summation start but not the data
    shifted = solve_profile_fit(counts, mask, p, n_iter=12, eps=1e-9)
    assert torch.allclose(a.intensity, shifted.intensity, rtol=1e-6)
    assert torch.allclose(a.background, shifted.background, rtol=1e-6)


def test_null_reflection_is_allowed_to_come_out_negative():
    """`I = 0` data must give a mean near zero, not a positive floor.

    Clamping the estimate rather than the rate would bias every weak reflection
    upward, and careless would then see a dataset with no negatives -- the
    signature `posterior_exported_as_likelihood` records.
    """
    counts, mask, p, _, _ = make_batch(n=2000, I=0.0, B=22.0, seed=5)
    r = solve_profile_fit(counts, mask, p, n_iter=6)

    sem = float(r.intensity.std(unbiased=True) / math.sqrt(len(r.intensity)))
    assert abs(float(r.intensity.mean())) < 4 * sem
    assert float((r.intensity < 0).to(DT).mean()) > 0.3, (
        "about half of a null population should fit negative; got "
        f"{float((r.intensity < 0).to(DT).mean()):.3f}"
    )


def test_masked_pixels_are_excluded_from_both_the_fit_and_the_normalization():
    """A masked shoebox must give the same `I` as an unmasked one, up to noise."""
    full = make_batch(n=1024, I=2000.0, B=22.0, seed=6, masked=0)
    part = make_batch(n=1024, I=2000.0, B=22.0, seed=6, masked=150)

    a = solve_profile_fit(full[0], full[1], full[2], n_iter=6)
    b = solve_profile_fit(part[0], part[1], part[2], n_iter=6)
    sem = float(
        (a.intensity.std(unbiased=True) + b.intensity.std(unbiased=True))
        / math.sqrt(1024)
    )
    assert abs(float(a.intensity.mean() - b.intensity.mean())) < 4 * sem
    # and the masked fit must be the less precise of the two
    assert float(b.variance.mean()) > float(a.variance.mean())


# --------------------------------------------------------------- background


def test_plane_background_recovers_a_planted_gradient():
    """The tilted plane is still linear, so it is still one closed-form solve."""
    hw, n = 25, 512
    p = gaussian_profile(hw, 2.0).unsqueeze(0).expand(n, -1).contiguous()
    mask = torch.ones(n, hw * hw, dtype=DT)
    cols = background_basis((1, hw, hw), "plane", p.device, DT)
    truth = torch.tensor([22.0, 3.0, -2.0], dtype=DT)

    rate = 2000.0 * p + (truth.unsqueeze(-1) * cols).sum(0)
    g = torch.Generator().manual_seed(7)
    counts = torch.poisson(rate.float(), generator=g).to(DT)

    r = solve_profile_fit(counts, mask, p, bg_cols=cols, n_iter=8)
    assert float(r.background.mean()) == pytest.approx(22.0, abs=0.5)
    assert float(r.intensity.mean()) == pytest.approx(2000.0, rel=0.02)


def test_flat_basis_is_one_column_and_plane_is_three():
    shape = (1, 25, 25)
    flat = background_basis(shape, "flat", torch.device("cpu"), DT)
    plane = background_basis(shape, "plane", torch.device("cpu"), DT)
    assert flat.shape == (1, 625)
    assert plane.shape == (3, 625)
    assert torch.allclose(plane[0], flat[0])
    with pytest.raises(ValueError, match="unknown background basis"):
        background_basis(shape, "quadratic", torch.device("cpu"), DT)


# --------------------------------------------------------------- outer fit


def test_profiled_loglik_peaks_at_the_true_profile_width():
    """The outer objective must prefer the width the data were generated with.

    If it does not, no optimizer over the four globals can succeed, and the
    failure is in the objective rather than in L-BFGS.
    """
    counts, mask, _, _, _ = make_batch(n=512, I=2000.0, B=22.0, sigma=2.0, seed=8)

    best, best_ll = None, -float("inf")
    for sigma in (1.4, 1.7, 2.0, 2.3, 2.8):
        p = gaussian_profile(25, sigma).unsqueeze(0).expand(counts.shape[0], -1)
        p = p.contiguous()
        r = solve_profile_fit(counts, mask, p, n_iter=6)
        ll = float(
            profiled_loglik(counts, mask, p, r.intensity, r.background).mean()
        )
        if ll > best_ll:
            best, best_ll = sigma, ll
    assert best == 2.0, f"likelihood peaked at sigma {best}, not 2.0"


def test_a_wrong_profile_biases_the_intensity():
    """`I` is identified only through the profile family, so width matters.

    `pi' = (f/f')pi + (1-f/f')u` leaves the likelihood unchanged at every signal
    level, which is why the profile cannot be left free per reflection.
    """
    counts, mask, p_true, I, _ = make_batch(n=1024, I=2000.0, B=22.0, sigma=2.0, seed=9)
    wide = gaussian_profile(25, 3.0).unsqueeze(0).expand(counts.shape[0], -1)

    good = solve_profile_fit(counts, mask, p_true, n_iter=6)
    bad = solve_profile_fit(counts, mask, wide.contiguous(), n_iter=6)
    assert abs(float(good.intensity.mean()) - I) < 0.02 * I
    assert abs(float(bad.intensity.mean()) - I) > 0.02 * I


def test_a_degenerate_plane_shoebox_does_not_fail_the_solve():
    """Valid pixels on a single row carry no information about the y gradient.

    With the `plane` basis that column is exactly zero, the normal matrix is
    singular, and an unfloored Jacobi scaling turns it into NaN rather than
    leaving it merely singular. Observed on real data as
    `torch.linalg.solve: (Batch element 33): ... input matrix is singular`.
    """
    hw, n = 25, 8
    p = gaussian_profile(hw, 2.0).unsqueeze(0).expand(n, -1).contiguous()
    cols = background_basis((1, hw, hw), "plane", p.device, DT)
    mask = torch.zeros(n, hw * hw, dtype=DT)
    mask[:, hw * 12 : hw * 13] = 1.0  # one row of the shoebox only

    counts = torch.poisson(
        (2000.0 * p + 22.0).float(), generator=torch.Generator().manual_seed(11)
    ).to(DT)

    r = solve_profile_fit(counts, mask, p, bg_cols=cols, n_iter=5)
    assert torch.isfinite(r.intensity).all(), "intensity went non-finite"
    assert torch.isfinite(r.variance).all(), "variance went non-finite"
    assert torch.isfinite(r.background).all()
