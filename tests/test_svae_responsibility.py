"""The SVAE's gate, and what its conjugate intensity update does and does not assume.

Three properties, all checkable on synthetic data with the latents known, so a
failure separates "the parameterization is wrong" from "it trains badly".

The model is `c_i ~ Poisson(I p_i + B)` over a shoebox, with `p` normalized over
the valid pixels. `SVAEIntegrator` splits each count into signal and background
with a gate `g_i`, then forms

    alpha' = alpha + sum(g_i c_i),   beta' = beta + sum(p_i m_i).

That pair is the exact Poisson-Gamma posterior for the AUGMENTED model, the one
in which the signal counts `s_i` are observed. Two consequences follow from the
algebra alone and are asserted here:

  - `sum(p m) == 1` by normalization, so `beta'` is the same constant for every
    reflection and the posterior Fano factor is fixed by the prior. Precision
    cannot vary per reflection in this branch, whatever the data says.
  - the substitution `s_i -> g_i c_i` is unbiased only when `g` is the true
    responsibility `I p_i / (I p_i + B)`. A free gate carries no such guarantee,
    and `cfg.gate == "learned"` is a free gate.
"""

import numpy as np
import pytest
import torch

DT = torch.float64


def responsibility(I, p, B):
    """Exact posterior share of a count belonging to signal."""
    return I * p / (I * p + B)


def make_box(hw=21, I=5000.0, B=20.0, sigma=2.0, seed=0):
    """A Gaussian spot on a flat background, with the latents known."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.arange(hw, dtype=DT), torch.arange(hw, dtype=DT), indexing="ij"
    )
    c = (hw - 1) / 2.0
    p = torch.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * sigma**2))
    p = (p / p.sum()).reshape(-1)
    rate = I * p + B
    counts = torch.poisson(rate.float(), generator=g).to(DT)
    return p, counts, rate


# --------------------------------------------------------------- test 1

def test_responsibility_update_is_unbiased_for_the_intensity():
    """`sum(r_i c_i)` recovers `I` when `r` is the true responsibility.

    This is the identity the conjugate update rests on. Averaged over
    realizations the estimator must return `I` with no fitted quantity
    anywhere, so a failure here is a plumbing error and not a training one.
    """
    I, B, hw = 5000.0, 20.0, 21
    p, _, _ = make_box(hw=hw, I=I, B=B)
    r = responsibility(I, p, B)

    # E[sum r c] = sum r (I p + B) = I exactly, by construction of r
    exact = float((r * (I * p + B)).sum())
    assert exact == pytest.approx(I, rel=1e-12)

    # and the sample mean converges to it
    est = []
    for s in range(400):
        _, counts, _ = make_box(hw=hw, I=I, B=B, seed=s)
        est.append(float((r * counts).sum()))
    est = np.asarray(est)
    sem = est.std(ddof=1) / np.sqrt(len(est))
    assert abs(est.mean() - I) < 4 * sem, (
        f"mean {est.mean():.1f} vs I={I}, sem {sem:.1f}"
    )


def test_a_gate_that_is_not_the_responsibility_is_biased():
    """A free gate has no unbiasedness guarantee, and generically misses.

    The learned gate is a free per-pixel output with no structural tie to the
    profile. Standing in for it with any other non-negative field shows that
    the identity above is a property of `r`, not of gating in general.
    """
    I, B, hw = 5000.0, 20.0, 21
    p, _, _ = make_box(hw=hw, I=I, B=B)
    r = responsibility(I, p, B)

    # a plausible but wrong gate: the right shape, slightly too broad
    p_wide, _, _ = make_box(hw=hw, I=I, B=B, sigma=3.0)
    g = responsibility(I, p_wide, B)

    exact_r = float((r * (I * p + B)).sum())
    exact_g = float((g * (I * p + B)).sum())
    assert exact_r == pytest.approx(I, rel=1e-12)
    assert abs(exact_g - I) > 0.02 * I, (
        "a mis-specified gate should bias the intensity; got "
        f"{exact_g:.1f} against I={I}"
    )


# --------------------------------------------------------------- test 2

def test_photons_are_conserved_by_the_split():
    """`sum(g c) + sum((1-g) c) == sum(c)`, for any gate in [0, 1].

    The signal and background posteriors are each built from one side of this
    split, so if it does not partition the counts the two are reading
    overlapping or incomplete evidence.
    """
    hw = 21
    p, counts, _ = make_box(hw=hw)
    mask = torch.ones_like(counts, dtype=torch.bool)
    mask[:7] = False  # some invalid pixels, as a real shoebox has

    for g in (
        responsibility(5000.0, p, 20.0),
        torch.sigmoid(torch.randn(hw * hw, dtype=DT)),
        torch.zeros(hw * hw, dtype=DT),
        torch.ones(hw * hw, dtype=DT),
    ):
        m = mask.to(DT)
        sig = (g * counts * m).sum()
        bg = ((1.0 - g) * counts * m).sum()
        assert float(sig + bg) == pytest.approx(float((counts * m).sum()), rel=1e-12)


# --------------------------------------------------------------- test 3

def test_conjugate_rate_is_constant_so_the_fano_factor_is_prior_fixed():
    """`beta' = beta + sum(p m)` and `sum(p m) == 1`, for every reflection.

    The profile is renormalized over the valid pixels, so the conjugate rate
    cannot depend on the data. `Var/mean = 1/(beta+1)` is then fixed before any
    shoebox is seen, which is why this branch cannot express a per-reflection
    precision and why `intensity_variance: fisher` exists.
    """
    beta_prior = 0.01
    rates, fanos = [], []
    for s, (I, B, sigma) in enumerate(
        [(50.0, 20.0, 1.5), (5000.0, 20.0, 2.0), (200000.0, 50.0, 3.0)]
    ):
        p, counts, _ = make_box(I=I, B=B, sigma=sigma, seed=s)
        mask = torch.ones_like(counts, dtype=torch.bool)
        mask[:7] = False
        # renormalize over the valid pixels, as `renormalize_masked` does
        pm = p * mask.to(DT)
        pm = pm / pm.sum()
        beta_post = beta_prior + float((pm * mask.to(DT)).sum())
        alpha_post = 1.0 + float((responsibility(I, pm, B) * counts * mask).sum())
        rates.append(beta_post)
        fanos.append((alpha_post / beta_post**2) / (alpha_post / beta_post))

    assert rates == pytest.approx([beta_prior + 1.0] * 3, rel=1e-12), (
        f"conjugate rate should be beta+1 regardless of data; got {rates}"
    )
    # the Fano factor is then 1/(beta+1) for all three, spanning I = 50 to 2e5
    assert fanos == pytest.approx([1.0 / (beta_prior + 1.0)] * 3, rel=1e-12), (
        f"Fano should be prior-fixed; got {fanos}"
    )


def test_coherence_residual_is_zero_only_for_the_responsibility_gate():
    """How far a gate sits from the responsibility implied by its own profile.

    This is the diagnostic to run on a trained checkpoint: under
    `gate: responsibility` it is zero by construction, and under
    `gate: learned` it measures how far the two accounts of where the spot is
    have drifted apart.
    """
    I, B = 5000.0, 20.0
    p, _, _ = make_box(I=I, B=B)
    r = responsibility(I, p, B)

    assert float((r - responsibility(I, p, B)).abs().max()) == pytest.approx(0.0)

    free = torch.sigmoid(torch.randn(p.numel(), dtype=DT, generator=
                                     torch.Generator().manual_seed(3)))
    resid = float((free - r).abs().mean())
    assert resid > 0.1, f"a free gate should not coincide with r; got {resid}"


# --------------------------------------------------------------- test 4

def test_background_seed_median_ignores_masked_pixels():
    """`b0` is the median over VALID pixels, not over `counts * mask`.

    Multiplying first turns every masked pixel into a zero that then enters the
    median, biasing it down with the masked fraction and sending it to exactly
    zero once more than half the box is masked. `b0 = 0` drives
    `r = I p / (I p + b0)` to 1 wherever the profile is non-zero, so the gate
    saturates and the intensity collapses to the box total.
    """
    hw = 21
    _, counts, _ = make_box(hw=hw, I=5000.0, B=20.0)
    n = counts.numel()

    for masked_frac in (0.0, 0.2, 0.4, 0.6, 0.86):
        m = torch.ones(n, dtype=DT)
        k = int(round(masked_frac * n))
        m[:k] = 0.0

        wrong = (counts * m).median()
        right = counts.masked_fill(m <= 0, float("nan")).nanmedian()

        ref = counts[m > 0].median()
        assert float(right) == pytest.approx(float(ref), rel=1e-12)
        if masked_frac > 0.5:
            assert float(wrong) == 0.0, "the old form should collapse to zero"
            assert float(right) > 10.0, "the fixed form must still see background"


def test_saturated_gate_turns_the_intensity_into_the_box_total():
    """What `b0 = 0` costs, stated as the intensity it produces.

    With the background seed at zero the responsibility is 1 on every pixel
    carrying profile mass, so `sum(r c)` returns the whole box rather than the
    signal in it -- on a shoebox that is 96.6% background that is wrong by
    more than an order of magnitude.
    """
    I, B, hw = 5000.0, 20.0, 21
    p, counts, _ = make_box(hw=hw, I=I, B=B)

    good = float((responsibility(I, p, B) * counts).sum())
    bad = float((responsibility(I, p, 1e-6) * counts).sum())

    assert good == pytest.approx(I, rel=0.1)
    assert bad > 2.0 * I, f"saturated gate should over-count; got {bad:.0f}"


def test_background_seed_accepts_integer_counts():
    """The loader hands counts over as int32; a NaN cannot go into an int tensor.

    Regression: the masked-median fix was first written as
    `raw_counts.masked_fill(mask <= 0, nan)` and died at the first training
    step with "value cannot be converted to type int without overflow",
    because every test fed it float counts.
    """
    n = 64
    counts_int = torch.randint(0, 50, (2, n), dtype=torch.int32)
    m = torch.ones(2, n, dtype=DT)
    m[:, : n // 3] = 0.0

    # the profile's dtype is the only one guaranteed floating point: counts
    # are int32 and the mask is bool, and a NaN fits in neither
    prf = torch.rand(2, n, dtype=DT)
    for mask in (m, m > 0):                      # float mask and bool mask
        b0 = (
            counts_int.to(prf.dtype)
            .masked_fill(mask <= 0, float("nan"))
            .nanmedian(dim=-1, keepdim=True)
            .values
        )
        assert torch.isfinite(b0).all()
        for i in range(2):
            ref = counts_int[i][m[i] > 0].to(DT).median()
            assert float(b0[i]) == pytest.approx(float(ref), rel=1e-12)
