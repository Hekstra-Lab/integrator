"""The Rao-Blackwellized intensity export, checked against known truth.

The claim being tested is calibration: `sqrt(var_rb)` should match the actual
scatter of the estimate around the true intensity. A plug-in sigma that conditions
on one profile understates it, which is what the downstream error model has been
inflating away.
"""

import pytest
import torch

from integrator.model.integrators.integrator_utils import (
    rao_blackwellized_intensity,
)


def _shoeboxes(n=4000, P=441, i_true=400.0, b_true=1.0, sigma_px=2.0, seed=0):
    """Poisson shoeboxes with a known Gaussian spot, plus the true profile."""
    g = torch.Generator().manual_seed(seed)
    side = int(P**0.5)
    y, x = torch.meshgrid(
        torch.arange(side) - (side - 1) / 2,
        torch.arange(side) - (side - 1) / 2,
        indexing="ij",
    )
    pi = torch.exp(-0.5 * (x**2 + y**2) / sigma_px**2).reshape(-1)
    pi = pi / pi.sum()
    rate = i_true * pi + b_true
    counts = torch.poisson(rate.expand(n, P), generator=g)
    return counts, torch.ones(n, P), pi, i_true, b_true


def test_recovers_the_true_intensity_with_the_true_profile():
    counts, mask, pi, i_true, b_true = _shoeboxes()
    prof = pi.expand(1, counts.shape[0], -1)
    bg = torch.full((1, counts.shape[0], 1), b_true)

    out = rao_blackwellized_intensity(counts, mask, prof, bg)

    assert out["i_rb"].mean().item() == pytest.approx(i_true, rel=0.02)
    # one profile sample means no profile uncertainty to add
    assert torch.allclose(out["var_rb_profile"], torch.zeros_like(out["var_rb"]))


def test_sigma_is_calibrated_against_the_observed_scatter():
    """With the true profile the reported sigma must match the real spread: that
    ratio is exactly the `a` the downstream error model fits."""
    counts, mask, pi, i_true, _ = _shoeboxes(n=8000)
    prof = pi.expand(1, counts.shape[0], -1)
    bg = torch.full((1, counts.shape[0], 1), 1.0)

    out = rao_blackwellized_intensity(counts, mask, prof, bg)
    observed = (out["i_rb"] - i_true).std()
    claimed = out["var_rb"].sqrt().mean()

    a = (observed / claimed).item()
    assert a == pytest.approx(1.0, abs=0.08), f"a = {a:.3f}"


def test_schur_exceeds_the_conditional_fisher_variance():
    """Not knowing the background costs precision; charging nothing for it is the
    understatement the plug-in sigma makes."""
    counts, mask, pi, _, b_true = _shoeboxes(sigma_px=6.0, i_true=80.0)
    prof = pi.expand(1, counts.shape[0], -1)
    bg = torch.full((1, counts.shape[0], 1), b_true)

    out = rao_blackwellized_intensity(counts, mask, prof, bg)

    rate = (out["i_rb"].unsqueeze(-1) * pi + b_true).clamp_min(1e-6)
    f_ii = (pi * pi / rate).sum(-1)
    conditional = 1.0 / f_ii
    assert (out["var_rb_cond"] > conditional).all()


def test_profile_uncertainty_raises_sigma():
    """Spread in the profile samples must appear in the reported variance, which is
    the term the plug-in export drops entirely."""
    counts, mask, pi, _, b_true = _shoeboxes(n=2000)
    g = torch.Generator().manual_seed(1)
    n = counts.shape[0]

    tight = pi.expand(4, n, -1)
    jitter = pi.expand(4, n, -1) * (
        1.0 + 0.3 * torch.randn(4, n, pi.numel(), generator=g)
    )
    jitter = jitter.clamp_min(1e-9)
    jitter = jitter / jitter.sum(-1, keepdim=True)
    bg = torch.full((4, n, 1), b_true)

    a = rao_blackwellized_intensity(counts, mask, tight, bg)
    b = rao_blackwellized_intensity(counts, mask, jitter, bg)

    assert b["var_rb_profile"].mean() > a["var_rb_profile"].mean()
    assert b["var_rb"].mean() > a["var_rb"].mean()


def test_gate_damps_the_profile_term_at_low_signal():
    """The profile term is only as good as q(h); below a few photons its spread is
    an over-broad posterior, not real uncertainty, so it gets damped."""
    counts, mask, pi, _, b_true = _shoeboxes(n=500, i_true=5.0)
    g = torch.Generator().manual_seed(2)
    prof = pi.expand(8, 500, -1) * (
        1.0 + 0.5 * torch.randn(8, 500, pi.numel(), generator=g)
    )
    prof = prof.clamp_min(1e-9)
    prof = prof / prof.sum(-1, keepdim=True)
    bg = torch.full((8, 500, 1), b_true)

    ungated = rao_blackwellized_intensity(counts, mask, prof, bg)
    gated = rao_blackwellized_intensity(
        counts, mask, prof, bg, gate_half_signal=100.0
    )

    assert (gated["rb_gate"] < 1.0).all()
    assert gated["var_rb"].mean() < ungated["var_rb"].mean()
    assert torch.allclose(gated["var_rb_cond"], ungated["var_rb_cond"])


def test_all_outputs_are_per_reflection_and_finite():
    counts, mask, pi, _, b_true = _shoeboxes(n=32)
    prof = pi.expand(3, 32, -1)
    bg = torch.full((3, 32, 1), b_true)
    out = rao_blackwellized_intensity(counts, mask, prof, bg)
    for k, v in out.items():
        assert v.shape == (32,), f"{k} has shape {tuple(v.shape)}"
        assert torch.isfinite(v).all(), k
    assert (out["var_rb"] > 0).all()


def test_poisson_noise_model_is_the_default_and_is_bit_identical():
    """`(a, b) = (1, 0)` must reproduce the Poisson path exactly.

    The variance function is an added option, so every run that does not
    set it has to export the number it exported before -- not close to it.
    """
    counts, mask, pi, _, b_true = _shoeboxes(n=48)
    prof = pi.expand(5, 48, -1)
    bg = torch.full((5, 48, 1), b_true)
    base = rao_blackwellized_intensity(counts, mask, prof, bg)
    same = rao_blackwellized_intensity(counts, mask, prof, bg,
                                       noise_a=1.0, noise_b=0.0)
    for k in base:
        assert torch.equal(base[k], same[k]), k


def test_sub_poisson_noise_sharpens_sigma_and_super_poisson_widens_it():
    """`Var = a * rate + b` moves sigma the way the information says.

    With `a < 1` each pixel is more informative than Poisson, so the
    conditional variance falls; with `a > 1` it rises. This is the whole
    point of the option: on hewl_1118 the held-out chi-square per pixel is
    0.24 at high resolution, so Poisson weights are too cautious there.
    """
    counts, mask, pi, _, b_true = _shoeboxes(n=48)
    prof = pi.expand(5, 48, -1)
    bg = torch.full((5, 48, 1), b_true)
    poisson = rao_blackwellized_intensity(counts, mask, prof, bg)
    sharp = rao_blackwellized_intensity(counts, mask, prof, bg, noise_a=0.25)
    wide = rao_blackwellized_intensity(counts, mask, prof, bg, noise_a=4.0)

    assert (sharp["var_rb_cond"] < poisson["var_rb_cond"]).all()
    assert (wide["var_rb_cond"] > poisson["var_rb_cond"]).all()
    # a pure scale on the variance scales the conditional variance with it
    assert torch.allclose(sharp["var_rb_cond"],
                          0.25 * poisson["var_rb_cond"], rtol=1e-4)


def test_read_noise_floors_the_variance_on_empty_pixels():
    """`b > 0` is a floor: it cannot make a pixel more informative."""
    counts, mask, pi, _, b_true = _shoeboxes(n=48)
    prof = pi.expand(5, 48, -1)
    bg = torch.full((5, 48, 1), b_true)
    base = rao_blackwellized_intensity(counts, mask, prof, bg)
    floored = rao_blackwellized_intensity(counts, mask, prof, bg,
                                          noise_b=25.0)
    assert (floored["var_rb_cond"] > base["var_rb_cond"]).all()


def test_pedestal_default_is_off_and_bit_identical():
    """`pedestal="none"` must reproduce the previous export exactly."""
    counts, mask, pi, _, b_true = _shoeboxes(n=48)
    prof = pi.expand(5, 48, -1)
    bg = torch.full((5, 48, 1), b_true)
    base = rao_blackwellized_intensity(counts, mask, prof, bg)
    same = rao_blackwellized_intensity(counts, mask, prof, bg,
                                       pedestal="none")
    for k in base:
        assert torch.equal(base[k], same[k]), k
    assert (base["rb_pedestal"] == 0).all()


def test_pedestal_is_recovered_from_a_shoebox_that_has_one():
    """An offset carrying no photons comes back as `c0 = mean - var`.

    Built so the answer is known: the background is `c0 + Poisson(rate)`,
    so a correct estimator returns `c0` whatever `c0` is, and the signal
    pixels must not drag it.
    """
    torch.manual_seed(0)
    n, npx, c0, bg_rate = 64, 625, 9.0, 1.6
    pi = torch.zeros(n, npx)
    centre = npx // 2
    pi[:, centre - 13:centre + 13] = 1.0
    pi = pi / pi.sum(-1, keepdim=True)
    counts = (torch.poisson(torch.full((n, npx), bg_rate)) + c0
              + torch.poisson(400.0 * pi))
    mask = torch.ones(n, npx)
    prof = pi.unsqueeze(0).expand(3, n, -1)
    bg = torch.full((3, n, 1), c0 + bg_rate)

    out = rao_blackwellized_intensity(counts, mask, prof, bg,
                                      pedestal="per_reflection",
                                      noise_b=0.5)
    got = out["rb_pedestal"]
    assert abs(float(got.median()) - c0) < 0.6, float(got.median())


def test_the_pedestal_sharpens_sigma_because_the_pixels_are_quieter():
    """Removing an offset that carries no photons lowers the variance.

    This is the whole point: a background pixel reading 10.5 ADU on a
    9 ADU pedestal holds 1.5 photons, not 10.5, so it is seven times less
    noisy than Poisson-on-ADU claims and carries seven times the weight.
    """
    torch.manual_seed(1)
    n, npx, c0 = 64, 625, 9.0
    pi = torch.zeros(n, npx)
    centre = npx // 2
    pi[:, centre - 13:centre + 13] = 1.0
    pi = pi / pi.sum(-1, keepdim=True)
    counts = (torch.poisson(torch.full((n, npx), 1.6)) + c0
              + torch.poisson(400.0 * pi))
    mask = torch.ones(n, npx)
    prof = pi.unsqueeze(0).expand(3, n, -1)
    bg = torch.full((3, n, 1), c0 + 1.6)

    plain = rao_blackwellized_intensity(counts, mask, prof, bg,
                                        noise_b=0.5)
    ped = rao_blackwellized_intensity(counts, mask, prof, bg,
                                      pedestal="per_reflection",
                                      noise_b=0.5)
    assert (ped["var_rb_cond"] < plain["var_rb_cond"]).all()
    assert torch.isfinite(ped["var_rb"]).all()
    assert (ped["var_rb"] > 0).all()


def test_pedestal_rejects_an_unknown_mode():
    counts, mask, pi, _, b_true = _shoeboxes(n=8)
    prof = pi.expand(2, 8, -1)
    bg = torch.full((2, 8, 1), b_true)
    with pytest.raises(ValueError, match="pedestal must be"):
        rao_blackwellized_intensity(counts, mask, prof, bg, pedestal="auto",
                                    noise_b=0.5)


def test_a_negative_variance_floor_is_refused():
    """`noise_b` is read-noise variance; negative is not a variance."""
    counts, mask, pi, _, b_true = _shoeboxes(n=8)
    prof = pi.expand(2, 8, -1)
    bg = torch.full((2, 8, 1), b_true)
    with pytest.raises(ValueError, match="cannot be negative"):
        rao_blackwellized_intensity(counts, mask, prof, bg, noise_b=-9.04)


def test_an_offset_without_a_floor_is_refused():
    """Subtracting c0 with no floor is the 1e6-weight path."""
    counts, mask, pi, _, b_true = _shoeboxes(n=8)
    prof = pi.expand(2, 8, -1)
    bg = torch.full((2, 8, 1), b_true)
    with pytest.raises(ValueError, match="pedestal needs noise_b"):
        rao_blackwellized_intensity(counts, mask, prof, bg,
                                    pedestal="per_reflection")


def test_fixed_bg_is_smaller_because_it_stops_double_charging():
    """`sample_bg` prices the background twice; `fixed_bg` once.

    `schur` is already `[F^-1]_II`, which charges for B being unknown.
    Each latent sample also carries its own background draw, so the
    spread of `i_hat` across samples charges for it again. Refitting
    every profile sample against one background leaves the spread as
    profile uncertainty alone.
    """
    torch.manual_seed(3)
    n, npx = 64, 625
    pi = torch.zeros(n, npx)
    centre = npx // 2
    pi[:, centre - 20:centre + 20] = 1.0
    pi = pi / pi.sum(-1, keepdim=True)
    counts = torch.poisson(600.0 * pi + 12.0)
    mask = torch.ones(n, npx)
    prof = (pi.unsqueeze(0) * (1.0 + 0.03 * torch.randn(6, n, 1))).clamp_min(1e-8)
    prof = prof / prof.sum(-1, keepdim=True)
    bg = 12.0 + 0.4 * torch.randn(6, n, 1)

    both = rao_blackwellized_intensity(counts, mask, prof, bg)
    once = rao_blackwellized_intensity(counts, mask, prof, bg,
                                       background_term="fixed_bg")
    assert torch.equal(both["var_rb_cond"], once["var_rb_cond"])
    assert (once["var_rb_profile"] <= both["var_rb_profile"] + 1e-6).all()
    assert once["var_rb"].mean() < both["var_rb"].mean()


def test_sample_bg_is_the_default_and_bit_identical():
    counts, mask, pi, _, b_true = _shoeboxes(n=32)
    prof = pi.expand(4, 32, -1)
    bg = torch.full((4, 32, 1), b_true)
    base = rao_blackwellized_intensity(counts, mask, prof, bg)
    same = rao_blackwellized_intensity(counts, mask, prof, bg,
                                       background_term="sample_bg")
    for k in base:
        assert torch.equal(base[k], same[k]), k


def test_background_term_rejects_an_unknown_mode():
    counts, mask, pi, _, b_true = _shoeboxes(n=8)
    prof = pi.expand(2, 8, -1)
    bg = torch.full((2, 8, 1), b_true)
    with pytest.raises(ValueError, match="background_term must be"):
        rao_blackwellized_intensity(counts, mask, prof, bg,
                                    background_term="auto")
