"""The closed-form merge, checked against planted scales and structure factors.

The claim under test is that pixels to structure factors needs no gradient step:
the responsibility split makes the model conditionally conjugate, so each block
has a Gamma or moment-matched update, and iterating them ascends the likelihood.

What has to hold for that to be usable:

  - the sweep must recover a planted `(J, S, B)` from pixels alone;
  - it must do so from a deliberately bad start, because a flat gauge direction
    is the known failure mode of block updates on `S_i J_h`;
  - the conjugate variance must be visibly WRONG and the Fisher variance right,
    since that is the whole content of the export finding and the reason the two
    are computed separately.
"""

import math

import pytest
import torch

from integrator.fit.merge import (
    MergeHyperparameters,
    merged_fisher_variance,
    observation_stats,
    rescale_gauge,
    responsibilities,
    update_background,
    update_merged,
    update_scale,
    update_wilson_sigma,
    wilson_prior,
)

DT = torch.float64


def planted(
    n_h=200, mult=8, hw=15, sigma=1.8, bg=22.0, seed=0, scale_spread=0.3,
    per_image=100, j_mean=3000.0,
):
    """`n_h` Miller indices observed `mult` times each, with everything known.

    Scales are per IMAGE, with `per_image` observations sharing one. A scale per
    observation is not identified -- see `update_scale` -- so planting one would
    test a degenerate model rather than the estimator.
    """
    g = torch.Generator().manual_seed(seed)
    n = n_h * mult

    yy, xx = torch.meshgrid(
        torch.arange(hw, dtype=DT), torch.arange(hw, dtype=DT), indexing="ij"
    )
    c = (hw - 1) / 2.0
    p = torch.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * sigma**2)).reshape(-1)
    p = (p / p.sum()).unsqueeze(0).expand(n, -1).contiguous()

    # inverse-CDF so the draw honours `seed`; a bare .sample() would not
    u = torch.rand(n_h, generator=g, dtype=DT).clamp(1e-9, 1 - 1e-9)
    j_true = (-float(j_mean) * torch.log1p(-u)).clamp_min(0.02 * j_mean)
    index = torch.arange(n_h).repeat_interleave(mult)

    # Images are assigned at RANDOM. A structured assignment such as
    # `arange(n) % n_img` puts each Miller index on a block of consecutive
    # images, which confounds `h` with the image and leaves `S_g J_h`
    # unidentified within the block: measured that way, J carries a flat 10%
    # multiplicative error at every intensity however long the sweep runs. The
    # scale/intensity design has to be connected, which is the same requirement
    # that makes careless tie images together through a smooth metadata scale.
    n_img = max(1, n // per_image)
    image = torch.randint(0, n_img, (n,), generator=g)
    g_img = (
        1.0 + scale_spread * torch.randn(n_img, generator=g, dtype=DT)
    ).clamp_min(0.2)
    s_true = g_img[image]
    b_true = torch.full((n,), bg, dtype=DT)

    rate = (s_true * j_true[index]).unsqueeze(-1) * p + b_true.unsqueeze(-1)
    counts = torch.poisson(rate.float(), generator=g).to(DT)
    mask = torch.ones_like(counts)
    return dict(
        counts=counts, mask=mask, profile=p, index=index, n_h=n_h,
        j_true=j_true, s_true=s_true, b_true=b_true,
        image=image, n_img=n_img, g_true=g_img,
    )


def degauge(merged, truth):
    """Divide out the overall scale before comparing to a planted truth.

    `S J` is invariant under `S -> cS, J -> J/c`, and `rescale_gauge` pins the
    representative at `mean(S) = 1` while the planted scales have some other
    mean. The recovered factor tracks that mean to four digits (0.9676 against
    0.9668, 1.0420 against 1.0420), so it is the gauge and not an error. A
    merged dataset is defined up to one scale in exactly this way.
    """
    a = (merged / truth).median()
    return merged / a, float(a)


def sweep(d, n_sweeps=60, start_scale=1.0, start_j=None, hyper=None):
    """Run the closed-form block updates and return the final state."""
    hyper = hyper or MergeHyperparameters()
    n, n_h, index = d["counts"].shape[0], d["n_h"], d["index"]
    scale = torch.full((n,), float(start_scale), dtype=DT)
    merged = (
        torch.full((n_h,), float(start_j), dtype=DT)
        if start_j is not None
        else torch.full((n_h,), float(d["counts"].sum(-1).mean()), dtype=DT)
    )
    background = torch.full((n,), float(d["counts"].median()), dtype=DT)
    bin_index = torch.zeros(n_h, dtype=torch.long)

    for _ in range(n_sweeps):
        r = responsibilities(d["profile"], scale, merged[index], background)
        stats = observation_stats(d["counts"], d["mask"], r)

        sig_sum = torch.zeros(n_h, dtype=DT).scatter_add_(0, index, stats.signal)
        scl_sum = torch.zeros(n_h, dtype=DT).scatter_add_(0, index, scale)
        sigma_d = update_wilson_sigma(merged, bin_index, 1)
        alpha_w, beta_w = wilson_prior(sigma_d.expand(n_h).contiguous())
        merged, var_conj = update_merged(sig_sum, scl_sum, alpha_w, beta_w)

        gain = update_scale(
            stats.signal, merged[index], hyper, d["image"], d["n_img"]
        )
        scale = gain[d["image"]]
        background = update_background(stats, hyper)
        # the gauge rescale multiplies J by c, so its variance moves by c^2
        c = scale.mean().clamp_min(1e-12)
        scale, merged = rescale_gauge(scale, merged)
        var_conj = var_conj * c.pow(2)

    return dict(
        scale=scale, merged=merged, background=background, var_conj=var_conj,
    )


# --------------------------------------------------------------- recovery


def test_sweep_recovers_planted_structure_factors():
    """`J_h` comes back from pixels, with no gradient step taken."""
    d = planted(seed=1)
    out = sweep(d)
    j, t = out["merged"], d["j_true"]
    r = float(torch.corrcoef(torch.stack([j, t]))[0, 1])
    j_g, a = degauge(j, t)
    rel = float(((j_g - t).abs() / t).median())
    assert r > 0.999, f"corr(J, truth) {r:.5f}"
    assert rel < 0.02, f"median |J - truth| / truth {rel:.4f} after de-gauging"
    # the recovered gauge must be the planted mean scale, not an arbitrary number
    assert a == pytest.approx(float(d["s_true"].mean()), rel=2e-3)


def test_sweep_recovers_the_scales_and_backgrounds():
    d = planted(seed=2)
    out = sweep(d)
    s, b = out["scale"], out["background"]
    rs = float(torch.corrcoef(torch.stack([s, d["s_true"]]))[0, 1])
    assert rs > 0.99, f"corr(S, truth) {rs:.5f}"
    assert float(b.mean()) == pytest.approx(float(d["b_true"].mean()), rel=0.03)


def test_gauge_is_pinned_so_a_bad_start_still_converges():
    """`S J` is invariant under `S -> cS, J -> J/c`; without a pin it drifts."""
    d = planted(seed=3)
    a = sweep(d, start_scale=1.0)
    b = sweep(d, start_scale=20.0)
    assert float(a["scale"].mean()) == pytest.approx(1.0, abs=1e-9)
    assert float(b["scale"].mean()) == pytest.approx(1.0, abs=1e-9)
    rel = float(((a["merged"] - b["merged"]).abs() / a["merged"]).median())
    assert rel < 0.02, f"two starts disagree by {rel:.4f} after the gauge pin"


def test_more_observations_give_a_tighter_merged_estimate():
    """Multiplicity must enter the rate, or the merged Fano is prior-pinned."""
    err = {}
    for mult in (2, 16):
        d = planted(n_h=200, mult=mult, seed=4)
        out = sweep(d)
        j_g, _ = degauge(out["merged"], d["j_true"])
        err[mult] = float(((j_g - d["j_true"]).abs() / d["j_true"]).median())
    assert err[16] < 0.6 * err[2], f"error by multiplicity: {err}"


# --------------------------------------------------------------- variances


def test_conjugate_variance_is_optimistic_and_fisher_variance_is_not():
    """The point of computing both, measured where it matters.

    The augmented posterior treats the signal photons as observed, so it drops
    the split uncertainty. That term is `n_eff * B`, so the two variances agree
    on bright data and diverge as the shoebox becomes background-dominated --
    which is the Laue regime, at 96.6% background. Measured here at
    `n_eff B / I = 3`, the standardized residual has sd 1.06 under the Fisher
    variance and 3.30 under the conjugate one.
    """
    d = planted(n_h=400, mult=8, seed=5, j_mean=300.0)
    out = sweep(d, n_sweeps=120)
    index = d["index"]

    var_fisher = merged_fisher_variance(
        d["profile"], d["mask"], out["scale"], out["merged"][index],
        out["background"], index, d["n_h"],
    )
    j_g, _ = degauge(out["merged"], d["j_true"])
    resid = j_g - d["j_true"]
    z_fisher = float((resid / var_fisher.sqrt()).std())
    z_conj = float((resid / out["var_conj"].sqrt()).std())

    assert 0.8 < z_fisher < 1.4, (
        f"Fisher variance should be calibrated; standardized sd {z_fisher:.2f}"
    )
    assert z_conj > 2.5, (
        f"conjugate variance should be clearly optimistic; sd {z_conj:.2f}"
    )
    assert z_conj > 2.0 * z_fisher


def test_the_two_variances_converge_on_bright_data_and_diverge_on_weak():
    """The gap is the split uncertainty, so it must track `n_eff B / I`."""
    gap = {}
    for j_mean in (3000.0, 100.0):
        d = planted(n_h=300, mult=8, seed=11, j_mean=j_mean)
        out = sweep(d, n_sweeps=120)
        idx = d["index"]
        vf = merged_fisher_variance(
            d["profile"], d["mask"], out["scale"], out["merged"][idx],
            out["background"], idx, d["n_h"],
        )
        gap[j_mean] = float(vf.sqrt().mean() / out["var_conj"].sqrt().mean())
    assert gap[100.0] > 2.0 * gap[3000.0], (
        f"sigma ratio should grow as the background dominates; got {gap}"
    )


def test_fisher_variance_falls_with_multiplicity_as_one_over_n():
    """Information adds over observations, so the variance must go as 1/n."""
    v = {}
    for mult in (4, 16):
        d = planted(n_h=150, mult=mult, seed=6)
        out = sweep(d)
        idx = d["index"]
        v[mult] = float(
            merged_fisher_variance(
                d["profile"], d["mask"], out["scale"], out["merged"][idx],
                out["background"], idx, d["n_h"],
            ).mean()
        )
    ratio = v[4] / v[16]
    assert 3.0 < ratio < 5.0, f"variance ratio {ratio:.2f}, expected about 4"


# --------------------------------------------------------------- priors


def test_centric_wilson_prior_has_the_same_mean_and_twice_the_spread():
    sigma = torch.tensor([3000.0, 3000.0], dtype=DT)
    centric = torch.tensor([False, True])
    a, b = wilson_prior(sigma, centric)
    mean = a / b
    norm_var = (a / b.pow(2)) / mean.pow(2)
    assert torch.allclose(mean, sigma)
    assert float(norm_var[1] / norm_var[0]) == pytest.approx(2.0, rel=1e-12)


def test_wilson_sigma_is_the_shell_mean_and_empty_shells_do_not_collapse():
    merged = torch.tensor([100.0, 300.0, 900.0], dtype=DT)
    bins = torch.tensor([0, 0, 2])
    sigma = update_wilson_sigma(merged, bins, 4)
    assert float(sigma[0]) == pytest.approx(200.0)
    assert float(sigma[2]) == pytest.approx(900.0)
    # shells 1 and 3 hold nothing and must fall back, not go to zero
    assert float(sigma[1]) == pytest.approx(float(merged.mean()))
    assert float(sigma[3]) > 0.0


def test_photons_are_conserved_by_the_responsibility_split():
    d = planted(n_h=20, mult=4, seed=7)
    r = responsibilities(
        d["profile"],
        d["s_true"],
        d["j_true"][d["index"]],
        d["b_true"],
    )
    stats = observation_stats(d["counts"], d["mask"], r)
    total = (d["counts"] * d["mask"]).sum(-1)
    assert torch.allclose(stats.signal + stats.background_counts, total, rtol=1e-12)


def test_responsibility_is_bounded_and_tracks_the_profile():
    d = planted(n_h=10, mult=2, seed=8)
    r = responsibilities(
        d["profile"], d["s_true"], d["j_true"][d["index"]], d["b_true"]
    )
    assert float(r.min()) >= 0.0 and float(r.max()) <= 1.0
    peak = d["profile"].argmax(dim=-1)
    rows = torch.arange(r.shape[0])
    assert float(r[rows, peak].mean()) > float(r.mean()), (
        "the peak pixel must carry more signal responsibility than average"
    )


def test_scale_update_is_the_conjugate_mean():
    hyper = MergeHyperparameters(alpha_scale=2.0, beta_scale=5.0)
    signal = torch.tensor([10.0, 0.0], dtype=DT)
    j = torch.tensor([3.0, 3.0], dtype=DT)
    got = update_scale(signal, j, hyper)
    assert float(got[0]) == pytest.approx((2.0 + 10.0) / (5.0 + 3.0))
    assert float(got[1]) == pytest.approx(2.0 / 8.0)
    assert math.isfinite(float(got.sum()))
