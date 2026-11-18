"""The Poisson split and the detector block, driven through the whole sweep.

The unit tests pin each piece: `anomalous_split` gets the fractional difference
from a Binomial with a known offset, and `detector_design`'s odd harmonics are
the part of a position-dependent scale that is antisymmetric between Friedel
mates. These check that the sweep wires them together -- that `split_anomalous`
is off by default and changes nothing when off, that it recovers what the
Newton block recovers when on, and that the detector columns ride on the
existing linear scale solve and remove a planted position-dependent error.

The detector block needs no new machinery in `run_sweep`: the scale solve is a
weighted least squares linear in its coefficients, so a wider design is the
whole change. That is why these tests pass `cheb_design` rather than a new
argument.
"""

import math

import pytest
import torch

from integrator.fit.merge import (
    combined_scale_design,
    detector_design,
    mate_antisymmetric_columns,
    scale_design,
)
from integrator.fit.sweep import equal_count_bins, run_sweep

DT = torch.float32
BC = (2000.0, 2000.0)
RMAX = 1600.0


def synth(
    n_pair=300, mult=10, n_img=20, hw=11, bg=8.0, j_mean=900.0, frac=0.10,
    seed=0, scale_spread=0.2, position_amp=0.0, harmonic=1,
):
    """Pairs with known intensities, differences, scales and positions.

    `position_amp` plants a scale error in `cos(harmonic * phi)` on top of the
    per-image level. Friedel mates are placed at opposite azimuth, which is the
    geometry that makes an odd harmonic antisymmetric between them.
    """
    g = torch.Generator().manual_seed(seed)
    n = n_pair * mult

    u = torch.rand(n_pair, generator=g, dtype=DT).clamp(1e-6, 1 - 1e-6)
    j_true = (-j_mean * torch.log1p(-u)).clamp_min(0.05 * j_mean)
    shell = torch.arange(n_pair) % 2
    f = torch.tensor([float(frac), 0.0], dtype=DT)[shell]
    delta_true = f * j_true * torch.sign(torch.randn(n_pair, generator=g, dtype=DT))

    pair = torch.arange(n_pair).repeat_interleave(mult)
    mate = torch.where(
        torch.arange(n) % 2 == 0, torch.ones(n, dtype=DT), -torch.ones(n, dtype=DT)
    )
    group = torch.randint(0, n_img, (n,), generator=g)
    # SMOOTH in image index, because that is what `scale_design` models: a
    # low-order Chebyshev. Planting an independent random scale per image
    # instead leaves the image block unable to fit it at any degree, and the
    # residual misfit then dominates everything downstream -- it put the
    # centric null at 3.8 on clean data and hid the detector term entirely.
    t_img = torch.linspace(-1.0, 1.0, n_img, dtype=DT)
    s_img = (
        1.0 + scale_spread * (0.7 * t_img + 0.5 * t_img.pow(2) - 0.3 * t_img.pow(3))
    ).clamp_min(0.3)

    # the mate sits at the opposite azimuth, same radius
    r = RMAX * (0.2 + 0.7 * torch.rand(n, generator=g, dtype=DT))
    phi0 = 2 * math.pi * torch.rand(n, generator=g, dtype=DT)
    phi = torch.where(mate > 0, phi0, phi0 + math.pi)
    x = BC[0] + r * torch.cos(phi)
    y = BC[1] + r * torch.sin(phi)
    s_pos = 1.0 + position_amp * torch.cos(harmonic * phi)

    yy, xx = torch.meshgrid(
        torch.arange(hw, dtype=DT), torch.arange(hw, dtype=DT), indexing="ij"
    )
    c = (hw - 1) / 2.0
    p = torch.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * 1.4**2)).reshape(-1)
    p = (p / p.sum()).unsqueeze(0).expand(n, -1).contiguous()

    j_obs = j_true[pair] + 0.5 * mate * delta_true[pair]
    rate = (s_img[group] * s_pos * j_obs).unsqueeze(-1) * p + bg
    counts = torch.poisson(rate, generator=g)

    d = torch.where(shell == 0, 3.0, 1.8)[pair]
    s2 = (1.0 / d.pow(2))
    image_u = (2.0 * group.to(DT) / max(n_img - 1, 1) - 1.0)
    centric = torch.zeros(n_pair, dtype=torch.bool)
    return dict(
        counts=counts, mask=torch.ones_like(counts), profile=p, pair=pair,
        mate=mate, group=group, n_pair=n_pair, n_img=n_img, n_obs=n,
        j_true=j_true, delta_true=delta_true, s_img=s_img, shell=shell, d=d,
        x=x, y=y, s2=s2, image_u=image_u, centric=centric, frac_true=f,
    )


def make_batches(d, batch=512):
    def factory():
        for k in range(0, d["n_obs"], batch):
            sl = torch.arange(k, min(k + batch, d["n_obs"]))
            yield sl, d["counts"][sl], d["mask"][sl], {"rows": sl}

    return factory


def run(d, n_sweeps=20, split=False, design=None, centric=None):
    bins = equal_count_bins(d["d"][::1][: d["n_pair"] * 0 + d["n_pair"]], 2) \
        if False else equal_count_bins(
            torch.where(d["shell"] == 0, torch.tensor(3.0), torch.tensor(1.8)), 2
        )
    return run_sweep(
        make_batches(d),
        profile_fn=lambda md, _m: d["profile"][md["rows"]],
        pair=d["pair"],
        mate=d["mate"],
        group=d["group"],
        bin_index=bins,
        centric_pair=d["centric"] if centric is None else centric,
        n_pair=d["n_pair"],
        n_groups=d["n_img"],
        n_bins=2,
        n_obs=d["n_obs"],
        n_sweeps=n_sweeps,
        cheb_design=design,
        split_anomalous=split,
        log_every=0,
    )


def degauge(x, truth):
    return x / (x / truth).median()


# ------------------------------------------------------------- off by default


def test_split_is_off_by_default_and_changes_nothing_when_off():
    d = synth(seed=1)
    a = run(d, split=False)
    b = run(d, split=False)
    assert torch.equal(a.state.merged, b.state.merged)
    assert a.frac is None and a.null_z is None


def test_the_split_recovers_what_the_newton_block_recovers():
    d = synth(seed=2, frac=0.10)
    old = run(d, split=False)
    new = run(d, split=True)

    for out, name in ((old, "newton"), (new, "split")):
        j = degauge(out.state.merged, d["j_true"])
        r = float(torch.corrcoef(torch.stack([j, d["j_true"]]))[0, 1])
        assert r > 0.99, f"{name}: corr(J, truth) {r:.4f}"

    assert new.frac is not None and new.var_frac is not None
    has = d["shell"] == 0
    # the SIGNED fractional truth; `frac_true` carries only the magnitude
    truth = d["delta_true"] / (2.0 * d["j_true"])
    r_new = float(torch.corrcoef(torch.stack([new.frac[has], truth[has]]))[0, 1])
    assert r_new > 0.3, f"split corr(frac, planted shell) {r_new:.3f}"


def test_the_split_finds_the_difference_only_where_it_was_planted():
    d = synth(seed=3, frac=0.12)
    out = run(d, split=True)
    assert out.tau is not None
    assert float(out.tau[0]) > 3.0 * float(out.tau[1]), (
        f"per-shell tau on the fractional scale: {out.tau.tolist()}"
    )


def test_no_planted_difference_gives_no_tau():
    d = synth(seed=4, frac=0.0)
    out = run(d, split=True)
    assert float(out.tau.max()) < 0.03, f"spurious tau {out.tau.tolist()}"


# ---------------------------------------------------------- the detector block


def test_the_combined_design_is_the_base_plus_the_detector_columns():
    d = synth(n_pair=40, mult=4, seed=5)
    base, tb = scale_design(d["image_u"], d["s2"], 4, 2)
    comb, tc = combined_scale_design(
        d["image_u"], d["s2"], d["x"], d["y"], BC, RMAX,
        degree_image=4, degree_s2=2, degree_r=2, n_harmonic=2,
    )
    det, td = detector_design(d["x"], d["y"], BC, RMAX, 2, 2)
    # the default is the tensor product, one detector block per image order
    assert comb.shape[1] == base.shape[1] + 5 * det.shape[1]
    assert len(tc) == len(tb) + 5 * len(td)
    # and the appended form is still available
    add, ta = combined_scale_design(
        d["image_u"], d["s2"], d["x"], d["y"], BC, RMAX,
        degree_image=4, degree_s2=2, degree_r=2, n_harmonic=2,
        multiplicative=False,
    )
    assert add.shape[1] == base.shape[1] + det.shape[1]
    assert len(ta) == len(tb) + len(td)
    assert torch.allclose(comb[:, : base.shape[1]], base)
    # the mask must not fire on the image block, which carries no azimuth
    for terms, n_base in ((tc, base.shape[1]), (ta, base.shape[1])):
        mask = mate_antisymmetric_columns(terms)
        assert not mask[:n_base].any()
        assert mask[n_base:].any()


def test_without_the_detector_block_a_position_error_is_left_in():
    """The baseline the next test improves on."""
    d = synth(seed=6, frac=0.0, position_amp=0.15, harmonic=1)
    design, _ = scale_design(d["image_u"], d["s2"], 4, 2)
    out = run(d, split=True, design=design)
    assert float(out.frac.abs().median()) > 0.01, (
        "a 15% odd-harmonic scale error should leak into the difference when "
        "the scale cannot see detector position"
    )


def test_the_detector_block_removes_a_planted_position_error():
    d = synth(seed=6, frac=0.0, position_amp=0.15, harmonic=1)
    plain, _ = scale_design(d["image_u"], d["s2"], 4, 2)
    added, _ = combined_scale_design(
        d["image_u"], d["s2"], d["x"], d["y"], BC, RMAX,
        degree_image=4, degree_s2=2, degree_r=2, n_harmonic=3,
        multiplicative=False,
    )
    crossed, _ = combined_scale_design(
        d["image_u"], d["s2"], d["x"], d["y"], BC, RMAX,
        degree_image=4, degree_s2=2, degree_r=2, n_harmonic=3,
        multiplicative=True,
    )
    leaks = {
        name: float(run(d, split=True, design=des).frac.abs().median())
        for name, des in (
            ("none", plain), ("added", added), ("crossed", crossed)
        )
    }
    # the detector response multiplies the image scale, so the appended form
    # can only fit the product on average and leaves `(g - mean g) h` behind
    assert leaks["crossed"] < 0.6 * leaks["none"], f"leaks {leaks}"
    assert leaks["crossed"] < leaks["added"], f"leaks {leaks}"


def test_the_detector_block_does_not_invent_signal_when_there_is_none():
    d = synth(seed=7, frac=0.0, position_amp=0.0)
    plain, _ = scale_design(d["image_u"], d["s2"], 4, 2)
    with_det, _ = combined_scale_design(
        d["image_u"], d["s2"], d["x"], d["y"], BC, RMAX,
        degree_image=4, degree_s2=2, degree_r=2, n_harmonic=3,
    )
    a = run(d, split=True, design=plain)
    b = run(d, split=True, design=with_det)
    assert float(b.frac.abs().median()) < 2.0 * float(a.frac.abs().median()) + 1e-4


def test_the_detector_block_keeps_the_intensities():
    """Extra scale columns must not eat the signal."""
    d = synth(seed=8, frac=0.0, position_amp=0.10)
    with_det, _ = combined_scale_design(
        d["image_u"], d["s2"], d["x"], d["y"], BC, RMAX,
        degree_image=4, degree_s2=2, degree_r=2, n_harmonic=3,
    )
    out = run(d, split=True, design=with_det)
    j = degauge(out.state.merged, d["j_true"])
    r = float(torch.corrcoef(torch.stack([j, d["j_true"]]))[0, 1])
    assert r > 0.98, f"corr(J, truth) with the detector block {r:.4f}"


# ------------------------------------------------------------- the exact null


def test_the_binomial_null_is_reported_for_centrics():
    d = synth(seed=9, frac=0.0, position_amp=0.0)
    centric = torch.zeros(d["n_pair"], dtype=torch.bool)
    centric[::3] = True
    out = run(d, split=True, centric=centric)
    assert out.null_z is not None and out.theta_null is not None
    sd = float(out.null_z[centric].std())
    assert 0.5 < sd < 2.0, f"standardized null spread {sd:.3f}"


def test_the_null_widens_when_the_scale_discriminates_the_mates():
    centric = torch.zeros(300, dtype=torch.bool)
    centric[::3] = True
    clean = synth(seed=10, frac=0.0, position_amp=0.0)
    dirty = synth(seed=10, frac=0.0, position_amp=0.40, harmonic=1)
    design, _ = scale_design(clean["image_u"], clean["s2"], 4, 2)
    a = run(clean, split=True, design=design, centric=centric)
    b = run(dirty, split=True, design=design, centric=centric)
    sd_a = float(a.null_z[centric].std())
    sd_b = float(b.null_z[centric].std())
    # NOT calibrated at 1 here, and it should not be expected to be: the
    # Binomial null is exact for integer Poisson counts, and the sweep feeds it
    # the responsibility-weighted soft count `sum_j c_ij r_ij`, whose variance
    # is not `T theta (1 - theta)`. On this synthetic data the clean spread is
    # about 3.8. So the null is read as a RATIO against a run with the scale
    # change removed, exactly as the existing anchor is, and making it exact
    # needs the split taken on integer foreground counts instead.
    assert sd_b > 1.3 * sd_a, f"clean {sd_a:.2f}, position error {sd_b:.2f}"
