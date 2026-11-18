"""The whole sweep, driven over batches, from pixels to structure factors.

The unit tests pin each block; this one checks that the driver wires them
together: that it converges from a cold start with no truth supplied anywhere,
that the state it carries between passes is indexed correctly when the loader
hands batches back in a different order, and that what comes out is usable --
merged intensities correlated with the planted ones, a Bijvoet difference on
the shell that has one, and variances that match the scatter.
"""

import math

import pytest
import torch

from integrator.fit.sweep import SweepState, equal_count_bins, run_sweep

DT = torch.float32


def synth(
    n_pair=250, mult=8, n_img=24, hw=13, bg=10.0, j_mean=1500.0,
    frac=(0.10, 0.0), seed=0, scale_spread=0.25,
):
    """A small dataset with scales, backgrounds, intensities and differences known.

    Two resolution shells: the first carries a fractional anomalous difference
    and the second carries none, so the per-shell scale has something to find
    and something to shrink.
    """
    g = torch.Generator().manual_seed(seed)
    n = n_pair * mult

    u = torch.rand(n_pair, generator=g, dtype=DT).clamp(1e-6, 1 - 1e-6)
    j_true = (-j_mean * torch.log1p(-u)).clamp_min(0.05 * j_mean)

    shell = torch.arange(n_pair) % 2
    f = torch.tensor([float(frac[0]), float(frac[1])], dtype=DT)[shell]
    delta_true = (
        f * j_true
        * torch.sign(torch.randn(n_pair, generator=g, dtype=DT))
        * torch.rand(n_pair, generator=g, dtype=DT).add(0.5)
    )

    pair = torch.arange(n_pair).repeat_interleave(mult)
    mate = torch.where(
        torch.arange(n) % 2 == 0, torch.ones(n, dtype=DT), -torch.ones(n, dtype=DT)
    )
    group = torch.randint(0, n_img, (n,), generator=g)
    s_true = (
        1.0 + scale_spread * torch.randn(n_img, generator=g, dtype=DT)
    ).clamp_min(0.25)

    yy, xx = torch.meshgrid(
        torch.arange(hw, dtype=DT), torch.arange(hw, dtype=DT), indexing="ij"
    )
    c = (hw - 1) / 2.0
    p = torch.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * 1.5**2)).reshape(-1)
    p = (p / p.sum()).unsqueeze(0).expand(n, -1).contiguous()

    j_obs = j_true[pair] + 0.5 * mate * delta_true[pair]
    rate = (s_true[group] * j_obs).unsqueeze(-1) * p + bg
    counts = torch.poisson(rate, generator=g)

    # shell 0 is "low resolution" so equal_count_bins puts it in bin 0
    d = torch.where(shell == 0, 3.0, 1.8)[pair]
    return dict(
        counts=counts, mask=torch.ones_like(counts), profile=p, pair=pair,
        mate=mate, group=group, n_pair=n_pair, n_img=n_img, n_obs=n,
        j_true=j_true, delta_true=delta_true, s_true=s_true, bg=bg,
        shell=shell, d=d,
    )


def make_batches(d, batch=256, shuffle_seed=None):
    """A callable returning a fresh iterator, as `run_sweep` expects.

    With `shuffle_seed` the rows come back in a different order every pass,
    which is what the `obs_index` argument exists to survive.
    """

    def factory():
        n = d["n_obs"]
        order = (
            torch.randperm(n, generator=torch.Generator().manual_seed(shuffle_seed))
            if shuffle_seed is not None
            else torch.arange(n)
        )
        for k in range(0, n, batch):
            sl = order[k : k + batch]
            yield sl, d["counts"][sl], d["mask"][sl], {"rows": sl}

    return factory


def profile_fn_for(d):
    return lambda md, _mask: d["profile"][md["rows"]]


def run(d, n_sweeps=25, shuffle_seed=None, tau_scale=None):
    bins = equal_count_bins(
        torch.where(d["shell"] == 0, torch.tensor(3.0), torch.tensor(1.8)), 2
    )
    return run_sweep(
        make_batches(d, shuffle_seed=shuffle_seed),
        profile_fn=profile_fn_for(d),
        pair=d["pair"],
        mate=d["mate"],
        group=d["group"],
        bin_index=bins,
        centric_pair=torch.zeros(d["n_pair"], dtype=torch.bool),
        n_pair=d["n_pair"],
        n_groups=d["n_img"],
        n_bins=2,
        n_obs=d["n_obs"],
        n_sweeps=n_sweeps,
        tau_scale=tau_scale,
        log_every=0,
    )


def degauge(x, truth):
    return x / (x / truth).median()


# --------------------------------------------------------------- convergence


def test_sweep_converges_from_a_cold_start():
    """No truth is supplied anywhere: scales, backgrounds and J are all seeded."""
    d = synth(seed=1)
    out = run(d)
    j = degauge(out.state.merged, d["j_true"])

    r = float(torch.corrcoef(torch.stack([j, d["j_true"]]))[0, 1])
    rel = float(((j - d["j_true"]).abs() / d["j_true"]).median())
    assert r > 0.995, f"corr(J, truth) {r:.4f}"
    assert rel < 0.06, f"median relative error {rel:.4f}"


def test_scales_and_backgrounds_come_back():
    d = synth(seed=2)
    out = run(d)
    rs_ = float(torch.corrcoef(torch.stack([out.state.scale, d["s_true"]]))[0, 1])
    assert rs_ > 0.97, f"corr(S, truth) {rs_:.4f}"
    assert float(out.state.background.median()) == pytest.approx(d["bg"], rel=0.1)
    assert float(out.state.scale.mean()) == pytest.approx(1.0, abs=1e-5)


def test_batch_order_does_not_change_the_answer():
    """`obs_index` carries the per-observation state, so a shuffled loader is fine.

    Without it the background vector would be written at the wrong rows as soon
    as the loader stopped returning a stable order, and the failure would look
    like slow convergence rather than a wiring bug.
    """
    d = synth(seed=3)
    a = run(d, shuffle_seed=None)
    b = run(d, shuffle_seed=7)
    rel = float(
        ((a.state.merged - b.state.merged).abs() / a.state.merged).median()
    )
    # not bit-identical: float32 `scatter_add` accumulates in whatever order the
    # batches arrive, and the cold-start seed is taken per batch. The tolerance
    # is for that, not for a wiring error, which shows up as NaN or as a
    # percent-level disagreement.
    assert rel < 1e-3, f"ordered and shuffled disagree by {rel:.2e}"


def test_more_sweeps_do_not_move_a_converged_answer():
    d = synth(seed=4)
    a, b = run(d, n_sweeps=20), run(d, n_sweeps=40)
    rel = float(
        ((a.state.merged - b.state.merged).abs() / a.state.merged).median()
    )
    assert rel < 5e-3, f"not converged at 20 sweeps: {rel:.2e}"


# --------------------------------------------------------------- anomalous


def test_the_anomalous_difference_is_found_only_where_it_was_planted():
    d = synth(seed=5, frac=(0.12, 0.0))
    out = run(d)
    assert float(out.tau[0]) > 5.0 * float(out.tau[1]), (
        f"per-shell tau {out.tau.tolist()}; shell 0 carries the signal"
    )

    has = d["shell"] == 0
    j = degauge(out.state.merged, d["j_true"])
    a = (out.delta_shrunk / out.state.merged * j)[has]
    r = float(torch.corrcoef(torch.stack([a, d["delta_true"][has]]))[0, 1])
    assert r > 0.7, f"corr(Delta, truth) on the signal shell {r:.3f}"


def test_no_anomalous_signal_gives_no_difference():
    d = synth(seed=6, frac=(0.0, 0.0))
    out = run(d)
    assert float(out.tau.max()) < 0.01, f"spurious tau {out.tau.tolist()}"
    ratio = float((out.delta_shrunk.abs() / out.state.merged).median())
    assert ratio < 0.01, f"spurious |Delta|/J {ratio:.4f}"


# --------------------------------------------------------------- variances


def test_reported_variance_matches_the_scatter():
    d = synth(n_pair=400, seed=7)
    out = run(d)
    j = degauge(out.state.merged, d["j_true"])
    z = (j - d["j_true"]) / out.var_merged.clamp_min(1e-12).sqrt()
    finite = torch.isfinite(z)
    assert finite.float().mean() > 0.95
    assert 0.7 < float(z[finite].std()) < 1.8, (
        f"standardized residual sd {float(z[finite].std()):.2f}"
    )


def test_variance_falls_with_multiplicity():
    v = {}
    for mult in (4, 16):
        d = synth(n_pair=200, mult=mult, seed=8)
        v[mult] = float(run(d).var_merged.median())
    assert 3.0 < v[4] / v[16] < 5.0, f"ratio {v[4] / v[16]:.2f}"


# --------------------------------------------------------------- plumbing


def test_equal_count_bins_are_equal_and_ordered_low_resolution_first():
    d = torch.tensor([9.0, 1.0, 5.0, 3.0, 7.0, 2.0], dtype=DT)
    b = equal_count_bins(d, 3)
    assert b[0] == 0 and b[1] == 2
    counts = torch.bincount(b, minlength=3)
    assert int(counts.max() - counts.min()) <= 1


def test_state_is_the_documented_size():
    """Two numbers per pair, one per image, one per observation."""
    d = synth(n_pair=50, mult=4, n_img=5, seed=9)
    out = run(d, n_sweeps=3)
    s: SweepState = out.state
    assert s.merged.shape == (50,) and s.delta.shape == (50,)
    assert s.scale.shape == (5,)
    assert s.background.shape == (d["n_obs"],)
    assert all(math.isfinite(float(x)) for x in (s.merged.sum(), s.scale.sum()))


# --------------------------------------------------------------- centric anchor


def synth_with_centrics(
    n_pair=300, mult=8, n_img=20, hw=13, bg=10.0, j_mean=1500.0,
    centric_frac=0.3, mate_bias=0.0, seed=0,
):
    """As `synth`, with a block of centric pairs and an optional scale artifact.

    `mate_bias` multiplies the intensity of every `pseudo +` observation and
    divides every `pseudo -` one, for centric and acentric pairs alike. It is a
    stand-in for a scale that has learned to discriminate the two groups: the
    crystal cannot produce a centric Bijvoet difference, so whatever the anchor
    recovers is this artifact and nothing else.
    """
    g = torch.Generator().manual_seed(seed)
    n = n_pair * mult
    u = torch.rand(n_pair, generator=g, dtype=DT).clamp(1e-6, 1 - 1e-6)
    j_true = (-j_mean * torch.log1p(-u)).clamp_min(0.05 * j_mean)
    centric = torch.zeros(n_pair, dtype=torch.bool)
    centric[: int(centric_frac * n_pair)] = True

    pair = torch.arange(n_pair).repeat_interleave(mult)
    pseudo = torch.where(
        torch.arange(n) % 2 == 0, torch.ones(n, dtype=DT), -torch.ones(n, dtype=DT)
    )
    # the REAL mate is +1 everywhere on a centric, which is what makes its
    # difference zero by construction
    mate = torch.where(centric[pair], torch.ones(n, dtype=DT), pseudo)
    group = torch.randint(0, n_img, (n,), generator=g)
    s_true = (1.0 + 0.2 * torch.randn(n_img, generator=g, dtype=DT)).clamp_min(0.3)

    yy, xx = torch.meshgrid(
        torch.arange(hw, dtype=DT), torch.arange(hw, dtype=DT), indexing="ij"
    )
    c = (hw - 1) / 2.0
    p = torch.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * 1.5**2)).reshape(-1)
    p = (p / p.sum()).unsqueeze(0).expand(n, -1).contiguous()

    bias = (1.0 + mate_bias) ** pseudo
    rate = (s_true[group] * j_true[pair] * bias).unsqueeze(-1) * p + bg
    counts = torch.poisson(rate, generator=g)
    return dict(
        counts=counts, mask=torch.ones_like(counts), profile=p, pair=pair,
        mate=mate, pseudo=pseudo, group=group, n_pair=n_pair, n_img=n_img,
        n_obs=n, j_true=j_true, centric=centric, bg=bg, s_true=s_true,
    )


def run_anchor(d, n_sweeps=20):
    from integrator.fit.sweep import run_sweep as _rs

    return _rs(
        make_batches(d),
        profile_fn=profile_fn_for(d),
        pair=d["pair"],
        mate=d["mate"],
        group=d["group"],
        bin_index=torch.zeros(d["n_pair"], dtype=torch.long),
        centric_pair=d["centric"],
        n_pair=d["n_pair"],
        n_groups=d["n_img"],
        n_bins=1,
        n_obs=d["n_obs"],
        n_sweeps=n_sweeps,
        pseudo_mate=d["pseudo"],
        log_every=0,
    )


def test_anchor_is_quiet_when_the_scale_invents_nothing():
    """With no mate-discriminating artifact the centric spread is noise alone."""
    out = run_anchor(synth_with_centrics(seed=21, mate_bias=0.0))
    a = out.anchor
    assert a is not None and a["n"] > 50
    assert 0.6 < a["z_sd"] < 1.6, f"standardized sd {a['z_sd']:.3f}, expected ~1"
    assert float(a["tau"][0]) < 0.02, f"fake fractional signal {float(a['tau'][0]):.4f}"


def test_anchor_detects_a_scale_that_discriminates_the_mates():
    """A 5% mate-dependent scale error must show up where truth is zero."""
    clean = run_anchor(synth_with_centrics(seed=22, mate_bias=0.0)).anchor
    dirty = run_anchor(synth_with_centrics(seed=22, mate_bias=0.05)).anchor
    assert dirty["z_sd"] > 3.0 * clean["z_sd"], (
        f"clean sd {clean['z_sd']:.2f}, biased sd {dirty['z_sd']:.2f}"
    )
    assert float(dirty["tau"][0]) > 0.02, (
        f"fake fractional signal {float(dirty['tau'][0]):.4f} should be visible"
    )


def test_anchor_magnitude_tracks_the_artifact():
    sds = [
        run_anchor(synth_with_centrics(seed=23, mate_bias=b)).anchor["z_sd"]
        for b in (0.0, 0.02, 0.06)
    ]
    assert sds[0] < sds[1] < sds[2], f"standardized sd by artifact size: {sds}"


def test_anchor_is_absent_unless_asked_for():
    """It costs a responsibility pass, so it is opt-in."""
    d = synth(seed=24)
    assert run(d, n_sweeps=3).anchor is None


# --------------------------------------------------------------- the spectrum


def synth_with_spectrum(
    n_pair=300, mult=10, n_img=20, n_lam=12, hw=13, bg=10.0, j_mean=1500.0,
    spectrum_depth=0.6, seed=0,
):
    """As `synth`, with a wavelength-dependent incident intensity planted.

    On Laue data one Miller index is measured at many wavelengths, so `G(lambda)`
    is a per-observation factor that the per-image scale cannot represent: two
    observations of the same pair on the same image at different wavelengths
    carry different scales. Leaving it out puts that variation into the
    residual.
    """
    g = torch.Generator().manual_seed(seed)
    n = n_pair * mult
    u = torch.rand(n_pair, generator=g, dtype=DT).clamp(1e-6, 1 - 1e-6)
    j_true = (-j_mean * torch.log1p(-u)).clamp_min(0.05 * j_mean)

    pair = torch.arange(n_pair).repeat_interleave(mult)
    mate = torch.where(
        torch.arange(n) % 2 == 0, torch.ones(n, dtype=DT), -torch.ones(n, dtype=DT)
    )
    group = torch.randint(0, n_img, (n,), generator=g)
    lam = torch.randint(0, n_lam, (n,), generator=g)
    s_true = (1.0 + 0.2 * torch.randn(n_img, generator=g, dtype=DT)).clamp_min(0.3)
    # a smooth band, brightest in the middle, as an undulator spectrum is
    x = torch.linspace(-1.0, 1.0, n_lam, dtype=DT)
    g_true = 1.0 + spectrum_depth * (1.0 - x.pow(2)) - spectrum_depth / 2.0
    g_true = g_true / g_true.mean()

    yy, xx = torch.meshgrid(
        torch.arange(hw, dtype=DT), torch.arange(hw, dtype=DT), indexing="ij"
    )
    c = (hw - 1) / 2.0
    p = torch.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * 1.5**2)).reshape(-1)
    p = (p / p.sum()).unsqueeze(0).expand(n, -1).contiguous()

    rate = (s_true[group] * g_true[lam] * j_true[pair]).unsqueeze(-1) * p + bg
    counts = torch.poisson(rate, generator=g)
    return dict(
        counts=counts, mask=torch.ones_like(counts), profile=p, pair=pair,
        mate=mate, group=group, lam=lam, n_pair=n_pair, n_img=n_img,
        n_lam=n_lam, n_obs=n, j_true=j_true, s_true=s_true, g_true=g_true, bg=bg,
    )


def run_spectrum(d, n_sweeps=25, with_spectrum=True):
    from integrator.fit.sweep import run_sweep as _rs

    return _rs(
        make_batches(d),
        profile_fn=profile_fn_for(d),
        pair=d["pair"],
        mate=d["mate"],
        group=d["group"],
        bin_index=torch.zeros(d["n_pair"], dtype=torch.long),
        centric_pair=torch.zeros(d["n_pair"], dtype=torch.bool),
        n_pair=d["n_pair"],
        n_groups=d["n_img"],
        n_bins=1,
        n_obs=d["n_obs"],
        n_sweeps=n_sweeps,
        lambda_bin=d["lam"] if with_spectrum else None,
        n_lambda=d["n_lam"] if with_spectrum else 0,
        log_every=0,
    )


def test_planted_spectrum_is_recovered():
    d = synth_with_spectrum(seed=31)
    out = run_spectrum(d)
    assert out.state.spectrum is not None
    got = out.state.spectrum / out.state.spectrum.mean()
    want = d["g_true"] / d["g_true"].mean()
    rel = float((got - want).abs().max() / want.mean())
    assert rel < 0.05, f"spectrum off by {rel:.4f}: {got.tolist()}"


def test_the_spectrum_improves_the_merged_intensities():
    """The per-image scale cannot represent a per-observation wavelength term."""
    d = synth_with_spectrum(seed=32, spectrum_depth=0.6)
    err = {}
    for flag in (False, True):
        out = run_spectrum(d, with_spectrum=flag)
        j = degauge(out.state.merged, d["j_true"])
        err[flag] = float(((j - d["j_true"]).abs() / d["j_true"]).median())
    assert err[True] < 0.6 * err[False], f"median relative error: {err}"


def test_a_flat_spectrum_is_found_when_there_is_none():
    d = synth_with_spectrum(seed=33, spectrum_depth=0.0)
    out = run_spectrum(d)
    g = out.state.spectrum / out.state.spectrum.mean()
    assert float((g - 1.0).abs().max()) < 0.03, f"spurious spectrum {g.tolist()}"


def test_per_image_scale_survives_the_split():
    """The image factor must still come back once the spectrum takes its share."""
    d = synth_with_spectrum(seed=34)
    out = run_spectrum(d)
    r = float(torch.corrcoef(torch.stack([out.state.scale, d["s_true"]]))[0, 1])
    assert r > 0.95, f"corr(S_image, truth) {r:.4f}"
    assert float(out.state.scale.mean()) == pytest.approx(1.0, abs=1e-4)


def test_spectrum_is_off_by_default():
    d = synth(seed=35)
    assert run(d, n_sweeps=3).state.spectrum is None


def test_a_grouped_spectrum_cannot_absorb_the_anomalous_difference():
    """Why a grouped factor is safer than a learned scale over the same inputs.

    Absorption is the recorded failure of a learned scale: merged-mean
    correlation 0.98 with anomalous-difference correlation 0.14. The capacity
    for it comes from per-reflection resolution. A binned `G(lambda)` is ONE
    number shared by every Miller pair in the bin, while `Delta_h` carries a
    different sign for each pair, so the factor cannot track them -- measured
    here with the wavelength tied perfectly to the mate, which is the worst
    case: tau 0.0634 either way, unchanged to three figures.

    An MLP over `(lambda, x, y, d)` is a different matter, because `x` and `y`
    locate individual reflections. This test is the reason to prefer the
    grouped factorization, and the reason to re-run it and the centric anchor
    before adding any scale term with per-reflection resolution.
    """
    n_lam = 12
    g = torch.Generator().manual_seed(42)

    def build(tied):
        d = synth(n_pair=400, mult=10, seed=42, frac=(0.12, 0.12))
        if tied:
            half = n_lam // 2
            plus = d["mate"] > 0
            lam = torch.where(
                plus,
                torch.randint(half, n_lam, (d["n_obs"],), generator=g),
                torch.randint(0, half, (d["n_obs"],), generator=g),
            )
        else:
            lam = torch.randint(0, n_lam, (d["n_obs"],), generator=g)
        d["lam"], d["n_lam"] = lam, n_lam
        return d

    from integrator.fit.sweep import run_sweep as _rs

    def go(d):
        return _rs(
            make_batches(d),
            profile_fn=profile_fn_for(d),
            pair=d["pair"],
            mate=d["mate"],
            group=d["group"],
            bin_index=torch.zeros(d["n_pair"], dtype=torch.long),
            centric_pair=torch.zeros(d["n_pair"], dtype=torch.bool),
            n_pair=d["n_pair"],
            n_groups=d["n_img"],
            n_bins=1,
            n_obs=d["n_obs"],
            n_sweeps=20,
            lambda_bin=d["lam"],
            n_lambda=d["n_lam"],
            log_every=0,
        )

    free, tied = go(build(False)), go(build(True))
    assert float(free.tau[0]) > 0.03, (
        f"the planted difference should be found when lambda is independent; "
        f"tau {float(free.tau[0]):.4f}"
    )
    assert float(tied.tau[0]) == pytest.approx(float(free.tau[0]), rel=0.05), (
        f"a grouped spectrum has no per-reflection resolution and must not "
        f"change the difference: tau {float(free.tau[0]):.4f} -> "
        f"{float(tied.tau[0]):.4f}"
    )


def test_a_one_sided_pair_keeps_its_intensity_and_loses_only_the_difference():
    """Dropping those pairs is what cut hewl_1118 from 1.5 A to 1.81 A.

    A pair measured under one mate only determines `J + Delta/2`. The
    difference is unmeasured and must come back infinite, so the shrink sends
    it to the base model. But `Delta` is then pinned at zero, and the single
    side does determine `J`, so its variance must be finite -- otherwise the
    merge discards every such reflection, and at high resolution that is most
    of them.
    """
    d = synth(n_pair=200, mult=8, seed=51)
    mate = d["mate"].clone()
    one_sided = d["pair"] < 40
    mate[one_sided] = 1.0  # pairs 0-39 are seen on the + side only

    from integrator.fit.sweep import run_sweep as _rs

    out = _rs(
        make_batches(d),
        profile_fn=profile_fn_for(d),
        pair=d["pair"],
        mate=mate,
        group=d["group"],
        bin_index=torch.zeros(d["n_pair"], dtype=torch.long),
        centric_pair=torch.zeros(d["n_pair"], dtype=torch.bool),
        n_pair=d["n_pair"],
        n_groups=d["n_img"],
        n_bins=1,
        n_obs=d["n_obs"],
        n_sweeps=20,
        log_every=0,
    )

    sel = torch.zeros(d["n_pair"], dtype=torch.bool)
    sel[:40] = True
    assert torch.isfinite(out.var_merged[sel]).all(), (
        "a one-sided pair must keep a finite intensity variance"
    )
    assert float(out.delta_shrunk[sel].abs().max()) == 0.0, (
        "and no difference, since the data cannot measure one"
    )
    # the rest are two-sided and must be unaffected
    assert torch.isfinite(out.var_merged[~sel]).all()
    j = degauge(out.state.merged, d["j_true"])
    rel = float(((j[sel] - d["j_true"][sel]).abs() / d["j_true"][sel]).median())
    assert rel < 0.08, f"one-sided intensities off by {rel:.4f}"


def test_chebyshev_scale_recovers_an_image_dependent_resolution_ramp():
    """The IDENTIFIABLE part of a decay term, end to end.

    A ramp in `s^2` that is the same on every image is degenerate with `J` --
    `s^2` is a property of the Miller index -- and refinement absorbs it with
    its own overall B, so `scale_design` drops those columns. What is left is a
    ramp whose slope varies image to image, which is what radiation damage
    looks like and what refinement cannot absorb. Measured on hewl_1118 that
    variation is sd 7.1 against a mean of 37.1, so 19% of the ramp is real.
    """
    from integrator.fit.merge import scale_design
    from integrator.fit.sweep import run_sweep as _rs

    d = synth(n_pair=300, mult=12, seed=61)
    n = d["n_obs"]
    g = torch.Generator().manual_seed(61)
    s2 = torch.where(
        d["shell"] == 0, 1.0 / (4 * 3.0**2), 1.0 / (4 * 1.8**2)
    )[d["pair"]]
    img_u = d["group"].to(DT) / max(int(d["group"].max()), 1)
    # B varies linearly across the run: the part a single overall B cannot fit
    b_img = 6.0 + 24.0 * img_u
    falloff = torch.exp(-2.0 * b_img * s2)
    counts = torch.poisson(
        (falloff * d["j_true"][d["pair"]]).unsqueeze(-1) * d["profile"] + 10.0,
        generator=g,
    )
    dd = dict(d, counts=counts, mask=torch.ones_like(counts))
    design, terms = scale_design(img_u, s2, 3, 2)
    assert (0, 1) not in terms and (0, 2) not in terms

    def go(use_cheb):
        return _rs(
            make_batches(dd),
            profile_fn=profile_fn_for(dd),
            pair=dd["pair"], mate=dd["mate"], group=dd["group"],
            bin_index=torch.zeros(dd["n_pair"], dtype=torch.long),
            centric_pair=torch.zeros(dd["n_pair"], dtype=torch.bool),
            n_pair=dd["n_pair"], n_groups=dd["n_img"], n_bins=1,
            n_obs=n, n_sweeps=25,
            cheb_design=design if use_cheb else None,
            log_every=0,
        )

    off, on = go(False), go(True)
    assert on.state.cheb_coeff is not None

    def err(res):
        j = degauge(res.state.merged, dd["j_true"])
        return float(((j - dd["j_true"]).abs() / dd["j_true"]).median())

    e_off, e_on = err(off), err(on)
    assert e_on < 0.8 * e_off, (
        f"median relative error, per-image scalar {e_off:.4f} -> "
        f"Chebyshev {e_on:.4f}"
    )
