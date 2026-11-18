"""A scale that depends on where the observation landed on the detector.

The scale model carries a per-image level and a resolution ramp. It has no
term in detector position, which is the one thing careless's scale has that
ours does not, and the geometry says that omission lands specifically on the
anomalous signal: a Friedel mate scatters with `q -> -q`, so to first order it
sits at the same radius and the opposite azimuth, and the odd azimuthal
harmonics of any position-dependent scale error are therefore antisymmetric
between the mates -- indistinguishable from a Bijvoet difference.

These tests pin that chain end to end:

  - the odd harmonics are antisymmetric under `phi -> phi + pi` and the even
    ones are not, which is the claim the whole argument rests on;
  - a planted odd-harmonic scale error manufactures a Bijvoet difference where
    the truth is zero, and the exact Binomial null detects it;
  - a planted EVEN-harmonic error of the same size does not, which is what
    separates this mechanism from "any scale error hurts";
  - the design is identifiable against `J`, unlike an `s^2`-only column.
"""

import math

import pytest
import torch

from integrator.fit.merge import (
    anomalous_split,
    detector_design,
    mate_antisymmetric_columns,
)

DT = torch.float64
BC = (2000.0, 2000.0)
RMAX = 1600.0


def positions(n, seed=0, r_lo=0.15, r_hi=0.95):
    """Detector positions, and the positions of their Friedel mates.

    The mate is placed at the same radius and `phi + pi`, which is the
    first-order geometry the argument uses.
    """
    g = torch.Generator().manual_seed(seed)
    r = RMAX * (r_lo + (r_hi - r_lo) * torch.rand(n, generator=g, dtype=DT))
    phi = 2 * math.pi * torch.rand(n, generator=g, dtype=DT)
    xp = BC[0] + r * torch.cos(phi)
    yp = BC[1] + r * torch.sin(phi)
    xm = BC[0] + r * torch.cos(phi + math.pi)
    ym = BC[1] + r * torch.sin(phi + math.pi)
    return (xp, yp), (xm, ym), r, phi


# ------------------------------------------------------- the geometric claim


def test_odd_harmonics_flip_sign_between_mates_and_even_ones_do_not():
    (xp, yp), (xm, ym), _r, _phi = positions(400, seed=1)
    dp, terms = detector_design(xp, yp, BC, RMAX, degree_r=2, n_harmonic=4)
    dm, _ = detector_design(xm, ym, BC, RMAX, degree_r=2, n_harmonic=4)

    odd = mate_antisymmetric_columns(terms)
    assert odd.any() and (~odd).any()
    for k, (j, m, kind) in enumerate(terms):
        a, b = dp[:, k], dm[:, k]
        if m % 2 == 1:
            assert torch.allclose(a, -b, atol=1e-9), f"odd {(j, m, kind)} not odd"
        else:
            assert torch.allclose(a, b, atol=1e-9), f"even {(j, m, kind)} not even"


def test_the_radial_part_is_common_to_the_pair():
    """So it cannot create a difference; it goes into the total."""
    (xp, yp), (xm, ym), _r, _phi = positions(300, seed=2)
    dp, terms = detector_design(xp, yp, BC, RMAX, degree_r=3, n_harmonic=0)
    dm, _ = detector_design(xm, ym, BC, RMAX, degree_r=3, n_harmonic=0)
    assert dp.shape[1] == 3, f"radial-only design has {dp.shape[1]} columns"
    assert torch.allclose(dp, dm, atol=1e-9)
    assert not mate_antisymmetric_columns(terms).any()


def test_dropping_the_radial_block_leaves_only_harmonics():
    (xp, yp), _, _, _ = positions(100, seed=3)
    _d, terms = detector_design(
        xp, yp, BC, RMAX, degree_r=3, n_harmonic=2, include_radial=False
    )
    assert all(m > 0 for _j, m, _k in terms)
    # every radial order still modulates each harmonic
    assert len(terms) == 4 * 2 * 1 or len(terms) == (3 + 1) * 2 * 2


def test_the_design_is_identifiable_against_a_miller_index():
    """Unlike `s^2`, position is not a property of the reflection.

    The same Miller index appears at different detector positions in different
    images, so a position column has within-reflection variation and cannot be
    absorbed into `J`. An `s^2` column has none, which is why it is dropped.
    """
    n_pair, mult = 200, 6
    g = torch.Generator().manual_seed(4)
    r = RMAX * (0.2 + 0.7 * torch.rand(n_pair * mult, generator=g, dtype=DT))
    phi = 2 * math.pi * torch.rand(n_pair * mult, generator=g, dtype=DT)
    x, y = BC[0] + r * torch.cos(phi), BC[1] + r * torch.sin(phi)
    d, _ = detector_design(x, y, BC, RMAX, degree_r=2, n_harmonic=2)
    pair = torch.arange(n_pair).repeat_interleave(mult)

    # within-reflection variance of each column, against its total variance
    for k in range(d.shape[1]):
        col = d[:, k]
        means = torch.zeros(n_pair, dtype=DT).scatter_add_(0, pair, col) / mult
        within = (col - means[pair]).pow(2).mean()
        assert within > 0.05 * col.var(), (
            f"column {k} is nearly constant within a reflection: "
            f"within {within:.4f} against total {col.var():.4f}"
        )


# ----------------------------------------- it manufactures anomalous signal


def plant_with_position_error(
    n=6000, j_mean=400.0, mult=12, amp=0.05, harmonic=1, seed=0
):
    """Pairs with NO anomalous difference and a position-dependent scale.

    The counts are generated with the true position-dependent scale; the merge
    is then told the scale is a per-image constant, which is the omission under
    test. `harmonic` picks whether the planted error is odd (antisymmetric
    between mates) or even (common to them).
    """
    g = torch.Generator().manual_seed(seed)
    j = (-j_mean * torch.rand(n, generator=g, dtype=DT).clamp(1e-9, 1).log())

    kp = torch.zeros(n, dtype=DT)
    km = torch.zeros(n, dtype=DT)
    ep = torch.zeros(n, dtype=DT)
    em = torch.zeros(n, dtype=DT)
    for _ in range(mult):
        (xp, yp), (xm, ym), _r, phi = positions(
            n, seed=int(torch.randint(0, 10**6, (1,), generator=g))
        )
        s_p = 1.0 + amp * torch.cos(harmonic * phi)
        s_m = 1.0 + amp * torch.cos(harmonic * (phi + math.pi))
        kp += torch.poisson((j * s_p).float(), generator=g).to(DT)
        km += torch.poisson((j * s_m).float(), generator=g).to(DT)
        # the merge believes the scale is 1 everywhere
        ep += torch.ones(n, dtype=DT)
        em += torch.ones(n, dtype=DT)
    return dict(kp=kp, km=km, ep=ep, em=em, j=j, n=n)


def null_spread(d):
    """Standardized deviation of the observed split from the exact null."""
    out = anomalous_split(
        d["kp"], d["km"], d["ep"], d["em"],
        torch.full_like(d["kp"], 1e-6), torch.full_like(d["kp"], 10.0),
    )
    t, th = out["total"], out["theta_null"]
    z = (d["kp"] - t * th) / (t * th * (1 - th)).clamp_min(1e-9).sqrt()
    return float(z.std()), out


# How big the manufactured difference is, exactly. With amplitude `A` on
# `cos(phi)` the true exposures are `mult (1 +/- A m)` for the two mates, where
# `m` is the mean of `cos(phi)` over that pair's observations, because
# `cos(phi + pi) = -cos(phi)`. The merge is told both exposures are `mult`, so
# it attributes the whole excess to the intensity fraction and reports
#
#     frac = A m,     sd(frac) = A / sqrt(2 mult)
#
# since `var(cos phi) = 1/2`. At `A = 0.05`, `mult = 12` that is 1.0%, against
# a measurement noise of 1.25% -- real but not separable, which is the 1.42
# null spread the first version of this test measured and wrongly demanded 2.0
# of. `A = 0.12` gives 2.45% against the same noise and separates.
#
# Real data averages down less, because a reflection recurs at similar detector
# positions rather than at independent ones.
AMP = 0.12


def test_an_odd_harmonic_scale_error_manufactures_a_bijvoet_difference():
    clean = plant_with_position_error(amp=0.0, seed=11)
    dirty = plant_with_position_error(amp=AMP, harmonic=1, seed=11)

    sd_clean, out_clean = null_spread(clean)
    sd_dirty, out_dirty = null_spread(dirty)

    assert 0.8 < sd_clean < 1.25, f"clean null is not calibrated: {sd_clean:.3f}"
    assert sd_dirty > 2.0 * sd_clean, (
        f"a {AMP:.0%} odd-harmonic scale error gave {sd_dirty:.2f} "
        f"against {sd_clean:.2f}"
    )
    # and it shows up as apparent anomalous signal where there is none.
    # Expected ratio sqrt(1 + (0.0245/0.0125)^2) = 2.2 by the arithmetic above
    assert float(out_dirty["frac"].abs().median()) > 2.0 * float(
        out_clean["frac"].abs().median()
    )


def test_an_even_harmonic_error_of_the_same_size_does_not():
    """The mechanism is the mate antisymmetry, not scale error in general."""
    odd = plant_with_position_error(amp=AMP, harmonic=1, seed=12)
    even = plant_with_position_error(amp=AMP, harmonic=2, seed=12)
    clean = plant_with_position_error(amp=0.0, seed=12)

    sd_clean, _ = null_spread(clean)
    sd_odd, _ = null_spread(odd)
    sd_even, _ = null_spread(even)

    assert sd_odd > 2.0 * sd_clean, f"odd {sd_odd:.2f} vs clean {sd_clean:.2f}"
    assert sd_even < 1.3 * sd_clean, (
        f"an even harmonic should be common to the pair: "
        f"even {sd_even:.2f} vs clean {sd_clean:.2f}, odd {sd_odd:.2f}"
    )


def test_the_artifact_grows_with_the_error():
    sds = [
        null_spread(plant_with_position_error(amp=a, harmonic=1, seed=13))[0]
        for a in (0.0, 0.02, 0.06)
    ]
    assert sds[0] < sds[1] < sds[2], f"null spread by error size: {sds}"


def test_knowing_the_position_scale_removes_the_artifact():
    """The fix, stated as a test: with `E+/E-` right, the null is restored.

    This plants the error and then hands the merge the true per-observation
    scales, which is what fitting a detector term is for. It isolates the
    question "would modelling position fix this" from "can we fit it".
    """
    g = torch.Generator().manual_seed(14)
    n, mult, amp = 6000, 12, 0.05
    j = (-400.0 * torch.rand(n, generator=g, dtype=DT).clamp(1e-9, 1).log())
    kp = torch.zeros(n, dtype=DT)
    km = torch.zeros(n, dtype=DT)
    ep = torch.zeros(n, dtype=DT)
    em = torch.zeros(n, dtype=DT)
    for _ in range(mult):
        _, _, _r, phi = positions(n, seed=int(torch.randint(0, 10**6, (1,), generator=g)))
        s_p = 1.0 + amp * torch.cos(phi)
        s_m = 1.0 + amp * torch.cos(phi + math.pi)
        kp += torch.poisson((j * s_p).float(), generator=g).to(DT)
        km += torch.poisson((j * s_m).float(), generator=g).to(DT)
        ep += s_p
        em += s_m

    out = anomalous_split(
        kp, km, ep, em,
        torch.full_like(kp, 1e-6), torch.full_like(kp, 10.0),
    )
    t, th = out["total"], out["theta_null"]
    z = (kp - t * th) / (t * th * (1 - th)).clamp_min(1e-9).sqrt()
    assert 0.8 < float(z.std()) < 1.25, (
        f"with the true scales the null should be calibrated, got {float(z.std()):.3f}"
    )
    assert float(out["frac"].abs().median()) < 0.01, (
        f"residual fake signal {float(out['frac'].abs().median()):.4f}"
    )
