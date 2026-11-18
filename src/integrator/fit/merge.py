"""Pixels to structure factors in closed form, with no gradient step anywhere.

The model carries no intermediate per-reflection intensity:

    c_ij ~ Poisson(S_i J_h p_ij + B_i)

for observation `i`, pixel `j`, Miller index `h(i)`, scale `S_i`, flat background
`B_i`, merged intensity `J_h = |F_h|^2`, and a profile `p` that is a function of
geometry and four global scalars (`integrator.fit.direct`).

Why it is closed form. Split each count into signal and background,
`s_ij ~ Poisson(S_i J_h p_ij)` and `b_ij ~ Poisson(B_i)`. The profile is
normalized over the valid pixels, so `sum_j p_ij = 1` and observation `i` carries
`s_i ~ Poisson(S_i J_h)`. Pooling the observations of one Miller index,
`sum_i s_i ~ Poisson(J_h sum_i S_i)`, and the Wilson prior is a Gamma, which is
conjugate to a Poisson rate:

    J_h | .  ~ Gamma(alpha_W + sum_i s_i,  beta_W + sum_i S_i)
    S_i | .  ~ Gamma(alpha_S + s_i,        beta_S + J_h)
    B_i | .  ~ Gamma(alpha_B + sum_j (1-r_ij) c_ij,  beta_B + n_i)

with the responsibility `r_ij = S_i J_h p_ij / (S_i J_h p_ij + B_i)` supplying the
split. Iterating the responsibility and these three updates is EM, so it ascends
the observed-data likelihood; every step is arithmetic on sufficient statistics.

**The variance from those posteriors is not the one to export.** The augmented
model treats the signal photons as observed, so `Var(J) ~ J / sum_i S_i` is
Poisson on the signal alone and drops the split uncertainty, which dominates on a
shoebox that is 96.6% background. `merged_fisher_variance` supplies the
observed-data quantity instead: the `J` information with each observation's
background Schur-complemented out. `docs/rb_export_2026-11-12.md` records what
believing the conjugate variance costs -- IQR log sigma 1.56 against a truth of
0.25, and with it 0.0073 of R-free and 0.02 of outer-shell CC-half.

**The gauge is fixed explicitly.** `S_i J_h` is invariant under
`S -> cS, J -> J/c`, so the likelihood is flat along that direction and block
updates drift along it rather than converging. `rescale_gauge` pins it by holding
the mean scale at one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "MergeHyperparameters",
    "ObservationStats",
    "accumulate_scale_normals",
    "anomalous_fisher_variance",
    "centric_anchor",
    "chebyshev_design",
    "fit_and_shrink_difference",
    "merged_fisher_variance",
    "rescale_gauge",
    "responsibilities",
    "scale_design",
    "shrink_difference",
    "solve_normals",
    "solve_scale_factors",
    "tensor_design",
    "split_by_mate",
    "update_background",
    "update_merged",
    "update_merged_anomalous",
    "update_scale",
    "update_tau",
    "update_wilson_sigma",
    "wilson_prior",
]


@dataclass
class MergeHyperparameters:
    """Gamma prior parameters for the three blocks.

    `alpha_w` is the Wilson shape: 1 for acentric reflections and 1/2 for
    centric, which is the chi^2_1 form with the same mean and twice the
    normalized variance.
    """

    alpha_scale: float = 1e-3
    beta_scale: float = 1e-3
    alpha_background: float = 1e-3
    beta_background: float = 1e-3
    eps: float = 1e-9


@dataclass
class ObservationStats:
    """Per-observation sufficient statistics from one responsibility pass."""

    signal: Tensor
    background_counts: Tensor
    valid_pixels: Tensor


def wilson_prior(
    sigma_d: Tensor, centric: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Wilson prior on `J` as a Gamma, shape and rate, per reflection.

    Acentric: `Gamma(1, 1/Sigma)`, the exponential with mean `Sigma`.
    Centric: `Gamma(1/2, 1/(2 Sigma))`, the same mean with twice the normalized
    variance. Both are Gammas, so both are conjugate to the Poisson signal total.
    """
    alpha = torch.full_like(sigma_d, 1.0)
    if centric is not None:
        alpha = torch.where(centric.to(torch.bool), torch.full_like(alpha, 0.5), alpha)
    beta = alpha / sigma_d.clamp_min(1e-12)
    return alpha, beta


def responsibilities(
    profile: Tensor, scale: Tensor, merged: Tensor, background: Tensor, eps: float = 1e-9
) -> Tensor:
    """Posterior share of each count belonging to signal, shape `(N, P)`.

    `scale` and `merged` are per observation, `background` per observation, and
    `profile` per observation per pixel. This is the E step; it is exact given
    the current parameters, and it is the only place the counts and the
    parameters meet.
    """
    sig = (scale * merged).unsqueeze(-1) * profile
    return sig / (sig + background.unsqueeze(-1)).clamp_min(eps)


def observation_stats(
    counts: Tensor, mask: Tensor, resp: Tensor
) -> ObservationStats:
    """Signal photons, background photons and valid-pixel count per observation.

    Photons are conserved by construction: `sum(r c) + sum((1-r) c) == sum(c)`
    for any `r` in `[0, 1]`, so the two blocks read a partition of the counts
    rather than overlapping evidence.
    """
    m = mask.to(counts.dtype)
    c = counts * m
    signal = (resp * c).sum(-1)
    return ObservationStats(
        signal=signal,
        background_counts=(c.sum(-1) - signal).clamp_min(0.0),
        valid_pixels=m.sum(-1),
    )


def update_merged(
    signal_sum: Tensor,
    scale_sum: Tensor,
    alpha_w: Tensor,
    beta_w: Tensor,
) -> tuple[Tensor, Tensor]:
    """Posterior Gamma for `J_h`, returned as `(mean, shape/rate^2 variance)`.

    `signal_sum` and `scale_sum` are summed over the observations of each Miller
    index. Unlike the per-reflection conjugate update, the rate here grows with
    multiplicity, so the merged Fano factor is not pinned by the prior the way
    `beta + sum(p m) = beta + 1` pinned it per reflection. It is still the
    augmented-model variance, so it is a diagnostic rather than the export.
    """
    shape = alpha_w + signal_sum
    rate = (beta_w + scale_sum).clamp_min(1e-12)
    return shape / rate, shape / rate.pow(2)


def update_scale(
    signal: Tensor,
    merged_of_obs: Tensor,
    hyper: MergeHyperparameters,
    group: Tensor | None = None,
    n_groups: int | None = None,
) -> Tensor:
    """Posterior mean scale, `Gamma(a + sum s, b + sum J)` over a group.

    **`group` is not optional in practice.** A free scale per observation leaves
    the model over-parameterized -- `n` observations against `n + n_h` unknowns
    -- so `S_i` absorbs that observation's own noise and the averaging that
    should determine `J_h` never happens. Measured on planted data at
    multiplicity 8: free per-observation scales give a median error of 8% where
    the statistical limit is 0.7%, and the Fisher variance is then 14x
    optimistic because it conditions on scales that are themselves fitted to
    noise. Pooling over an image, or any group large enough to be identified,
    is what makes the block well posed; it is the same role careless's smooth
    metadata scale plays.

    Omitting `group` returns the per-observation update, which is correct only
    when the scale really is known to vary that fast.

    Returns a tensor of length `n_groups` when grouped, else one per observation.
    """
    if group is None:
        return (hyper.alpha_scale + signal) / (
            hyper.beta_scale + merged_of_obs
        ).clamp_min(hyper.eps)

    n_groups = int(n_groups if n_groups is not None else int(group.max()) + 1)
    sig = torch.zeros(n_groups, dtype=signal.dtype, device=signal.device)
    jsum = torch.zeros_like(sig)
    sig.scatter_add_(0, group, signal)
    jsum.scatter_add_(0, group, merged_of_obs)
    return (hyper.alpha_scale + sig) / (hyper.beta_scale + jsum).clamp_min(hyper.eps)


def update_background(
    stats: ObservationStats, hyper: MergeHyperparameters
) -> Tensor:
    """Posterior mean background per pixel per observation."""
    return (hyper.alpha_background + stats.background_counts) / (
        hyper.beta_background + stats.valid_pixels
    ).clamp_min(hyper.eps)


def update_wilson_sigma(
    merged: Tensor, bin_index: Tensor, n_bins: int, eps: float = 1e-12
) -> Tensor:
    """Per-shell Wilson scale by moment matching, shape `(n_bins,)`.

    `Sigma_d = <J>` over the shell, which is the maximum-likelihood scale of the
    exponential and the empirical-Bayes choice. Shells with no reflections keep
    the dataset mean rather than collapsing to zero.
    """
    total = torch.zeros(n_bins, dtype=merged.dtype, device=merged.device)
    count = torch.zeros_like(total)
    total.scatter_add_(0, bin_index, merged)
    count.scatter_add_(0, bin_index, torch.ones_like(merged))
    overall = merged.mean()
    return torch.where(count > 0, total / count.clamp_min(eps), overall).clamp_min(eps)


def rescale_gauge(scale: Tensor, merged: Tensor) -> tuple[Tensor, Tensor]:
    """Pin `S J` to a representative by holding the mean scale at one.

    The likelihood is invariant under `S -> cS, J -> J/c`, so without this the
    block updates wander along a flat direction and the sweep looks like it is
    failing to converge when it is only moving in the gauge.
    """
    c = scale.mean().clamp_min(1e-12)
    return scale / c, merged * c


def merged_fisher_variance(
    profile: Tensor,
    mask: Tensor,
    scale: Tensor,
    merged_of_obs: Tensor,
    background: Tensor,
    index: Tensor,
    n_merged: int,
    eps: float = 1e-9,
) -> Tensor:
    """Observed-data variance of `J_h`, backgrounds Schur-complemented out.

    For one observation, `dlambda/dJ = S_i p_ij` and `dlambda/dB = 1`, so

        F_JJ^i - (F_JB^i)^2 / F_BB^i

    is that observation's information about `J` after paying for its own unknown
    background. Information adds over observations, so

        Var(J_h) = 1 / sum_{i in h} [ F_JJ^i - (F_JB^i)^2 / F_BB^i ].

    This is the quantity the export needs. It conditions on `S` and on the
    profile, so it is a lower bound on the full uncertainty; the law of total
    variance over the scale is the next term and is not formed here.
    """
    m = mask.to(profile.dtype)
    rate = ((scale * merged_of_obs).unsqueeze(-1) * profile + background.unsqueeze(-1))
    inv = m / rate.clamp_min(eps)
    g = scale.unsqueeze(-1) * profile

    f_jj = (inv * g * g).sum(-1)
    f_jb = (inv * g).sum(-1)
    f_bb = inv.sum(-1).clamp_min(eps)
    per_obs = (f_jj - f_jb * f_jb / f_bb).clamp_min(0.0)

    info = torch.zeros(n_merged, dtype=profile.dtype, device=profile.device)
    info.scatter_add_(0, index, per_obs)
    return 1.0 / info.clamp_min(eps)


# ----------------------------------------------------------------- anomalous


def split_by_mate(
    signal: Tensor,
    scale: Tensor,
    index: Tensor,
    mate: Tensor,
    n_merged: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Reduce the observations of each Miller pair to four numbers.

    In the augmented model an observation contributes only its signal total and
    its scale, and `sum_i s_i ~ Poisson(J^± sum_i S_i)` separately for each mate.
    So the whole anomalous block runs on `(s+, s-, a+, a-)` per pair -- it is
    O(number of Miller indices), not O(pixels), however deep the dataset is.

    `mate` is +1 for `F(+)` and -1 for `F(-)`.
    """
    plus = (mate > 0).to(signal.dtype)
    out = []
    for value in (signal * plus, signal * (1 - plus), scale * plus, scale * (1 - plus)):
        acc = torch.zeros(n_merged, dtype=signal.dtype, device=signal.device)
        acc.scatter_add_(0, index, value)
        out.append(acc)
    return out[0], out[1], out[2], out[3]


def detector_design(
    x: Tensor,
    y: Tensor,
    beam_center: tuple[float, float] | list[float],
    r_max: float,
    degree_r: int = 3,
    n_harmonic: int = 3,
    include_radial: bool = True,
) -> tuple[Tensor, list[tuple[int, int, str]]]:
    """Smooth scale over the detector face, in `(r, phi)`, `(N, K)`.

    The scale model has a per-image level and a resolution ramp but nothing
    that depends on WHERE on the detector an observation landed, which is the
    one term careless has that we do not: its positional encoding on
    `xcal, ycal` is a flexible 2D map per image. Absorption, detector
    non-uniformity and the polarization residual all live there.

    Built in `(r, phi)` rather than `(x, y)` because the two directions are not
    interchangeable here.

    **Radial.** The detector radius is a monotone function of the scattering
    angle at fixed wavelength, so on MONOCHROMATIC data a radial term is
    degenerate with `s^2` and therefore, by the same argument that removes the
    `s^2`-only columns, degenerate with `J`. On Laue data `r` depends on both
    `s^2` and `lambda`, so the two separate and the radial term is identifiable.
    `include_radial=False` drops it, which is what a rotation dataset wants.

    **Azimuthal.** A Friedel mate scatters with `q -> -q`, so its in-plane
    component flips and, to first order in the scattering angle, the mate sits
    at the same radius and the opposite azimuth, `phi + pi`. Under that flip
    `cos(m phi)` and `sin(m phi)` change sign for ODD `m` and are unchanged for
    even `m`. So the odd harmonics are exactly the part of a detector-position
    scale error that is ANTISYMMETRIC between mates, and antisymmetric scale
    error is indistinguishable from a Bijvoet difference. The even harmonics
    and the radial part are common to the pair to the same order and go into
    the total instead.

    That is the mechanism the centric anchor has been measuring. It also says
    the fix is small: the anomalous-relevant part of a detector map is a
    handful of odd harmonics, not a 10-layer network.

    The constant column is dropped, since it is the per-image level again.

    Args:
        x, y: Detector position in pixels, `(N,)`.
        beam_center: Beam centre in the same pixels.
        r_max: Normalizing radius; `r / r_max` is mapped onto `[-1, 1]`.
        degree_r: Chebyshev degree in the radial coordinate.
        n_harmonic: Highest azimuthal harmonic `m`.
        include_radial: Keep the `m = 0` columns. False on monochromatic data,
            where they are degenerate with `s^2`.

    Returns:
        The design and, per column, `(radial_order, m, 'cos'|'sin'|'const')`.
    """
    bx, by = float(beam_center[0]), float(beam_center[1])
    dx = x.reshape(-1) - bx
    dy = y.reshape(-1) - by
    r = torch.sqrt(dx * dx + dy * dy)
    phi = torch.atan2(dy, dx)
    # map r onto [-1, 1] for the Chebyshev recursion
    u = (2.0 * r / max(float(r_max), 1e-9) - 1.0).clamp(-1.0, 1.0)
    phi_r = chebyshev_design(u, int(degree_r))

    cols, terms = [], []
    for j in range(int(degree_r) + 1):
        if include_radial:
            if j > 0:  # j = 0 with m = 0 is the constant: the image level
                cols.append(phi_r[:, j])
                terms.append((j, 0, "const"))
        for m in range(1, int(n_harmonic) + 1):
            cols.append(phi_r[:, j] * torch.cos(m * phi))
            terms.append((j, m, "cos"))
            cols.append(phi_r[:, j] * torch.sin(m * phi))
            terms.append((j, m, "sin"))
    return torch.stack(cols, dim=-1), terms


def combined_scale_design(
    image_u: Tensor,
    s2: Tensor,
    x: Tensor | None = None,
    y: Tensor | None = None,
    beam_center: tuple[float, float] | list[float] | None = None,
    r_max: float | None = None,
    degree_image: int = 6,
    degree_s2: int = 3,
    degree_r: int = 3,
    n_harmonic: int = 3,
    include_radial: bool = True,
    multiplicative: bool = True,
) -> tuple[Tensor, list[tuple]]:
    """`scale_design` with the detector block, `(N, K)`.

    The scale solve is a weighted least squares LINEAR in its coefficients, so
    adding detector columns needs no change to `run_sweep`, only a wider
    design. The two blocks live in one function so the gauge rules stay in one
    place -- `scale_design` drops the columns constant in image and varying in
    `s^2`, `detector_design` drops its own constant, and what remains is the
    single overall constant the sweep's mean-scale pin handles.

    **`multiplicative` matters and defaults on.** A detector response
    multiplies the image's scale rather than adding to it: the physical form is
    `S = g(image) h(r, phi)`, not `g + h`. Appending the columns fits the sum,
    which cannot represent the product, and the residual is
    `(g - mean g) h` -- proportional to how much the image scale varies.
    Measured on planted data with a 20% image spread and a 15% `cos(phi)`
    error, the appended form removed only 13% of the manufactured anomalous
    signal. The tensor product `phi_j(image) psi_m(r, phi)` is the
    linear-in-coefficients way to write a product of two smooth functions, and
    is what this builds unless `multiplicative=False`.

    Returns the design and the per-column terms: `(j, k)` pairs for the image
    block, then `(radial_order, m, kind)` triples for the detector block, or
    `(j, k, radial_order, m, kind)` for a tensor-product column.
    `mate_antisymmetric_columns` reads `m` from either shape.
    """
    base, terms_base = scale_design(image_u, s2, degree_image, degree_s2)
    if x is None or y is None or beam_center is None or r_max is None:
        return base, list(terms_base)
    det, terms_det = detector_design(
        x, y, beam_center, r_max, degree_r, n_harmonic, include_radial
    )
    det = det.to(base.dtype)
    if not multiplicative:
        return torch.cat([base, det], dim=-1), [*terms_base, *terms_det]

    # the image block alone, without the s^2 factor, is what the detector
    # response multiplies; crossing it with the full (image, s^2) block would
    # square the column count for no physical reason
    phi_i = chebyshev_design(image_u, degree_image).to(base.dtype)
    cross = tensor_design(phi_i, det)
    terms_cross = [
        (j, 0, r_ord, m, kind)
        for j in range(degree_image + 1)
        for (r_ord, m, kind) in terms_det
    ]
    return torch.cat([base, cross], dim=-1), [*terms_base, *terms_cross]


def mate_antisymmetric_columns(
    terms: list[tuple[int, int, str]],
) -> Tensor:
    """Mask of the `detector_design` columns that can manufacture anomalous signal.

    Odd azimuthal harmonics, by the `phi -> phi + pi` argument in
    `detector_design`. Useful two ways: to penalise exactly those columns when
    the centric anchor says the scale is inventing a difference, and to confirm
    that a fitted scale's anomalous leakage lives where the geometry says it
    must rather than somewhere unaccounted for.
    """
    out = []
    for t in terms:
        if len(t) == 2:
            # the image block: (j, k), no azimuth, symmetric between mates
            out.append(False)
        elif len(t) == 3:
            out.append(int(t[1]) % 2 == 1)        # (radial, m, kind)
        else:
            out.append(int(t[3]) % 2 == 1)        # (j, k, radial, m, kind)
    return torch.tensor(out, dtype=torch.bool)


def anomalous_split(
    signal_plus: Tensor,
    signal_minus: Tensor,
    scale_plus: Tensor,
    scale_minus: Tensor,
    beta_w: Tensor,
    tau: Tensor,
    centric: Tensor | None = None,
    n_iter: int = 12,
    eps: float = 1e-12,
) -> dict[str, Tensor]:
    """Bijvoet pair by Poisson splitting: a total and a split, solved apart.

    Poisson thinning factorizes the pair's likelihood exactly. With
    `mu+ = J+ E+` and `mu- = J- E-`, the total `T = k+ + k-` is
    `Poisson(mu+ + mu-)` and, conditional on it,

        k+ | T ~ Binomial(T, theta),    theta = mu+ / (mu+ + mu-)

    so the common intensity and the anomalous difference are carried by
    different statistics. This is the whole point: the difference is read off a
    Binomial split that knows nothing about the overall intensity, instead of
    off a 2x2 Newton step in `(J, Delta)` that couples them and whose Hessian
    goes rank-1 the moment one mate is missing.

    The natural unknown is the INTENSITY fraction `rho = J+ / (J+ + J-)`, not
    `theta`, because the exposures differ:

        logit(theta) = logit(rho) + log(E+ / E-)

    which is a Binomial likelihood with a KNOWN offset. In `eta = logit(rho)`
    the log-likelihood is strictly concave --
    `d^2/d eta^2 = -T theta (1 - theta) < 0` -- so a Newton iteration converges
    from any start with no damping, no determinant guard and no special case
    for a missing mate: with `E- = 0` the offset is infinite, `theta` is one,
    the data say nothing, and the prior returns `rho = 1/2`.

    `2 rho - 1` is exactly `(I+ - I-) / (I+ + I-)`, so the prior sits on the
    FRACTIONAL difference by construction. That is the scale measured to be the
    right one -- shrinking a planted 3% difference on the absolute scale raised
    the mean squared error from 1162 to 1719 while the fractional scale lowered
    it to 895 -- and here it is not a choice but what the parameterization is.

    Given `rho`, the total is conjugate: `T ~ Poisson(2 J [rho E+ + (1-rho) E-])`
    against the Wilson Gamma gives a Gamma posterior for `J` in closed form,
    with an exact variance rather than a Fisher approximation. The two blocks
    are exactly independent when `E+ = E-`, and coupled only through
    `(E+ - E-)(2 rho - 1)` otherwise -- which is the same `E+/E-` ratio that is
    the only route by which a scale error can reach the anomalous signal.

    Args:
        signal_plus, signal_minus: Summed signal counts per pair, `(P,)`.
        scale_plus, scale_minus: Summed scales `E+`, `E-`, `(P,)`.
        beta_w: Wilson rate per pair, `(P,)`; the Gamma prior's rate on `J`.
        tau: Per-shell prior sd of the FRACTIONAL difference `2 rho - 1`,
            broadcast per pair, `(P,)`. Enters as a Gaussian on `eta` with
            sd `2 tau`, since `d eta / d(2 rho - 1) = 2` at `rho = 1/2`.
        centric: Pairs held at `rho = 1/2` exactly, `(P,)` bool.
        n_iter: Newton steps on the 1-D concave problem.
        eps: Floor.

    Returns:
        Dict with `j`, `var_j` (Gamma posterior mean and variance), `rho`,
        `var_rho`, `frac` = `2 rho - 1` and its `var_frac`, `delta` = `J+ - J-`
        and `var_delta`, and `theta_null` -- the split a centric must show,
        `E+/(E+ + E-)`, which is the anchor's exact null.
    """
    work = torch.float64
    out_dtype = scale_plus.dtype
    kp = signal_plus.to(work).clamp_min(0.0)
    km = signal_minus.to(work).clamp_min(0.0)
    ep = scale_plus.to(work).clamp_min(0.0)
    em = scale_minus.to(work).clamp_min(0.0)
    bw = beta_w.to(work)
    t_total = kp + km

    # log(E+/E-) as a known offset; a missing mate sends it to +/- inf, which
    # the sigmoid saturates and the prior then decides, so no branch is needed
    off = torch.log(ep.clamp_min(eps)) - torch.log(em.clamp_min(eps))
    off = torch.where(ep > eps, off, torch.full_like(off, -80.0))
    off = torch.where(em > eps, off, torch.full_like(off, 80.0))

    # Gaussian prior on eta with sd 2 tau: eta = logit(rho), and near rho = 1/2
    # d eta = 2 d(2 rho - 1), so a prior sd of tau on the fractional difference
    # is a prior sd of 2 tau on eta
    sd_eta = (2.0 * tau.to(work)).clamp_min(1e-6)
    prec = 1.0 / sd_eta.pow(2)

    eta = torch.zeros_like(kp)
    for _ in range(max(1, n_iter)):
        theta = torch.sigmoid(eta + off)
        grad = kp - t_total * theta - prec * eta
        hess = -(t_total * theta * (1.0 - theta)).clamp_min(eps) - prec
        eta = eta - grad / hess
        eta = eta.clamp(-40.0, 40.0)

    if centric is not None:
        eta = torch.where(centric.to(torch.bool), torch.zeros_like(eta), eta)

    theta = torch.sigmoid(eta + off)
    rho = torch.sigmoid(eta)
    # observed information of the 1-D concave problem, exact
    info_eta = (t_total * theta * (1.0 - theta)).clamp_min(eps) + prec
    var_eta = 1.0 / info_eta
    d_rho = rho * (1.0 - rho)
    var_rho = d_rho.pow(2) * var_eta

    # the total, conjugate given rho
    exposure = (2.0 * (rho * ep + (1.0 - rho) * em)).clamp_min(eps)
    alpha_post = 1.0 + t_total
    beta_post = (bw + exposure).clamp_min(eps)
    j = alpha_post / beta_post
    var_j = alpha_post / beta_post.pow(2)

    frac = 2.0 * rho - 1.0
    var_frac = 4.0 * var_rho
    delta = 2.0 * j * frac
    # Var(2 J F) with J and F independent, which they are exactly at E+ = E-
    var_delta = 4.0 * (
        j.pow(2) * var_frac + frac.pow(2) * var_j + var_j * var_frac
    )
    if centric is not None:
        c = centric.to(torch.bool)
        frac = torch.where(c, torch.zeros_like(frac), frac)
        delta = torch.where(c, torch.zeros_like(delta), delta)
        var_frac = torch.where(c, torch.zeros_like(var_frac), var_frac)
        var_delta = torch.where(c, torch.zeros_like(var_delta), var_delta)

    theta_null = ep / (ep + em).clamp_min(eps)
    return {
        k: v.to(out_dtype)
        for k, v in dict(
            j=j, var_j=var_j, rho=rho, var_rho=var_rho, frac=frac,
            var_frac=var_frac, delta=delta, var_delta=var_delta,
            theta=theta, theta_null=theta_null, total=t_total,
        ).items()
    }


def update_merged_anomalous(
    signal_plus: Tensor,
    signal_minus: Tensor,
    scale_plus: Tensor,
    scale_minus: Tensor,
    beta_w: Tensor,
    tau: Tensor,
    centric: Tensor | None = None,
    n_iter: int = 25,
    eps: float = 1e-9,
) -> tuple[Tensor, Tensor]:
    """MAP common intensity and Bijvoet difference, `(J, Delta)`, per Miller pair.

    Parameterized as `J = (J+ + J-)/2` and `Delta = J+ - J-`, so the Poisson rate
    is linear in both and

        f(J, D) = s+ log(J + D/2) - a+ (J + D/2)
                + s- log(J - D/2) - a- (J - D/2)
                - beta_W J - D^2 / (2 tau^2)

    is jointly concave: `log` of a positive linear function is concave, the rest
    is linear or a negative quadratic. A damped Newton step therefore converges
    to the unique maximum.

    The Wilson rate acts on the COMMON intensity only and the difference carries
    a Gaussian prior with a per-shell scale. Shrinking `J+` and `J-` separately
    toward a Wilson prior is what destroys the anomalous signal; the measured
    rule is that any Bijvoet shrink must be per shell, because the
    per-reflection Bijvoet signal-to-noise averaged over all acentrics floors
    `tau` and zeroes the difference everywhere.

    Centric reflections have `Delta = 0` by symmetry and are held there.
    """
    # This block is O(Miller indices), so double precision here is free, and it
    # is necessary: in float32 a `J` sitting on the `eps` floor makes `s/J`
    # about 1e9 and the Hessian entry `s/J^2` about 1e18, which overflows to
    # `inf`, and `det` then comes out `inf - inf = NaN`. Seen as three pairs
    # going NaN on the first sweep and spreading to every pair by the second.
    work = torch.float64
    out_dtype = scale_plus.dtype
    signal_plus = signal_plus.to(work)
    signal_minus = signal_minus.to(work)
    beta_w = beta_w.to(work)
    tau = tau.to(work)

    a_p = scale_plus.to(work).clamp_min(0.0)
    a_m = scale_minus.to(work).clamp_min(0.0)
    both = (a_p > eps) & (a_m > eps)

    j = ((signal_plus + signal_minus + 1.0) / (a_p + a_m + beta_w).clamp_min(eps))
    j = j.clamp_min(eps)
    delta = torch.zeros_like(j)
    inv_tau2 = 1.0 / tau.clamp_min(eps).pow(2)
    if centric is not None:
        both = both & ~centric.to(torch.bool)

    for _ in range(n_iter):
        jp = (j + 0.5 * delta).clamp_min(eps)
        jm = (j - 0.5 * delta).clamp_min(eps)
        rp, rm = signal_plus / jp, signal_minus / jm

        g_j = rp - a_p + rm - a_m - beta_w
        g_d = 0.5 * (rp - a_p) - 0.5 * (rm - a_m) - delta * inv_tau2

        h_jj = -(rp / jp + rm / jm)
        h_jd = -0.5 * (rp / jp) + 0.5 * (rm / jm)
        h_dd = -0.25 * (rp / jp + rm / jm) - inv_tau2

        # The Hessian is negative definite where both mates are present, so
        # det > 0. Where only one mate was seen the two columns are identical
        # and det is EXACTLY zero: the pair determines `J + Delta/2` and nothing
        # else. Nudging det to a negative number there flips the Newton step and
        # drives J to zero, which then corrupts the scales and backgrounds for
        # the whole dataset. Those pairs get the one-dimensional step instead.
        det = (h_jj * h_dd - h_jd * h_jd).clamp_min(eps)
        step_j_2d = -(h_dd * g_j - h_jd * g_d) / det
        step_d_2d = -(-h_jd * g_j + h_jj * g_d) / det
        step_j_1d = -g_j / h_jj.clamp_max(-eps)

        step_j = torch.where(both, step_j_2d, step_j_1d)
        step_d = torch.where(both, step_d_2d, torch.zeros_like(step_d_2d))

        j_new = (j + step_j).clamp_min(eps)
        d_new = torch.where(both, delta + step_d, torch.zeros_like(delta))
        # keep both mates positive: |Delta| < 2J is the feasible region
        d_new = d_new.clamp(min=-1.98 * j_new, max=1.98 * j_new)
        j, delta = j_new, d_new

    j = torch.nan_to_num(j, nan=eps, posinf=eps, neginf=eps).clamp_min(eps)
    delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
    return j.to(out_dtype), torch.where(both, delta, torch.zeros_like(delta)).to(
        out_dtype
    )


def anomalous_fisher_variance(
    profile: Tensor,
    mask: Tensor,
    scale: Tensor,
    merged_of_obs: Tensor,
    background: Tensor,
    index: Tensor,
    mate: Tensor,
    n_merged: int,
    eps: float = 1e-9,
) -> tuple[Tensor, Tensor]:
    """Observed-data variances of `J` and `Delta`, backgrounds Schur'd out.

    Within one observation the two columns are collinear -- an observation of the
    `+` mate sees only `J + Delta/2` -- so the pair is separated by having both
    mates, and the algebra collapses. Writing `q_i` for that observation's
    background-corrected information (the same scalar `merged_fisher_variance`
    sums), and `Q+`, `Q-` for its totals over each mate,

        Var(J)     = (Q+ + Q-) / (4 Q+ Q-)
        Var(Delta) = (Q+ + Q-) / (Q+ Q-)

    so `Var(Delta) = 4 Var(J)`, and both diverge when a mate is unobserved, which
    is the honest answer rather than a number borrowed from the prior.
    """
    m = mask.to(profile.dtype)
    rate = ((scale * merged_of_obs).unsqueeze(-1) * profile + background.unsqueeze(-1))
    inv = m / rate.clamp_min(eps)
    g = scale.unsqueeze(-1) * profile

    f_jj = (inv * g * g).sum(-1)
    f_jb = (inv * g).sum(-1)
    f_bb = inv.sum(-1).clamp_min(eps)
    q = (f_jj - f_jb * f_jb / f_bb).clamp_min(0.0)

    plus = (mate > 0).to(q.dtype)
    q_p = torch.zeros(n_merged, dtype=q.dtype, device=q.device)
    q_m = torch.zeros_like(q_p)
    q_p.scatter_add_(0, index, q * plus)
    q_m.scatter_add_(0, index, q * (1 - plus))

    # A pair observed on one side only determines `J + Delta/2` and neither `J`
    # nor `Delta` alone, so both variances are infinite. That is the value to
    # return: flooring the product instead reports a large finite number whose
    # size is set by the floor, and a downstream merge cannot tell it from a
    # measurement. The caller drops those reflections from the anomalous columns,
    # or supplies the prior and uses the posterior variance.
    total = q_p + q_m
    prod = q_p * q_m
    both = (q_p > eps) & (q_m > eps)
    inf = torch.full_like(total, float("inf"))
    var_j = torch.where(both, total / (4.0 * prod.clamp_min(eps)), inf)
    var_d = torch.where(both, total / prod.clamp_min(eps), inf)
    return var_j, var_d


def update_tau(
    delta: Tensor,
    var_delta: Tensor,
    bin_index: Tensor,
    n_bins: int,
    tau_scale: float | None = None,
    tau_prob: float = 0.01,
    n_iter: int = 40,
    eps: float = 1e-12,
) -> Tensor:
    """Per-shell scale of the Bijvoet difference, shape `(n_bins,)`.

    **`delta` must be the UNSHRUNK, likelihood-only estimate** -- the one from
    `update_merged_anomalous` run with `tau = inf` -- because empirical Bayes
    reads the scale off the marginal of the raw estimate. Passing a posterior
    mean that has already been shrunk toward zero, while passing the unshrunk
    `var_delta`, is inconsistent and drives `tau` to zero: measured on a planted
    3% difference it returned 2.9 against a true 94.1, having been given
    estimates that the previous sweep's prior had already pulled in. A sweep
    therefore solves for the difference without its prior, fits `tau`, then
    shrinks with `shrink_difference`.

    With `Delta_h ~ N(0, tau_d^2)` and measurement variance `v_h`, the marginal
    is `N(0, tau_d^2 + v_h)`, and the shell's scale is estimated from

        sum_h [ Delta_h^2 / (tau^2+v_h)^2 - 1 / (tau^2+v_h) ] = theta / tau,

    which is the marginal maximum likelihood when `theta = 0` -- the empirical
    Bayes estimate -- and the penalised-complexity MAP otherwise. The PC prior of
    Simpson et al. (Statistical Science 32(1), 2017) takes `Delta = 0` as the
    base model and penalises departure from it with an exponential on `tau`,
    `theta = -log(tau_prob) / tau_scale`, so the shrinkage is set by one
    statement: the difference exceeds `tau_scale` with probability `tau_prob`.

    Solved by bisection on `tau`, which needs no derivative and cannot overshoot
    past zero. A shell whose data carry no anomalous signal returns `tau -> 0`
    and its differences are shrunk away, which is correct and is why the scale
    must be per shell rather than global.
    """
    theta = 0.0 if tau_scale is None else -torch.log(
        torch.tensor(float(tau_prob), dtype=delta.dtype, device=delta.device)
    ).item() / float(tau_scale)

    def score(t: Tensor) -> Tensor:
        """Shell-wise `dl/d(tau^2) * 2 tau`, zero at the optimum."""
        out = torch.zeros(n_bins, dtype=delta.dtype, device=delta.device)
        denom = (t[bin_index].pow(2) + var_delta).clamp_min(eps)
        term = delta.pow(2) / denom.pow(2) - 1.0 / denom
        out.scatter_add_(0, bin_index, term)
        return out * t - theta

    hi = torch.full((n_bins,), float(delta.abs().max().clamp_min(1.0) * 4.0),
                    dtype=delta.dtype, device=delta.device)
    lo = torch.zeros_like(hi)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        s = score(mid)
        hi = torch.where(s < 0, mid, hi)
        lo = torch.where(s < 0, lo, mid)
    return 0.5 * (lo + hi)


def shrink_difference(
    delta_ml: Tensor, var_delta: Tensor, tau_of_h: Tensor, eps: float = 1e-12
) -> tuple[Tensor, Tensor]:
    """Shrink a likelihood-only Bijvoet difference toward the base model.

    For a Gaussian likelihood `N(Delta_ML, v)` and a `N(0, tau^2)` prior the
    posterior is available in closed form, so the shrinkage is a weight rather
    than another solve:

        w = tau^2 / (tau^2 + v),   Delta = w Delta_ML,   Var = w v.

    `w` is the quantity the earlier post-hoc merger reported per shell, running
    0 -> 0.38 -> 0.57 -> 0.77 toward low resolution on S-SAD data: the shells
    where the anomalous signal exceeds its noise keep most of their difference
    and the rest are shrunk away.

    An unobserved mate has `v = inf`, which gives `w = 0` and a zero difference
    with zero variance -- the base model, which is the right answer when the
    data carry no information about the difference.
    """
    w = tau_of_h.pow(2) / (tau_of_h.pow(2) + var_delta).clamp_min(eps)
    w = torch.where(torch.isfinite(var_delta), w, torch.zeros_like(w))
    return w * torch.nan_to_num(delta_ml, nan=0.0, posinf=0.0, neginf=0.0), w * torch.nan_to_num(
        var_delta, nan=0.0, posinf=0.0, neginf=0.0
    )


def fit_and_shrink_difference(
    delta_ml: Tensor,
    var_delta: Tensor,
    merged: Tensor,
    bin_index: Tensor,
    n_bins: int,
    tau_scale: float | None = None,
    tau_prob: float = 0.01,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fit the per-shell difference scale and shrink, on the FRACTIONAL scale.

    The anomalous difference is proportional to the intensity -- a substructure
    gives a roughly constant `Delta / J` across a shell, not a constant `Delta`
    -- so the prior belongs on `delta = Delta / 2J`. Putting a single absolute
    scale on `Delta` instead mis-specifies the prior for every reflection whose
    intensity is away from the shell mean, and makes the shrink actively
    harmful: measured on a planted 3% difference, shrinking on the absolute
    scale raised the mean squared error from 1162 to 1719 while shrinking on
    the fractional scale lowered it to 895.

    This is the scale the earlier post-hoc merger used, shrinking
    `(I+ - I-)/(I+ + I-)`.

    Returns the shrunk difference, its variance, and the per-shell fractional
    scale `tau_d`, which is directly readable as "the anomalous signal is this
    fraction of the intensity in this shell".
    """
    j = merged.clamp_min(eps)
    frac = delta_ml / (2.0 * j)
    var_frac = var_delta / (4.0 * j.pow(2))

    tau = update_tau(
        frac, var_frac, bin_index, n_bins, tau_scale=tau_scale, tau_prob=tau_prob
    )
    frac_post, var_post = shrink_difference(frac, var_frac, tau[bin_index])
    return frac_post * 2.0 * j, var_post * 4.0 * j.pow(2), tau


# ------------------------------------------------------------ centric anchor


def centric_anchor(
    signal_plus: Tensor,
    signal_minus: Tensor,
    scale_plus: Tensor,
    scale_minus: Tensor,
    q_plus: Tensor,
    q_minus: Tensor,
    beta_w: Tensor,
    centric: Tensor,
    bin_index: Tensor,
    n_bins: int,
    eps: float = 1e-9,
) -> dict:
    """How much anomalous difference the scale model manufactures, where truth is 0.

    A centric reflection is its own Friedel mate, so its Bijvoet difference is
    zero by symmetry. Splitting its observations by the parity of the symmetry
    operation that brought them into the asymmetric unit -- the same rule that
    separates `F(+)` from `F(-)` for an acentric -- produces two groups that are
    physically identical but sit at different detector positions, wavelengths
    and images. Anything the pair block then recovers as a difference came from
    the scale model, not from the crystal.

    This is the measurement that bounds how much flexibility the scale can carry.
    A scale over Friedel-discriminating inputs can absorb `I(+) - I(-)`: the
    recorded diagnosis of a learned scale that kept merged-mean correlation 0.98
    while its anomalous-difference correlation fell to 0.14. Adding a wavelength
    or position dependence is exactly what makes a scale able to do that, so the
    anchor should be read before and after any such change.

    Returns the standardized spread of the fake difference, which is 1 when the
    centric differences are consistent with measurement noise alone and larger
    when the scale is creating them, together with the per-shell fractional
    scale that `fit_and_shrink_difference` would assign to signal that cannot
    exist.
    """
    sel = centric.to(torch.bool)
    n = int(sel.sum())
    if n == 0:
        zero = torch.zeros(n_bins, dtype=signal_plus.dtype, device=signal_plus.device)
        return {"n": 0, "z_sd": float("nan"), "tau": zero, "n_usable": 0}

    j, delta = update_merged_anomalous(
        signal_plus[sel],
        signal_minus[sel],
        scale_plus[sel],
        scale_minus[sel],
        beta_w[sel],
        torch.full((n,), float("inf"), dtype=signal_plus.dtype,
                   device=signal_plus.device),
        centric=None,  # the whole point: let the difference be nonzero
    )

    qp, qm = q_plus[sel], q_minus[sel]
    usable = (qp > eps) & (qm > eps)
    var = torch.where(
        usable, (qp + qm) / (qp * qm).clamp_min(eps), torch.full_like(qp, float("inf"))
    )

    z = delta / var.sqrt()
    z = z[torch.isfinite(z)]
    n_usable = int(usable.sum())
    if n_usable == 0:
        # every centric pair was seen under one symmetry-operation parity only,
        # so the split has nothing to compare and the anchor cannot be formed
        return {
            "n": n,
            "n_usable": 0,
            "z_sd": float("nan"),
            "z_median_abs": float("nan"),
            "tau": torch.zeros(n_bins, dtype=delta.dtype, device=delta.device),
            "median_abs_fraction": float("nan"),
        }
    tau = update_tau(
        (delta / (2.0 * j.clamp_min(eps)))[usable],
        (var / (4.0 * j.clamp_min(eps).pow(2)))[usable],
        bin_index[sel][usable],
        n_bins,
    )
    return {
        "n": n,
        "n_usable": n_usable,
        "z_sd": float(z.std()) if z.numel() > 1 else float("nan"),
        "z_median_abs": float(z.abs().median()) if z.numel() else float("nan"),
        "tau": tau,
        "median_abs_fraction": float(
            (delta[usable].abs() / (2.0 * j[usable].clamp_min(eps))).median()
        )
        if n_usable
        else float("nan"),
    }


def solve_scale_factors(
    signal_a: Tensor,
    signal_b: Tensor,
    table: Tensor,
    hyper: MergeHyperparameters,
    n_iter: int = 30,
    eps: float = 1e-9,
) -> tuple[Tensor, Tensor]:
    """Two multiplicative scale factors from a joint table, without the data.

    The scale is a product, `S_i = A_{a(i)} B_{b(i)}`, and each factor's Gamma
    conditional has the same shape as a one-factor scale:

        A_a = (alpha + sum_{i in a} s_i) / (beta + sum_{i in a} J_i B_{b(i)})

    The denominator needs `J` summed per `(a, b)` cell, nothing finer, so one
    streaming pass can accumulate `table[a, b] = sum J_i` and the two factors
    can then be cycled to convergence on that table alone. For a wavelength
    grouping against an image grouping the table is a few tens of thousands of
    entries, so the whole scale block costs nothing after the pass.

    Cycling rather than updating both at once matters: the system is bilinear,
    and simultaneous updates oscillate where alternating ones converge.

    Returns `(A, B)`, gauge-fixed so that `B` has mean one; the constant is
    moved into `A`, and the product's own gauge is left to the caller.
    """
    a = torch.ones(table.shape[0], dtype=table.dtype, device=table.device)
    b = torch.ones(table.shape[1], dtype=table.dtype, device=table.device)
    for _ in range(max(1, n_iter)):
        a = (hyper.alpha_scale + signal_a) / (
            hyper.beta_scale + table @ b
        ).clamp_min(eps)
        b = (hyper.alpha_scale + signal_b) / (
            hyper.beta_scale + table.transpose(0, 1) @ a
        ).clamp_min(eps)
        c = b.mean().clamp_min(eps)
        b = b / c
        a = a * c
    return a, b


# --------------------------------------------------------- Chebyshev scale


def chebyshev_design(
    values: Tensor, degree: int, lo: float | None = None, hi: float | None = None
) -> Tensor:
    """Chebyshev basis `T_0..T_degree` on values mapped to `[-1, 1]`, `(N, d+1)`.

    The recurrence is `ChebyshevSpectrum._chebyshev`; it is repeated here only
    so this module does not import the SGD scaling path, which wraps the basis
    in a softplus.
    """
    lo = float(values.min()) if lo is None else float(lo)
    hi = float(values.max()) if hi is None else float(hi)
    mid, half = 0.5 * (lo + hi), max(0.5 * (hi - lo), 1e-12)
    x = ((values - mid) / half).clamp(-1.0, 1.0)
    t = [torch.ones_like(x)]
    if degree >= 1:
        t.append(x)
    for k in range(2, degree + 1):
        t.append(2.0 * x * t[k - 1] - t[k - 2])
    return torch.stack(t, dim=-1)


def tensor_design(phi_a: Tensor, phi_b: Tensor) -> Tensor:
    """Row-wise outer product of two bases, `(N, Ka*Kb)`."""
    return (phi_a.unsqueeze(-1) * phi_b.unsqueeze(-2)).reshape(phi_a.shape[0], -1)


def accumulate_scale_normals(
    counts: Tensor,
    mask: Tensor,
    profile: Tensor,
    merged_of_obs: Tensor,
    background: Tensor,
    scale: Tensor,
    design: Tensor,
    eps: float = 1e-9,
) -> tuple[Tensor, Tensor]:
    """One batch's contribution to the normal equations of a LINEAR scale.

    With `S_i = phi_i . c` the rate `(phi_i . c) a_ij + B_i` is linear in `c`,
    where `a_ij = J_h p_ij`, so the Poisson log-likelihood is concave in `c` and
    IRLS is a weighted least squares. The existing `ChebyshevScale` applies a
    softplus to the same basis, which makes the rate convex in the coefficients
    and gives that up.

    The basis depends on the observation and not the pixel, so the pixel sums
    collapse to two scalars per observation before the outer product is formed:
    the whole block is `phi^T diag(q_aa) phi` and `phi^T q_ab`, whatever the
    shoebox size.

    Returns `(K, K)` and `(K,)` to be summed across batches.
    """
    b = counts.shape[0]
    m = mask.reshape(b, -1).to(profile.dtype)
    c = counts.reshape(b, -1).to(profile.dtype) * m
    a = merged_of_obs.unsqueeze(-1) * profile.reshape(b, -1)
    rate = (scale.unsqueeze(-1) * a + background.unsqueeze(-1)).clamp_min(eps)
    w = m / rate

    q_aa = (w * a * a).sum(-1)
    q_ab = (w * a * (c - background.unsqueeze(-1) * m)).sum(-1)
    return design.transpose(0, 1) @ (q_aa.unsqueeze(-1) * design), design.transpose(
        0, 1
    ) @ q_ab


def solve_normals(
    a: Tensor, b: Tensor, ridge: float = 1e-8, fallback: Tensor | None = None
) -> Tensor:
    """Solve `A c = b`, Jacobi-scaled, falling back when it is singular.

    The basis columns differ in scale, so the damping is applied on the unit
    diagonal rather than against raw entries, for the same reason
    `solve_profile_fit` does it.
    """
    work = torch.float64
    a64, b64 = a.to(work), b.to(work)
    dg = a64.diagonal().clamp_min(torch.finfo(work).tiny)
    s = dg.rsqrt()
    scaled = s.unsqueeze(-1) * a64 * s.unsqueeze(-2)
    scaled = scaled + ridge * torch.eye(
        scaled.shape[0], dtype=work, device=a.device
    )
    try:
        c = s * torch.linalg.solve(scaled, s * b64)
    except Exception:
        return b if fallback is None else fallback
    if not torch.isfinite(c).all():
        return b if fallback is None else fallback
    return c.to(b.dtype)


def scale_design(
    image_u: Tensor,
    s2: Tensor,
    degree_image: int = 6,
    degree_s2: int = 3,
) -> tuple[Tensor, list[tuple[int, int]]]:
    """Identifiable Chebyshev scale design in (image, `s^2`), `(N, K)`.

    **The columns constant in image and varying in `s^2` are dropped.** `s^2`
    is a property of the Miller index, so a scale depending on it alone is
    exactly degenerate with `J`: the solve wanders along it, and on planted
    data it came back inverted -- fitted 0.794/1.206 against a true 0.513/0.157.
    A resolution ramp common to every image is also absorbed by refinement's
    own overall B, so it costs nothing to leave unmodelled.

    What remains identifiable is how that ramp VARIES between images, which
    refinement cannot absorb because it fits one B for the whole dataset.
    Measured on hewl_1118 the per-image slope of `log r` against `s^2` has mean
    -37.1 and sd 7.1 across 992 images, so 81% of the ramp is common gauge and
    19% is real.

    Returns the design and the `(j, k)` degree pair behind each kept column.
    """
    phi_i = chebyshev_design(image_u, degree_image)
    phi_s = chebyshev_design(s2, degree_s2)
    full = tensor_design(phi_i, phi_s)
    keep, terms = [], []
    for j in range(degree_image + 1):
        for k in range(degree_s2 + 1):
            if j == 0 and k > 0:
                continue  # constant in image, varying in s^2: pure gauge
            keep.append(j * (degree_s2 + 1) + k)
            terms.append((j, k))
    return full[:, keep], terms
