"""The full pixels-to-structure-factors sweep, driven over a dataset.

One pass of the loop is:

  E     responsibilities from the current `(S, J, Delta, B)`, which splits every
        count into signal and background;
  obs   per-observation background, in closed form from its own split;
  pair  `(J_h, Delta_h)` from the per-mate signal and scale totals, a 2-D
        concave solve;
  group `S_g` from the same totals pooled over an image, a Gamma mean;
  shell `Sigma_d` by moment matching, and the gauge pinned at `mean(S) = 1`.

Nothing here takes a gradient step. The state is small -- one background per
observation, two numbers per Miller pair, one per image -- so the cost is the
streaming pass over the shoeboxes, which the responsibilities need.

`Delta` is solved WITHOUT its prior throughout and shrunk once at the end.
Re-fitting `tau` from an already shrunk difference drives it to zero, and the
difference is a few per cent of the intensity, so leaving it unshrunk inside the
loop changes the responsibilities negligibly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch import Tensor

from .merge import (
    MergeHyperparameters,
    accumulate_scale_normals,
    anomalous_split,
    centric_anchor,
    fit_and_shrink_difference,
    observation_stats,
    responsibilities,
    shrink_difference,
    solve_normals,
    solve_scale_factors,
    split_by_mate,
    update_background,
    update_merged_anomalous,
    update_tau,
    update_wilson_sigma,
    wilson_prior,
)

_logger = logging.getLogger(__name__)

__all__ = [
    "SweepResult",
    "SweepState",
    "equal_count_bins",
    "friedel_pseudo_mate",
    "pair_index",
    "run_sweep",
]


def pair_index(hkl, spacegroup) -> tuple[Tensor, Tensor, Tensor, int]:
    """Friedel-pooled group id, mate sign, and centric flag per observation.

    The pair id pools Friedel mates so one Miller pair carries both `F(+)` and
    `F(-)`; the sign says which side an observation sits on. A centric
    reflection is its own mate and is marked so the difference can be held at
    zero -- negating it would split one reflection into two ids and invent an
    anomalous difference that cannot exist.
    """
    import numpy as np
    import reciprocalspaceship as rs

    from integrator.utils.asu import canonical_hkl

    h = np.asarray(hkl, dtype=np.int32)
    pooled = canonical_hkl(h, spacegroup, anomalous=False)
    _, isym = rs.utils.hkl_to_asu(h, spacegroup)
    centric = np.asarray(rs.utils.is_centric(pooled, spacegroup), dtype=bool)

    _, pair = np.unique(pooled, axis=0, return_inverse=True)
    pair = np.asarray(pair, dtype=np.int64).ravel()
    mate = np.where((isym % 2 == 1) | centric, 1.0, -1.0)

    n_pair = int(pair.max()) + 1
    cen_pair = np.zeros(n_pair, dtype=bool)
    cen_pair[pair] = centric
    return (
        torch.from_numpy(pair),
        torch.from_numpy(mate).float(),
        torch.from_numpy(cen_pair),
        n_pair,
    )


def friedel_pseudo_mate(hkl, spacegroup) -> Tensor:
    """Mate sign from the symmetry-operation parity alone, centrics included.

    `pair_index` marks every centric observation as `+1`, because a centric is
    its own Friedel mate and splitting it would invent a difference that cannot
    exist. This function does not apply that override, so a centric pair is
    split into two physically identical groups that nevertheless sit at
    different detector positions, wavelengths and images. The difference the
    pair block then recovers is the scale model's doing; `centric_anchor`
    measures it.

    For an acentric reflection this returns exactly the real mate.
    """
    import numpy as np
    import reciprocalspaceship as rs

    h = np.asarray(hkl, dtype=np.int32)
    _, isym = rs.utils.hkl_to_asu(h, spacegroup)
    return torch.from_numpy(np.where(isym % 2 == 1, 1.0, -1.0)).float()


def equal_count_bins(d: Tensor, n_bins: int) -> Tensor:
    """Resolution bin per element, equal count, 0 at low resolution."""
    order = torch.argsort(d, descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(d.numel(), device=d.device)
    return (rank * n_bins // max(d.numel(), 1)).clamp_max(n_bins - 1)


@dataclass
class SweepState:
    """Everything the sweep carries between passes."""

    merged: Tensor
    delta: Tensor
    scale: Tensor
    background: Tensor
    sigma_d: Tensor
    spectrum: Tensor | None = None
    cheb_coeff: Tensor | None = None
    history: list[dict] = field(default_factory=list)


@dataclass
class SweepResult:
    state: SweepState
    delta_shrunk: Tensor
    var_merged: Tensor
    var_delta: Tensor
    tau: Tensor
    anchor: dict | None = None
    # present only with `split_anomalous`: the Poisson-split quantities, which
    # are exact rather than Fisher approximations. `frac` is
    # `(I+ - I-)/(I+ + I-)`, `theta_null` the split a centric must show, and
    # `null_z` the standardized deviation from it -- an exact Binomial test
    # where the anchor's `z_sd` is a heuristic read against a planted baseline.
    frac: Tensor | None = None
    var_frac: Tensor | None = None
    theta_null: Tensor | None = None
    null_z: Tensor | None = None


def run_sweep(
    batches,
    *,
    profile_fn,
    pair: Tensor,
    mate: Tensor,
    group: Tensor,
    bin_index: Tensor,
    centric_pair: Tensor,
    n_pair: int,
    n_groups: int,
    n_bins: int,
    n_obs: int,
    n_sweeps: int = 15,
    lambda_bin: Tensor | None = None,
    n_lambda: int = 0,
    cheb_design: Tensor | None = None,
    pseudo_mate: Tensor | None = None,
    hyper: MergeHyperparameters | None = None,
    tau_scale: float | None = None,
    tau_prob: float = 0.01,
    split_anomalous: bool = False,
    device=None,
    log_every: int = 1,
) -> SweepResult:
    """Run the block sweep to convergence and return the merged quantities.

    `batches` is a callable returning a fresh iterator of
    `(obs_index, counts, mask, metadata)` each time it is called, because the
    loop makes one streaming pass per sweep. `obs_index` places each row in the
    per-observation state, so the loader does not have to be order-stable.

    `profile_fn(metadata, mask)` returns the normalized profile; it is a pure
    function of geometry here, so it is recomputed per batch rather than cached.
    """
    hyper = hyper or MergeHyperparameters()
    dev = device or pair.device
    dt = torch.float32

    pair, mate = pair.to(dev), mate.to(dev).to(dt)
    group, bin_index = group.to(dev), bin_index.to(dev)
    centric_pair = centric_pair.to(dev)

    state = SweepState(
        merged=torch.ones(n_pair, device=dev, dtype=dt),
        delta=torch.zeros(n_pair, device=dev, dtype=dt),
        scale=torch.ones(n_groups, device=dev, dtype=dt),
        background=torch.ones(n_obs, device=dev, dtype=dt),
        sigma_d=torch.ones(n_bins, device=dev, dtype=dt),
    )
    use_spectrum = lambda_bin is not None and n_lambda > 0
    if use_spectrum:
        lambda_bin = lambda_bin.to(dev)
        state.spectrum = torch.ones(n_lambda, device=dev, dtype=dt)
    # A smooth scale in (image, s^2) replaces the free per-image scalar. It is
    # LINEAR in its coefficients, so the rate stays linear in them and the block
    # is a weighted least squares; the repo's ChebyshevScale applies a softplus
    # to the same basis and gives that up. It also carries the resolution
    # falloff, which the per-image scalar cannot represent at all.
    use_cheb = cheb_design is not None
    if use_cheb:
        cheb_design = cheb_design.to(dev).to(dt)
        state.cheb_coeff = torch.zeros(cheb_design.shape[1], device=dev, dtype=dt)
        state.cheb_coeff[0] = 1.0
    free_tau = torch.full((n_pair,), float("inf"), device=dev, dtype=dt)
    seeded = False

    for sweep in range(n_sweeps):
        sig_p = torch.zeros(n_pair, device=dev, dtype=dt)
        sig_m = torch.zeros_like(sig_p)
        scl_p = torch.zeros_like(sig_p)
        scl_m = torch.zeros_like(sig_p)
        sig_g = torch.zeros(n_groups, device=dev, dtype=dt)
        j_g = torch.zeros_like(sig_g)
        if use_spectrum:
            sig_l = torch.zeros(n_lambda, device=dev, dtype=dt)
            cnt_l = torch.zeros(n_lambda, device=dev, dtype=dt)
            den_l = torch.zeros(n_lambda, device=dev, dtype=dt)
            table = torch.zeros(n_lambda, n_groups, device=dev, dtype=dt)
        if use_cheb:
            k = cheb_design.shape[1]
            amat = torch.zeros(k, k, device=dev, dtype=dt)
            bvec = torch.zeros(k, device=dev, dtype=dt)
            phi_sum = torch.zeros(k, device=dev, dtype=dt)
            n_seen = 0

        for obs_idx, counts, mask, md in batches():
            obs_idx = obs_idx.to(dev)
            counts = counts.to(dev).to(dt)
            mask = mask.to(dev)
            prof = profile_fn(md, mask).to(dt)

            p_i, m_i = pair[obs_idx], mate[obs_idx]
            g_i = group[obs_idx]
            s_base = (
                cheb_design[obs_idx] @ state.cheb_coeff
                if use_cheb
                else state.scale[g_i]
            ).clamp_min(1e-4)
            s_i = s_base
            if use_spectrum:
                s_i = s_i * state.spectrum[lambda_bin[obs_idx]]
            j_i = (state.merged[p_i] + 0.5 * m_i * state.delta[p_i]).clamp_min(1e-6)

            if not seeded:
                # one median per shoebox, so the first responsibility is formed
                # from something data-driven rather than from `background = 1`
                mm = mask.reshape(counts.shape[0], -1).to(dt)
                b0 = counts.reshape(counts.shape[0], -1).masked_fill(
                    mm <= 0, float("nan")
                ).nanmedian(dim=-1).values
                state.background[obs_idx] = torch.nan_to_num(b0, nan=0.0).clamp_min(
                    1e-3
                )
                flat = ((counts.reshape(counts.shape[0], -1) * mm).sum(-1)
                        - state.background[obs_idx] * mm.sum(-1))
                state.merged.scatter_reduce_(
                    0, p_i, flat.clamp_min(1.0), reduce="mean", include_self=False
                )

            r = responsibilities(prof, s_i, j_i, state.background[obs_idx])
            stats = observation_stats(counts, mask, r)
            state.background[obs_idx] = update_background(stats, hyper).to(dt)

            sp, sm, ap, am = split_by_mate(
                stats.signal.clamp_min(0.0).to(dt), s_i, p_i, m_i, n_pair
            )
            sig_p += sp
            sig_m += sm
            scl_p += ap
            scl_m += am
            sig_obs = stats.signal.clamp_min(0.0).to(dt)
            sig_g.scatter_add_(0, g_i, sig_obs)
            j_g.scatter_add_(0, g_i, j_i)
            if use_spectrum:
                l_i = lambda_bin[obs_idx]
                sig_l.scatter_add_(0, l_i, sig_obs)
                cnt_l.scatter_add_(0, l_i, torch.ones_like(sig_obs))
                den_l.scatter_add_(0, l_i, j_i * s_base)
                table.view(-1).scatter_add_(0, l_i * n_groups + g_i, j_i)
            if use_cheb:
                # G(lambda) folded into the design column, so the solve is for
                # the Chebyshev coefficients alone
                gl = (
                    state.spectrum[lambda_bin[obs_idx]]
                    if use_spectrum
                    else torch.ones_like(j_i)
                )
                da, db = accumulate_scale_normals(
                    counts, mask, prof, j_i * gl, state.background[obs_idx],
                    s_base, cheb_design[obs_idx],
                )
                amat += da
                bvec += db
                phi_sum += cheb_design[obs_idx].sum(0)
                n_seen += int(obs_idx.numel())

        seeded = True
        _, beta_w = wilson_prior(
            state.sigma_d[bin_index].clamp_min(1e-6), centric_pair
        )
        if split_anomalous:
            # Poisson thinning: a Gamma posterior for the total and a 1-D
            # concave Binomial for the split, in place of the 2x2 Newton step
            # whose Hessian is rank-1 when a mate is missing. `free_tau` is
            # infinite here, so the split is the likelihood's alone and the
            # prior is applied once, after the loop, on the fractional scale.
            sp_out = anomalous_split(
                sig_p, sig_m, scl_p, scl_m, beta_w, free_tau,
                centric=centric_pair,
            )
            state.merged, state.delta = sp_out["j"], sp_out["delta"]
        else:
            state.merged, state.delta = update_merged_anomalous(
                sig_p, sig_m, scl_p, scl_m, beta_w, free_tau, centric=centric_pair
            )
        if use_cheb:
            state.cheb_coeff = solve_normals(amat, bvec, fallback=state.cheb_coeff)
            if use_spectrum:
                state.spectrum = (
                    (hyper.alpha_scale + sig_l)
                    / (hyper.beta_scale + den_l).clamp_min(hyper.eps)
                )
            # pin the gauge on the mean scale, exactly: mean(phi . c) over the
            # observations is (sum phi) . c, so no second pass is needed
            mean_s = (phi_sum @ state.cheb_coeff) / max(n_seen, 1)
            if use_spectrum:
                w = cnt_l / cnt_l.sum().clamp_min(1.0)
                mean_s = mean_s * (state.spectrum * w).sum()
                state.spectrum = state.spectrum / (state.spectrum * w).sum().clamp_min(
                    1e-12
                )
            mean_s = mean_s.clamp_min(1e-12)
            state.cheb_coeff = state.cheb_coeff / mean_s
            state.merged = state.merged * mean_s
            state.delta = state.delta * mean_s
            state.scale = torch.ones(n_groups, device=dev, dtype=dt)
        elif use_spectrum:
            # the per-image scale and the spectrum are cycled on the joint
            # (wavelength, image) table rather than re-read from the data.
            # `solve_scale_factors` returns the row factor first and pins the
            # column factor's mean at one, so the per-image scale comes back
            # already gauge-fixed and the constant sits in the spectrum.
            state.spectrum, state.scale = solve_scale_factors(
                sig_l, sig_g, table, hyper
            )
            w = cnt_l / cnt_l.sum().clamp_min(1.0)
            c_s = (state.spectrum * w).sum().clamp_min(1e-12)
            state.spectrum = state.spectrum / c_s
            state.merged = state.merged * c_s
            state.delta = state.delta * c_s
        else:
            state.scale = (hyper.alpha_scale + sig_g) / (
                hyper.beta_scale + j_g
            ).clamp_min(hyper.eps)

        # the per-image gauge. With the Chebyshev scale the per-image factor is
        # held at one and the gauge was already pinned above, so this is a
        # no-op there rather than a second, competing constraint.
        c = state.scale.mean().clamp_min(1e-12)
        state.scale = state.scale / c
        state.merged = state.merged * c
        state.delta = state.delta * c
        state.sigma_d = update_wilson_sigma(state.merged, bin_index, n_bins)

        if log_every and (sweep + 1) % log_every == 0:
            state.history.append(
                {
                    "sweep": sweep + 1,
                    "median_J": float(state.merged.median()),
                    "median_abs_delta": float(state.delta.abs().median()),
                    "mean_scale": float(state.scale.mean()),
                    "median_background": float(state.background.median()),
                }
            )
            _logger.info(
                "sweep %2d | median J %10.2f | median |Delta| %8.3f | "
                "median background %7.3f",
                sweep + 1,
                state.history[-1]["median_J"],
                state.history[-1]["median_abs_delta"],
                state.history[-1]["median_background"],
            )

    # one more pass for the observed-data information, and for the centric
    # anchor, which needs the same totals split by symmetry-operation parity
    q_num = torch.zeros(n_pair, device=dev, dtype=torch.float64)
    q_den = torch.zeros_like(q_num)
    anchor_acc = None
    if pseudo_mate is not None:
        pseudo_mate = pseudo_mate.to(dev).to(dt)
        anchor_acc = [torch.zeros(n_pair, device=dev, dtype=dt) for _ in range(6)]

    for obs_idx, counts, mask, md in batches():
        obs_idx = obs_idx.to(dev)
        counts = counts.to(dev).to(dt)
        mask = mask.to(dev)
        prof = profile_fn(md, mask).to(dt)
        p_i, m_i = pair[obs_idx], mate[obs_idx]
        s_i = state.scale[group[obs_idx]]
        if use_spectrum:
            s_i = s_i * state.spectrum[lambda_bin[obs_idx]]
        j_i = (state.merged[p_i] + 0.5 * m_i * state.delta[p_i]).clamp_min(1e-6)
        b_i = state.background[obs_idx]

        q = _observation_information(prof, mask, s_i, j_i, b_i)
        plus = (m_i > 0).to(torch.float64)
        q_num.scatter_add_(0, p_i, q.to(torch.float64) * plus)
        q_den.scatter_add_(0, p_i, q.to(torch.float64) * (1 - plus))

        if anchor_acc is not None:
            r = responsibilities(prof, s_i, j_i, b_i)
            sig = observation_stats(counts, mask, r).signal.clamp_min(0.0).to(dt)
            pm = pseudo_mate[obs_idx]
            sp, sm, ap, am = split_by_mate(sig, s_i, p_i, pm, n_pair)
            pplus = (pm > 0).to(dt)
            qp = torch.zeros(n_pair, device=dev, dtype=dt)
            qm = torch.zeros_like(qp)
            qp.scatter_add_(0, p_i, q.to(dt) * pplus)
            qm.scatter_add_(0, p_i, q.to(dt) * (1 - pplus))
            for acc, add in zip(anchor_acc, (sp, sm, ap, am, qp, qm), strict=False):
                acc += add

    both = (q_num > 0) & (q_den > 0)
    total, prodq = q_num + q_den, q_num * q_den
    inf = torch.full_like(total, float("inf"))
    # A pair seen under one mate only determines `J + Delta/2` and not the two
    # separately, so the DIFFERENCE is genuinely unmeasured and stays infinite,
    # which `shrink_difference` turns into the base model. But `Delta` is then
    # pinned at zero, and with it pinned the single side does determine `J`
    # with variance `1/Q`. Returning infinity for both instead drops every
    # one-sided pair from the merge, which on hewl_1118 cut the data from
    # 1.5 A to 1.81 A and 11,488 pairs to 9,548 -- most of the outer shells.
    var_j = torch.where(
        both,
        total / (4.0 * prodq.clamp_min(1e-30)),
        torch.where(total > 0, 1.0 / total.clamp_min(1e-30), inf),
    ).to(dt)
    var_d = torch.where(both, total / prodq.clamp_min(1e-30), inf).to(dt)

    frac_out = var_frac_out = theta_null_out = null_z_out = None
    if split_anomalous:
        # The split already works on the fractional scale, so the prior goes on
        # directly: no conversion from an absolute `Delta` and back, which is
        # where the absolute-versus-fractional mistake lived. `var_frac` is the
        # exact Binomial variance; `var_d` from the information pass is used
        # only to report `Var(Delta)`, because it charges for the background
        # where the Binomial conditions on it.
        _, beta_w = wilson_prior(
            state.sigma_d[bin_index].clamp_min(1e-6), centric_pair
        )
        sp_out = anomalous_split(
            sig_p, sig_m, scl_p, scl_m, beta_w,
            torch.full_like(state.merged, float("inf")), centric=centric_pair,
        )
        tau = update_tau(
            sp_out["frac"], sp_out["var_frac"].clamp_min(1e-12), bin_index,
            n_bins, tau_scale=tau_scale, tau_prob=tau_prob,
        )
        frac_shrunk, var_frac_post = shrink_difference(
            sp_out["frac"], sp_out["var_frac"].clamp_min(1e-12), tau[bin_index]
        )
        delta_shrunk = 2.0 * state.merged * frac_shrunk
        var_delta_post = (2.0 * state.merged).pow(2) * var_frac_post
        frac_out, var_frac_out = frac_shrunk, var_frac_post
        theta_null_out = sp_out["theta_null"]
        t, th = sp_out["total"], sp_out["theta_null"]
        null_z = (sig_p - t * th) / (
            t * th * (1.0 - th)
        ).clamp_min(1e-9).sqrt()
        null_z_out = null_z
        _logger.info(
            "Binomial null on centrics: n=%d | standardized sd %.3f "
            "(exactly 1 when the scale invents nothing)",
            int(centric_pair.sum()),
            float(null_z[centric_pair].std()) if bool(centric_pair.any()) else float("nan"),
        )
    else:
        delta_shrunk, var_delta_post, tau = fit_and_shrink_difference(
            state.delta, var_d, state.merged, bin_index, n_bins,
            tau_scale=tau_scale, tau_prob=tau_prob,
        )

    anchor = None
    if anchor_acc is not None:
        _, beta_w = wilson_prior(
            state.sigma_d[bin_index].clamp_min(1e-6), centric_pair
        )
        anchor = centric_anchor(
            anchor_acc[0], anchor_acc[1], anchor_acc[2], anchor_acc[3],
            anchor_acc[4], anchor_acc[5], beta_w, centric_pair,
            bin_index, n_bins,
        )
        # tau_centric is the absolute instrument: it is zero when the scale
        # manufactures nothing, and it is on the same fractional scale as the
        # acentric tau, so the ratio says how much of the apparent anomalous
        # signal could be scale artifact. The standardized sd is relative --
        # its no-artifact baseline is 0.64 on planted data, not 1 -- so it is
        # read against a run with the scale change removed, not against unity.
        ratio = [
            float(c) / float(t) if float(t) > 0 else float("nan")
            for c, t in zip(anchor["tau"], tau, strict=False)
        ]
        anchor["tau_acentric"] = tau.detach().cpu()
        anchor["fraction_of_tau"] = ratio
        _logger.info(
            "centric anchor: n=%d usable=%d | standardized sd %.3f | "
            "tau_centric %s | as a fraction of tau_acentric %s",
            anchor["n"], anchor["n_usable"], anchor["z_sd"],
            " ".join(f"{float(x):.4f}" for x in anchor["tau"]),
            " ".join("nan" if x != x else f"{x:.2f}" for x in ratio),
        )

    return SweepResult(
        state=state,
        delta_shrunk=delta_shrunk,
        var_merged=var_j,
        var_delta=var_delta_post,
        tau=tau,
        anchor=anchor,
        frac=frac_out,
        var_frac=var_frac_out,
        theta_null=theta_null_out,
        null_z=null_z_out,
    )


def _observation_information(
    profile: Tensor,
    mask: Tensor,
    scale: Tensor,
    merged_of_obs: Tensor,
    background: Tensor,
    eps: float = 1e-9,
) -> Tensor:
    """One observation's information about its own intensity, background paid for."""
    m = mask.reshape(profile.shape[0], -1).to(profile.dtype)
    rate = (scale * merged_of_obs).unsqueeze(-1) * profile + background.unsqueeze(-1)
    inv = m / rate.clamp_min(eps)
    g = scale.unsqueeze(-1) * profile
    f_jj = (inv * g * g).sum(-1)
    f_jb = (inv * g).sum(-1)
    f_bb = inv.sum(-1).clamp_min(eps)
    return (f_jj - f_jb * f_jb / f_bb).clamp_min(0.0)
