"""The anomalous block: a common intensity and a Bijvoet difference, in closed form.

The parameterization is `J = (J+ + J-)/2` and `Delta = J+ - J-`, which keeps the
Poisson rate linear in both unknowns and so keeps the block concave. The prior
sits on the COMMON intensity, with a per-shell Gaussian on the difference --
shrinking the two mates separately toward Wilson is what destroys the anomalous
signal, and a GLOBAL difference scale zeroes it everywhere because the
per-reflection Bijvoet signal-to-noise averaged over all acentrics is below one.

What has to hold:

  - a planted `Delta` comes back, and a planted zero stays zero;
  - the reported `Var(Delta)` is four times `Var(J)` and diverges when a mate is
    unobserved, rather than quietly inheriting the prior;
  - the per-shell scale finds signal in the shell that has it and shrinks the
    shell that does not, which is the property a global scale cannot have;
  - centric reflections are held at `Delta = 0`.
"""

import pytest
import torch

from integrator.fit.merge import (
    anomalous_fisher_variance,
    fit_and_shrink_difference,
    observation_stats,
    responsibilities,
    shrink_difference,
    split_by_mate,
    update_merged_anomalous,
    update_tau,
)

DT = torch.float64


def two_pass(d, tau_scale=None):
    """Solve without the difference prior, fit `tau` per shell, then shrink.

    Empirical Bayes reads `tau` off the marginal of the UNSHRUNK estimate, so
    the difference has to be solved for once with no prior on it. Feeding an
    already-shrunk difference back in drives `tau` to zero.
    """
    var = anomalous_fisher_variance(
        d["profile"], d["mask"], d["scale"], d["j_true"][d["index"]],
        d["background"], d["index"], d["mate"], d["n_h"],
    )[1]
    free = torch.full((d["n_h"],), float("inf"), dtype=DT)
    _, delta_ml = reduce_and_solve(d, free)
    j, _ = reduce_and_solve(d, free)
    delta, var_post, tau = fit_and_shrink_difference(
        delta_ml, var, j, d["bin_index"], d["n_bins"], tau_scale=tau_scale
    )
    return dict(
        delta_ml=delta_ml, delta=delta, var=var, var_post=var_post, tau=tau,
        merged=j,
    )


def planted_pairs(
    n_h=300, mult=6, j_mean=2000.0, frac=0.06, seed=0, shells=None, hw=13, bg=10.0
):
    """Miller pairs with a planted fractional Bijvoet difference.

    `frac` is the ratio `Delta / J`, the size anomalous differences actually
    have: a few per cent. `shells` gives a per-shell fraction instead, so a
    dataset can carry signal in one shell and none in another.
    """
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n_h, generator=g, dtype=DT).clamp(1e-9, 1 - 1e-9)
    j_true = (-j_mean * torch.log1p(-u)).clamp_min(0.05 * j_mean)

    n_bins = 1 if shells is None else len(shells)
    bin_index = torch.arange(n_h) % n_bins
    f = (
        torch.full((n_h,), float(frac), dtype=DT)
        if shells is None
        else torch.tensor([float(s) for s in shells], dtype=DT)[bin_index]
    )
    delta_true = f * j_true * torch.sign(
        torch.randn(n_h, generator=g, dtype=DT)
    ) * torch.rand(n_h, generator=g, dtype=DT).add(0.5)

    n = n_h * mult
    index = torch.arange(n_h).repeat_interleave(mult)
    mate = torch.where(
        torch.arange(n) % 2 == 0, torch.ones(n, dtype=DT), -torch.ones(n, dtype=DT)
    )
    scale = torch.ones(n, dtype=DT)

    j_obs = j_true[index] + 0.5 * mate * delta_true[index]
    yy, xx = torch.meshgrid(
        torch.arange(hw, dtype=DT), torch.arange(hw, dtype=DT), indexing="ij"
    )
    c = (hw - 1) / 2.0
    p = torch.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * 1.6**2)).reshape(-1)
    p = (p / p.sum()).unsqueeze(0).expand(n, -1).contiguous()
    counts = torch.poisson(
        (j_obs.unsqueeze(-1) * p + bg).float(), generator=g
    ).to(DT)

    return dict(
        counts=counts, mask=torch.ones_like(counts), profile=p, index=index,
        mate=mate, scale=scale, n_h=n_h, j_true=j_true, delta_true=delta_true,
        bin_index=bin_index, n_bins=n_bins, background=torch.full((n,), bg, dtype=DT),
    )


def reduce_and_solve(d, tau, centric=None, beta_w=None, n_iter=40, n_outer=25):
    """Iterate responsibilities and the 2-D concave solve: the actual sweep.

    The responsibilities must be formed from the CURRENT estimates, not from the
    truth and not once. One pass with exact responsibilities is the gated
    estimator `sum(r c)`, whose variance is `sum(r^2 lambda)` and which is not
    the maximum-likelihood profile fit; iterating is EM, and EM's fixed point
    solves the same score equation as the profile fit, so its scatter is the one
    `anomalous_fisher_variance` reports. Measured with the one-pass version the
    two disagreed by a factor 0.74 and shrinkage looked harmful.
    """
    if beta_w is None:
        beta_w = torch.full((d["n_h"],), 1.0 / float(d["j_true"].mean()), dtype=DT)
    n_pix = d["mask"].sum(-1)
    j = ((d["counts"] * d["mask"]).sum(-1) - d["background"] * n_pix).clamp_min(1.0)
    j = torch.zeros(d["n_h"], dtype=DT).scatter_add_(0, d["index"], j)
    j = (j / torch.zeros(d["n_h"], dtype=DT).scatter_add_(
        0, d["index"], torch.ones_like(d["index"], dtype=DT)
    )).clamp_min(1.0)
    delta = torch.zeros_like(j)

    for _ in range(n_outer):
        j_obs = (j[d["index"]] + 0.5 * d["mate"] * delta[d["index"]]).clamp_min(1e-6)
        resp = responsibilities(d["profile"], d["scale"], j_obs, d["background"])
        stats = observation_stats(d["counts"], d["mask"], resp)
        sp, sm, ap, am = split_by_mate(
            stats.signal.clamp_min(0.0), d["scale"], d["index"], d["mate"], d["n_h"]
        )
        j, delta = update_merged_anomalous(
            sp, sm, ap, am, beta_w, tau, centric=centric, n_iter=n_iter
        )
    return j, delta


# --------------------------------------------------------------- recovery


def test_planted_bijvoet_difference_is_recovered():
    d = planted_pairs(n_h=400, mult=8, frac=0.08, seed=1)
    tau = torch.full((d["n_h"],), float((0.08 * d["j_true"]).mean()), dtype=DT)
    j, delta = reduce_and_solve(d, tau)

    rj = float(torch.corrcoef(torch.stack([j, d["j_true"]]))[0, 1])
    rd = float(torch.corrcoef(torch.stack([delta, d["delta_true"]]))[0, 1])
    assert rj > 0.999, f"corr(J, truth) {rj:.4f}"
    assert rd > 0.80, f"corr(Delta, truth) {rd:.4f}"


def test_no_planted_difference_gives_no_recovered_difference():
    """The base model must be reachable, or every dataset grows an anomalous signal."""
    d = planted_pairs(n_h=400, mult=8, frac=0.0, seed=2)
    var = anomalous_fisher_variance(
        d["profile"], d["mask"], d["scale"], d["j_true"][d["index"]],
        d["background"], d["index"], d["mate"], d["n_h"],
    )[1]
    tau = update_tau(
        torch.zeros(d["n_h"], dtype=DT), var, d["bin_index"], d["n_bins"]
    )
    j, delta = reduce_and_solve(d, tau[d["bin_index"]].clamp_min(1e-6))
    scatter = float(delta.std() / d["j_true"].mean())
    assert scatter < 0.02, f"spurious Delta/J scatter {scatter:.4f}"


def test_centric_reflections_are_held_at_zero():
    d = planted_pairs(n_h=200, mult=6, frac=0.08, seed=3)
    centric = torch.zeros(d["n_h"], dtype=torch.bool)
    centric[::3] = True
    tau = torch.full((d["n_h"],), float((0.08 * d["j_true"]).mean()), dtype=DT)
    _, delta = reduce_and_solve(d, tau, centric=centric)
    assert float(delta[centric].abs().max()) == 0.0
    assert float(delta[~centric].abs().max()) > 0.0


# --------------------------------------------------------------- variance


def test_delta_variance_is_four_times_the_common_variance():
    """`J` is the mean of the mates and `Delta` their difference."""
    d = planted_pairs(n_h=100, mult=8, seed=4)
    vj, vd = anomalous_fisher_variance(
        d["profile"], d["mask"], d["scale"], d["j_true"][d["index"]],
        d["background"], d["index"], d["mate"], d["n_h"],
    )
    assert torch.allclose(vd, 4.0 * vj, rtol=1e-10)


def test_an_unobserved_mate_gives_an_infinite_difference_variance():
    """A pair seen on one side says nothing about the difference.

    Reporting a finite number there would hand the merge a difference that came
    entirely from the prior and let it be read as signal.
    """
    d = planted_pairs(n_h=40, mult=6, seed=5)
    mate = d["mate"].clone()
    mate[d["index"] == 0] = 1.0  # every observation of pair 0 is the + mate
    vj, vd = anomalous_fisher_variance(
        d["profile"], d["mask"], d["scale"], d["j_true"][d["index"]],
        d["background"], d["index"], mate, d["n_h"],
    )
    assert torch.isinf(vd[0]) and torch.isinf(vj[0])
    assert torch.isfinite(vd[1:]).all(), "only the one-sided pair may be infinite"


def test_difference_variance_falls_with_multiplicity():
    v = {}
    for mult in (4, 16):
        d = planted_pairs(n_h=100, mult=mult, seed=6)
        v[mult] = float(
            anomalous_fisher_variance(
                d["profile"], d["mask"], d["scale"], d["j_true"][d["index"]],
                d["background"], d["index"], d["mate"], d["n_h"],
            )[1].median()
        )
    assert 3.0 < v[4] / v[16] < 5.0, f"ratio {v[4] / v[16]:.2f}, expected about 4"


# --------------------------------------------------------------- the scale


def test_tau_finds_signal_in_one_shell_and_shrinks_the_other():
    """The property a GLOBAL difference scale cannot have.

    A global scale averages the shell with signal against the shell without,
    floors, and zeroes the anomalous difference everywhere -- the recorded
    failure of a naive fully-Bayesian merge.
    """
    d = planted_pairs(n_h=600, mult=8, seed=7, shells=(0.12, 0.0))
    out = two_pass(d)
    per_shell = out["tau"]
    assert float(per_shell[0]) > 5.0 * float(per_shell[1]), (
        f"shell scales {per_shell.tolist()}; shell 0 has the signal"
    )

    # the same estimate with ONE bin, so both are on the fractional scale
    glob = fit_and_shrink_difference(
        out["delta_ml"], out["var"], out["merged"],
        torch.zeros_like(d["bin_index"]), 1,
    )[2]
    assert float(glob[0]) < float(per_shell[0]), (
        f"a global scale ({float(glob[0]):.4f}) must under-state the shell "
        f"that has signal ({float(per_shell[0]):.4f})"
    )


def test_tau_recovers_the_planted_fraction():
    """`tau` is the fractional scale, so it is readable against the planted one.

    The planted difference is `f J (U + 1/2)` with a random sign and `U`
    uniform, so `delta = Delta / 2J` has standard deviation about `0.52 f`.
    """
    for f in (0.08, 0.15):
        d = planted_pairs(n_h=400, mult=8, frac=f, seed=12)
        out = two_pass(d)
        assert float(out["tau"][0]) == pytest.approx(0.52 * f, rel=0.2), (
            f"planted fraction {f}, tau {float(out['tau'][0]):.4f}"
        )


def test_pc_prior_shrinks_tau_and_the_base_model_is_reachable():
    """Simpson et al.'s penalty, with `Delta = 0` as the base model.

    A tighter `tau_scale` pulls harder toward no anomalous signal, and on data
    with none the free estimate already reaches the base model exactly.
    """
    none = two_pass(planted_pairs(n_h=400, mult=8, frac=0.0, seed=8))
    assert float(none["tau"][0]) < 1e-3, "the base model must be reachable"

    d = planted_pairs(n_h=400, mult=8, frac=0.08, seed=8)
    free = float(two_pass(d)["tau"][0])
    loose = float(two_pass(d, tau_scale=0.5)["tau"][0])
    tight = float(two_pass(d, tau_scale=0.02)["tau"][0])
    assert tight < loose <= free + 1e-9, (
        f"free {free:.2f}, loose {loose:.2f}, tight {tight:.2f}"
    )


def test_tau_is_monotone_in_the_planted_difference():
    taus = [
        float(two_pass(planted_pairs(n_h=400, mult=8, frac=f, seed=9))["tau"][0])
        for f in (0.0, 0.05, 0.15)
    ]
    assert taus[0] < taus[1] < taus[2], f"tau by planted fraction: {taus}"


def test_shrinkage_weight_is_the_signal_to_noise_ratio():
    """`w = tau^2 / (tau^2 + v)`, and an unobserved mate gives `w = 0`."""
    delta = torch.tensor([100.0, 100.0, 100.0], dtype=DT)
    var = torch.tensor([0.0, 100.0, float("inf")], dtype=DT)
    tau = torch.tensor([10.0, 10.0, 10.0], dtype=DT)
    out, var_post = shrink_difference(delta, var, tau)
    assert float(out[0]) == pytest.approx(100.0)
    assert float(out[1]) == pytest.approx(50.0)
    assert float(out[2]) == 0.0, "no information means the base model"
    assert float(var_post[2]) == 0.0
    assert torch.isfinite(out).all() and torch.isfinite(var_post).all()


def test_shrinkage_lowers_the_error_and_the_magnitude():
    """The trade the shrink buys: less error, at the cost of magnitude.

    It only buys it on the fractional scale. Shrinking a constant absolute
    scale raises the error instead, because the anomalous difference is
    proportional to the intensity.
    """
    for frac in (0.03, 0.08, 0.15):
        d = planted_pairs(n_h=400, mult=8, frac=frac, seed=13)
        out = two_pass(d)
        t = d["delta_true"]
        mse_ml = float((out["delta_ml"] - t).pow(2).mean())
        mse_post = float((out["delta"] - t).pow(2).mean())
        assert mse_post < mse_ml, (
            f"frac {frac}: shrunk {mse_post:.1f} vs unshrunk {mse_ml:.1f}"
        )
        assert float(out["delta"].abs().mean()) < float(
            out["delta_ml"].abs().mean()
        )


def test_split_by_mate_partitions_the_observations():
    d = planted_pairs(n_h=50, mult=6, seed=10)
    sig = torch.rand(d["index"].shape[0], dtype=DT)
    sp, sm, ap, am = split_by_mate(
        sig, d["scale"], d["index"], d["mate"], d["n_h"]
    )
    total = torch.zeros(d["n_h"], dtype=DT).scatter_add_(0, d["index"], sig)
    assert torch.allclose(sp + sm, total, rtol=1e-12)
    assert float((ap + am).sum()) == pytest.approx(float(d["scale"].sum()))
