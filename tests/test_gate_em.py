"""Refining the responsibility gate toward the exact Poisson E-step.

A Poisson count whose rate is a sum splits Binomially: with
`rate = s I p + b`, the signal's share of pixel `p` is

    r* = s I p / (s I p + b),

and the model's `sigmoid(log p + g)` equals that exactly when
`g = log(s I / b)`.  The amortized gate produces `g` from the shoebox alone,
so it cannot know `s`, `I_h` (a quantity aggregating ~116 observations) or
`b` -- it matches the E-step only by coincidence.  Measured on hewl_1118 it
keeps 45% of the signal below `I/sigma` 0.5 and 149% near 2, and because
brightness within a reflection tracks the spectrum, that error is
wavelength-shaped and corrupts any scale fitted from these counts.

`gate_em_iters` recomputes `r` from the model's own detached estimates. These
tests check it on synthetic shoeboxes where the truth is known.
"""

import pytest
import torch


def _exact_r(scale, intensity, profile, bg_rate):
    sig = scale * intensity * profile
    return sig / (sig + bg_rate)


def _shoebox(n_obs=64, n_pix=100, intensity=40.0, bg=8.0, seed=0):
    """Counts from a known rate, plus the exact responsibility."""
    g = torch.Generator().manual_seed(seed)
    centre = torch.linspace(-2.0, 2.0, n_pix)
    profile = torch.exp(-0.5 * centre.pow(2))
    profile = (profile / profile.sum()).repeat(n_obs, 1)
    scale = torch.rand(n_obs, generator=g) * 1.5 + 0.5
    rate = scale[:, None] * intensity * profile + bg
    counts = torch.poisson(rate, generator=g)
    r_true = _exact_r(scale[:, None], intensity, profile, bg)
    return counts, profile, scale, r_true


def test_the_exact_e_step_is_what_sigmoid_of_log_p_plus_g_computes():
    """g = log(s I / b) reproduces the Binomial split exactly."""
    _, profile, scale, r_true = _shoebox()
    intensity, bg = 40.0, 8.0
    g = torch.log(scale * intensity / bg)
    r_param = torch.sigmoid(torch.log(profile) + g[:, None])
    assert torch.allclose(r_param, r_true, atol=1e-6)


def test_a_biased_gate_misattributes_and_the_e_step_recovers_it():
    """The measured failure: a gate off by a constant in log-odds."""
    counts, profile, scale, r_true = _shoebox(seed=3)
    intensity, bg = 40.0, 8.0
    signal_true = (r_true * counts).sum(-1)

    # the amortized gate, wrong by +1 in log-odds (it over-attributes)
    g_bad = torch.log(scale * intensity / bg) + 1.0
    r_bad = torch.sigmoid(torch.log(profile) + g_bad[:, None])
    signal_bad = (r_bad * counts).sum(-1)

    # one exact E-step from the model's own (correct) estimates
    r_em = _exact_r(scale[:, None], intensity, profile, bg)
    signal_em = (r_em * counts).sum(-1)

    err_bad = float((signal_bad / signal_true - 1.0).abs().mean())
    err_em = float((signal_em / signal_true - 1.0).abs().mean())
    assert err_bad > 0.10, f"the biased gate was not biased enough to test ({err_bad:.4f})"
    assert err_em < 0.01, f"the E-step did not recover the split ({err_em:.4f})"


def test_the_split_error_grows_as_background_takes_over():
    """Why this matters on Laue and not on 9b7c.

    The same gate error costs `sum_p r(1-r) c_p delta`, and at 93% background
    those are mostly background counts. At 0% background the split is free.
    """
    intensity = 40.0
    errors = {}
    for bg in (0.01, 1.0, 8.0, 40.0):
        counts, profile, scale, r_true = _shoebox(
            bg=bg, intensity=intensity, seed=5
        )
        signal_true = (r_true * counts).sum(-1)
        g_bad = torch.log(scale * intensity / bg) + 0.5
        r_bad = torch.sigmoid(torch.log(profile) + g_bad[:, None])
        signal_bad = (r_bad * counts).sum(-1)
        errors[bg] = float((signal_bad / signal_true - 1.0).abs().mean())
    assert errors[0.01] < errors[1.0] < errors[8.0] < errors[40.0], errors
    assert errors[0.01] < 0.02, f"a near-background-free split should be ~free: {errors}"


def test_iterating_converges_to_the_truth_from_a_bad_start():
    """The fixed point is the E-step, so iterating must reach it and settle.

    From a 8x-too-small start the approach is slow -- a small `I` makes `r`
    small, which makes the next `I` small -- so this asserts the properties
    that matter: the steps shrink monotonically and the limit is the truth,
    not that it arrives within some chosen number of iterations.
    """
    counts, profile, scale, _ = _shoebox(seed=7)
    bg = 8.0
    i_hat = torch.tensor(5.0)
    history = []
    for _ in range(400):
        r = _exact_r(scale[:, None], i_hat, profile, bg)
        sig = (r * counts).sum(-1)
        # the conjugate M-step for a shared intensity, with exposures = scale
        i_hat = sig.sum() / scale.sum()
        history.append(float(i_hat))
    steps = [abs(b - a) for a, b in zip(history[:-1], history[1:])]
    assert steps[-1] <= steps[0], "the iteration is not contracting"
    assert steps[-1] < 1e-4 * history[-1], f"not settled: {steps[-3:]}"
    assert history[-1] == pytest.approx(40.0, rel=0.10), history[-1]


def test_the_split_variance_is_available_and_nonzero():
    """`sum_p c r (1-r)` is the split's own variance, missing from the merge."""
    counts, profile, scale, r_true = _shoebox(seed=11)
    var = (counts * r_true * (1.0 - r_true)).sum(-1)
    signal = (r_true * counts).sum(-1)
    assert (var > 0).all()
    # it is a real fraction of the signal's own counting variance
    ratio = float((var / signal.clamp(min=1e-6)).median())
    assert 0.01 < ratio < 1.0, ratio


class _Merge:
    """`_responsibility_merge` on stubs, so the real E-step code runs.

    Only the priors are stubbed: everything the E-step reads -- above all the
    background posterior's parameterisation -- is the shipped code.
    """

    from integrator.model.scaling.svae_merging import (  # noqa: E402
        SVAEMergingIntegrator as _S,
    )

    _responsibility_merge = _S._responsibility_merge

    def __init__(self, gate_em_iters):
        self.gate_em_iters = gate_em_iters
        self.wilson_prior_per_obs = False
        self.alpha_W = 1.0

    def _bg_prior(self, group_labels, n, device):
        weak = torch.full((n,), 1e-3, dtype=torch.float32)
        return weak, weak

    def _wilson_tau(self, d, extra=None):
        return torch.full_like(d, 1e-6)


def _background_dominated(
    n_hkl=24,
    per_hkl=12,
    n_pix=100,
    i_true=300.0,
    bg=60.0,
    gate_bias=None,
    seed=0,
):
    """Laue-like shoeboxes: 95% of the counts are background.

    `gate_bias` sets what the amortized gate is told, as an error in log-odds
    against the value that makes `sigmoid(log p + g)` exact: `None` gives a
    gate of zeros, i.e. one that knows nothing.
    """
    g = torch.Generator().manual_seed(seed)
    n_obs = n_hkl * per_hkl
    centre = torch.linspace(-2.5, 2.5, n_pix)
    prf = torch.exp(-0.5 * centre.pow(2))
    prf = (prf / prf.sum()).repeat(n_obs, 1)
    scale = torch.rand(n_obs, generator=g) * 0.8 + 0.6
    rate = scale[:, None] * i_true * prf + bg
    gate = (
        torch.zeros(n_obs)
        if gate_bias is None
        else torch.log(scale * i_true / bg) + gate_bias
    )
    return {
        "logits": torch.log(prf),
        "gate": gate,
        "counts": torch.poisson(rate, generator=g),
        "scale": scale,
        "profile_mean": prf,
        "mask": torch.ones(n_obs, n_pix),
        "asu_ids": torch.arange(n_hkl).repeat_interleave(per_hkl),
        "d_per_obs": torch.full((n_obs,), 2.0),
        "group_labels": None,
    }


def _merged_intensity(iters, **kwargs):
    _, alpha_h, beta_h, *_ = _Merge(iters)._responsibility_merge(
        **_background_dominated(**kwargs)
    )
    return float((alpha_h / beta_h).median())


ITERS = 20
"""Enough for the joint `(I_h, b_i)` fixed point from a gate off by e^2.

Measured at the real batch shape (2048 x 1323) it costs 52 ms a batch on CPU,
about 40 minutes over a 50-epoch run and far less on a GPU.
"""


def test_the_e_step_keeps_the_background_out_of_the_signal():
    """The regression the synthetic tests above could not see.

    `qbg`'s rate is `bg_b0 + n_pix`, so its mean is counts per pixel -- the
    quantity the ELBO adds to `zI_scaled * zp`.  Dividing it by `n_pix` a
    second time drove `r` to 1 and merged the whole shoebox as signal: at 95%
    background that is a ~20x intensity inflation, and it took the arm's
    R-free from 0.20 to 0.39.
    """
    i_true, bg, n_pix = 300.0, 60.0, 100
    kwargs = dict(i_true=i_true, bg=bg, n_pix=n_pix)
    kw = _background_dominated(**kwargs)
    *_, r = _Merge(ITERS)._responsibility_merge(**kw)

    counts = kw["counts"]
    got = float((r * counts).sum() / counts.sum())
    want = i_true / (i_true + bg * n_pix)
    assert got == pytest.approx(want, rel=0.25), (
        f"the split called {got:.3f} of the counts signal, against a true "
        f"signal fraction of {want:.3f}"
    )
    assert _merged_intensity(ITERS, **kwargs) == pytest.approx(i_true, rel=0.15)


@pytest.mark.parametrize("gate_bias", [-2.0, -1.0, 0.0, 1.0, 2.0, None])
def test_the_solve_does_not_depend_on_what_the_gate_believed(gate_bias):
    """The property that makes this a fixed point rather than a nudge.

    Two ways of getting this wrong both leave a solve that inherits the gate's
    bias, and both looked fine against a hand-written formula: accelerating by
    the linear-space derivative `V_h / (I_h beta_h)`, which is not bounded by
    1 below the root, and holding `b_i` at the gate's own value while solving
    for `I_h`, when the two are one joint fixed point.
    """
    i_true = 300.0
    got = _merged_intensity(ITERS, i_true=i_true, gate_bias=gate_bias, seed=2)
    assert got == pytest.approx(i_true, rel=0.15), (
        f"a gate biased by {gate_bias} in log-odds converged to {got:.1f}, "
        f"not {i_true:.1f}"
    )


@pytest.mark.parametrize("gate_bias", [-6.0, 6.0])
def test_even_an_absurd_start_reaches_the_same_fixed_point(gate_bias):
    """A start off by 400x takes more steps, and lands in the same place.

    This is what says the solve has one fixed point rather than a basin per
    start, and it fixes the number: 20 steps cover the range a trained gate
    produces, 50 cover anything.
    """
    got = _merged_intensity(50, i_true=300.0, gate_bias=gate_bias, seed=2)
    settled = _merged_intensity(50, i_true=300.0, gate_bias=0.0, seed=2)
    assert got == pytest.approx(settled, rel=5e-3)
    assert got == pytest.approx(300.0, rel=0.15), got


def test_the_e_step_improves_on_a_biased_gate():
    """What the key is for: the gate's error is what it has to remove."""
    i_true = 300.0
    for bias in (-1.0, 1.0):
        kwargs = dict(i_true=i_true, gate_bias=bias, seed=2)
        off = _merged_intensity(0, **kwargs)
        on = _merged_intensity(ITERS, **kwargs)
        assert abs(on / i_true - 1.0) < 0.5 * abs(off / i_true - 1.0), (
            f"at bias {bias:+.1f} the E-step went {off:.1f} -> {on:.1f} "
            f"against a true {i_true:.1f}"
        )


def test_a_few_iterations_are_not_enough_and_the_count_is_the_knob():
    """Why `gate_em_iters` is 20 and not 2.

    The EM map's derivative in log `I_h` is `V_h / alpha_h`, which at 95%
    background sits just under 1; even with the `1/(1-K)` correction the joint
    `(I_h, b_i)` solve needs iterations, and a value of 2 is close to a no-op.
    """
    kwargs = dict(i_true=300.0, gate_bias=1.0, seed=2)
    err = [abs(_merged_intensity(k, **kwargs) / 300.0 - 1.0) for k in (0, 2, ITERS)]
    assert err[2] < err[1] < err[0], err
    assert err[1] > 0.5, f"2 iterations already fixed it, so this test is stale: {err}"


def test_the_key_defaults_off():
    from integrator.configs.integrator import IntegratorCfg

    assert IntegratorCfg.__dataclass_fields__["gate_em_iters"].default == 0
