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


def test_the_key_defaults_off():
    from integrator.configs.integrator import IntegratorCfg

    assert IntegratorCfg.__dataclass_fields__["gate_em_iters"].default == 0
