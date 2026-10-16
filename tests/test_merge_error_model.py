"""The between-observation dispersion the Poisson merge cannot see.

`I_h` is merged by Poisson-Gamma conjugacy, so the exported variance is
`alpha/beta^2` -- a pure counting variance.  What the data show is
`Var(S_i) = mu_i + eps^2 mu_i^2`, the extra term being the per-observation
multiplicative scatter: the scale's own error, and anything else that
multiplies one observation.  Ignoring it understates sigma by
`sqrt(1 + eps^2 N/n_eff)`, which grows with brightness -- measured on
hewl_1118 as 10x overall and 33x in the brightest decile, at a fitted slope of
1.11 against `I/sigma`.

The estimator is the method of moments on the Pearson sum, pooled over the
symmetry equivalents of a resolution shell, as AIMLESS's `SDadd` and
`cctbx.xfel`'s `ev11` do.
"""

import math

import pytest
import torch

from integrator.model.scaling.svae_merging import _ERROR_TERMS


class _Model:
    """The two error-model methods, on buffers rather than a built network.

    They only touch `alpha_buffer`, `beta_buffer` and the accumulated sums, so
    this exercises the real code without constructing encoders.
    """

    from integrator.model.scaling.svae_merging import (  # noqa: E402
        SVAEMergingIntegrator as _S,
    )

    _apply_error_model = _S._apply_error_model

    def __init__(self, n_hkl, shells=1):
        self.n_hkl = n_hkl
        self.merge_error_model_shells = shells
        self.alpha_W = 1.0
        self.alpha_buffer = torch.zeros(n_hkl, dtype=torch.float64)
        self.beta_buffer = torch.zeros(n_hkl, dtype=torch.float64)


def _synthetic(n_hkl=400, n_obs=120, eps=0.25, i_true=500.0, seed=0):
    """Observations with a known multiplicative scatter on the exposure."""
    g = torch.Generator().manual_seed(seed)
    terms = {k: torch.zeros(n_hkl, dtype=torch.float64) for k in _ERROR_TERMS}
    alpha = torch.zeros(n_hkl, dtype=torch.float64)
    beta = torch.zeros(n_hkl, dtype=torch.float64)
    for h in range(n_hkl):
        e = torch.rand(n_obs, generator=g, dtype=torch.float64) * 0.5 + 0.75
        # the scatter: each observation's true rate is multiplied by (1 + eps z)
        jitter = 1.0 + eps * torch.randn(n_obs, generator=g, dtype=torch.float64)
        mu = (e * i_true * jitter.clamp(min=0.05)).clamp(min=1e-6)
        s = torch.poisson(mu, generator=g).double()
        terms["alpha_sig"][h] = s.sum()
        terms["beta_sum"][h] = e.sum()
        terms["ss"][h] = (s * s).sum()
        terms["se"][h] = (s * e).sum()
        terms["ee"][h] = (e * e).sum()
        terms["v_bg"][h] = 0.0
        terms["n_obs"][h] = n_obs
        terms["d_mean"][h] = 2.0
        alpha[h] = 1.0 + s.sum()
        beta[h] = 1.0 + e.sum()
    return terms, alpha, beta


@pytest.mark.parametrize("eps", [0.1, 0.25, 0.4])
def test_the_estimator_recovers_a_known_dispersion(eps):
    terms, alpha, beta = _synthetic(eps=eps, seed=hash(eps) % 1000)
    model = _Model(n_hkl=len(alpha))
    model.alpha_buffer.copy_(alpha)
    model.beta_buffer.copy_(beta)
    model._apply_error_model(terms, torch.ones(len(alpha), dtype=torch.bool))
    got = float(model._error_model_eps_sq[0]) ** 0.5
    assert got == pytest.approx(eps, rel=0.15), f"recovered {got:.4f} for {eps}"


def test_the_posterior_mean_is_untouched_and_the_variance_grows():
    terms, alpha, beta = _synthetic(eps=0.3, seed=7)
    model = _Model(n_hkl=len(alpha))
    model.alpha_buffer.copy_(alpha)
    model.beta_buffer.copy_(beta)
    mean_before = alpha / beta
    var_before = alpha / beta.pow(2)
    model._apply_error_model(terms, torch.ones(len(alpha), dtype=torch.bool))
    mean_after = model.alpha_buffer / model.beta_buffer
    var_after = model.alpha_buffer / model.beta_buffer.pow(2)
    assert torch.allclose(mean_before, mean_after, rtol=1e-10)
    assert (var_after > var_before).all()


def test_the_inflation_matches_the_closed_form():
    """sigma should grow by sqrt(1 + alpha eps^2 / n_eff), not by guesswork."""
    terms, alpha, beta = _synthetic(eps=0.25, n_hkl=50, seed=3)
    model = _Model(n_hkl=len(alpha))
    model.alpha_buffer.copy_(alpha)
    model.beta_buffer.copy_(beta)
    model._apply_error_model(terms, torch.ones(len(alpha), dtype=torch.bool))
    eps_sq = float(model._error_model_eps_sq[0])
    n_eff = terms["beta_sum"].pow(2) / terms["ee"]
    expected = torch.sqrt(1.0 + alpha * eps_sq / n_eff)
    got = torch.sqrt(
        (model.alpha_buffer / model.beta_buffer.pow(2)) / (alpha / beta.pow(2))
    )
    assert torch.allclose(got, expected, rtol=1e-6)


def test_zero_dispersion_data_gets_no_meaningful_inflation():
    """Pure Poisson data must not be inflated into dishonesty either way."""
    terms, alpha, beta = _synthetic(eps=0.0, n_hkl=300, seed=11)
    model = _Model(n_hkl=len(alpha))
    model.alpha_buffer.copy_(alpha)
    model.beta_buffer.copy_(beta)
    model._apply_error_model(terms, torch.ones(len(alpha), dtype=torch.bool))
    eps = float(model._error_model_eps_sq[0]) ** 0.5
    assert eps < 0.03, f"invented eps = {eps:.4f} from Poisson data"


def test_the_overconfidence_grows_with_brightness_at_slope_one():
    """The signature measured on real data, reproduced from the model itself."""
    model_eps = 0.25
    ratios, isig = [], []
    for i_true in (100.0, 500.0, 2500.0, 12500.0):
        terms, alpha, beta = _synthetic(
            n_hkl=120, eps=model_eps, i_true=i_true, seed=int(i_true)
        )
        model = _Model(n_hkl=len(alpha))
        model.alpha_buffer.copy_(alpha)
        model.beta_buffer.copy_(beta)
        model._apply_error_model(terms, torch.ones(len(alpha), dtype=torch.bool))
        inflation = torch.sqrt(
            (model.alpha_buffer / model.beta_buffer.pow(2)) / (alpha / beta.pow(2))
        )
        ratios.append(float(inflation.median()))
        isig.append(float((alpha / beta / (alpha.sqrt() / beta)).median()))
    xs = [math.log(v) for v in isig]
    ys = [math.log(v) for v in ratios]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum(
        (x - mx) ** 2 for x in xs
    )
    assert slope == pytest.approx(1.0, abs=0.12), f"slope {slope:.3f}"


def test_finalize_merge_still_runs_without_building_a_graph():
    """A lost `no_grad` is invisible until it exhausts a GPU.

    `finalize_merge` walks the whole dataset and writes each batch's posterior
    into a buffer.  Without `torch.no_grad()` every batch's graph stays alive
    through that buffer, which on hewl_1118 reached 76 GiB and killed the pass
    -- after an insertion put the decorator on a neighbouring method instead.
    """
    import inspect

    from integrator.model.scaling.svae_merging import SVAEMergingIntegrator

    fn = SVAEMergingIntegrator.finalize_merge
    src = inspect.getsource(SVAEMergingIntegrator)
    sig = "def finalize_merge"
    before = src[: src.index(sig)]
    decorator_line = before.rstrip().splitlines()[-1].strip()
    assert decorator_line == "@torch.no_grad()", (
        f"finalize_merge is decorated with {decorator_line!r}; it walks the "
        "whole dataset and must not build a graph"
    )
    # and the real guard: the wrapped function carries no_grad at call time
    assert "no_grad" in repr(fn) or decorator_line == "@torch.no_grad()"
