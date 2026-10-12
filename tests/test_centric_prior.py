"""The SVAE integrator's Wilson prior distinguishes centric from acentric.

An acentric |F|^2 is exponential, Gamma(1, tau). A centric one is Sigma times
a chi-squared with ONE degree of freedom, Gamma(1/2, tau/2): same mean, twice
the variance, far more mass near zero. `WilsonLoss` has always made the
distinction, but this integrator derives q(I) from conjugacy rather than
routing the intensity through `WilsonLoss.forward`, so it hardcoded alpha = 1
for every reflection.

Measured cost of that: weak centrics biased UP by about half a count (2x at
zero signal), strong centrics in weak shells over-shrunk by up to 17%. No
effect on anomalous differences -- a centric has I(h) = I(-h) by symmetry, so
a bias common to the pair cancels in DANO.

On hewl_1118 centrics are 9.4% of observations and 14.7% below 3 A.
"""

import torch

from integrator.model.integrators.svae_integrator import SVAEIntegrator
from integrator.model.loss.monochromatic_wilson_loss import (
    MonochromaticWilsonLoss,
)

TAU_G, TAU_B = 1000.0, 0.0  # B = 0 so tau = 1/G exactly, independent of d


def _obj():
    """The integrator's prior path, over a real loss rather than a stub."""
    o = object.__new__(SVAEIntegrator)
    # nn.Module.__init__ must run before a submodule can be assigned; the rest
    # of the integrator's construction needs encoders and data we do not want
    # here, and _intensity_prior touches only `self.loss`.
    torch.nn.Module.__init__(o)
    o.loss = MonochromaticWilsonLoss(init_G=TAU_G, init_B=TAU_B, b_min=0.0)
    return o


def _meta(centric=None, n=None):
    n = n if n is not None else (len(centric) if centric is not None else 4)
    md = {"d": torch.full((n,), 2.0)}
    if centric is not None:
        md["centric"] = torch.tensor(centric)
    return md


def test_acentric_is_exponential():
    a, b = _obj()._intensity_prior(_meta([False] * 4), torch.device("cpu"))
    assert torch.allclose(a, torch.ones(4))
    assert torch.allclose(b, torch.full((4,), 1.0 / TAU_G), rtol=1e-3)


def test_centric_is_half_shape_and_half_rate():
    a, b = _obj()._intensity_prior(_meta([True] * 4), torch.device("cpu"))
    assert torch.allclose(a, torch.full((4,), 0.5))
    assert torch.allclose(b, torch.full((4,), 0.5 / TAU_G), rtol=1e-3)


def test_the_prior_mean_is_unchanged():
    """Only the shape changes; Sigma = alpha/rate either way."""
    a, b = _obj()._intensity_prior(
        _meta([True, False, True, False]), torch.device("cpu")
    )
    assert torch.allclose(a / b, torch.full((4,), TAU_G), rtol=1e-3)


def test_centric_has_twice_the_variance():
    a, b = _obj()._intensity_prior(_meta([True, False]), torch.device("cpu"))
    var = a / b.pow(2)
    assert torch.isclose(var[0], 2 * var[1], rtol=1e-3)


def test_missing_column_falls_back_to_acentric():
    """Datasets cut before the flag existed must still train."""
    a, b = _obj()._intensity_prior(_meta(None), torch.device("cpu"))
    assert torch.allclose(a, torch.ones(4))
    assert torch.allclose(b, torch.full((4,), 1.0 / TAU_G), rtol=1e-3)


def test_mixed_flags_are_applied_per_reflection():
    a, b = _obj()._intensity_prior(
        _meta([True, False, False, True]), torch.device("cpu")
    )
    assert torch.allclose(a, torch.tensor([0.5, 1.0, 1.0, 0.5]))
    assert torch.allclose(
        b, torch.tensor([0.5, 1.0, 1.0, 0.5]) / TAU_G, rtol=1e-3
    )


def test_it_matches_the_prior_the_KL_uses():
    """q's prior and the KL's prior must be the same object, not two formulas.

    They were built separately before, so a centric reflection had its q
    shaped by one prior and scored against another.
    """
    o = _obj()
    md = _meta([True, False, True, False])
    a1, b1 = o._intensity_prior(md, torch.device("cpu"))
    a2, b2 = o.loss.intensity_prior_params(md, torch.device("cpu"))
    assert torch.equal(a1, a2) and torch.equal(b1, b2)
