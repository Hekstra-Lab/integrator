"""The three things that put the SVAE merger's intensities on the wrong scale.

1. The Wilson prior's log-normalizer was missing from the intensity ELBO. It is
   constant in `q` but not in `G` and `B`, and it carries their whole gradient:
   `d/dtau [log tau - tau E[I]] = 1/tau - E[I]`, zero at `tau = 1/E[I]`, which
   is the Wilson plot fit. Without it the derivative is `-E[I]`, negative
   whatever `tau` is, so `tau` only falls and `G` only climbs, with no fixed
   point.

2. The amortized anchor used `a_i / e_i` and called it the prior-free MLE, but
   `a_i = nu_i + sigma_i` carries the random-effect pseudo-count.

3. d-spacing and detector radius are the same for every symmetry mate, so a
   scale built from them is constant within each merging group. `s -> c(d) s`,
   `I_h -> I_h / c(d)` leaves the likelihood and the merging consistency
   untouched, which is why the merged intensities came out on the prior's line.
"""

import math

import pytest
import torch

from integrator.model.scaling.chebyshev_scale import MLPScale


# ------------------------------------------------------- the log tau gradient


def _i_block_with_norm(tau, e_I, alpha_W=1.0):
    """`E_q[log p(I)]` keeping the prior's normalizer."""
    return alpha_W * torch.log(tau) - math.lgamma(alpha_W) - tau * e_I


def _i_block_without_norm(tau, e_I):
    """What the code computed before the fix."""
    return -tau * e_I


@pytest.mark.parametrize("e_I", [1.0, 100.0, 1000.0])
def test_prior_normalizer_creates_the_wilson_fixed_point(e_I):
    """With the term, dELBO/dtau vanishes exactly at tau = 1/E[I]."""
    tau = torch.tensor(1.0 / e_I, requires_grad=True)
    _i_block_with_norm(tau, torch.tensor(e_I)).backward()
    assert tau.grad.abs() < 1e-4 * max(1.0, e_I)


def test_without_the_normalizer_the_gradient_never_changes_sign():
    """The old form pushes tau down from everywhere, so G climbs forever."""
    e_I = torch.tensor(1000.0)
    for tau_val in (1e-6, 1e-3, 1.0, 1e3):
        tau = torch.tensor(tau_val, requires_grad=True)
        _i_block_without_norm(tau, e_I).backward()
        assert tau.grad < 0

    # The corrected form reverses once tau is below the fit, which is what
    # lets it come back up. The fit here is tau = 1/E[I] = 1e-3, so 1e-4 is
    # an order of magnitude below it.
    tau = torch.tensor(1e-4, requires_grad=True)
    _i_block_with_norm(tau, e_I).backward()
    assert tau.grad > 0

    # and still pushes down from above it
    tau = torch.tensor(1e-2, requires_grad=True)
    _i_block_with_norm(tau, e_I).backward()
    assert tau.grad < 0


def test_integrator_adds_the_normalizer():
    """`_wilson_log_norm` is alpha log tau - lgamma(alpha), per HKL."""
    from integrator.model.scaling.hierarchical_svae_integrator import (
        HierarchicalSVAEIntegrator,
    )

    obj = object.__new__(HierarchicalSVAEIntegrator)
    obj.alpha_W = 1.0
    tau = torch.tensor([1e-3, 1.0, 42.5])
    got = HierarchicalSVAEIntegrator._wilson_log_norm(obj, tau)
    assert torch.allclose(got, tau.log(), atol=1e-6)

    obj.alpha_W = 0.5
    got = HierarchicalSVAEIntegrator._wilson_log_norm(obj, tau)
    want = 0.5 * tau.log() - math.lgamma(0.5)
    assert torch.allclose(got, want, atol=1e-6)


# -------------------------------------------------------------- the MLP gauge


def _scale_inputs(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "frame": torch.randint(0, 360, (n,), generator=g).float(),
        "x": torch.rand(n, generator=g) * 2000,
        "y": torch.rand(n, generator=g) * 2000,
        "lp": torch.rand(n, generator=g),
        "d": 1.5 + torch.rand(n, generator=g) * 20,
    }


def test_excluding_hkl_constant_inputs_shrinks_the_input_layer():
    kept = MLPScale(beam_center=[1000.0, 1000.0])
    dropped = MLPScale(beam_center=[1000.0, 1000.0], exclude_hkl_constant=True)
    assert kept.net[0].in_features == 4
    assert dropped.net[0].in_features == 2


def test_excluded_scale_cannot_depend_on_resolution():
    """The point of the flag: vary only d, the scale must not move."""
    torch.manual_seed(0)
    scale = MLPScale(
        beam_center=[1000.0, 1000.0], exclude_hkl_constant=True,
        head_init_std=0.5,
    )
    args = _scale_inputs()
    base = scale(**args)
    moved = scale(**{**args, "d": args["d"] * 3.0})
    assert torch.allclose(base, moved, atol=1e-7)


def test_excluded_scale_still_depends_on_frame():
    """It must keep the directions that DO vary between symmetry mates."""
    torch.manual_seed(0)
    scale = MLPScale(
        beam_center=[1000.0, 1000.0], exclude_hkl_constant=True,
        head_init_std=0.5,
    )
    args = _scale_inputs()
    base = scale(**args)
    moved = scale(**{**args, "frame": (args["frame"] + 180.0) % 360})
    assert not torch.allclose(base, moved, atol=1e-4)


def test_default_scale_still_sees_resolution():
    """Unchanged behaviour for every existing scaling config."""
    torch.manual_seed(0)
    scale = MLPScale(beam_center=[1000.0, 1000.0], head_init_std=0.5)
    args = _scale_inputs()
    assert not torch.allclose(
        scale(**args), scale(**{**args, "d": args["d"] * 3.0}), atol=1e-4
    )


# ------------------------------------------------------------- freezing G, B


@pytest.mark.parametrize("learn_G", [True, False])
def test_learn_g_controls_the_scale_gradient(learn_G):
    from integrator.model.loss.monochromatic_wilson_loss import (
        MonochromaticWilsonLoss,
    )

    loss = MonochromaticWilsonLoss(learn_G=learn_G, init_G=1000.0)
    assert loss.raw_G.requires_grad is learn_G
    # frozen or not, it stays a Parameter so checkpoints still load
    assert isinstance(loss.raw_G, torch.nn.Parameter)
