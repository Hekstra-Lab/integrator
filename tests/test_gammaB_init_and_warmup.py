"""Test the gammaB head zero-init (when mean_init is set) and the
step-level lr warmup scheduler.

Together these target the seed-dependent NaN-or-not behavior the init
diagnostic surfaced: random Kaiming init on linear_mu/linear_fano times
encoder features produces ~1000× spread in initial qi_linear_mu_grad_max
across seeds. Zero-init kills the per-reflection variance at step 0;
step warmup keeps Adam's running moments from being whacked by a single
big-grad batch in the first ~100 steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from integrator.model.distributions.gamma import GammaDistributionRepamB


# ---------- gammaB head zero-init ----------


def test_zero_init_inactive_when_mean_init_none():
    """`mean_init=None` keeps the existing random Kaiming weights —
    backward compat with `mean_init: null` runs."""
    torch.manual_seed(0)
    layer = GammaDistributionRepamB(
        in_features=8, separate_inputs=True, mean_init=None
    )
    # Default Kaiming init produces nonzero weights with bound 1/sqrt(8).
    assert torch.any(layer.linear_mu.weight != 0)
    assert torch.any(layer.linear_fano.weight != 0)


def test_zero_init_separate_inputs():
    """When mean_init is provided, BOTH linear_mu.weight AND
    linear_fano.weight are zeroed."""
    layer = GammaDistributionRepamB(
        in_features=8, separate_inputs=True, mean_init=50.0
    )
    assert torch.all(layer.linear_mu.weight == 0)
    assert torch.all(layer.linear_fano.weight == 0)
    # Bias is set to map to the target.
    raw_mu_at_init = float(layer.linear_mu.bias.item())
    mu_at_init = torch.nn.functional.softplus(
        torch.tensor(raw_mu_at_init)
    ) + layer.eps
    assert abs(mu_at_init.item() - 50.0) < 1e-3


def test_zero_init_fc_path():
    """Same for the non-separate-inputs path: fc.weight is zeroed."""
    layer = GammaDistributionRepamB(
        in_features=8, separate_inputs=False, mean_init=50.0
    )
    assert torch.all(layer.fc.weight == 0)


def test_zero_init_makes_initial_mu_uniform_across_batch():
    """The whole point: every reflection in the batch sees the same mu
    at step 0 because raw_mu = bias + 0 @ x = bias for all x."""
    layer = GammaDistributionRepamB(
        in_features=8,
        separate_inputs=True,
        mean_init=50.0,
        mu_parameterization="log",
    )
    # Random encoder features, varying per "reflection"
    x = torch.randn(16, 8) * 5.0
    dist = layer(x, x)
    mu_predicted = dist.concentration / dist.rate
    # All 16 mu's should be identical (= 50, the init target).
    assert torch.allclose(
        mu_predicted, mu_predicted[0].expand_as(mu_predicted), atol=1e-4
    ), f"mu varies across reflections: {mu_predicted}"
    assert abs(float(mu_predicted[0]) - 50.0) < 1e-2


def test_zero_init_qi_linear_mu_grad_is_seed_independent():
    """The init diagnostic showed qi_linear_mu_grad_max varies ~1000×
    across seeds with the random-Kaiming default. With zero-init the
    spread should be much tighter — driven only by encoder feature
    variation, not by the head weights."""
    grad_max_per_seed = []
    for seed in range(8):
        torch.manual_seed(seed)
        layer = GammaDistributionRepamB(
            in_features=8,
            separate_inputs=True,
            mean_init=50.0,
            mu_parameterization="log",
        )
        x = torch.randn(16, 8, requires_grad=True)
        dist = layer(x, x)
        sample = dist.rsample()
        loss = sample.sum()
        loss.backward()
        grad_max_per_seed.append(
            float(layer.linear_mu.weight.grad.abs().max())
        )
    spread = max(grad_max_per_seed) / max(min(grad_max_per_seed), 1e-12)
    assert spread < 5.0, (
        f"With zero-init, the seed-to-seed grad-max spread on "
        f"linear_mu.weight should be modest. Got {spread:.1f}× "
        f"({grad_max_per_seed})"
    )


# ---------- step-level lr warmup ----------


@dataclass
class _StubCfg:
    """Minimal cfg shim for exercising the warmup lambda. The full
    IntegratorCfg has a lot of required fields; we only need a couple."""
    warmup_steps: int = 100
    lr: float = 1e-3
    lr_min: float = 1e-5


class _IntegratorShim:
    """Just enough of BaseIntegrator to call _step_linear_warmup_lambda."""

    def __init__(self, warmup_steps: int):
        self.warmup_steps = warmup_steps

    # Pull the actual implementation in so we're testing real code, not a
    # reimplementation.
    from integrator.model.integrators.base_integrator import BaseIntegrator
    _step_linear_warmup_lambda = BaseIntegrator._step_linear_warmup_lambda


def test_step_warmup_ramps_linearly_to_1():
    sched = _IntegratorShim(warmup_steps=100)._step_linear_warmup_lambda()
    # Step 0 → 1/100, step 49 → 50/100, step 99 → 100/100, step 200 → 1.0
    assert abs(sched(0) - 0.01) < 1e-9
    assert abs(sched(49) - 0.50) < 1e-9
    assert abs(sched(99) - 1.00) < 1e-9
    assert sched(200) == 1.0


def test_step_warmup_zero_means_passthrough():
    """warmup_steps=0 returns lr_lambda that always returns 1 — equivalent
    to no scheduler. Needed so omitting the field doesn't error."""
    sched = _IntegratorShim(warmup_steps=0)._step_linear_warmup_lambda()
    assert sched(0) == 1.0
    assert sched(100) == 1.0


def test_step_warmup_step_immediately_nonzero():
    """At step 0 the lr multiplier is 1/warmup, NOT 0, so the optimizer
    takes a real step on the very first batch (zero step is wasted)."""
    sched = _IntegratorShim(warmup_steps=500)._step_linear_warmup_lambda()
    assert sched(0) > 0


if __name__ == "__main__":
    for name, fn in [
        ("zero-init inactive when mean_init=None", test_zero_init_inactive_when_mean_init_none),
        ("zero-init separate_inputs", test_zero_init_separate_inputs),
        ("zero-init fc path", test_zero_init_fc_path),
        ("initial mu uniform across batch", test_zero_init_makes_initial_mu_uniform_across_batch),
        ("seed-independent grad with zero-init", test_zero_init_qi_linear_mu_grad_is_seed_independent),
        ("warmup ramps linearly to 1", test_step_warmup_ramps_linearly_to_1),
        ("warmup_steps=0 is passthrough", test_step_warmup_zero_means_passthrough),
        ("warmup step 0 is nonzero", test_step_warmup_step_immediately_nonzero),
    ]:
        fn()
        print(f"PASSED: {name}")
