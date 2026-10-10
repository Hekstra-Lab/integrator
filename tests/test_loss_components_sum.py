"""Every KL term a loss reports must actually be in the loss.

The scaling merge deleted `kl = kl + kl_prf` from `WilsonLoss.forward`. The
term kept being computed and kept appearing in the returned `kl_prf_mean`, so
no log and no test looked wrong, but it contributed nothing to the gradient.
Models trained for two days with an unregularised profile posterior before the
numbers were traced back to it.

Neither branch was wrong on its own -- ours accumulated incrementally
(`kl = kl + kl_prf`), theirs in one line (`kl = kl_prf + kl_i + kl_bg`) -- so
no textual check could have caught it: no shared line was removed. What makes
it detectable is the invariant below, that the reported components add up to
the reported total.
"""
import pytest
import torch
from torch.distributions import Gamma

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)
from integrator.model.loss.monochromatic_wilson_loss import (
    MonochromaticWilsonLoss,
)
from integrator.model.loss.polychromatic_wilson_loss import (
    PolychromaticWilsonLoss,
)

B, P, LATENT = 32, 441, 8


def _inputs(poly: bool):
    torch.manual_seed(0)
    md = {
        "d": torch.full((B,), 2.0),
        "group_label": torch.zeros(B, dtype=torch.long),
    }
    if poly:
        md["wavelength"] = torch.full((B,), 1.1)
    qp = ProfileSurrogateOutput(
        zp=torch.full((1, B, P), 1.0 / P),
        mean_profile=torch.full((B, P), 1.0 / P),
        loc=torch.randn(B, LATENT) * 0.7,
        scale=torch.rand(B, LATENT) * 0.5 + 0.1,
        kl=None,
    )
    rate = torch.rand(B, 1, P) * 5 + 1
    return dict(
        rate=rate,
        counts=torch.poisson(rate.squeeze(1)),
        qp=qp,
        qi=Gamma(torch.full((B,), 5.0), torch.full((B,), 0.5)),
        qbg=Gamma(torch.full((B,), 2.0), torch.full((B,), 2.0)),
        mask=torch.ones(B, P, 1),
        group_labels=md["group_label"],
        metadata=md,
    )


@pytest.mark.parametrize(
    "cls,poly",
    [(MonochromaticWilsonLoss, False), (PolychromaticWilsonLoss, True)],
)
def test_kl_components_sum_to_total(cls, poly):
    """`kl_mean` is the sum of every KL component the loss reports."""
    loss = cls(bg_rate=1.0, bg_concentration=1.0, n_bins=1)
    out = loss(**_inputs(poly))

    parts = {
        k: float(v)
        for k, v in out.items()
        if k.startswith("kl_") and k.endswith("_mean") and k != "kl_mean"
    }
    assert "kl_prf_mean" in parts, "the profile KL is not even reported"
    total = float(out["kl_mean"])
    assert total == pytest.approx(sum(parts.values()), rel=1e-4, abs=1e-5), (
        f"{cls.__name__}: kl_mean={total:.5f} but the reported components sum "
        f"to {sum(parts.values()):.5f}. A term is computed and logged without "
        f"entering the loss. Components: "
        f"{ {k: round(v, 5) for k, v in parts.items()} }"
    )


@pytest.mark.parametrize(
    "cls,poly",
    [(MonochromaticWilsonLoss, False), (PolychromaticWilsonLoss, True)],
)
def test_profile_kl_reaches_the_gradient(cls, poly):
    """The profile KL must move the loss, not just the logs.

    Stronger than the sum check: raising `profile_kl_weight` with everything
    else fixed has to change the total. A term that is summed but multiplied by
    a dead weight would pass the first test and fail this one.
    """
    kw = dict(bg_rate=1.0, bg_concentration=1.0, n_bins=1)
    lo = cls(profile_kl_weight=1.0, **kw)(**_inputs(poly))["loss"]
    hi = cls(profile_kl_weight=10.0, **kw)(**_inputs(poly))["loss"]
    assert float(hi) > float(lo), (
        f"{cls.__name__}: profile_kl_weight 1 -> 10 left the loss at "
        f"{float(lo):.5f} -> {float(hi):.5f}; the profile KL is not in it"
    )
