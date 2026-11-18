"""Tests for the Poisson / Negative Binomial pixel count likelihood."""

import pytest
import torch
from torch.distributions import Dirichlet, Gamma, Poisson

from integrator.model.loss.count_likelihood import CountLikelihood
from integrator.model.loss.monochromatic_wilson_loss import (
    MonochromaticWilsonLoss,
)


def _reference_poisson_neg_ll(rate, counts, mask):
    ll = Poisson(rate.clamp(min=1e-12)).log_prob(counts.unsqueeze(1))
    ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
    return (-ll_mean).sum(1)


@pytest.fixture
def synthetic():
    torch.manual_seed(0)
    b, s, p = 5, 8, 9
    rate = torch.rand(b, s, p) * 20 + 0.1
    counts = torch.randint(0, 25, (b, p)).float()
    mask = torch.ones(b, p, 1)
    mask[0, -2:] = 0.0  # exercise masking
    return rate, counts, mask


def test_poisson_matches_reference(synthetic):
    rate, counts, mask = synthetic
    lik = CountLikelihood("poisson")
    got = lik.neg_ll(rate, counts, mask)
    ref = _reference_poisson_neg_ll(rate, counts, mask)
    torch.testing.assert_close(got, ref)


def test_poisson_has_no_parameters():
    """Default likelihood must add nothing to the state dict (checkpoint compat)."""
    lik = CountLikelihood("poisson")
    assert list(lik.parameters()) == []
    assert lik.state_dict() == {}


def test_normal_coupled_matches_reference(synthetic):
    """Var = rate + read_noise^2, evaluated as a Normal on the (real-valued) counts."""
    rate, counts, mask = synthetic
    counts = counts - 3.0  # push some pixels negative: the real-valued regime
    lik = CountLikelihood("normal", read_noise=0.5, variance="coupled")
    std = (rate.clamp(min=1e-12) + 0.5**2).sqrt()
    ll = torch.distributions.Normal(rate.clamp(min=1e-12), std).log_prob(
        counts.unsqueeze(1)
    )
    ref = (-(torch.mean(ll, dim=1) * mask.squeeze(-1))).sum(1)
    torch.testing.assert_close(lik.neg_ll(rate, counts, mask), ref)


def test_normal_coupled_is_checkpoint_clean():
    """Coupled Normal adds no saved state, exactly like Poisson."""
    lik = CountLikelihood("normal", read_noise=0.024, variance="coupled")
    assert list(lik.parameters()) == []
    assert lik.state_dict() == {}  # read_noise is a non-persistent buffer


def test_normal_free_learns_sigma(synthetic):
    rate, counts, mask = synthetic
    lik = CountLikelihood("normal", variance="free", sigma_init=2.0)
    assert "raw_sigma" in dict(lik.named_parameters())
    assert abs(float(lik.sigma().detach()) - 2.0) < 1e-3
    loss = lik.neg_ll(rate, counts, mask).sum()
    loss.backward()
    assert lik.raw_sigma.grad is not None
    assert torch.isfinite(lik.raw_sigma.grad).all()
    assert lik.raw_sigma.grad.abs().sum() > 0


def test_normal_finite_on_negative_counts(synthetic):
    """The whole point: a Normal likelihood is defined where Poisson is not."""
    rate, _, mask = synthetic
    neg = torch.full(rate.shape[:1] + rate.shape[2:], -0.5)
    for variance in ("coupled", "free"):
        lik = CountLikelihood("normal", read_noise=0.024, variance=variance)
        assert torch.isfinite(lik.neg_ll(rate, neg, mask)).all()


def test_normal_rejects_bad_variance():
    with pytest.raises(ValueError):
        CountLikelihood("normal", variance="bogus")


def test_nb_reduces_to_poisson_in_limit(synthetic):
    """As dispersion r -> inf the NB reparameterization must match Poisson.

    Checked in float64: at very large r the float32 NB log_prob loses precision
    to lgamma cancellation, which is a numerical artifact, not a reparameterization
    error (learned r stays modest in practice).
    """
    rate, counts, mask = (t.double() for t in synthetic)
    nb = CountLikelihood("negative_binomial", dispersion_init=1e8).double()
    poi = CountLikelihood("poisson")
    torch.testing.assert_close(
        nb.neg_ll(rate, counts, mask),
        poi.neg_ll(rate, counts, mask),
        rtol=1e-4,
        atol=1e-4,
    )


def test_nb_dispersion_is_learnable(synthetic):
    rate, counts, mask = synthetic
    nb = CountLikelihood("negative_binomial", dispersion_init=10.0)
    assert "raw_dispersion" in dict(nb.named_parameters())
    loss = nb.neg_ll(rate, counts, mask).sum()
    loss.backward()
    grad = nb.raw_dispersion.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_fixed_dispersion_is_a_buffer_not_a_parameter():
    nb = CountLikelihood(
        "negative_binomial", dispersion_init=7.0, learn_dispersion=False
    )
    assert list(nb.parameters()) == []  # nothing to learn
    assert "raw_dispersion" in dict(nb.named_buffers())
    assert not nb.raw_dispersion.requires_grad
    assert abs(float(nb.dispersion()) - 7.0) < 1e-2
    assert "raw_dispersion" in nb.state_dict()  # still saved/restored


def test_diagnostics():
    assert CountLikelihood("poisson").diagnostics() == {}
    nb = CountLikelihood("negative_binomial", dispersion_init=8.0)
    d = nb.diagnostics()
    assert set(d) == {"nb_dispersion"}
    assert abs(float(d["nb_dispersion"]) - 8.0) < 1e-2


def test_loss_diagnostics_exposes_G_B_and_dispersion():
    poi = MonochromaticWilsonLoss(bg_rate=1.0, bg_concentration=1.0)
    assert set(poi.diagnostics()) == {"wilson_G", "wilson_B"}
    nb = MonochromaticWilsonLoss(
        bg_rate=1.0, bg_concentration=1.0,
        likelihood="negative_binomial", nb_dispersion_init=5.0,
    )
    d = nb.diagnostics()
    assert set(d) == {"wilson_G", "wilson_B", "nb_dispersion"}
    assert abs(float(d["nb_dispersion"]) - 5.0) < 1e-2


def test_wilson_loss_fixed_dispersion_via_kwargs():
    loss = MonochromaticWilsonLoss(
        bg_rate=1.0, bg_concentration=1.0,
        likelihood="negative_binomial", nb_dispersion_init=10.0,
        nb_learn_dispersion=False,
    )
    assert "count_likelihood.raw_dispersion" not in dict(loss.named_parameters())
    assert "count_likelihood.raw_dispersion" in dict(loss.named_buffers())


def test_nb_overdispersed_data_prefers_nb():
    """For counts more spread than Poisson at a fixed rate, NB fits better."""
    torch.manual_seed(1)
    b, s, p = 3, 4, 400
    rate = torch.full((b, s, p), 8.0)
    # NB-distributed counts (overdispersed relative to Poisson mean 8)
    r_true = 2.0
    logits = torch.log(rate[:, 0]) - torch.log(torch.tensor(r_true))
    counts = torch.distributions.NegativeBinomial(
        total_count=r_true, logits=logits
    ).sample()
    mask = torch.ones(b, p, 1)
    poi = CountLikelihood("poisson").neg_ll(rate, counts, mask).sum()
    nb = (
        CountLikelihood("negative_binomial", dispersion_init=r_true)
        .neg_ll(rate, counts, mask)
        .sum()
    )
    assert nb < poi


def test_per_bin_dispersion_selects_by_group():
    nb = CountLikelihood(
        "negative_binomial", dispersion_scope="per_bin", n_bins=4
    )
    with torch.no_grad():
        nb.raw_dispersion.copy_(torch.tensor([-5.0, 0.0, 2.0, 5.0]))
    groups = torch.tensor([0, 3, 1])
    disp = nb.dispersion(groups)
    assert disp.shape == (3,)
    # bin 3 (raw 5.0) must have larger dispersion than bin 0 (raw -5.0)
    assert disp[1] > disp[0]

    with pytest.raises(ValueError):
        nb.dispersion(group_labels=None)


def _wilson_forward(loss, b=6, s=8, p=9, seed=0):
    torch.manual_seed(seed)
    rate = torch.rand(b, s, p) * 15 + 0.1
    counts = torch.randint(0, 20, (b, p)).float()
    mask = torch.ones(b, p, 1)
    qp = Dirichlet(torch.ones(b, p))
    qi = Gamma(torch.rand(b) + 1.0, torch.rand(b) + 1.0)
    qbg = Gamma(torch.rand(b) + 1.0, torch.rand(b) + 1.0)
    metadata = {"d": torch.rand(b) * 3 + 1.0}
    group_labels = torch.zeros(b, dtype=torch.long)
    return loss(
        rate=rate,
        counts=counts,
        qp=qp,
        qi=qi,
        qbg=qbg,
        mask=mask,
        group_labels=group_labels,
        metadata=metadata,
    )


def test_wilson_loss_poisson_is_default():
    loss = MonochromaticWilsonLoss()
    assert loss.count_likelihood.name == "poisson"
    assert "count_likelihood.raw_dispersion" not in loss.state_dict()
    out = _wilson_forward(loss)
    assert torch.isfinite(out["loss"])


def test_wilson_loss_negative_binomial_via_factory_kwargs():
    """Construct exactly as the factory would from loss.args, then train-step."""
    loss = MonochromaticWilsonLoss(
        lp_correction=False,
        init_B=30.0,
        b_min=0.0,
        intensity_kl_weight=1.0,
        background_kl_weight=1.0,
        profile_kl_weight=1.0,
        bg_rate=1.0,
        bg_concentration=1.0,
        n_bins=1,
        likelihood="negative_binomial",
        nb_dispersion_init=10.0,
        nb_dispersion_scope="global",
    )
    assert "count_likelihood.raw_dispersion" in loss.state_dict()

    out = _wilson_forward(loss)
    assert torch.isfinite(out["loss"])

    out["loss"].backward()
    grad = loss.count_likelihood.raw_dispersion.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().sum() > 0
