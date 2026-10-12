"""1.5-entmax: a simplex map that can say "no mass here".

`softmax` is strictly positive everywhere, so a Bragg profile built from it
cannot put exactly zero on a pixel, and the gradient pi (1 - pi) vanishes
before the mass is gone. Over a 625-pixel shoebox the residue adds up: the
softmax profile families hold 4-13% of their mass more than 5 px from the spot
centre, against 0.2% for the families that normalize a Gaussian-tailed
amplitude instead.

Entmax (Peters, Niculae & Martins 2019, arXiv:1905.05702) interpolates
softmax (alpha = 1) and sparsemax (alpha = 2); alpha = 1.5 has a closed form
and yields exact zeros.
"""

import pytest
import torch

from integrator.model.distributions.profile_surrogates import (
    entmax15,
    get_normalizer,
    masked_entmax15,
    masked_softmax,
)


def test_it_lies_on_the_simplex():
    y = entmax15(torch.randn(7, 40))
    assert torch.allclose(y.sum(-1), torch.ones(7), atol=1e-5)
    assert (y >= 0).all()


def test_it_produces_exact_zeros_where_softmax_cannot():
    """The whole point: a peaked input must leave most pixels at exactly 0."""
    z = torch.full((1, 200), -8.0)
    z[0, 100] = 8.0
    y = entmax15(z)
    s = masked_softmax(z, None)
    assert (y == 0).sum() > 190
    assert (s > 0).all()


def test_the_tail_is_orders_of_magnitude_smaller():
    """What this was added to fix, on a real 25x25 box and the real cutoff."""
    side = 25
    c = (side - 1) / 2
    yy, xx = torch.meshgrid(
        torch.arange(side).float(), torch.arange(side).float(), indexing="ij"
    )
    r = ((yy - c) ** 2 + (xx - c) ** 2).sqrt().reshape(-1)
    z = (-((r / 1.5) ** 2) / 2).unsqueeze(0)  # a spot about 1.5 px wide
    far = (r > 5.0).unsqueeze(0)

    e = entmax15(z)
    sm = masked_softmax(z, None)
    assert e[far].sum() == 0.0
    assert sm[far].sum() > 0.0
    assert e[far].sum() < sm[far].sum()


def test_it_is_order_preserving_on_its_support():
    """Zeros tie, so the ordering claim only holds where the profile is live."""
    z = torch.randn(1, 50)
    y = entmax15(z)
    live = y[0] > 0
    assert live.sum() >= 2
    zl, yl = z[0][live], y[0][live]
    assert torch.equal(yl.argsort(), zl.argsort())
    assert y.argmax() == z.argmax()


def test_a_uniform_input_gives_a_uniform_profile():
    y = entmax15(torch.zeros(1, 16))
    assert torch.allclose(y, torch.full((1, 16), 1 / 16), atol=1e-6)


def test_the_closed_form_backward_matches_numerical_gradients():
    """The backward is written by hand, so check it against finite differences."""
    torch.manual_seed(0)
    z = torch.randn(1, 12, dtype=torch.double, requires_grad=True)
    # a non-symmetric downstream scalar, so every component contributes
    w = torch.randn(1, 12, dtype=torch.double)

    def f(x):
        return (entmax15(x) * w).sum()

    assert torch.autograd.gradcheck(f, (z,), eps=1e-6, atol=1e-7)


def test_gradient_is_zero_off_the_support():
    """A pixel at exactly zero should not be pushed around by the loss."""
    z = torch.full((1, 30), -6.0, requires_grad=True)
    with torch.no_grad():
        z[0, 15] = 6.0
    z.retain_grad()
    y = entmax15(z)
    y.sum().backward()
    off = y.detach() == 0
    assert off.sum() > 20
    assert torch.allclose(z.grad[off], torch.zeros_like(z.grad[off]))


def test_masking_keeps_the_profile_on_the_valid_pixels():
    z = torch.randn(3, 20)
    mask = torch.ones(3, 20)
    mask[:, 10:] = 0
    y = masked_entmax15(z, mask)
    assert torch.allclose(y.sum(-1), torch.ones(3), atol=1e-5)
    assert (y[:, 10:] == 0).all()


def test_a_fully_masked_shoebox_stays_finite():
    """`masked_softmax` returns the uniform profile here; entmax must not NaN."""
    y = masked_entmax15(torch.randn(2, 12), torch.zeros(2, 12))
    assert torch.isfinite(y).all()
    assert torch.allclose(y.sum(-1), torch.ones(2), atol=1e-5)


def test_it_carries_a_sample_axis():
    y = masked_entmax15(torch.randn(4, 3, 18), torch.ones(3, 18))
    assert y.shape == (4, 3, 18)
    assert torch.allclose(y.sum(-1), torch.ones(4, 3), atol=1e-5)


def test_the_registry_resolves_both_and_rejects_the_rest():
    assert get_normalizer("softmax") is masked_softmax
    assert get_normalizer("entmax15") is masked_entmax15
    with pytest.raises(ValueError, match="unknown profile normalization"):
        get_normalizer("sparsemax")
