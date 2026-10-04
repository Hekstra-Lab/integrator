"""Tests for the variational tilted-plane background.

The properties that matter are the ones the rest of the model relies on: the Poisson
rate must stay positive for every draw, the tilt must preserve the box total so that
`qbg.mean` keeps its meaning as the mean background per pixel, and a config without the
plane must behave exactly as it did before.
"""

import pytest
import torch

from integrator.model.distributions.background_plane import (
    BackgroundPlaneSurrogate,
    compute_background_plane_kl,
)
from integrator.utils.factory_utils import construct_integrator, resolve_config

SBOX = [1, 21, 21]
NPIX = 441


def _surrogate(**kw):
    torch.manual_seed(0)
    return BackgroundPlaneSurrogate(in_features=8, sbox_shape=SBOX, **kw)


def test_tilt_preserves_the_box_total():
    """(u, v) are centred, so the tilt averages to one however steep it is."""
    s = _surrogate(zero_init=False)
    out = s(torch.randn(16, 8), mc_samples=4)
    assert out.tilt.shape == (16, 4, NPIX)
    assert torch.allclose(
        out.tilt.mean(-1), torch.ones(16, 4), atol=1e-5
    )
    assert torch.allclose(out.mean_tilt.mean(-1), torch.ones(16), atol=1e-5)


def test_tilt_stays_positive_under_extreme_slopes():
    """The bound is what keeps a Poisson rate legal, so push the head hard."""
    s = _surrogate(zero_init=False, max_gradient=0.45, init_scale=1.0)
    with torch.no_grad():
        s.loc_head.weight.fill_(50.0)
        s.loc_head.bias.fill_(50.0)
    out = s(torch.randn(32, 8) * 10.0, mc_samples=8)
    # |g_x| + |g_y| <= 2 * max_gradient = 0.9, so the floor is 1 - 0.9
    assert out.tilt.min().item() > 0.0
    assert out.tilt.min().item() >= 0.1 - 1e-4


def test_max_gradient_must_keep_the_rate_positive():
    for bad in (0.5, 0.75, 1.0, 0.0, -0.1):
        with pytest.raises(ValueError, match="max_gradient"):
            _surrogate(max_gradient=bad)


def test_zero_init_starts_flat():
    """Training should begin from the flat background the model had before."""
    s = _surrogate(zero_init=True)
    out = s(torch.randn(8, 8), mc_samples=1)
    assert torch.allclose(out.mean_tilt, torch.ones(8, NPIX), atol=1e-6)
    assert torch.allclose(out.gradient, torch.zeros(8, 2), atol=1e-6)


def test_kl_matches_the_analytic_normal_value():
    s = _surrogate(zero_init=True, init_scale=0.05, prior_scale=0.1)
    out = s(torch.randn(4, 8), mc_samples=1)
    kl = compute_background_plane_kl(out, 1.0, 4, torch.device("cpu"))
    # loc is 0 at init, so KL per dim is log(p/q) + q^2/(2 p^2) - 1/2
    import math

    expected = 2 * (math.log(0.1 / 0.05) + 0.05**2 / (2 * 0.1**2) - 0.5)
    assert kl.shape == (4,)
    assert kl.mean().item() == pytest.approx(expected, rel=1e-4)


def test_kl_is_zero_when_absent_or_unweighted():
    dev = torch.device("cpu")
    assert torch.equal(
        compute_background_plane_kl(None, 1.0, 5, dev), torch.zeros(5)
    )
    out = _surrogate()(torch.randn(5, 8), mc_samples=1)
    assert torch.equal(
        compute_background_plane_kl(out, 0.0, 5, dev), torch.zeros(5)
    )


def _cfg(with_plane: bool) -> dict:
    cfg = {
        "integrator": {
            "name": "hierarchical",
            "args": {
                "data_dim": "2d", "d": 1, "h": 21, "w": 21,
                "encoder_out": 64, "mc_samples": 2,
            },
        },
        "loss": {
            "name": "polychromatic_wilson",
            "args": {
                "degree": 4, "lambda_min": 0.95, "lambda_max": 1.2,
                "init_B": 30.0, "n_bins": 1,
            },
        },
        "data_loader": {"name": "polychromatic_data", "args": {"data_dir": None}},
        "optimizer": {"lr": 1e-3},
    }
    if with_plane:
        cfg["surrogates"] = {
            "qbg_plane": {"name": "background_plane", "args": {"prior_scale": 0.1}}
        }
    return cfg


def _batch(b=6):
    torch.manual_seed(1)
    counts = torch.poisson(torch.full((b, NPIX), 12.0))
    meta = {
        "d": torch.full((b,), 2.5),
        "group_label": torch.zeros(b),
        "wavelength": torch.full((b,), 1.05),
    }
    return counts, counts.clone(), torch.ones(b, NPIX), meta


@pytest.mark.parametrize("with_plane", [False, True])
def test_full_factory_path_and_backward(with_plane):
    """Go through resolve_config/construct_integrator, not the class directly."""
    cfg = resolve_config(_cfg(with_plane))
    if with_plane:
        args = cfg["surrogates"]["qbg_plane"]["args"]
        # resolve_config supplies these, so a config never has to
        assert args["sbox_shape"] == SBOX
        assert args["in_features"] == 64

    model = construct_integrator(cfg)
    counts, shoebox, mask, meta = _batch()
    out = model(counts, shoebox, mask, meta)
    fo = out["forward_out"]

    assert (fo["rates"] > 0).all()
    plane = out.get("qbg_plane")
    assert (plane is not None) == with_plane

    loss = model.loss(
        rate=fo["rates"], counts=fo["counts"], qp=out["qp"], qi=out["qi"],
        qbg=out["qbg"], mask=fo["mask"],
        group_labels=meta["group_label"].long(), metadata=meta,
        qbg_plane=plane,
    )
    assert torch.isfinite(loss["loss"])
    loss["loss"].backward()

    if with_plane:
        # the per-pixel background must average back to the scalar level
        assert fo["bg_plane_mean"].shape == (6, NPIX)
        assert torch.allclose(
            fo["bg_plane_mean"].mean(-1), fo["qbg_mean"].reshape(-1), rtol=1e-5
        )
        assert fo["bg_gradient"].shape == (6, 2)
        grad = model.surrogates["qbg_plane"].loc_head.weight.grad
        assert grad is not None and grad.abs().sum() > 0
    else:
        assert loss["kl_bg_plane_mean"].item() == 0.0
        assert "bg_plane_mean" not in fo
