"""Coordinate channels for the profile encoder.

The channels put every reflection in the beam-centred radial/tangential frame, so
one kernel can express an elongated spot at any azimuth. These check the frame is
actually the one claimed, that the feature is inert when off, and that the factory
widens the encoder to match -- the failure that would otherwise appear only at the
first batch.
"""

import math

import pytest
import torch

from integrator.model.integrators.integrator_utils import coordinate_channels

BC = (2000.0, 2000.0)


def meta(x, y):
    return {
        "xyzcal.px.0": torch.tensor(x, dtype=torch.float32),
        "xyzcal.px.1": torch.tensor(y, dtype=torch.float32),
    }


def test_shape_2d_and_3d():
    m = meta([2100.0], [2000.0])
    assert coordinate_channels(m, (7, 7), BC).shape == (1, 2, 7, 7)
    assert coordinate_channels(m, (3, 7, 7), BC).shape == (1, 2, 3, 7, 7)


def test_radial_runs_along_x_at_zero_azimuth():
    ch = coordinate_channels(meta([2100.0], [2000.0]), (5, 5), BC)
    # radial increases left-to-right, and is flat down a column
    assert ch[0, 0, 2, -1] > ch[0, 0, 2, 0]
    assert torch.allclose(ch[0, 0, :, 2], torch.zeros(5), atol=1e-6)


def test_radial_runs_along_y_at_ninety_degrees():
    ch = coordinate_channels(meta([2000.0], [2100.0]), (5, 5), BC)
    assert ch[0, 0, -1, 2] > ch[0, 0, 2, 2]
    assert torch.allclose(ch[0, 0, 2, :], torch.zeros(5), atol=1e-6)


def test_frame_is_orthonormal_everywhere():
    """Radial and tangential must stay a rotation: lengths and angle preserved."""
    m = meta([2100.0, 2000.0, 1900.0, 2070.0], [2000.0, 2100.0, 1950.0, 1930.0])
    ch = coordinate_channels(m, (9, 9), BC)
    r, t = ch[:, 0], ch[:, 1]
    yy, xx = torch.meshgrid(torch.arange(9.0), torch.arange(9.0), indexing="ij")
    du, dv = (xx - 4) / 4.0, (yy - 4) / 4.0
    want = du**2 + dv**2
    for i in range(ch.shape[0]):
        assert torch.allclose(r[i] ** 2 + t[i] ** 2, want, atol=1e-5)


def test_depth_slices_are_identical():
    ch = coordinate_channels(meta([2100.0], [2000.0]), (4, 5, 5), BC)
    for d in range(1, 4):
        assert torch.allclose(ch[0, :, 0], ch[0, :, d])


def test_channels_are_order_one():
    ch = coordinate_channels(meta([2100.0], [2000.0]), (21, 21), BC)
    assert ch.abs().max() <= math.sqrt(2.0) + 1e-5


def test_missing_geometry_is_a_clear_error():
    with pytest.raises(KeyError, match="coord_channels"):
        coordinate_channels({}, (5, 5), BC)
    with pytest.raises(KeyError, match="coord_channels"):
        coordinate_channels(meta([2100.0], [2000.0]), (5, 5), None)


def test_factory_widens_the_profile_encoder_only():
    from integrator.utils.factory_utils import _inject_coord_channels

    cfg = {
        "integrator": {"args": {"profile_coord_channels": True,
                                "beam_center": [1.0, 2.0]}},
        "encoders": [
            {"name": "profile_encoder", "args": {"in_channels": 1}},
            {"name": "intensity_encoder", "args": {"in_channels": 1}},
        ],
    }
    _inject_coord_channels(cfg)
    assert cfg["encoders"][0]["args"]["in_channels"] == 3
    assert cfg["encoders"][1]["args"]["in_channels"] == 1


def test_factory_is_inert_when_disabled():
    from integrator.utils.factory_utils import _inject_coord_channels

    cfg = {
        "integrator": {"args": {}},
        "encoders": [{"name": "profile_encoder", "args": {"in_channels": 1}}],
    }
    _inject_coord_channels(cfg)
    assert cfg["encoders"][0]["args"]["in_channels"] == 1
