"""Reading the per-observation scale from metadata instead of modelling it.

The merge is closed-form given the scale -- `I_h` is a ratio of two scatter
sums -- so pinning the scale to an external reference and finalising, with no
retraining, separates "our scale is wrong" from "our merge is wrong".  That
distinction is the whole polychromatic question: on a still a Friedel pair is
never excited at the same wavelength, so unlike the monochromatic case the
scale's error enters `I(+) - I(-)` with no common-mode cancellation.

Built through `construct_integrator`, since a config key that is only read is
not tested.
"""

import pytest
import torch

from integrator.model.scaling import AmortizedMergingIntegrator
from integrator.utils.factory_utils import construct_integrator

COL = "careless_scale"


def _cfg(**integrator_args):
    args = {
        "data_dim": "2d",
        "d": 1,
        "h": 21,
        "w": 21,
        "mc_samples": 3,
        "n_hkl": 64,
        "wilson_alpha": 1.0,
        "scale_laue_mlp": True,
        "scale_n_images": 8,
    }
    args.update(integrator_args)
    return {
        "integrator": {"name": "amortized_merging", "args": args},
        "optimizer": {"lr": 1e-3},
        # amortized_merging takes five: profile, intensity, background,
        # and the two the merge head reads
        "encoders": [
            {"name": "profile_encoder", "args": {"data_dim": "2d"}},
            {"name": "intensity_encoder", "args": {"data_dim": "2d"}},
            {"name": "intensity_encoder", "args": {"data_dim": "2d"}},
            {"name": "intensity_encoder", "args": {"data_dim": "2d"}},
            {"name": "intensity_encoder", "args": {"data_dim": "2d"}},
        ],
        "surrogates": {
            "qp": {
                "name": "learned_basis_profile",
                "args": {"input_dim": 64, "output_dim": 441},
            },
            "qbg": {"name": "gammaA", "args": {"in_features": 64}},
        },
        "loss": {
            "name": "polychromatic_wilson",
            "args": {"eps": 1e-6, "intensity_kl_weight": 0.0},
        },
        "data_loader": {"name": "rotation_data", "args": {"data_dir": "/tmp"}},
    }


def _metadata(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "asu_id": torch.randint(0, 16, (n,), generator=g),
        "d": torch.rand(n, generator=g) * 2 + 1.5,
        "wavelength": torch.rand(n, generator=g) * 0.28 + 0.97,
        "xyzcal.px.0": torch.rand(n, generator=g) * 2000,
        "xyzcal.px.1": torch.rand(n, generator=g) * 2000,
        "xyzcal.px.2": torch.zeros(n),
        "image_num": torch.randint(0, 8, (n,), generator=g),
        "lp": torch.rand(n, generator=g) * 0.5 + 0.5,
        COL: torch.rand(n, generator=g) * 40 + 1.0,
    }


def _build(**over):
    torch.manual_seed(0)
    return construct_integrator(_cfg(**over))


def test_the_key_defaults_off_so_existing_runs_are_unchanged():
    model = _build()
    assert isinstance(model, AmortizedMergingIntegrator)
    assert model.scale_from_metadata is None


def test_the_override_returns_the_column_itself():
    model = _build(scale_from_metadata=COL)
    md = _metadata()
    got = model._get_scale(md, torch.device("cpu")).reshape(-1)
    assert torch.allclose(got, md[COL].float(), rtol=0, atol=0)


def test_without_the_override_the_scale_model_is_what_runs():
    """The default path must not be reachable from the metadata column."""
    model = _build()
    md = _metadata()
    got = model._get_scale(md, torch.device("cpu")).reshape(-1)
    assert got.shape == md[COL].shape
    assert not torch.allclose(got, md[COL].float())


def test_a_pinned_scale_is_what_the_merge_divides_by():
    """`beta` carries the scale, so pinning it has to change the merge."""
    md = _metadata()
    pinned = _build(scale_from_metadata=COL)._get_scale(md, torch.device("cpu"))
    learned = _build()._get_scale(md, torch.device("cpu"))
    # the merge's denominator is a scatter sum of scale * sum(profile); with a
    # shared profile the ratio of denominators is the ratio of scale sums
    assert not torch.isclose(pinned.sum(), learned.sum())
    assert float(pinned.min()) > 0.0


def test_non_positive_entries_are_floored_not_passed_through():
    """A non-positive scale would make the merge's beta non-positive."""
    model = _build(scale_from_metadata=COL)
    md = _metadata()
    md[COL] = torch.tensor([-1.0, 0.0] + [2.0] * (len(md["d"]) - 2))
    got = model._get_scale(md, torch.device("cpu")).reshape(-1)
    assert float(got.min()) > 0.0
    assert got[2:].eq(2.0).all()


def test_a_missing_column_says_which_one_and_what_to_do():
    model = _build(scale_from_metadata="not_a_column")
    with pytest.raises(KeyError, match="not_a_column"):
        model._get_scale(_metadata(), torch.device("cpu"))
