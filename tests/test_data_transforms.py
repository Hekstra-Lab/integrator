"""Tests for the count transforms, focused on the asinh transform for real-valued data."""

import numpy as np
import pytest
import torch

from integrator.data_loaders.data_module import RotationDataModule
from integrator.io.dataset import write_dataset_yaml


def _write_dataset(path, counts: np.ndarray, h=4, w=4):
    """Write a minimal accepted dataset (counts + masks + metadata + dataset.yaml)."""
    n = counts.shape[0]
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "counts.npy", counts.astype(np.float32))
    np.save(path / "masks.npy", np.ones((n, h * w), dtype=bool))
    np.save(
        path / "metadata.npy",
        {
            "intensity.prf.variance": np.zeros(n, dtype=np.float32),
            "d": np.ones(n, dtype=np.float32),
            "refl_ids": np.arange(n, dtype=np.int64),
        },
        allow_pickle=True,
    )
    valid = counts.reshape(-1)
    write_dataset_yaml(
        path,
        geometry={"d": 1, "h": h, "w": w, "data_dim": "2d"},
        n_reflections=n,
        polychromatic=False,
        anscombe=False,
        files={"counts": "counts.npy", "masks": "masks.npy", "reference": "metadata.npy"},
        stats={"raw": [float(valid.mean()), float(valid.var())], "anscombe": [0.0, 1.0]},
    )


def _setup(path, **kwargs):
    dm = RotationDataModule(data_dir=str(path), batch_size=4, min_valid_pixels=1, **kwargs)
    dm.setup()
    return dm.full_dataset


def test_asinh_is_accepted():
    RotationDataModule(data_dir=".", transform="asinh")  # no ValueError at construction


def test_asinh_rejects_bad_scale():
    with pytest.raises(ValueError):
        RotationDataModule(data_dir=".", transform="asinh", asinh_scale=0.0)


def test_asinh_matches_formula_and_is_finite_on_negatives(tmp_path):
    rng = np.random.default_rng(0)
    counts = rng.normal(2.0, 5.0, size=(8, 16)).astype(np.float32)  # ~40% negative
    assert (counts < 0).mean() > 0.2
    _write_dataset(tmp_path, counts)

    ds = _setup(tmp_path, transform="asinh", asinh_scale=2.0)
    got = ds.standardized_counts
    expected = torch.asinh(ds.counts / 2.0) * ds.masks
    torch.testing.assert_close(got, expected)
    assert torch.isfinite(got).all()  # the whole point: defined where counts < 0


def test_asinh_scale_sets_the_knee(tmp_path):
    counts = np.full((4, 16), 10.0, dtype=np.float32)
    _write_dataset(tmp_path, counts)
    small = _setup(tmp_path, transform="asinh", asinh_scale=1.0).standardized_counts
    large = _setup(tmp_path, transform="asinh", asinh_scale=5.0).standardized_counts
    # larger scale keeps more of the count in the linear regime -> larger transformed value
    assert float(large.mean()) < float(small.mean())  # asinh(10/5) < asinh(10/1)
    assert abs(float(small[0, 0]) - float(np.arcsinh(10.0))) < 1e-5
