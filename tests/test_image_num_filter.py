"""The loader can cut a dataset to a range of images.

Until now the batch cut was baked in at shoebox-build time by
`refltorch.mksbox-laue --max-images`, so comparing a model against a baseline
MTZ cut to a different batch range meant regenerating 6.6 GB of shoeboxes. The
range belongs in the config: one dataset covering every frame then serves a
360-frame run and a 999-frame run, and can be matched to whatever the baseline
used.

Half-open [min, max), so the bound reads like a frame count: `max_image_num
360` keeps frames 0..359.
"""

import numpy as np
import pytest
import torch
import yaml

from integrator.data_loaders.data_module import RotationDataModule
from integrator.data_loaders.poly_data_module import PolychromaticDataModule

N, P = 40, 25


def _write(tmp_path, n=N):
    """A dataset whose image_num runs 0..n-1, one reflection per image."""
    counts = np.arange(n * P, dtype=np.float32).reshape(n, P)
    masks = np.ones((n, P), dtype=np.float32)
    np.save(tmp_path / "counts.npy", counts)
    np.save(tmp_path / "masks.npy", masks)
    reference = {
        "image_num": torch.arange(n, dtype=torch.float32),
        "d": torch.full((n,), 2.0),
        "asu_id": torch.arange(n),
        "H": torch.zeros(n),
        "K": torch.zeros(n),
        "L": torch.zeros(n),
        "intensity.sum.value": torch.ones(n),
        "intensity.sum.variance": torch.ones(n),
        "intensity.prf.value": torch.ones(n),
        "intensity.prf.variance": torch.ones(n),
    }
    torch.save(reference, tmp_path / "metadata.pt")
    (tmp_path / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "geometry": {"d": 1, "h": 5, "w": 5, "data_dim": "2d"},
                "n_reflections": n,
                "polychromatic": True,
                "anscombe": False,
                "files": {
                    "counts": "counts.npy",
                    "masks": "masks.npy",
                    "reference": "metadata.pt",
                },
                "stats": {"raw": [0.0, 1.0], "anscombe": [0.0, 1.0]},
            }
        )
    )
    return {
        "counts": "counts.npy",
        "masks": "masks.npy",
        "reference": "metadata.pt",
    }


def _kept(dm):
    dm.setup()
    img = dm.full_dataset.reference["image_num"]
    return np.asarray(img, dtype=float)


@pytest.mark.parametrize(
    "lo,hi,want",
    [
        (None, 10, list(range(10))),
        (5, 10, [5, 6, 7, 8, 9]),
        (30, None, list(range(30, N))),
        (None, None, list(range(N))),
    ],
)
def test_range_is_half_open(tmp_path, lo, hi, want):
    files = _write(tmp_path)
    dm = RotationDataModule(
        data_dir=str(tmp_path),
        shoebox_file_names=files,
        batch_size=4,
        min_valid_pixels=1,
        min_image_num=lo,
        max_image_num=hi,
    )
    assert sorted(_kept(dm)) == want


def test_counts_and_metadata_stay_aligned(tmp_path):
    """The cut must slice counts, masks and every metadata column together."""
    files = _write(tmp_path)
    dm = RotationDataModule(
        data_dir=str(tmp_path),
        shoebox_file_names=files,
        batch_size=4,
        min_valid_pixels=1,
        min_image_num=7,
        max_image_num=12,
    )
    dm.setup()
    ds = dm.full_dataset
    img = np.asarray(ds.reference["image_num"], dtype=float)
    # counts row i was built as arange, so its first pixel is image_num * P
    first_pixel = np.asarray(ds.counts[:, 0], dtype=float)
    assert np.allclose(first_pixel, img * P)
    assert len(ds.counts) == len(ds.masks) == len(img) == 5


def test_empty_selection_is_an_error_not_a_crash_later(tmp_path):
    files = _write(tmp_path)
    dm = RotationDataModule(
        data_dir=str(tmp_path),
        shoebox_file_names=files,
        batch_size=4,
        min_valid_pixels=1,
        min_image_num=10_000,
    )
    with pytest.raises(ValueError, match="selected no reflections"):
        dm.setup()


def test_missing_image_num_column_is_reported(tmp_path):
    files = _write(tmp_path)
    reference = torch.load(tmp_path / "metadata.pt", weights_only=False)
    del reference["image_num"]
    torch.save(reference, tmp_path / "metadata.pt")
    dm = RotationDataModule(
        data_dir=str(tmp_path),
        shoebox_file_names=files,
        batch_size=4,
        min_valid_pixels=1,
        max_image_num=10,
    )
    with pytest.raises(KeyError, match="image_num"):
        dm.setup()


def test_polychromatic_loader_cuts_the_same_way(tmp_path):
    """The Laue loader is a separate class, so it needs its own coverage."""
    files = _write(tmp_path)
    dm = PolychromaticDataModule(
        data_dir=str(tmp_path),
        shoebox_file_names=files,
        batch_size=4,
        min_valid_pixels=1,
        max_image_num=10,
    )
    dm.setup()
    img = np.asarray(dm.full_dataset.reference["image_num"], dtype=float)
    assert sorted(img) == list(range(10))
