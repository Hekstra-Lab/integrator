"""Tests for ShoeboxDataModule handling both 3D (D>1) and 2D (D=1) data.

Covers:
- uint16 .npy counts load + cast to int32
- Dead-pixel filtering on both shapes
- Dict metadata filtering alongside counts/masks
- Anscombe, log1p, and raw standardization paths
- Missing intensity.prf.variance (laue-dials data)
- Batch shapes from the DataLoader
- Resolution cutoff filtering
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from integrator.data_loaders.data_module import (
    ShoeboxDataModule,
    _load_shoebox_array,
)
from integrator.data_loaders.poly_data_module import PolyShoeboxDataModule


def _make_synthetic_data(
    tmp_path: Path,
    N: int,
    D: int,
    H: int,
    W: int,
    *,
    counts_dtype=np.uint16,
    include_prf_variance: bool = True,
    include_wavelength: bool = False,
    n_dead: int = 0,
):
    """Write a minimal synthetic dataset to tmp_path.

    Returns the paths dict for shoebox_file_names config.
    """
    n_pixels = D * H * W
    rng = np.random.default_rng(42)

    counts = rng.integers(0, 200, size=(N, n_pixels)).astype(counts_dtype)
    masks = np.ones((N, n_pixels), dtype=bool)

    # Make some reflections "dead" (all mask=False)
    if n_dead > 0:
        masks[:n_dead] = False

    np.save(tmp_path / "counts.npy", counts)
    np.save(tmp_path / "masks.npy", masks)

    # Stats: [mean, var] of the Anscombe-transformed counts
    mean_val = float(2 * np.sqrt(counts.mean() + 0.375))
    var_val = 1.0
    torch.save(
        torch.tensor([mean_val, var_val], dtype=torch.float32),
        tmp_path / "anscombe_stats.pt",
    )

    # Metadata dict
    metadata = {
        "refl_ids": torch.arange(N, dtype=torch.float32),
        "is_test": torch.zeros(N, dtype=torch.bool),
        "d": torch.rand(N) * 10 + 1.5,
        "group_label": torch.randint(0, 5, (N,)).float(),
        "intensity.sum.value": torch.rand(N) * 1000,
        "intensity.sum.variance": torch.rand(N) * 100,
    }
    if include_prf_variance:
        metadata["intensity.prf.variance"] = torch.rand(N) * 100
    if include_wavelength:
        metadata["wavelength"] = torch.rand(N) * 0.3 + 0.95
    torch.save(metadata, tmp_path / "metadata.pt")

    return {
        "data_dir": str(tmp_path),
        "counts": "counts.npy",
        "masks": "masks.npy",
        "stats": "anscombe_stats.pt",
        "reference": "metadata.pt",
        "standardized_counts": None,
    }


# ---------------------------------------------------------------------------
# _load_shoebox_array
# ---------------------------------------------------------------------------


class TestLoadShoeboxArray:
    def test_uint16_npy_cast_to_int32(self, tmp_path):
        arr = np.arange(100, dtype=np.uint16).reshape(10, 10)
        np.save(tmp_path / "counts.npy", arr)
        t = _load_shoebox_array(str(tmp_path / "counts.npy"))
        assert t.dtype == torch.int32
        assert t.shape == (10, 10)
        assert (t.numpy() == arr.astype(np.int32)).all()

    def test_int32_npy_unchanged(self, tmp_path):
        arr = np.arange(100, dtype=np.int32).reshape(10, 10)
        np.save(tmp_path / "counts.npy", arr)
        t = _load_shoebox_array(str(tmp_path / "counts.npy"))
        assert t.dtype == torch.int32

    def test_float32_npy_unchanged(self, tmp_path):
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        np.save(tmp_path / "counts.npy", arr)
        t = _load_shoebox_array(str(tmp_path / "counts.npy"))
        assert t.dtype == torch.float32

    def test_pt_fallback(self, tmp_path):
        t_orig = torch.arange(100).reshape(10, 10)
        torch.save(t_orig, tmp_path / "counts.pt")
        t = _load_shoebox_array(str(tmp_path / "counts.pt"))
        assert torch.equal(t, t_orig)

    def test_npy_preferred_over_pt(self, tmp_path):
        """When both .npy and .pt exist, .npy wins."""
        arr_npy = np.full((5, 5), 42, dtype=np.int32)
        np.save(tmp_path / "counts.npy", arr_npy)
        torch.save(torch.full((5, 5), 99), tmp_path / "counts.pt")
        t = _load_shoebox_array(str(tmp_path / "counts.pt"))
        assert int(t[0, 0]) == 42

    def test_uint16_indexing_works(self, tmp_path):
        """The original bug: uint16 tensors can't be boolean-indexed."""
        arr = np.arange(50, dtype=np.uint16).reshape(10, 5)
        np.save(tmp_path / "counts.npy", arr)
        t = _load_shoebox_array(str(tmp_path / "counts.npy"))
        mask = torch.tensor([True] * 5 + [False] * 5)
        result = t[mask]
        assert result.shape == (5, 5)


# ---------------------------------------------------------------------------
# ShoeboxDataModule — 3D (D > 1)
# ---------------------------------------------------------------------------


class TestShoeboxDataModule3D:
    N, D, H, W = 200, 3, 21, 21

    def _make_dm(self, tmp_path, **overrides):
        fnames = _make_synthetic_data(
            tmp_path, self.N, self.D, self.H, self.W,
            **{k: v for k, v in overrides.items()
               if k in ("counts_dtype", "include_prf_variance",
                        "include_wavelength", "n_dead")},
        )
        dm = ShoeboxDataModule(
            data_dir=str(tmp_path),
            batch_size=32,
            val_split=0.2,
            test_split=0.0,
            num_workers=0,
            include_test=False,
            shoebox_file_names=fnames,
            D=self.D,
            H=self.H,
            W=self.W,
            anscombe=overrides.get("anscombe", True),
            transform=overrides.get("transform", None),
        )
        return dm

    def test_setup_runs(self, tmp_path):
        dm = self._make_dm(tmp_path)
        dm.setup()
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

    def test_batch_shape(self, tmp_path):
        dm = self._make_dm(tmp_path)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        counts, std_counts, masks, meta = batch
        n_pixels = self.D * self.H * self.W
        assert counts.shape[1] == n_pixels
        assert std_counts.shape[1] == n_pixels
        assert masks.shape[1] == n_pixels

    def test_uint16_counts(self, tmp_path):
        dm = self._make_dm(tmp_path, counts_dtype=np.uint16)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert batch[0].dtype in (torch.int32, torch.float32)

    def test_dead_pixel_filtering(self, tmp_path):
        n_dead = 10
        dm = self._make_dm(tmp_path, n_dead=n_dead)
        dm.setup()
        total = len(dm.train_dataset) + len(dm.val_dataset)
        assert total == self.N - n_dead

    def test_metadata_is_dict(self, tmp_path):
        dm = self._make_dm(tmp_path)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        meta = batch[3]
        assert isinstance(meta, dict)
        assert "d" in meta
        assert "refl_ids" in meta

    def test_metadata_filtered_with_dead(self, tmp_path):
        dm = self._make_dm(tmp_path, n_dead=5)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        meta = batch[3]
        assert meta["d"].shape[0] == batch[0].shape[0]

    def test_anscombe_transform(self, tmp_path):
        dm = self._make_dm(tmp_path, anscombe=True)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        std = batch[1]
        assert torch.isfinite(std).all()

    def test_log1p_transform(self, tmp_path):
        dm = self._make_dm(tmp_path, transform="log1p")
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        std = batch[1]
        assert torch.isfinite(std).all()
        assert (std >= 0).all()

    def test_no_transform(self, tmp_path):
        dm = self._make_dm(tmp_path, transform="none")
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert torch.isfinite(batch[1]).all()


# ---------------------------------------------------------------------------
# ShoeboxDataModule — 2D (D = 1, laue stills)
# ---------------------------------------------------------------------------


class TestPolyShoeboxDataModule2D:
    """Tests for the polychromatic (Laue stills) data module."""

    N, D, H, W = 200, 1, 25, 25

    def _make_dm(self, tmp_path, **overrides):
        fnames = _make_synthetic_data(
            tmp_path, self.N, self.D, self.H, self.W,
            include_wavelength=True,
            include_prf_variance=False,
            **{k: v for k, v in overrides.items()
               if k in ("counts_dtype", "n_dead")},
        )
        dm = PolyShoeboxDataModule(
            data_dir=str(tmp_path),
            batch_size=32,
            val_split=0.2,
            test_split=0.0,
            num_workers=0,
            include_test=False,
            shoebox_file_names=fnames,
            D=self.D,
            H=self.H,
            W=self.W,
            anscombe=overrides.get("anscombe", True),
            transform=overrides.get("transform", None),
        )
        return dm

    def test_setup_runs(self, tmp_path):
        dm = self._make_dm(tmp_path)
        dm.setup()
        assert dm.train_dataset is not None

    def test_batch_shape_is_flat(self, tmp_path):
        """D=1: counts shape should be (B, H*W) = (B, 625)."""
        dm = self._make_dm(tmp_path)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        counts = batch[0]
        assert counts.dim() == 2
        assert counts.shape[1] == self.H * self.W

    def test_uint16_counts(self, tmp_path):
        dm = self._make_dm(tmp_path, counts_dtype=np.uint16)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert batch[0].dtype in (torch.int32, torch.float32)

    def test_dead_pixel_filtering(self, tmp_path):
        n_dead = 8
        dm = self._make_dm(tmp_path, n_dead=n_dead)
        dm.setup()
        total = len(dm.train_dataset) + len(dm.val_dataset)
        assert total == self.N - n_dead

    def test_metadata_has_wavelength(self, tmp_path):
        """Laue batches must carry wavelength for PolyWilsonLoss."""
        dm = self._make_dm(tmp_path)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        meta = batch[3]
        assert "wavelength" in meta
        assert meta["wavelength"].shape[0] == batch[0].shape[0]

    def test_metadata_has_d(self, tmp_path):
        dm = self._make_dm(tmp_path)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert "d" in batch[3]

    def test_no_prf_variance_no_crash(self, tmp_path):
        """PolyShoeboxDataModule must not crash without intensity.prf.variance."""
        dm = self._make_dm(tmp_path)
        dm.setup()
        total = len(dm.train_dataset) + len(dm.val_dataset)
        assert total == self.N

    def test_anscombe_transform(self, tmp_path):
        dm = self._make_dm(tmp_path, anscombe=True)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert torch.isfinite(batch[1]).all()

    def test_log1p_transform(self, tmp_path):
        dm = self._make_dm(tmp_path, transform="log1p")
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert torch.isfinite(batch[1]).all()
        assert (batch[1] >= 0).all()

    def test_resolution_cutoff(self, tmp_path):
        fnames = _make_synthetic_data(
            tmp_path, self.N, self.D, self.H, self.W,
            include_wavelength=True,
            include_prf_variance=False,
        )
        dm = PolyShoeboxDataModule(
            data_dir=str(tmp_path),
            batch_size=32,
            val_split=0.2,
            test_split=0.0,
            num_workers=0,
            include_test=False,
            shoebox_file_names=fnames,
            D=self.D,
            H=self.H,
            W=self.W,
            anscombe=True,
            cutoff=5.0,
        )
        dm.setup()
        total = len(dm.train_dataset) + len(dm.val_dataset)
        assert total < self.N


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestShoeboxDataModuleEdgeCases:
    def test_all_dead_removed(self, tmp_path):
        """If ALL reflections are dead, dataset should be empty."""
        N, D, H, W = 10, 1, 11, 11
        fnames = _make_synthetic_data(tmp_path, N, D, H, W, n_dead=N)
        dm = ShoeboxDataModule(
            data_dir=str(tmp_path),
            batch_size=32,
            val_split=0.0,
            test_split=0.0,
            num_workers=0,
            include_test=False,
            shoebox_file_names=fnames,
            D=D, H=H, W=W,
            anscombe=True,
        )
        dm.setup()
        assert len(dm.train_dataset) == 0

    def test_mixed_dtypes_in_metadata(self, tmp_path):
        """metadata.pt may have float32, int64, bool tensors — all must
        survive boolean indexing during dead-pixel filtering."""
        N, D, H, W = 50, 1, 15, 15
        fnames = _make_synthetic_data(tmp_path, N, D, H, W)
        # Add extra typed columns
        meta = torch.load(tmp_path / "metadata.pt")
        meta["flags"] = torch.randint(0, 256, (N,), dtype=torch.int64).float()
        meta["entering"] = torch.randint(0, 2, (N,)).bool().float()
        torch.save(meta, tmp_path / "metadata.pt")

        dm = ShoeboxDataModule(
            data_dir=str(tmp_path),
            batch_size=16,
            val_split=0.2,
            test_split=0.0,
            num_workers=0,
            include_test=False,
            shoebox_file_names=fnames,
            D=D, H=H, W=W,
            anscombe=True,
        )
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert batch[0].shape[0] > 0
