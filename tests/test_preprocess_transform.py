"""
Verify that _apply_transform_np (vectorised numpy, used by integrator.preprocess)
produces numerically identical output to IntegratorDataset._transform_counts
(the per-reflection torch reference used during live training).

Five transforms (anscombe, log1p, asinh, log_softplus, sqrt_squareplus) are
cross-checked against _transform_counts.  The input deliberately includes
negative pixel values to expose any wrong pre-clamping.

Standardisation is tested separately: its masking was intentionally corrected
in the preprocess path (masked pixels -> 0.0) relative to the original loader
which produces -mean/std for masked pixels.  The corrected formula is verified
directly rather than compared against the original.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# Transforms verified numerically against _transform_counts
CROSS_CHECKED_TRANSFORMS = [
    "anscombe",
    "log1p",
    "asinh",
    "log_softplus",
    "sqrt_squareplus",
]

MEAN = 30.0
VAR = 200.0


@pytest.fixture(scope="module")
def sample_data():
    """8 reflections x 441 pixels with negative values to expose wrong clamping."""
    rng = np.random.default_rng(42)
    counts = rng.uniform(-5.0, 100.0, size=(8, 441)).astype(np.float32)
    masks = rng.uniform(0.0, 1.0, size=(8, 441)) > 0.2  # ~80 % valid
    return counts, masks


class _TransformStub:
    """Minimal stand-in for IntegratorDataset exposing .transform and .stats."""

    def __init__(self, transform: str, mean: float, var: float):
        self.transform = transform
        self.stats = torch.tensor([mean, var], dtype=torch.float32)


@pytest.mark.parametrize("transform", CROSS_CHECKED_TRANSFORMS)
def test_numpy_matches_torch(sample_data, transform):
    """_apply_transform_np must be numerically identical to _transform_counts."""
    from integrator.cli.preprocess import _apply_transform_np
    from integrator.data_loaders.data_module import IntegratorDataset

    counts_np, masks_np = sample_data
    stub = _TransformStub(transform, MEAN, VAR)

    # Vectorised numpy path (full batch)
    out_np = _apply_transform_np(counts_np, masks_np, MEAN, VAR, transform)

    # Reference: original per-row torch path called row by row
    out_ref = np.stack(
        [
            IntegratorDataset._transform_counts(
                stub,
                torch.from_numpy(counts_np[i].copy()).float(),
                torch.from_numpy(masks_np[i].copy()).bool(),
            ).numpy()
            for i in range(len(counts_np))
        ]
    )

    np.testing.assert_allclose(
        out_np,
        out_ref,
        rtol=1e-5,
        atol=1e-6,
        err_msg=(
            f"Transform '{transform}': _apply_transform_np output differs "
            "from _transform_counts reference."
        ),
    )


def test_standardization_corrected_masking(sample_data):
    """Standardisation: masked pixels must be 0.0 (corrected from original loader).

    The original _transform_counts returns ((counts*masks)-mean)/std, which
    gives -mean/std for masked pixels.  The preprocess path uses
    ((x-mean)/std)*m so masked pixels are exactly 0.0.
    """
    from integrator.cli.preprocess import _apply_transform_np

    counts_np, masks_np = sample_data
    out = _apply_transform_np(
        counts_np, masks_np, MEAN, VAR, "standardization"
    )

    # Masked pixels must be exactly zero
    assert np.all(out[~masks_np] == 0.0), (
        "Masked pixels are not 0.0 for standardization. "
        "Check that masking is applied after normalisation: ((x-mean)/std)*m"
    )

    # Unmasked pixels must match the corrected formula
    expected = (counts_np - MEAN) / np.sqrt(VAR)
    np.testing.assert_allclose(
        out[masks_np],
        expected[masks_np],
        rtol=1e-5,
        atol=1e-6,
        err_msg="Unmasked pixels do not match (x-mean)/std for standardization.",
    )
