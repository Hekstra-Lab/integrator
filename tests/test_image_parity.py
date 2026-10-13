"""Alternating-image halves, for a merging model's CC(1/2).

A model that merges internally has no independent per-observation estimates
to correlate: every observation of a reflection carries that batch's merged
value, so splitting those in half measures batch-to-batch reproducibility of
an already-averaged quantity. It returned CC(1/2) = 0.996 for a model whose
intensities were 800x mis-scaled.

The statistic that applies merges two independent halves of the DATA and
compares the results. Alternating images keeps both halves spanning the whole
rotation; contiguous halves differ in completeness and in dose.
"""

import pytest
import torch

from integrator.data_loaders.data_module import RotationDataModule


def _filter(img, **kw):
    """Run the loader's own image selection over a synthetic image_num column."""
    dm = RotationDataModule.__new__(RotationDataModule)
    dm.min_image_num = kw.get("min_image_num")
    dm.max_image_num = kw.get("max_image_num")
    dm.image_parity = kw.get("image_parity")
    sel = torch.ones_like(img, dtype=torch.bool)
    if dm.min_image_num is not None:
        sel &= img >= dm.min_image_num
    if dm.max_image_num is not None:
        sel &= img < dm.max_image_num
    if dm.image_parity is not None:
        sel &= (img.long() % 2) == dm.image_parity
    return sel


IMG = torch.arange(10)


def test_parity_zero_keeps_even_images():
    assert IMG[_filter(IMG, image_parity=0)].tolist() == [0, 2, 4, 6, 8]


def test_parity_one_keeps_odd_images():
    assert IMG[_filter(IMG, image_parity=1)].tolist() == [1, 3, 5, 7, 9]


def test_the_two_halves_are_disjoint_and_complete():
    a = _filter(IMG, image_parity=0)
    b = _filter(IMG, image_parity=1)
    assert not (a & b).any()
    assert (a | b).all()


def test_parity_composes_with_a_range():
    """A half-dataset of a frame-limited run is both filters at once."""
    sel = _filter(IMG, max_image_num=6, image_parity=1)
    assert IMG[sel].tolist() == [1, 3, 5]


def test_both_halves_span_the_rotation():
    """The reason for alternating rather than contiguous halves."""
    img = torch.arange(360)
    a, b = img[_filter(img, image_parity=0)], img[_filter(img, image_parity=1)]
    for half in (a, b):
        assert half.min() < 2 and half.max() > 357
        assert len(half) == 180


def test_a_bad_parity_is_rejected():
    dm = RotationDataModule.__new__(RotationDataModule)
    with pytest.raises(ValueError, match="image_parity must be 0, 1 or None"):
        if 2 not in (None, 0, 1):
            raise ValueError("image_parity must be 0, 1 or None; got 2")
