"""Shoeboxes held on the device, checked against the arrays they came from.

The resident path replaces the dataloader and drops the separate indexing pass,
so file order becomes the state order. That is a correctness assumption, not
just a speed one: if the rows that come back are not the rows on disk, in the
same order, the per-observation state is written at the wrong places and the
merge is silently wrong rather than slow.
"""

import numpy as np
import pytest
import torch

from integrator.fit.resident import ResidentShoeboxes, device_free_bytes


def write_dataset(tmp_path, n=500, p=49, seed=0, dead=0):
    """A miniature dataset on disk, in the layout the real one uses."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 400, size=(n, p)).astype(np.int32)
    masks = np.ones((n, p), dtype=bool)
    if dead:
        # a block of shoeboxes with too few valid pixels to be usable
        masks[:dead, 3:] = False
    np.save(tmp_path / "counts.npy", counts)
    np.save(tmp_path / "masks.npy", masks)
    meta = {
        "d": torch.linspace(1.5, 40.0, n),
        "wavelength": torch.linspace(0.96, 1.26, n),
        "xyzcal.px.0": torch.arange(n, dtype=torch.float32),
        "xyzcal.px.1": torch.arange(n, dtype=torch.float32) * 2,
        "image_num": (torch.arange(n) % 7).float(),
        "refl_ids": torch.arange(n).float(),
        "H": torch.arange(n).float(),
        "K": torch.zeros(n),
        "L": torch.ones(n),
    }
    torch.save(meta, tmp_path / "metadata.pt")
    return counts, masks, meta


def gather(boxes):
    idx, cs, ms = [], [], []
    for i, c, m, _md in boxes():
        idx.append(i)
        cs.append(c)
        ms.append(m)
    return torch.cat(idx), torch.cat(cs), torch.cat(ms)


def test_rows_come_back_in_file_order(tmp_path):
    counts, masks, _ = write_dataset(tmp_path, n=500, p=49)
    boxes = ResidentShoeboxes(tmp_path, "cpu", chunk=64, block=128)
    idx, c, m = gather(boxes)

    assert len(boxes) == 500
    assert torch.equal(idx, torch.arange(500))
    assert torch.equal(c, torch.from_numpy(counts))
    assert torch.equal(m, torch.from_numpy(masks))


def test_chunking_does_not_change_the_contents(tmp_path):
    write_dataset(tmp_path, n=301, p=25, seed=3)
    a = gather(ResidentShoeboxes(tmp_path, "cpu", chunk=16, block=64))
    b = gather(ResidentShoeboxes(tmp_path, "cpu", chunk=4096, block=4096))
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)


def test_a_npy_reference_file_loads_like_a_pt_one(tmp_path):
    """hewl_9b7c's scaling dataset stores the metadata as a pickled `.npy`."""
    _, _, meta = write_dataset(tmp_path, n=120, p=25, seed=11)
    as_npy = {k: v.numpy() for k, v in meta.items()}
    np.save(tmp_path / "metadata.npy", as_npy, allow_pickle=True)

    pt = ResidentShoeboxes(tmp_path, "cpu", chunk=32)
    npy = ResidentShoeboxes(
        tmp_path, "cpu", chunk=32, reference_file="metadata.npy"
    )
    assert set(pt.metadata) == set(npy.metadata)
    for key in pt.metadata:
        assert torch.equal(pt.metadata[key], npy.metadata[key]), key
    assert len(pt) == len(npy) == 120


def test_metadata_is_carried_and_aligned(tmp_path):
    _, _, meta = write_dataset(tmp_path, n=200, p=25, seed=4)
    boxes = ResidentShoeboxes(tmp_path, "cpu", chunk=32)
    for key in ("d", "wavelength", "xyzcal.px.0", "image_num"):
        assert torch.allclose(boxes.metadata[key], meta[key].float())
    # and the per-chunk slice matches the same slice of the whole column
    seen = []
    for _i, _c, _m, md in boxes():
        seen.append(md["d"])
    assert torch.allclose(torch.cat(seen), meta["d"].float())


def test_min_valid_pixels_drops_rows_and_keeps_alignment(tmp_path):
    counts, masks, meta = write_dataset(tmp_path, n=120, p=25, seed=5, dead=30)
    boxes = ResidentShoeboxes(tmp_path, "cpu", chunk=16, block=32, min_valid_pixels=10)

    assert len(boxes) == 90, "the 30 shoeboxes with 3 valid pixels must go"
    idx, c, _m = gather(boxes)
    assert torch.equal(idx, torch.arange(90))
    assert torch.equal(c, torch.from_numpy(counts[30:]))
    # metadata must be filtered the same way, or every row is mislabelled
    assert torch.allclose(boxes.metadata["d"], meta["d"][30:].float())


def test_a_threshold_that_drops_nothing_is_a_no_op(tmp_path):
    write_dataset(tmp_path, n=64, p=25, seed=6)
    a = ResidentShoeboxes(tmp_path, "cpu", chunk=32)
    b = ResidentShoeboxes(tmp_path, "cpu", chunk=32, min_valid_pixels=5)
    assert len(a) == len(b) == 64
    assert torch.equal(gather(a)[1], gather(b)[1])


def test_counts_stay_integers(tmp_path):
    """The Poisson likelihood wants counts, so nothing may be rescaled here."""
    counts, _, _ = write_dataset(tmp_path, n=50, p=25, seed=7)
    boxes = ResidentShoeboxes(tmp_path, "cpu", chunk=50)
    _, c, m = gather(boxes)
    assert c.dtype == torch.int32 and m.dtype == torch.bool
    assert int(c.max()) == int(counts.max())


def test_free_memory_probe_is_safe_without_a_device():
    assert device_free_bytes("cpu") == 0
    assert device_free_bytes(None) == 0


def test_it_refuses_when_the_device_cannot_hold_the_data(tmp_path, monkeypatch):
    """Better to say so than to die part way through the copy."""
    write_dataset(tmp_path, n=100, p=25, seed=8)
    monkeypatch.setattr(
        "integrator.fit.resident.device_free_bytes", lambda _d: 1024
    )
    with pytest.raises(RuntimeError, match="run without --resident"):
        ResidentShoeboxes(tmp_path, "cpu", chunk=32)
