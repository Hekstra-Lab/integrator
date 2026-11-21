"""`_stats_from_memmap` on counts below zero: the Anscombe stats stay finite and
match the data module's clamped transform; non-negative counts are unchanged."""

import numpy as np

from integrator.cli.make_shoeboxes import _stats_from_memmap


def _write(tmp_path, counts, masks):
    np.save(tmp_path / "counts.npy", counts)
    np.save(tmp_path / "masks.npy", masks)
    return _stats_from_memmap(tmp_path / "counts.npy", tmp_path / "masks.npy")


def test_negative_float_counts_give_finite_anscombe(tmp_path):
    rng = np.random.default_rng(0)
    c = (rng.poisson(5.0, size=(50, 441)) * 0.7 + 10 + rng.normal(0, 3, size=(50, 441)) - 12).astype(np.float32)
    assert (c < 0).any()
    m = np.ones_like(c, dtype=bool)
    stats = _write(tmp_path, c, m)
    a = 2.0 * np.sqrt(np.clip(c.astype(np.float64), 0, None) + 0.375)
    assert np.isfinite(stats["anscombe"]).all()
    np.testing.assert_allclose(stats["anscombe"], [a.mean(), a.var()], rtol=1e-9)


def test_nonnegative_counts_unchanged(tmp_path):
    rng = np.random.default_rng(1)
    c = rng.poisson(8.0, size=(40, 441)).astype(np.int32)
    m = rng.random(c.shape) > 0.1
    stats = _write(tmp_path, c, m)
    cm = (c * m).astype(np.float64)
    a = 2.0 * np.sqrt(cm + 0.375)
    np.testing.assert_allclose(stats["raw"], [cm.mean(), cm.var()], rtol=1e-9)
    np.testing.assert_allclose(stats["anscombe"], [a.mean(), a.var()], rtol=1e-9)
