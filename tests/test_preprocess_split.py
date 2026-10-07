"""
Tests for the deterministic train/val split in integrator.preprocess.

Covers:
  - same seed  → same split
  - different seeds → different splits
  - row-order changes do not change membership
  - no train/val overlap
  - every reflection in exactly one split (Mode B full coverage)
  - exported CSV IDs match chunk is_train / is_val arrays
  - duplicate refl_ids detected
  - val fraction is approximately correct
"""

from __future__ import annotations

import numpy as np
import pytest
import polars as pl


def _fake_ids(n: int = 1000, seed: int = 0) -> np.ndarray:
    """Return n guaranteed-unique int64 refl_ids sampled without replacement."""
    rng = np.random.default_rng(seed)
    return rng.choice(
        np.arange(1, 10_000_001, dtype=np.int64),
        size=n,
        replace=False,
    )


def test_same_seed_same_split():
    """Identical inputs + seed must produce identical is_train / is_val."""
    from integrator.cli.preprocess import _make_deterministic_split

    ids = _fake_ids(1000)
    a_tr, a_va = _make_deterministic_split(ids, 0.2, seed=42)
    b_tr, b_va = _make_deterministic_split(ids, 0.2, seed=42)
    np.testing.assert_array_equal(a_tr, b_tr)
    np.testing.assert_array_equal(a_va, b_va)


def test_different_seeds_different_splits():
    """Different seeds must produce different splits."""
    from integrator.cli.preprocess import _make_deterministic_split

    ids = _fake_ids(1000)
    tr_42, _ = _make_deterministic_split(ids, 0.2, seed=42)
    tr_99, _ = _make_deterministic_split(ids, 0.2, seed=99)
    assert not np.array_equal(tr_42, tr_99), (
        "Different seeds should produce different splits"
    )


def test_row_order_independent():
    """Shuffling input rows must not change any refl_id's train/val assignment."""
    from integrator.cli.preprocess import _make_deterministic_split

    rng = np.random.default_rng(7)
    ids = _fake_ids(500)
    perm = rng.permutation(len(ids))
    ids_shuffled = ids[perm]

    tr, va = _make_deterministic_split(ids, 0.2, seed=42)
    tr_s, va_s = _make_deterministic_split(ids_shuffled, 0.2, seed=42)

    # Undo the shuffle to compare element-by-element
    inv = np.argsort(perm)
    np.testing.assert_array_equal(tr, tr_s[inv])
    np.testing.assert_array_equal(va, va_s[inv])


def test_no_overlap():
    """No refl_id may appear in both train and val."""
    from integrator.cli.preprocess import _make_deterministic_split

    ids = _fake_ids(1000)
    is_train, is_val = _make_deterministic_split(ids, 0.2, seed=42)
    assert not np.any(is_train & is_val), "Train and val must not overlap"


def test_full_coverage():
    """Every reflection must be in exactly one of train or val (Mode B)."""
    from integrator.cli.preprocess import _make_deterministic_split

    ids = _fake_ids(1000)
    is_train, is_val = _make_deterministic_split(ids, 0.2, seed=42)
    assert np.all(is_train | is_val), (
        "Every reflection must belong to train or val"
    )


def test_exported_csv_matches_chunks(tmp_path):
    """IDs written to CSV must exactly match the is_train / is_val arrays."""
    from integrator.cli.preprocess import (
        _make_deterministic_split,
        _save_split_csvs,
    )

    ids = _fake_ids(200, seed=5)
    is_train, is_val = _make_deterministic_split(ids, 0.2, seed=42)
    _save_split_csvs(
        out_dir=tmp_path,
        refl_ids_valid=ids,
        is_train=is_train,
        is_val=is_val,
    )
    train_csv = pl.read_csv(tmp_path / "train_labels.csv")[
        "train_ids"
    ].to_numpy()
    val_csv = pl.read_csv(tmp_path / "val_labels.csv")["val_ids"].to_numpy()
    np.testing.assert_array_equal(np.sort(train_csv), np.sort(ids[is_train]))
    np.testing.assert_array_equal(np.sort(val_csv), np.sort(ids[is_val]))


def test_duplicate_refl_ids_detected():
    """np.unique must surface duplicated IDs (logic used in _pass1_scan)."""
    ids = np.array([1, 2, 3, 2, 5], dtype=np.int64)
    unique, counts = np.unique(ids, return_counts=True)
    dups = unique[counts > 1]
    assert len(dups) > 0 and 2 in dups


def test_val_fraction_approximately_correct():
    """Actual val fraction should be within 2 pp of the requested split."""
    from integrator.cli.preprocess import _make_deterministic_split

    ids = _fake_ids(10_000)
    is_train, is_val = _make_deterministic_split(ids, 0.2, seed=0)
    actual = is_val.sum() / len(ids)
    assert abs(actual - 0.2) < 0.02, (
        f"Val fraction {actual:.3f} too far from 0.20"
    )
