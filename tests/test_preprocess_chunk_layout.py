"""
Tests for the consolidated chunk layout introduced in v6.

Covers:
  metadata.npz
    - contains all expected numeric/bool column names (no extras, no missing)
    - row_id dtype is int64; is_train / is_val dtype is bool
    - values round-trip correctly to source reference data
    - old-style col_*.npy / str_*.txt files are NOT written

  strings.parquet
    - contains all expected string column names
    - values round-trip correctly to source reference data
    - file is omitted entirely when string_columns is empty

  per-chunk image statistics in manifest.yaml
    - n_unique_images, min_image_id, max_image_id present in every chunk entry
    - values match np.unique of image_id for those rows
    - values are null (None) when image_id is absent from reference

  manifest schema
    - version field is 2
    - numeric_columns / string_columns are YAML sequences (lists), not mappings

  ChunkedDataset._load_chunk round-trips
    - numeric columns loaded correctly from metadata.npz
    - string columns loaded correctly from strings.parquet
    - works without error when strings.parquet is absent
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml


# ── Synthetic dataset ─────────────────────────────────────────────────────────

N_ROWS = 120  # total valid reflections across all chunks
N_PIXELS = 25  # pixels per shoebox
CHUNK_SIZE = 50  # reflections per chunk  →  3 chunks total


def _build_scan_and_data():
    """Return (ScanResult, counts, masks, reference) built from numpy only.

    No integrator imports needed: ScanResult is constructed directly so the
    tests can run without the full package installed.
    """
    from integrator.cli.preprocess import ScanResult

    rng = np.random.default_rng(0)
    n = N_ROWS

    valid_indices = np.arange(n, dtype=np.int64)
    refl_ids = rng.choice(
        np.arange(1, 1_000_001, dtype=np.int64), size=n, replace=False
    )

    # ~20 % val, rest train — no overlap guaranteed
    val_mask = rng.random(n) < 0.2
    is_train = ~val_mask
    is_val = val_mask

    group_label = rng.integers(0, 3, size=n, dtype=np.int64)
    image_ids = rng.integers(0, 10, size=n, dtype=np.int64)
    d_vals = rng.uniform(1.0, 5.0, size=n).astype(np.float32)

    counts = rng.uniform(0, 100, size=(n, N_PIXELS)).astype(np.float32)
    masks = (rng.uniform(0, 1, size=(n, N_PIXELS)) > 0.1).astype(bool)

    reference = {
        "refl_ids": refl_ids,
        "image_id": image_ids,
        "d": d_vals,
        "source_refl": np.array(
            [f"run{i % 3}.refl" for i in range(n)], dtype=object
        ),
        "source_expt": np.array(
            [f"run{i % 3}.expt" for i in range(n)], dtype=object
        ),
    }

    scan = ScanResult(
        valid_indices=valid_indices,
        refl_ids_valid=refl_ids,
        is_train=is_train,
        is_val=is_val,
        group_label=group_label,
        profile_group_label=None,
        stats=(30.0, 200.0),
        transform="anscombe",
        numeric_columns=["image_id", "d", "refl_ids", "group_label"],
        string_columns=["source_refl", "source_expt"],
        n_images=10,
        n_valid=n,
        n_train=int(is_train.sum()),
        n_val=int(is_val.sum()),
        split_meta={"mode": "generated", "validation_split": 0.2, "seed": 0},
    )
    return scan, counts, masks, reference


# ── metadata.npz ──────────────────────────────────────────────────────────────


def test_metadata_npz_contains_expected_columns(tmp_path):
    """metadata.npz must hold exactly numeric_columns + row_id + is_train + is_val."""
    from integrator.cli.preprocess import _pass2_write_chunks

    scan, counts, masks, reference = _build_scan_and_data()
    _pass2_write_chunks(
        scan=scan,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )

    meta = np.load(
        tmp_path / "chunk_00000" / "metadata.npz", allow_pickle=False
    )
    expected = set(scan.numeric_columns) | {"row_id", "is_train", "is_val"}
    assert set(meta.files) == expected, (
        f"metadata.npz keys mismatch.\n"
        f"  expected: {sorted(expected)}\n"
        f"  got:      {sorted(meta.files)}"
    )


def test_metadata_npz_dtypes(tmp_path):
    """row_id must be int64; is_train / is_val must be bool."""
    from integrator.cli.preprocess import _pass2_write_chunks

    scan, counts, masks, reference = _build_scan_and_data()
    _pass2_write_chunks(
        scan=scan,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )

    meta = np.load(
        tmp_path / "chunk_00000" / "metadata.npz", allow_pickle=False
    )
    assert meta["row_id"].dtype == np.int64, (
        f"row_id dtype: expected int64, got {meta['row_id'].dtype}"
    )
    assert meta["is_train"].dtype == bool, (
        f"is_train dtype: expected bool, got {meta['is_train'].dtype}"
    )
    assert meta["is_val"].dtype == bool, (
        f"is_val dtype: expected bool, got {meta['is_val'].dtype}"
    )


def test_metadata_npz_values_match_source(tmp_path):
    """Numeric column values in metadata.npz must match the source reference arrays."""
    from integrator.cli.preprocess import _pass2_write_chunks

    scan, counts, masks, reference = _build_scan_and_data()
    _pass2_write_chunks(
        scan=scan,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )

    rows = scan.valid_indices[:CHUNK_SIZE]
    meta = np.load(
        tmp_path / "chunk_00000" / "metadata.npz", allow_pickle=False
    )

    np.testing.assert_array_equal(
        meta["row_id"], rows, err_msg="row_id mismatch"
    )
    np.testing.assert_array_equal(
        meta["is_train"],
        scan.is_train[:CHUNK_SIZE],
        err_msg="is_train mismatch",
    )
    np.testing.assert_array_equal(
        meta["is_val"], scan.is_val[:CHUNK_SIZE], err_msg="is_val mismatch"
    )
    np.testing.assert_array_equal(
        meta["image_id"],
        reference["image_id"][rows],
        err_msg="image_id mismatch",
    )
    np.testing.assert_array_almost_equal(
        meta["d"], reference["d"][rows], err_msg="d mismatch"
    )
    np.testing.assert_array_equal(
        meta["group_label"],
        scan.group_label[rows],
        err_msg="group_label mismatch",
    )


def test_no_col_npy_or_str_txt_files_written(tmp_path):
    """Old-style col_*.npy and str_*.txt files must NOT appear in v6 chunks."""
    from integrator.cli.preprocess import _pass2_write_chunks

    scan, counts, masks, reference = _build_scan_and_data()
    _pass2_write_chunks(
        scan=scan,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )

    chunk_dir = tmp_path / "chunk_00000"
    old_style = list(chunk_dir.glob("col_*.npy")) + list(
        chunk_dir.glob("str_*.txt")
    )
    assert len(old_style) == 0, (
        f"Old-style per-column files found: {[f.name for f in old_style]}"
    )


# ── strings.parquet ───────────────────────────────────────────────────────────


def test_strings_parquet_contains_expected_columns(tmp_path):
    """strings.parquet must contain every column in string_columns."""
    from integrator.cli.preprocess import _pass2_write_chunks

    scan, counts, masks, reference = _build_scan_and_data()
    _pass2_write_chunks(
        scan=scan,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )

    pq_path = tmp_path / "chunk_00000" / "strings.parquet"
    assert pq_path.exists(), "strings.parquet not found in chunk_00000"
    df = pl.read_parquet(pq_path)
    assert set(df.columns) == set(scan.string_columns), (
        f"strings.parquet columns mismatch.\n"
        f"  expected: {sorted(scan.string_columns)}\n"
        f"  got:      {sorted(df.columns)}"
    )


def test_strings_parquet_values_match_source(tmp_path):
    """String values in strings.parquet must match the source reference arrays."""
    from integrator.cli.preprocess import _pass2_write_chunks

    scan, counts, masks, reference = _build_scan_and_data()
    _pass2_write_chunks(
        scan=scan,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )

    rows = scan.valid_indices[:CHUNK_SIZE]
    df = pl.read_parquet(tmp_path / "chunk_00000" / "strings.parquet")

    expected_refl = [str(reference["source_refl"][r]) for r in rows]
    assert df["source_refl"].to_list() == expected_refl, (
        "source_refl values mismatch"
    )

    expected_expt = [str(reference["source_expt"][r]) for r in rows]
    assert df["source_expt"].to_list() == expected_expt, (
        "source_expt values mismatch"
    )


def test_strings_parquet_omitted_when_no_string_cols(tmp_path):
    """strings.parquet must not be written when string_columns is empty."""
    from integrator.cli.preprocess import _pass2_write_chunks

    scan, counts, masks, reference = _build_scan_and_data()
    scan_no_str = replace(scan, string_columns=[])
    _pass2_write_chunks(
        scan=scan_no_str,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )

    pq_path = tmp_path / "chunk_00000" / "strings.parquet"
    assert not pq_path.exists(), (
        "strings.parquet must not exist when string_columns is empty"
    )


# ── Per-chunk image statistics ────────────────────────────────────────────────


def test_manifest_chunk_entries_have_image_stat_keys(tmp_path):
    """Each chunk entry in manifest.yaml must have n_unique_images, min/max image_id."""
    from integrator.cli.preprocess import _pass2_write_chunks, _write_manifest

    scan, counts, masks, reference = _build_scan_and_data()
    chunk_infos = _pass2_write_chunks(
        scan=scan,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_manifest(
        scan=scan,
        chunk_infos=chunk_infos,
        chunk_size=CHUNK_SIZE,
        n_bins=1,
        out_dir=tmp_path,
        data_dir=data_dir,
    )

    m = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    for entry in m["chunks"]:
        for key in ("n_unique_images", "min_image_id", "max_image_id"):
            assert key in entry, (
                f"key '{key}' missing from chunk entry: {entry}"
            )


def test_image_stats_values_correct(tmp_path):
    """Per-chunk image stats must match np.unique of image_id for those rows."""
    from integrator.cli.preprocess import _pass2_write_chunks, _write_manifest

    scan, counts, masks, reference = _build_scan_and_data()
    chunk_infos = _pass2_write_chunks(
        scan=scan,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_manifest(
        scan=scan,
        chunk_infos=chunk_infos,
        chunk_size=CHUNK_SIZE,
        n_bins=1,
        out_dir=tmp_path,
        data_dir=data_dir,
    )

    m = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    for i, entry in enumerate(m["chunks"]):
        start = i * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, scan.n_valid)
        rows = scan.valid_indices[start:end]
        img_ids = reference["image_id"][rows]
        unique = np.unique(img_ids)

        assert entry["n_unique_images"] == len(unique), (
            f"chunk {i}: n_unique_images {entry['n_unique_images']} != {len(unique)}"
        )
        assert entry["min_image_id"] == int(unique.min()), (
            f"chunk {i}: min_image_id {entry['min_image_id']} != {int(unique.min())}"
        )
        assert entry["max_image_id"] == int(unique.max()), (
            f"chunk {i}: max_image_id {entry['max_image_id']} != {int(unique.max())}"
        )


def test_image_stats_null_when_image_id_absent(tmp_path):
    """n_unique_images / min/max image_id must be null when image_id is absent."""
    from integrator.cli.preprocess import _pass2_write_chunks, _write_manifest

    scan, counts, masks, reference = _build_scan_and_data()
    ref_no_img = {k: v for k, v in reference.items() if k != "image_id"}
    scan_no_img = replace(
        scan,
        n_images=None,
        numeric_columns=[c for c in scan.numeric_columns if c != "image_id"],
    )
    chunk_infos = _pass2_write_chunks(
        scan=scan_no_img,
        counts=counts,
        masks=masks,
        reference=ref_no_img,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_manifest(
        scan=scan_no_img,
        chunk_infos=chunk_infos,
        chunk_size=CHUNK_SIZE,
        n_bins=1,
        out_dir=tmp_path,
        data_dir=data_dir,
    )

    m = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    for entry in m["chunks"]:
        assert entry["n_unique_images"] is None, (
            f"n_unique_images should be null, got {entry['n_unique_images']}"
        )
        assert entry["min_image_id"] is None, (
            f"min_image_id should be null, got {entry['min_image_id']}"
        )
        assert entry["max_image_id"] is None, (
            f"max_image_id should be null, got {entry['max_image_id']}"
        )


# ── Manifest version and schema ───────────────────────────────────────────────


def test_manifest_version_is_2(tmp_path):
    """manifest.yaml must declare version: 2."""
    from integrator.cli.preprocess import _pass2_write_chunks, _write_manifest

    scan, counts, masks, reference = _build_scan_and_data()
    chunk_infos = _pass2_write_chunks(
        scan=scan,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_manifest(
        scan=scan,
        chunk_infos=chunk_infos,
        chunk_size=CHUNK_SIZE,
        n_bins=1,
        out_dir=tmp_path,
        data_dir=data_dir,
    )

    m = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert m["version"] == 2, f"Expected version 2, got {m['version']}"


def test_manifest_columns_are_lists_not_dicts(tmp_path):
    """numeric_columns and string_columns in manifest.yaml must be lists."""
    from integrator.cli.preprocess import _pass2_write_chunks, _write_manifest

    scan, counts, masks, reference = _build_scan_and_data()
    chunk_infos = _pass2_write_chunks(
        scan=scan,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_manifest(
        scan=scan,
        chunk_infos=chunk_infos,
        chunk_size=CHUNK_SIZE,
        n_bins=1,
        out_dir=tmp_path,
        data_dir=data_dir,
    )

    m = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert isinstance(m["numeric_columns"], list), (
        f"numeric_columns should be list, got {type(m['numeric_columns']).__name__}"
    )
    assert isinstance(m["string_columns"], list), (
        f"string_columns should be list, got {type(m['string_columns']).__name__}"
    )
    # Sanity: expected names are present
    assert "group_label" in m["numeric_columns"]
    assert "source_refl" in m["string_columns"]


# ── ChunkedDataset._load_chunk round-trips ────────────────────────────────────


def test_load_chunk_round_trips_numeric_columns(tmp_path):
    """_load_chunk must restore numeric column values from metadata.npz."""
    from integrator.cli.preprocess import _pass2_write_chunks
    from integrator.data_loaders.chunked_data_module import ChunkedDataset

    scan, counts, masks, reference = _build_scan_and_data()
    _pass2_write_chunks(
        scan=scan,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )

    chunk_dir = tmp_path / "chunk_00000"
    chunk = ChunkedDataset._load_chunk(
        chunk_dir,
        numeric_columns=scan.numeric_columns,
        string_columns=scan.string_columns,
    )

    rows = scan.valid_indices[:CHUNK_SIZE]
    np.testing.assert_array_equal(
        chunk["row_id"], rows, err_msg="_load_chunk: row_id mismatch"
    )
    np.testing.assert_array_equal(
        chunk["is_train"],
        scan.is_train[:CHUNK_SIZE],
        err_msg="_load_chunk: is_train mismatch",
    )
    np.testing.assert_array_equal(
        chunk["image_id"],
        reference["image_id"][rows],
        err_msg="_load_chunk: image_id mismatch",
    )


def test_load_chunk_round_trips_string_columns(tmp_path):
    """_load_chunk must restore string column values from strings.parquet."""
    from integrator.cli.preprocess import _pass2_write_chunks
    from integrator.data_loaders.chunked_data_module import ChunkedDataset

    scan, counts, masks, reference = _build_scan_and_data()
    _pass2_write_chunks(
        scan=scan,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )

    chunk_dir = tmp_path / "chunk_00000"
    chunk = ChunkedDataset._load_chunk(
        chunk_dir,
        numeric_columns=scan.numeric_columns,
        string_columns=scan.string_columns,
    )

    rows = scan.valid_indices[:CHUNK_SIZE]
    expected_refl = [str(reference["source_refl"][r]) for r in rows]
    assert chunk["source_refl"] == expected_refl, (
        "_load_chunk: source_refl values mismatch"
    )

    expected_expt = [str(reference["source_expt"][r]) for r in rows]
    assert chunk["source_expt"] == expected_expt, (
        "_load_chunk: source_expt values mismatch"
    )


def test_load_chunk_works_without_strings_parquet(tmp_path):
    """_load_chunk must not raise when strings.parquet is absent."""
    from integrator.cli.preprocess import _pass2_write_chunks
    from integrator.data_loaders.chunked_data_module import ChunkedDataset

    scan, counts, masks, reference = _build_scan_and_data()
    scan_no_str = replace(scan, string_columns=[])
    _pass2_write_chunks(
        scan=scan_no_str,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )

    chunk_dir = tmp_path / "chunk_00000"
    assert not (chunk_dir / "strings.parquet").exists()

    # Must load without raising even though strings.parquet is absent
    chunk = ChunkedDataset._load_chunk(
        chunk_dir,
        numeric_columns=scan.numeric_columns,
        string_columns=[],
    )
    assert "source_refl" not in chunk
    assert "counts" in chunk
    assert "row_id" in chunk


# ── New tests for v7 robustness changes ───────────────────────────────────────


def test_load_chunk_raises_if_strings_parquet_missing(tmp_path):
    """FileNotFoundError when strings.parquet is absent but string_columns is set."""
    from integrator.cli.preprocess import _pass2_write_chunks
    from integrator.data_loaders.chunked_data_module import ChunkedDataset

    scan, counts, masks, reference = _build_scan_and_data()
    _pass2_write_chunks(
        scan=scan,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )

    # Simulate a corrupted / partially-written chunk by removing the file.
    (tmp_path / "chunk_00000" / "strings.parquet").unlink()

    with pytest.raises(FileNotFoundError, match="strings.parquet"):
        ChunkedDataset._load_chunk(
            tmp_path / "chunk_00000",
            numeric_columns=scan.numeric_columns,
            string_columns=scan.string_columns,  # non-empty — must raise
        )


def test_setup_raises_on_version_1_manifest(tmp_path):
    """setup() must raise ValueError when manifest.yaml declares version != 2."""
    import yaml
    from integrator.data_loaders.chunked_data_module import ChunkedDataModule

    # Write a minimal version-1 manifest (old col_*.npy layout).
    v1_manifest = {
        "version": 1,
        "chunk_size": CHUNK_SIZE,
        "n_chunks": 0,
        "n_valid_rows": 0,
        "n_train": 0,
        "n_val": 0,
        "n_images": None,
        "transform": "anscombe",
        "n_bins": 1,
        "numeric_columns": {},  # v1 used a dict  col_name -> filename
        "string_columns": {},
        "split": {"mode": "generated"},
        "chunks": [],
    }
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(v1_manifest))

    dm = ChunkedDataModule(chunk_dir=tmp_path, batch_size=10)
    with pytest.raises(ValueError, match="version 2 is required"):
        dm.setup()


def test_setup_raises_if_chunk_file_missing(tmp_path):
    """setup() must raise FileNotFoundError when a required file is missing."""
    from integrator.cli.preprocess import _pass2_write_chunks, _write_manifest
    from integrator.data_loaders.chunked_data_module import ChunkedDataModule

    scan, counts, masks, reference = _build_scan_and_data()
    chunk_infos = _pass2_write_chunks(
        scan=scan,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_manifest(
        scan=scan,
        chunk_infos=chunk_infos,
        chunk_size=CHUNK_SIZE,
        n_bins=1,
        out_dir=tmp_path,
        data_dir=data_dir,
    )

    # Simulate a partially-written chunk.
    (tmp_path / "chunk_00000" / "metadata.npz").unlink()

    dm = ChunkedDataModule(chunk_dir=tmp_path, batch_size=10)
    with pytest.raises(FileNotFoundError, match="metadata.npz"):
        dm.setup()


def test_load_chunk_raises_on_length_mismatch(tmp_path):
    """_load_chunk must raise ValueError when an array length mismatches counts."""
    from integrator.cli.preprocess import _pass2_write_chunks
    from integrator.data_loaders.chunked_data_module import ChunkedDataset

    scan, counts, masks, reference = _build_scan_and_data()
    _pass2_write_chunks(
        scan=scan,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )

    chunk_dir = tmp_path / "chunk_00000"

    # Overwrite metadata.npz with a truncated row_id to simulate a disk-full
    # write that left the array short by 5 rows.
    with np.load(chunk_dir / "metadata.npz", allow_pickle=False) as meta_orig:
        meta_bad = {k: meta_orig[k].copy() for k in meta_orig.files}
    meta_bad["row_id"] = meta_bad["row_id"][:-5]  # truncate
    np.savez(chunk_dir / "metadata.npz", **meta_bad)

    with pytest.raises(ValueError, match="Length mismatch"):
        ChunkedDataset._load_chunk(
            chunk_dir,
            numeric_columns=scan.numeric_columns,
            string_columns=scan.string_columns,
        )


# ── New tests for v8 robustness changes ─────────────────────────────────────────────


@pytest.mark.parametrize("key", ["shoebox", "mask"])
def test_load_chunk_raises_on_shoebox_mask_row_mismatch(tmp_path, key):
    """_load_chunk must raise ValueError when shoebox.npy or mask.npy first
    dimension does not match counts.npy."""
    from integrator.cli.preprocess import _pass2_write_chunks
    from integrator.data_loaders.chunked_data_module import ChunkedDataset

    scan, counts, masks, reference = _build_scan_and_data()
    _pass2_write_chunks(
        scan=scan,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )

    chunk_dir = tmp_path / "chunk_00000"

    # Truncate the target array by 5 rows to simulate a partial write.
    arr = np.load(chunk_dir / f"{key}.npy")
    np.save(chunk_dir / f"{key}.npy", arr[:-5])

    with pytest.raises(ValueError, match=key):
        ChunkedDataset._load_chunk(
            chunk_dir,
            numeric_columns=scan.numeric_columns,
            string_columns=scan.string_columns,
        )


def test_load_chunk_raises_on_missing_metadata_key(tmp_path):
    """_load_chunk must raise KeyError naming the missing key and chunk
    directory when metadata.npz is missing a required key."""
    from integrator.cli.preprocess import _pass2_write_chunks
    from integrator.data_loaders.chunked_data_module import ChunkedDataset

    scan, counts, masks, reference = _build_scan_and_data()
    _pass2_write_chunks(
        scan=scan,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )

    chunk_dir = tmp_path / "chunk_00000"

    # Re-save metadata.npz without 'is_val' to simulate a missing key.
    with np.load(chunk_dir / "metadata.npz", allow_pickle=False) as meta_orig:
        meta_bad = {
            k: meta_orig[k].copy() for k in meta_orig.files if k != "is_val"
        }
    np.savez(chunk_dir / "metadata.npz", **meta_bad)

    with pytest.raises(KeyError, match="is_val"):
        ChunkedDataset._load_chunk(
            chunk_dir,
            numeric_columns=scan.numeric_columns,
            string_columns=scan.string_columns,
        )


def test_load_chunk_raises_on_missing_string_column(tmp_path):
    """_load_chunk must raise ValueError naming the missing column and chunk
    directory when strings.parquet is missing a declared column."""
    from integrator.cli.preprocess import _pass2_write_chunks
    from integrator.data_loaders.chunked_data_module import ChunkedDataset

    scan, counts, masks, reference = _build_scan_and_data()
    _pass2_write_chunks(
        scan=scan,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )

    chunk_dir = tmp_path / "chunk_00000"

    # Overwrite strings.parquet without 'source_refl'.
    df_orig = pl.read_parquet(chunk_dir / "strings.parquet")
    df_orig.drop("source_refl").write_parquet(chunk_dir / "strings.parquet")

    with pytest.raises(ValueError, match="source_refl"):
        ChunkedDataset._load_chunk(
            chunk_dir,
            numeric_columns=scan.numeric_columns,
            string_columns=scan.string_columns,
        )


def test_manifest_has_source_data_dir(tmp_path):
    """_write_manifest records source_data_dir as an absolute resolved path."""
    from integrator.cli.preprocess import _pass2_write_chunks, _write_manifest

    scan, counts, masks, reference = _build_scan_and_data()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    chunk_infos = _pass2_write_chunks(
        scan=scan,
        counts=counts,
        masks=masks,
        reference=reference,
        chunk_size=CHUNK_SIZE,
        out_dir=tmp_path,
    )
    _write_manifest(
        scan=scan,
        chunk_infos=chunk_infos,
        chunk_size=CHUNK_SIZE,
        n_bins=1,
        out_dir=tmp_path,
        data_dir=data_dir,
    )

    m = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert "source_data_dir" in m, "manifest.yaml must contain source_data_dir"
    assert m["source_data_dir"] == str(data_dir.resolve()), (
        f"source_data_dir mismatch:\n"
        f"  expected: {data_dir.resolve()}\n"
        f"  got:      {m['source_data_dir']}"
    )
