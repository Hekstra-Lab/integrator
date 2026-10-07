"""
Tests for resolve_source_data_dir and the prediction-level validation helpers
_resolve_and_check_metadata and _resolve_and_check_mtz_sources.

Resolver tests (resolve_source_data_dir):
  - rotation_data: returns data_dir without manifest access
  - rotation_data: no manifest.yaml present — no raise
  - rotation_data: missing data_dir key → KeyError
  - chunked: one-chunk manifest → correct path
  - chunked: multi-chunk manifest (5 chunks) → correct path
  - chunked: automatic metadata resolution (metadata.npy reachable)
  - chunked: manifest missing source_data_dir → ValueError with re-run hint
  - chunked: manifest.yaml absent → FileNotFoundError

Prediction validation helper tests:
  - chunked MFX write-back: _resolve_and_check_metadata returns correct path
  - chunked MTZ export: _resolve_and_check_mtz_sources returns source_data_dir
  - missing metadata.npy: _resolve_and_check_metadata raises FileNotFoundError
    with the expected path in the message
  - missing dataset.yaml: _resolve_and_check_mtz_sources raises FileNotFoundError
    naming dataset.yaml
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_manifest(
    chunk_dir: Path,
    *,
    source_data_dir: Path,
    n_chunks: int = 1,
    include_source: bool = True,
) -> None:
    """Write a minimal v2 manifest.yaml to chunk_dir."""
    m: dict = {
        "version": 2,
        "chunk_size": 100,
        "n_chunks": n_chunks,
        "n_valid_rows": n_chunks * 100,
        "n_train": n_chunks * 80,
        "n_val": n_chunks * 20,
        "n_images": 5,
        "transform": "asinh",
        "n_bins": 1,
        "numeric_columns": ["group_label"],
        "string_columns": [],
        "split": {"mode": "generated", "validation_split": 0.2, "seed": 42},
        "chunks": [
            {
                "dir": f"chunk_{i:05d}",
                "n_rows": 100,
                "n_train": 80,
                "n_val": 20,
                "n_unique_images": 5,
                "min_image_id": 0,
                "max_image_id": 4,
            }
            for i in range(n_chunks)
        ],
    }
    if include_source:
        m["source_data_dir"] = str(Path(source_data_dir).resolve())
    (chunk_dir / "manifest.yaml").write_text(yaml.safe_dump(m))


def _chunked_cfg(chunk_dir: Path) -> dict:
    return {
        "data_loader": {
            "name": "chunked_rotation_data",
            "args": {"chunk_dir": str(chunk_dir)},
        }
    }


def _rotation_cfg(data_dir: Path) -> dict:
    return {
        "data_loader": {
            "name": "rotation_data",
            "args": {"data_dir": str(data_dir)},
        }
    }


# ── resolve_source_data_dir — rotation_data ───────────────────────────────────


def test_rotation_data_returns_data_dir(tmp_path):
    """rotation_data: resolver returns data_dir directly."""
    from integrator.utils import resolve_source_data_dir

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    assert resolve_source_data_dir(_rotation_cfg(data_dir)) == data_dir


def test_rotation_data_no_manifest_access(tmp_path):
    """rotation_data: no manifest.yaml present — resolver must not raise."""
    from integrator.utils import resolve_source_data_dir

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # No manifest.yaml written anywhere — rotation_data must not read one.
    resolve_source_data_dir(_rotation_cfg(data_dir))


def test_rotation_data_missing_data_dir_key_raises():
    """rotation_data: missing data_dir key → KeyError."""
    from integrator.utils import resolve_source_data_dir

    cfg = {"data_loader": {"name": "rotation_data", "args": {}}}
    with pytest.raises(KeyError):
        resolve_source_data_dir(cfg)


# ── resolve_source_data_dir — chunked_rotation_data ───────────────────────────


def test_chunked_one_chunk_manifest(tmp_path):
    """chunked: single-chunk manifest — returns source_data_dir as absolute Path."""
    from integrator.utils import resolve_source_data_dir

    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_manifest(chunk_dir, source_data_dir=data_dir, n_chunks=1)

    assert (
        resolve_source_data_dir(_chunked_cfg(chunk_dir)) == data_dir.resolve()
    )


def test_chunked_multi_chunk_manifest(tmp_path):
    """chunked: 5-chunk manifest — returns source_data_dir correctly."""
    from integrator.utils import resolve_source_data_dir

    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_manifest(chunk_dir, source_data_dir=data_dir, n_chunks=5)

    assert (
        resolve_source_data_dir(_chunked_cfg(chunk_dir)) == data_dir.resolve()
    )


def test_chunked_automatic_metadata_resolution(tmp_path):
    """chunked: resolved source_data_dir / metadata.npy gives correct path."""
    from integrator.utils import resolve_source_data_dir

    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.npy").write_bytes(b"")  # simulate file exists

    _write_manifest(chunk_dir, source_data_dir=data_dir, n_chunks=1)
    result = resolve_source_data_dir(_chunked_cfg(chunk_dir))

    assert (result / "metadata.npy").exists(), (
        "metadata.npy must be reachable via the resolved source_data_dir"
    )


def test_chunked_old_manifest_raises_valueerror(tmp_path):
    """chunked: manifest without source_data_dir raises ValueError with re-run hint."""
    from integrator.utils import resolve_source_data_dir

    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    _write_manifest(
        chunk_dir, source_data_dir=tmp_path / "data", include_source=False
    )

    with pytest.raises(ValueError, match="source_data_dir"):
        resolve_source_data_dir(_chunked_cfg(chunk_dir))


def test_chunked_missing_manifest_raises_filenotfounderror(tmp_path):
    """chunked: no manifest.yaml → FileNotFoundError with preprocess reminder."""
    from integrator.utils import resolve_source_data_dir

    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="manifest.yaml"):
        resolve_source_data_dir(_chunked_cfg(chunk_dir))


# ── Prediction validation helpers ─────────────────────────────────────────────


def test_resolve_and_check_metadata_chunked_returns_correct_path(tmp_path):
    """chunked MFX write-back: _resolve_and_check_metadata returns
    source_data_dir/metadata.npy when the file exists."""
    from integrator.cli.predict import _resolve_and_check_metadata

    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.npy").write_bytes(b"")  # file exists

    _write_manifest(chunk_dir, source_data_dir=data_dir, n_chunks=1)
    cfg = _chunked_cfg(chunk_dir)

    result = _resolve_and_check_metadata(cfg)
    assert result == (data_dir.resolve() / "metadata.npy")


def test_resolve_and_check_mtz_sources_chunked_returns_source_data_dir(
    tmp_path,
):
    """chunked MTZ export: _resolve_and_check_mtz_sources returns source_data_dir
    when both metadata.npy and dataset.yaml exist."""
    from integrator.cli.predict import _resolve_and_check_mtz_sources

    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.npy").write_bytes(b"")
    (data_dir / "dataset.yaml").write_bytes(b"")

    _write_manifest(chunk_dir, source_data_dir=data_dir, n_chunks=1)
    cfg = _chunked_cfg(chunk_dir)

    result = _resolve_and_check_mtz_sources(cfg)
    assert result == data_dir.resolve()


def test_resolve_and_check_metadata_raises_when_metadata_missing(tmp_path):
    """Missing metadata.npy: _resolve_and_check_metadata raises FileNotFoundError
    with the expected path in the message."""
    from integrator.cli.predict import _resolve_and_check_metadata

    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # metadata.npy intentionally NOT created

    _write_manifest(chunk_dir, source_data_dir=data_dir, n_chunks=1)
    cfg = _chunked_cfg(chunk_dir)

    expected_path = str(data_dir.resolve() / "metadata.npy")
    with pytest.raises(FileNotFoundError, match="metadata.npy"):
        _resolve_and_check_metadata(cfg)

    # Confirm the expected path appears in the error message
    try:
        _resolve_and_check_metadata(cfg)
    except FileNotFoundError as e:
        assert expected_path in str(e), (
            f"Expected path '{expected_path}' not found in error message:\n{e}"
        )


def test_resolve_and_check_mtz_raises_when_dataset_yaml_missing(tmp_path):
    """Missing dataset.yaml: _resolve_and_check_mtz_sources raises FileNotFoundError
    naming dataset.yaml (metadata.npy present, dataset.yaml absent)."""
    from integrator.cli.predict import _resolve_and_check_mtz_sources

    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "metadata.npy").write_bytes(b"")  # present
    # dataset.yaml intentionally NOT created

    _write_manifest(chunk_dir, source_data_dir=data_dir, n_chunks=1)
    cfg = _chunked_cfg(chunk_dir)

    with pytest.raises(FileNotFoundError, match="dataset.yaml"):
        _resolve_and_check_mtz_sources(cfg)
