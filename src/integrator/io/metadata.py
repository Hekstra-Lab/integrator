from pathlib import Path

import numpy as np
import reciprocalspaceship as rs
import reciprocalspaceship.io as rs_io
import torch

from .dtypes import DEFAULT_EXCLUDED_COLS, DEFAULT_REFL_COLS


def _to_numpy(v):
    return v.numpy() if torch.is_tensor(v) else np.asarray(v)


def save_data(obj, path) -> Path:
    """Save a tensor or dict-of-tensors.

    Writes `.npy` by default; writes `.pt` only when path ends
    in `.pt`. Returns the path actually written.
    """
    p = Path(path)
    if p.suffix == ".pt":
        torch.save(obj, p)
        return p
    p = p.with_suffix(".npy")
    if isinstance(obj, dict):
        np.save(
            p, {k: _to_numpy(v) for k, v in obj.items()}, allow_pickle=True
        )
    else:
        np.save(p, _to_numpy(obj))
    return p


def data_path(path) -> Path | None:
    p = Path(path)
    npy = p.with_suffix(".npy")
    if npy.exists():
        return npy
    pt = p.with_suffix(".pt")
    if pt.exists():
        return pt
    return p if p.exists() else None


def load_data(path, map_location="cpu"):
    """Load a tensor or dict-of-tensors"""
    target = data_path(path) or Path(path)
    if target.suffix == ".npy":
        arr = np.load(target, allow_pickle=True)
        if arr.dtype == object:
            obj = arr.item()
            if isinstance(obj, dict):
                return {k: torch.as_tensor(v) for k, v in obj.items()}
            return torch.as_tensor(obj)
        return torch.as_tensor(arr)
    try:
        return torch.load(target, weights_only=True, map_location=map_location)
    except Exception:
        return torch.load(
            target, weights_only=False, map_location=map_location
        )


def load_metadata(path, map_location="cpu") -> dict:
    """Load a per-reflection metadata dict."""
    return load_data(path, map_location=map_location)


def refl_as_pt(
    refl,
    column_names: list[str] = DEFAULT_REFL_COLS,
    excluded_columns: list[str] = DEFAULT_EXCLUDED_COLS,
    out_dir: Path | None = None,
    out_fname: str = "metadata.npy",
) -> dict:
    ds = rs_io.read_dials_stills(
        refl,
        extra_cols=column_names,
    )
    assert isinstance(ds, rs.DataSet)

    data = {}
    for k, v in ds.items():
        if k not in excluded_columns:
            data[k] = torch.tensor(v, dtype=torch.float32)

    if out_dir is not None:
        fname = Path(out_dir) / out_fname
    else:
        fname = Path(out_fname)
    save_data(data, fname)
    return data
