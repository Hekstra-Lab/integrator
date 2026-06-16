"""Convert per-reflection metadata between a DIALS `.refl` and torch/numpy.

`refl_as_pt` reads a `.refl` (via reciprocalspaceship) into a tensor dict and
optionally serializes it as `.pt` or `.npy`; `load_metadata` reads either format
back. Used by `integrator.mksbox` and the data loaders.
"""

from pathlib import Path

import numpy as np
import reciprocalspaceship as rs
import reciprocalspaceship.io as rs_io
import torch

from .dtypes import DEFAULT_EXCLUDED_COLS, DEFAULT_REFL_COLS


def refl_as_pt(
    refl,
    column_names: list[str] = DEFAULT_REFL_COLS,
    excluded_columns: list[str] = DEFAULT_EXCLUDED_COLS,
    out_dir: Path | None = None,
    out_fname: str = "metadata.pt",
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

    # write to output directory if specified
    if out_dir is not None:
        fname = Path(out_dir) / out_fname
    else:
        fname = Path(out_fname)

    if fname.suffix == ".npy":
        # numpy-native: a single pickled dict of arrays (loadable without
        # torch). Mirrors the .npy counts/masks; read back via load_metadata.
        np.save(
            fname,
            {k: v.numpy() for k, v in data.items()},
            allow_pickle=True,
        )
    else:
        torch.save(data, fname)
    return data


def load_metadata(path, map_location="cpu") -> dict:
    """Load a per-reflection metadata dict from `.npy` or `.pt`.

    Args:
        path: path to the metadata file (`.npy` or `.pt`).
        map_location: device mapping for the `.pt` path.

    Returns:
        Dict mapping column name to a torch tensor.
    """
    p = Path(path)
    npy = p.with_suffix(".npy")
    if npy.exists():
        d = np.load(npy, allow_pickle=True).item()
        return {k: torch.as_tensor(v) for k, v in d.items()}
    try:
        return torch.load(p, weights_only=True, map_location=map_location)
    except Exception:
        return torch.load(p, weights_only=False, map_location=map_location)
