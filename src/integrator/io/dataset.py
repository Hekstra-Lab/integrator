from __future__ import annotations

from pathlib import Path

import yaml

spec_NAME = "dataset.yaml"


def data_dim_for(d: int) -> str:
    return "2d" if int(d) == 1 else "3d"


def read_dataset_spec(data_dir) -> dict | None:
    """Return the parsed dataset.yaml under data_dir, or None if absent."""
    p = Path(data_dir) / spec_NAME
    if not p.exists():
        return None
    return yaml.safe_load(p.read_text()) or None


def _to_native(obj):
    """Recursively convert numpy scalars/arrays to native Python types.

    `yaml.safe_dump` cannot represent `np.float64`/`np.int64`/`np.ndarray`, which
    leak in from DIALS/numpy-derived geometry; this makes the spec safe to dump.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_dataset_yaml(
    out_dir,
    *,
    geometry: dict,
    n_reflections: int,
    polychromatic: bool,
    anscombe: bool,
    files: dict,
    crystal: dict | None = None,
    stats: dict | None = None,
    refl_file=None,
    n_hkl: dict | None = None,
    scale: dict | None = None,
) -> Path:
    """Write the dataset spec to out_dir/dataset.yaml."""
    d, h, w = (int(geometry[k]) for k in ("d", "h", "w"))
    spec: dict = {
        "geometry": {
            "d": d,
            "h": h,
            "w": w,
            "data_dim": geometry.get("data_dim") or data_dim_for(d),
        },
        "n_reflections": int(n_reflections),
        "polychromatic": bool(polychromatic),
        "anscombe": bool(anscombe),
        "files": dict(files),
    }
    if n_hkl is not None:
        spec["n_hkl"] = {k: int(v) for k, v in n_hkl.items()}
    if scale is not None:
        spec["scale"] = dict(scale)
    if crystal is not None:
        spec["crystal"] = dict(crystal)
    if stats is not None:
        spec["stats"] = dict(stats)
    if refl_file is not None:
        spec["refl_file"] = str(refl_file)

    p = Path(out_dir) / spec_NAME
    p.write_text(yaml.safe_dump(_to_native(spec), sort_keys=False))
    return p
