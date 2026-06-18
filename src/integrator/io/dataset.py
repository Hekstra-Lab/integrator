"""Read/write the consolidated per-dataset spec (dataset.yaml).

mksbox writes one dataset.yaml at the data directory holding everything that is a
property of the dataset rather than a modeling choice: shoebox geometry, the file
spec, crystal parameters, and normalization stats. The config loader reads it
to fill geometry/filenames so training YAMLs only carry model choices.
"""

from __future__ import annotations

from pathlib import Path

import yaml

spec_NAME = "dataset.yaml"


def data_dim_for(d: int) -> str:
    """Shoebox depth -> integrator data_dim ('2d' for a single z-slice)."""
    return "2d" if int(d) == 1 else "3d"


def read_dataset_spec(data_dir) -> dict | None:
    """Return the parsed dataset.yaml under data_dir, or None if absent."""
    p = Path(data_dir) / spec_NAME
    if not p.exists():
        return None
    return yaml.safe_load(p.read_text()) or None


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
) -> Path:
    """Write the consolidated dataset spec to out_dir/dataset.yaml.

    geometry must hold d, h, w; data_dim is derived when absent.
    """
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
    if crystal is not None:
        spec["crystal"] = dict(crystal)
    if stats is not None:
        spec["stats"] = dict(stats)
    if refl_file is not None:
        spec["refl_file"] = str(refl_file)

    p = Path(out_dir) / spec_NAME
    p.write_text(yaml.safe_dump(spec, sort_keys=False))
    return p
