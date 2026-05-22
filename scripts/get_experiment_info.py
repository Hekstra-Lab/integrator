"""Extract dmin and wavelength from DIALS experiment/reflection files.

Usage
-----
    uv run python scripts/get_experiment_info.py <expt_or_refl_file>

Example
-------
    uv run python scripts/get_experiment_info.py integrated.expt
    uv run python scripts/get_experiment_info.py integrated.refl
"""

import argparse
import json
from pathlib import Path


def from_expt(path: Path) -> dict:
    with open(path) as f:
        expt = json.load(f)

    info = {}

    beams = expt.get("beam", [])
    if beams:
        wl = beams[0].get("wavelength")
        if wl is not None:
            info["wavelength"] = float(wl)

    crystals = expt.get("crystal", [])
    if crystals:
        cell = crystals[0].get("real_space_a")
        if cell is not None:
            info["space_group"] = crystals[0].get(
                "space_group_hall_symbol", "unknown"
            )

    return info


def from_refl(path: Path) -> dict:
    import dials.array_family.flex  # noqa: F401
    from dxtbx.model.experiment_list import ExperimentListFactory

    try:
        from dials.util import sorry  # noqa: F401
    except ImportError:
        pass

    from dials.util.serialization import load

    refls = load(str(path))
    info = {}
    if "d" in refls:
        d_vals = refls["d"]
        info["dmin"] = float(min(d_vals))
        info["dmax"] = float(max(d_vals))
    return info


def from_metadata(path: Path) -> dict:
    import torch

    meta = torch.load(path, weights_only=False, map_location="cpu")
    info = {}
    if "d" in meta:
        d = meta["d"]
        info["dmin"] = float(d.min())
        info["dmax"] = float(d.max())
    if "wavelength" in meta:
        wl = meta["wavelength"]
        info["wavelength_min"] = float(wl.min())
        info["wavelength_max"] = float(wl.max())
        info["wavelength_mean"] = float(wl.mean())
    return info


def main():
    parser = argparse.ArgumentParser(
        description="Extract dmin/wavelength from DIALS or metadata files."
    )
    parser.add_argument("file", type=Path)
    args = parser.parse_args()

    path = args.file
    suffix = path.suffix.lower()

    if suffix == ".expt":
        info = from_expt(path)
    elif suffix == ".pt":
        info = from_metadata(path)
    else:
        print(f"Unsupported file type: {suffix}")
        print("Supported: .expt, .pt (metadata)")
        return

    for k, v in sorted(info.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
