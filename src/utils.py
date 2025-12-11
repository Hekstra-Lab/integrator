from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from importlib.resources.abc import Traversable
from pathlib import Path

# -
# ROOT_DIR = files("integrator")
ROOT_DIR = Path(__file__).parents[1]
# RESOURCES = ROOT_DIR / "resources"
CONFIGS = ROOT_DIR / "tests/configs"
DATA = ROOT_DIR / "tests/data"


def get_configs() -> Mapping[str, Traversable]:
    out: dict[str, Traversable] = {}
    for entry in CONFIGS.rglob("*.yaml"):
        if entry.is_file():
            out[entry.stem] = entry  # keep as Traversable
    return out


def get_data() -> Mapping[str, Mapping[str, Mapping[str, Traversable]]]:
    """
    Returns {dim: {dataset_name: {key: resource}}}
    """
    out: dict[str, dict[str, dict[str, Traversable]]] = defaultdict(dict)

    for dim_dir in DATA.iterdir():
        if not dim_dir.is_dir():
            continue

        by_dataset: dict[str, dict[str, Traversable]] = {}
        for ds_dir in dim_dir.iterdir():
            if not ds_dir.is_dir():
                continue
            entries: dict[str, Traversable] = {}
            for p in ds_dir.glob("*.pt"):
                key = p.name.split("_", 1)[0]
                entries[key] = p
            if entries:
                by_dataset[ds_dir.name] = entries

        if by_dataset:
            out[dim_dir.name] = by_dataset

    return out
