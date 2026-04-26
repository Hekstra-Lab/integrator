import argparse

from integrator.utils.refl_utils import DEFAULT_REFL_COLS

SCALAR_DTYPES = {
    "zeta": "double",
    "qe": "double",
    "profile.correlation": "double",
    "partiality": "double",
    "partial_id": "std::size_t",
    "panel": "std::size_t",
    "flags": "std::size_t",
    "num_pixels.valid": "int",
    "num_pixels.foreground": "int",
    "num_pixels.background_used": "int",
    "num_pixels.background": "int",
    "lp": "double",
    "intensity.prf.value": "double",
    "intensity.prf.variance": "double",
    "intensity.sum.value": "double",
    "intensity.sum.variance": "double",
    "imageset_id": "int",
    "entering": "bool",
    "d": "double",
    "background.mean": "double",
    "background.sum.value": "double",
    "background.sum.variance": "double",
    "refl_ids": "int",
}


VECTOR_COLUMNS = {
    "bbox": ("int6", 6),
    "s1": ("vec3<double>", 3),
    "xyzcal.mm": ("vec3<double>", 3),
    "xyzcal.px": ("vec3<double>", 3),
    "xyzobs.mm.value": ("vec3<double>", 3),
    "xyzobs.mm.variance": ("vec3<double>", 3),
    "xyzobs.px.value": ("vec3<double>", 3),
    "xyzobs.px.variance": ("vec3<double>", 3),
    "miller_index": ("cctbx::miller::index<>", 3),
}

DEFAULT_REFL_COLS = list(SCALAR_DTYPES.keys()) + list(VECTOR_COLUMNS.keys())

# Default columns to exclude
DEFAULT_EXCLUDED_COLS = ["BATCH", "PARTIAL"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a .refl file to a pytorch.pt file"
    )

    parser.add_argument(
        "--refl",
        type=str,
        help="Path to the .refl file",
    )
    parser.add_argument(
        "--fname",
        type=str,
        help="Name of out.pt file",
    )
    parser.add_argument(
        "--column-names",
        type=str,
        default=DEFAULT_REFL_COLS,
        help="Names DIALS .refl columns to extract",
    )
    parser.add_argument(
        "--excluded-columns",
        type=str,
        default=DEFAULT_EXCLUDED_COLS,
        help="Name of columns to exclude. Based off rs.read_dials_stills naming scheme",
    )
    return parser.parse_args()


def main():
    import reciprocalspaceship as rs
    import torch

    args = parse_args()
    ds = rs.io.read_dials_stills(
        args.refl,
        extra_cols=args.column_names,
    )

    data = {}
    for k, v in ds.items():
        if k not in args.excluded_columns:
            data[k] = torch.tensor(v, dtype=torch.float32)

    torch.save(data, args.fname)


def _refl_as_pt(
    refl,
    column_names=DEFAULT_REFL_COLS,
    excluded_columns=DEFAULT_EXCLUDED_COLS,
    out_dir: str | None = None,
):
    import reciprocalspaceship as rs
    import torch

    ds = rs.io.read_dials_stills(
        refl,
        extra_cols=column_names,
    )

    data = {}
    for k, v in ds.items():
        if k not in excluded_columns:
            data[k] = torch.tensor(v, dtype=torch.float32)

    if out_dir is not None:
        fname = Path(out_dir) / "metadata.pt"
    else:
        fname = "metadata.pt"

    torch.save(data, fname)


if __name__ == "__main__":
    main()
