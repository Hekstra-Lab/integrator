import argparse

DEFAULT_EXCLUDED_COLS = ["BATCH", "PARTIAL"]

EXTRA_COLS = [
    "background.mean",
    "background.sum.value",
    "background.sum.variance",
    "bbox",
    "d",
    "entering",
    "flags",
    "imageset_id",
    "intensity.prf.value",
    "intensity.prf.variance",
    "lp",
    "num_pixels.background",
    "num_pixels.background_used",
    "num_pixels.foreground",
    "num_pixels.valid",
    "panel",
    "partial_id",
    "partiality",
    "profile.correlation",
    "qe",
    "refl_ids",
    "xyzcal.mm",
    "xyzobs.mm.value",
    "xyzobs.mm.variance",
    "xyzobs.px.value",
    "xyzobs.px.variance",
    "zeta",
]


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
        default=EXTRA_COLS,
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


if __name__ == "__main__":
    main()
