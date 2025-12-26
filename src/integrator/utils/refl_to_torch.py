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


DEFAULT_DS_COLS = [
    "zeta",
    "xyzobs.px.variance.0",
    "xyzobs.px.variance.1",
    "xyzobs.px.variance.2",
    "xyzobs.px.value.0",
    "xyzobs.px.value.1",
    "xyzobs.px.value.2",
    "xyzobs.mm.variance.0",
    "xyzobs.mm.variance.1",
    "xyzobs.mm.variance.2",
    "xyzobs.mm.value.0",
    "xyzobs.mm.value.1",
    "xyzobs.mm.value.2",
    "xyzcal.mm.0",
    "xyzcal.mm.1",
    "xyzcal.mm.2",
    "refl_ids",
    "qe",
    "profile.correlation",
    "partiality",
    "partial_id",
    "panel",
    "num_pixels.valid",
    "num_pixels.foreground",
    "num_pixels.background_used",
    "num_pixels.background",
    "lp",
    "intensity.prf.variance",
    "intensity.prf.value",
    "imageset_id",
    "flags",
    "entering",
    "d",
    "bbox.0",
    "bbox.1",
    "bbox.2",
    "bbox.3",
    "bbox.4",
    "bbox.5",
    "background.sum.variance",
    "background.sum.value",
    "background.mean",
    "s1.0",
    "s1.1",
    "s1.2",
    "xyzcal.px.0",
    "xyzcal.px.1",
    "xyzcal.px.2",
    "intensity.sum.variance",
    "intensity.sum.value",
    "H",
    "K",
    "L",
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
