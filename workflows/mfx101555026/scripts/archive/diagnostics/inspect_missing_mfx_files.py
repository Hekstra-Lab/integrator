from pathlib import Path
import re
import csv
import traceback
import numpy as np

from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory


ORIGINAL_DIR = Path(
    "/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/outputs/r0269/018_rg070/out"
)

METADATA = Path(
    "/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/mfx_shoebox_r0269_018_rg070_combined_all/metadata.npy"
)

OUT_CSV = Path(
    "/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/diagnostics_missing_0269_files.csv"
)


def image_num_from_name(path: Path):
    match = re.search(r"idx-data_(\d+)_integrated\.refl$", path.name)
    if not match:
        return None
    return int(match.group(1))


def matching_expt(refl_path: Path):
    return refl_path.with_suffix(".expt")


def main():
    meta = np.load(METADATA, allow_pickle=True).item()
    represented = set(int(x) for x in np.unique(meta["image_num"]))

    refl_paths = sorted(ORIGINAL_DIR.glob("idx-data_*_integrated.refl"))

    rows = []

    for i, refl_path in enumerate(refl_paths, start=1):
        image_num = image_num_from_name(refl_path)
        expt_path = matching_expt(refl_path)

        status = "UNKNOWN"
        n_refl = None
        n_experiments = None
        raw_ok = False
        error_type = ""
        error_message = ""

        try:
            if image_num is None:
                status = "BAD_FILENAME"
                raise RuntimeError("Could not parse image_num from filename")

            if image_num in represented:
                status = "OK_IN_METADATA"
            else:
                status = "MISSING_FROM_METADATA"

            if not expt_path.exists():
                status = "MISSING_EXPT"
                raise FileNotFoundError(f"Missing matching expt: {expt_path}")

            refl = flex.reflection_table.from_file(str(refl_path))
            n_refl = len(refl)

            if n_refl == 0:
                status = "EMPTY_REFL"
                raise RuntimeError("Reflection table has zero rows")

            experiments = ExperimentListFactory.from_json_file(
                str(expt_path),
                check_format=True,
            )
            n_experiments = len(experiments)

            if n_experiments != 1:
                status = "BAD_EXPERIMENT_COUNT"
                raise RuntimeError(f"Expected 1 experiment, got {n_experiments}")

            imageset = experiments[0].imageset

            # This is the important test: can dxtbx/psana2 actually open raw data?
            raw = imageset.get_raw_data(0)
            raw_ok = True

            # If it was missing from metadata but all basic open tests pass,
            # then the reason is probably later filtering/cropping/mask logic.
            if image_num not in represented:
                status = "MISSING_FROM_METADATA_BUT_OPENS"

        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e).replace("\n", " ")[:500]

        rows.append(
            {
                "image_num": image_num,
                "refl_path": str(refl_path),
                "expt_path": str(expt_path),
                "status": status,
                "in_metadata": image_num in represented if image_num is not None else False,
                "n_refl": n_refl,
                "n_experiments": n_experiments,
                "raw_ok": raw_ok,
                "error_type": error_type,
                "error_message": error_message,
            }
        )

        if i % 100 == 0:
            print(f"Checked {i}/{len(refl_paths)} files...")

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_num",
                "status",
                "in_metadata",
                "n_refl",
                "n_experiments",
                "raw_ok",
                "error_type",
                "error_message",
                "refl_path",
                "expt_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {OUT_CSV}")

    # Summary
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    print("\nSummary:")
    for k, v in sorted(counts.items()):
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
