from pathlib import Path
import argparse
import re
import time
import numpy as np

from dxtbx.model.experiment_list import ExperimentListFactory


BASE = Path("/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx")
DEFAULT_IN_DIR = BASE / "outputs/r0269/018_rg070/out"
DEFAULT_OUT_DIR = BASE / "raw_panel_cache_r0269_018_rg070"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump MFX raw detector panels from .expt files into cached .npz files."
    )

    parser.add_argument(
        "--in-dir",
        type=str,
        default=str(DEFAULT_IN_DIR),
        help="Input folder containing idx-data_XXXXX_integrated.expt files.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Output folder for raw_panels_image_XXXXX.npz files.",
    )

    parser.add_argument(
        "--start-file",
        type=int,
        default=0,
        help="Starting index in the sorted .expt file list.",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of .expt files to process.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cached .npz files.",
    )

    return parser.parse_args()


def image_num_from_path(path: Path) -> int:
    m = re.search(r"idx-data_(\d+)_integrated\.expt$", path.name)
    if not m:
        raise ValueError(f"Cannot parse image number from {path}")
    return int(m.group(1))


def main():
    args = parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expt_paths = sorted(in_dir.glob("idx-data_*_integrated.expt"))

    start = args.start_file
    stop = None if args.max_files is None else start + args.max_files
    expt_paths = expt_paths[start:stop]

    print(f"Input dir: {in_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Start file: {args.start_file}")
    print(f"Max files: {args.max_files}")
    print(f"Selected {len(expt_paths)} expt files")

    total_load_expt = 0.0
    total_get_raw = 0.0
    total_save = 0.0
    total_wall_start = time.perf_counter()

    n_saved = 0
    n_skipped_existing = 0
    n_failed = 0

    for local_i, expt_path in enumerate(expt_paths):
        image_num = image_num_from_path(expt_path)
        out_path = out_dir / f"raw_panels_image_{image_num:05d}.npz"

        if out_path.exists() and not args.overwrite:
            n_skipped_existing += 1
            print(f"[{local_i+1}/{len(expt_paths)}] exists, skipping {out_path.name}")
            continue

        try:
            t0 = time.perf_counter()
            experiments = ExperimentListFactory.from_json_file(
                str(expt_path),
                check_format=True,
            )
            total_load_expt += time.perf_counter() - t0

            t0 = time.perf_counter()
            raw = experiments[0].imageset.get_raw_data(0)
            total_get_raw += time.perf_counter() - t0

            panels = {
                f"panel_{i:02d}": raw[i].as_numpy_array().astype(np.float32, copy=False)
                for i in range(len(raw))
            }

            t0 = time.perf_counter()

            # Uncompressed .npz: larger files, but faster save/load than savez_compressed.
            np.savez(out_path, **panels)

            total_save += time.perf_counter() - t0

            try:
                experiments[0].imageset.clear_cache()
            except Exception:
                pass

            size_mb = out_path.stat().st_size / 1024 / 1024
            n_saved += 1
            print(f"[{local_i+1}/{len(expt_paths)}] saved {out_path.name}: {size_mb:.1f} MB")

        except Exception as e:
            n_failed += 1
            print(f"[{local_i+1}/{len(expt_paths)}] FAILED {expt_path.name}: {type(e).__name__}: {e}")

    total_wall = time.perf_counter() - total_wall_start

    print("")
    print("Timing summary")
    print(f"load .expt:        {total_load_expt:.2f} sec")
    print(f"get_raw_data:      {total_get_raw:.2f} sec")
    print(f"save .npz:         {total_save:.2f} sec")
    print(f"accounted total:   {total_load_expt + total_get_raw + total_save:.2f} sec")
    print(f"wall time:         {total_wall:.2f} sec")
    print("")
    print(f"saved:             {n_saved}")
    print(f"skipped existing:  {n_skipped_existing}")
    print(f"failed:            {n_failed}")


if __name__ == "__main__":
    main()