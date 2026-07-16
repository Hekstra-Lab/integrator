"""Extract fixed-size shoebox windows for integrator training.

Reads a DIALS `integrated.refl` + `integrated.expt` and reconstructs a
fixed-size window around each predicted centroid directly from the raw image
data.

Two modes:
1. Default (rotation / sequence): one imageset with a rotation scan.
2. --laue: many single-frame stills from laue-dials. --d must be 1;

Examples: 

Run (rotation mode):
integrator.make_shoeboxes \
    --data-dir /n/.../dials \
    --refl integrated.refl \
    --expt integrated.expt \
    --out-dir /n/.../pytorch_data \
    --w 21 --h 21 --d 3

Run (laue mode): 
integrator.make_shoeboxes --laue \
    --data-dir /n/.../laue-dials \
    --refl integrated.refl \
    --expt integrated.expt \
    --out-dir /n/.../pytorch_data \
    --w 21 --h 21 --d 1 \
    --max-images 1000
"""
import time
import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

try:
    import torch
except ModuleNotFoundError:
    torch = None

from integrator.io import refl_as_pt, save_data
from dials.array_family import flex

# re to parse image numbers from laue-dials filenames
_TRAILING_INT_RE = re.compile(r"_(\d+)\.[A-Za-z0-9]+$")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="integrator.make_shoeboxes",
        description=(
            "Reconstruct fixed-size shoebox windows from a DIALS "
            "integrated.refl + integrated.expt. Default handles rotation "
            "sequences; --laue handles laue-dials stills."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--laue",
        action="store_true",
        help="extract from laue-dials single-frame stills instead of a "
        "rotation sequence",
    )

    
    #Thao: added MFX mode to process cctbx.xfel integrated refl/expt pairs for stills.
    parser.add_argument(
        "--mfx",
        action="store_true",
        help="Process MFX/cctbx.xfel serial stills output using *_integrated.refl/expt pairs.",
    )

    # Thao: added MFX combined mode to process one combined .expt/.refl pair
    # created by dials.combine_experiments.
    parser.add_argument(
        "--mfx-combined",
        action="store_true",
        help=(
            "Process one combined MFX/cctbx.xfel .expt/.refl pair. "
            "The combined .refl should contain an id column linking each "
            "reflection row to experiments[id] in the combined .expt file."
        ),
    )

    parser.add_argument(
        "--mfx-pattern",
        default="idx-data_*_integrated.refl",
        help="Glob pattern for MFX integrated reflection files inside data_dir.",
    )

    #Thao: Added start and max file arguments to allow batching of MFX integrated files for large datasets.
    parser.add_argument(
        "--start-file",
        type=int,
        default=0,
        help="Starting index in the sorted MFX integrated file list.",
    )

    #Thao: Added max-files argument to allow batching of MFX integrated files for large datasets.
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of MFX integrated files to process.",
    )

    #Thao: Added detector-mask argument to allow users to provide a cctbx/DIALS detector mask file for MFX mode.
    parser.add_argument(
        "--detector-mask",
        "--mask",
        dest="detector_mask",
        default=None,
        help=(
            "Optional cctbx/DIALS detector mask file for MFX mode. "
            "Example: hot_lines_combined5.mask. The file is loaded with "
            "libtbx.easy_pickle and should contain one boolean mask per panel, "
            "where True means good pixel and False means bad/masked pixel."
        ),
    )

    # Thao: optional raw-panel cache for MFX mode.
    # If this is supplied, run_mfx() loads raw detector panels from files like
    # raw_panels_image_01418.npz instead of calling get_raw_data(0).
    parser.add_argument(
        "--mfx-raw-cache-dir",
        type=str,
        default=None,
        help=(
            "Optional MFX raw-panel cache directory containing files named "
            "raw_panels_image_XXXXX.npz. When supplied with --mfx, the code "
            "loads cached panel arrays from this directory instead of calling "
            "ExperimentListFactory.from_json_file(..., check_format=True) and "
            "get_raw_data(0)."
        ),
    )

    common = parser.add_argument_group("common options")
    common.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="directory containing the .refl and .expt files",
    )
    common.add_argument(
        "--refl",
        type=str,
        default=None,
        help="refl filename",
    )
    common.add_argument(
        "--expt",
        type=str,
        default=None,
        help="expt filename",
    )
    common.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="output directory (default: out_dir)",
    )
    common.add_argument(
        "--w",
        type=int,
        default=21,
        help="window width (odd)",
    )
    common.add_argument(
        "--h", type=int, default=21, help="window height (odd)"
    )
    common.add_argument(
        "--d",
        type=int,
        default=1,
        help="window depth in frames (rotation: odd, e.g. 3; laue: 1)",
    )
    common.add_argument(
        "--counts-dtype",
        type=str,
        default=None,
        choices=["uint16", "int32", "float32"],
        help="storage dtype for pixel counts (rotation default: uint16; "
        "laue default: int32)",
    )
    common.add_argument(
        "--counts-fname",
        type=str,
        default="counts.npy",
    )
    common.add_argument(
        "--masks-fname",
        type=str,
        default="masks.npy",
    )
    common.add_argument(
        "--refl-fname",
        type=str,
        default="reflections_.refl",
        help="filename of output reflection table",
    )
    common.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="rotation: frames per worker block; laue: images per worker "
        "chunk",
    )
    common.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="max parallel workers (capped by os.cpu_count())",
    )
    common.add_argument(
        "--no-mask-overlap",
        action="store_true",
        help="skip overlap masking",
    )
    common.add_argument(
        "--shoebox-format",
        type=str,
        default="npy",
        choices=["npy", "pt"],
        help="storage format for counts and masks. npy uses streaming "
        "memmap writes. pt converts each memmap to a .pt tensor at the end ",
    )
    common.add_argument(
        "--no-stats",
        action="store_true",
        help="skip the normalization stats (stored in dataset.yaml) and "
        "concentration.npy (written by default)",
    )
    common.add_argument(
        "--stats-chunk",
        type=int,
        default=10_000,
    )

    #Thao: Added write-chunk-size argument to control how many shoeboxes are kept in memory before writing a chunk folder to disk in MFX mode.
    common.add_argument(
        "--write-chunk-size",
        type=int,
        default=10_000,
        help=(
            "MFX mode: number of shoeboxes to keep in memory before "
            "writing one chunk folder to disk."
        ),
    )
    common.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="fraction of reflections to flag as is_test (random)",
    )

    laue = parser.add_argument_group("laue mode (--laue)")
    laue.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="keep only reflections whose image_num is < this value",
    )
    return parser.parse_args()


def _require(args, names):
    """Raise if any of `names` is unset on `args` for the chosen mode."""
    missing = [
        f"--{n.replace('_', '-')}" for n in names if getattr(args, n) is None
    ]
    if missing:
        raise SystemExit(
            "integrator.make_shoeboxes: missing required argument(s) for this mode: "
            + ", ".join(missing)
        )


def main():
    args = parse_args()

    if args.mfx_combined:
        run_mfx_combined(args)
    elif args.mfx:
        run_mfx(args)
    elif args.laue:
        run_laue(args)
    else:
        run_dials(args)


# Example MFX test command used during development.
#
# cd /sdf/home/t/thaoh/s3df_practice/integrator
# export PYTHONPATH=/sdf/home/t/thaoh/s3df_practice/integrator/src:$PYTHONPATH
#
# dials.python src/integrator/cli/make_shoeboxes.py \
#   --mfx \
#   --data-dir /sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/outputs/r0269/012_rg058/out \
#   --out-dir /sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/mfx_shoebox_test_masked \
#   --mfx-pattern "idx-data_36001_integrated.refl" \
#   --detector-mask /sdf/data/lcls/ds/mfx/mfx101555026/results/pam/hot_lines_combined5.mask \
#   --w 25 \
#   --h 25 \
#   --d 1 \
#   --counts-dtype float32 \
#   --no-mask-overlap \
#   --no-stats
#
# Why dials.python? It runs Python with the DIALS/cctbx environment loaded.
# Why PYTHONPATH? It lets Python find this local integrator source tree.

def run_mfx(args):
    """Create an integrator-style shoebox dataset from MFX/cctbx.xfel outputs.

    Expected input folder:
        args.data_dir should point to a cctbx.xfel `out/` folder, for example:

            /.../outputs/r0269/012_rg058/out

    Expected files inside data_dir:
        idx-data_XXXXX_integrated.refl
        idx-data_XXXXX_integrated.expt

    Main idea:
        - The .refl file gives reflection metadata: panel, bbox, Miller index,
          intensity, background, etc.
        - The .expt file lets dxtbx load the actual detector pixels.
        - For a fixed-size integrator dataset, we crop a fixed window centered
          on xyzcal.px from the correct detector panel.

    Notes for this first prototype:
        - MFX still images should use --d 1.
        - Existing MFX integrated bbox values can have different sizes, so this
          function uses --w/--h fixed windows around xyzcal.px instead of
          stacking variable-size integration bboxes.
        - Edge reflections are padded with zeros and masked False where pixels
          fall outside the panel boundary.

    Streaming/chunking update:
        - Instead of storing every shoebox in Python lists until the end, this
          version writes chunk folders such as chunk_00000, chunk_00001, ...
          during extraction.
        - After all files are processed, the chunk folders are merged into final
          counts.npy, masks.npy, and metadata.npy.
        - This keeps memory lower during the MFX extraction loop.

    Skip-bad-file update:
        - Some .refl/.expt pairs can exist on disk but fail when dxtbx/psana2
          tries to open the image data referenced by the .expt file.
        - Instead of crashing the whole Slurm array task, this version prints a
          warning, records the bad pair, skips it, and continues to the next pair.

    Raw-panel-cache update:
        - If --mfx-raw-cache-dir is supplied, this function loads saved NumPy
          panel arrays such as raw_panels_image_01418.npz.
        - This skips the slow dxtbx/psana2 path:
              ExperimentListFactory.from_json_file(..., check_format=True)
              imageset.get_raw_data(0)
        - The .refl file is still used for reflection metadata, and the .expt
          JSON is read lightly for image index, wavelength, and unit cell.
    """
    # To read cctbx.xfel integrated .refl/.expt files, we need DIALS and dxtbx.
    # flex provides the DIALS array objects used by reflection tables.
    from dials.array_family import flex

    # ExperimentListFactory loads the .expt experiment list and lets us access
    # the underlying imageset/raw pixel data for the MFX stills.
    from dxtbx.model.experiment_list import ExperimentListFactory

    from integrator.io import write_dataset_yaml

    # MFX/cctbx.xfel output files are still images, so depth should be 1.
    if args.d != 1:
        raise ValueError(f"--d must be 1 for MFX still-image extraction (got {args.d})")

    # Fixed-size windows need a clear center pixel.
    if args.w % 2 == 0 or args.h % 2 == 0:
        raise ValueError(f"--w and --h must be odd (got w={args.w}, h={args.h})")

    # MFX mode only needs a folder of integrated refl/expt pairs.
    _require(args, ["data_dir"])

    # Convert input/output paths from strings to Path objects.
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir or "out_dir")

    # Create output folder if needed.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all integrated reflection tables in the cctbx.xfel out/ folder.
    # Example exact match: "idx-data_36001_integrated.refl"
    # Example wildcard match: "idx-data_*_integrated.refl"
    #
    # Path.glob(...) searches only inside data_dir for names matching the pattern.
    # sorted(...) makes the order deterministic.
    refl_paths = sorted(data_dir.glob(args.mfx_pattern))
    if not refl_paths:
        raise SystemExit(f"No MFX integrated reflection files found in {data_dir}")

    # Process only a slice of the sorted file list when batching MFX jobs.
    # Example: --start-file 25 --max-files 25 processes files 25 through 49.
    start = args.start_file
    stop = None if args.max_files is None else start + args.max_files
    refl_paths = refl_paths[start:stop]
    if not refl_paths:
        raise SystemExit(
            f"No MFX integrated reflection files selected after slicing "
            f"start={args.start_file}, max_files={args.max_files}"
        )

    # Storage dtype. Use float32 by default because Jungfrau/cctbx raw pixels can
    # be floating point and may include negative values.
    counts_dtype = np.dtype(args.counts_dtype or "float32")

    # Optional real detector mask for MFX data.
    #
    # Pam's/cctbx mask file loads as a tuple of 32 flex.bool arrays, one per
    # Jungfrau panel. Each panel mask has the same shape as the corresponding
    # raw detector panel, e.g. (514, 1030). In the inspected file, about 98.7%
    # of pixels were True, so we treat True as good/usable and False as
    # bad/hot/masked.
    #
    # If no detector mask is supplied, the MFX path falls back to the simpler
    # finite/padding mask: True for finite pixels inside the panel, False for
    # outside-panel padding or NaN/inf.
    detector_masks = None
    if args.detector_mask is not None:
        from libtbx import easy_pickle

        detector_masks = easy_pickle.load(args.detector_mask)
        print(f"loaded detector mask with {len(detector_masks)} panel mask(s): {args.detector_mask}")

    # Optional raw-panel cache for MFX mode.
    # This is the local-copy speed test suggested after the psana/dxtbx timing:
    #   slow path: .expt -> ExperimentListFactory -> get_raw_data(0)
    #   cache path: raw_panels_image_XXXXX.npz -> NumPy arrays
    #
    # The cache files are expected to contain panel_00, panel_01, ..., panel_31.
    raw_cache_dir = None
    if args.mfx_raw_cache_dir is not None:
        raw_cache_dir = Path(args.mfx_raw_cache_dir)
        if not raw_cache_dir.exists():
            raise SystemExit(f"MFX raw-panel cache directory not found: {raw_cache_dir}")
        print(f"using MFX raw-panel cache: {raw_cache_dir}")

    # Chunk buffers.
    # These replace the old full-run lists:
    #   counts = []
    #   masks = []
    #   metadata_rows = []
    #
    # Each buffer stores only up to args.write_chunk_size shoeboxes. When the
    # chunk is full, flush_chunk() writes it to disk and clears memory.
    chunk_counts = []
    chunk_masks = []
    chunk_metadata_rows = []

    # List of chunk folders written to disk. Used later by merge_mfx_chunks().
    chunk_dirs = []

    # Chunk counter: chunk_00000, chunk_00001, ...
    chunk_id = 0

    # Global row counter across the final merged dataset.
    # This keeps refl_ids unique across chunks.
    global_row = 0

    # Number of shoeboxes to hold in memory before writing one chunk.
    write_chunk_size = int(getattr(args, "write_chunk_size", 10_000))

    def flush_chunk():
        """Write current chunk buffers to disk and clear them from memory."""
        nonlocal chunk_id, global_row
        nonlocal chunk_counts, chunk_masks, chunk_metadata_rows

        # Nothing to write.
        if not chunk_counts:
            return

        # Create folder like chunk_00000.
        chunk_dir = out_dir / f"chunk_{chunk_id:05d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        # Stack chunk lists into arrays.
        # counts_chunk shape: (n_chunk, d*h*w), usually (n_chunk, 625)
        # masks_chunk shape:  (n_chunk, d*h*w), usually (n_chunk, 625)
        counts_chunk = np.stack(chunk_counts)
        masks_chunk = np.stack(chunk_masks).astype(np.bool_)

        # Convert this chunk's row metadata into Luis-style dict.
        # global_start tells the metadata helper where this chunk begins in the
        # final merged dataset, so refl_ids stay unique after merging.
        metadata_chunk = _mfx_metadata_rows_to_luis_dict(
            chunk_metadata_rows,
            args.test_fraction,
            global_start=global_row,
        )

        # This helps long MFX jobs avoid "Too many open files" during np.save.
        import gc
        gc.collect()

        # Save this chunk to disk.
        np.save(chunk_dir / args.counts_fname, counts_chunk)
        np.save(chunk_dir / args.masks_fname, masks_chunk)
        np.save(chunk_dir / "metadata.npy", metadata_chunk)

        # Remember this chunk for final merge.
        chunk_dirs.append(chunk_dir)

        n_chunk = counts_chunk.shape[0]
        print(f"wrote chunk {chunk_id}: {n_chunk} shoeboxes")

        # Advance global row counter.
        global_row += n_chunk

        # Move to next chunk ID.
        chunk_id += 1

        # Clear memory for next chunk.
        chunk_counts = []
        chunk_masks = []
        chunk_metadata_rows = []

    n_files = 0
    n_reflections_seen = 0
    n_reflections_saved = 0

    # New: keep a list of skipped bad file pairs for the final summary.
    skipped_files = []

    for refl_path in refl_paths:
        # Each integrated .refl should have a matching integrated .expt.
        expt_path = None

        # New: skip-bad-file protection.
        #
        # Some .refl/.expt pairs exist on disk, but dxtbx/psana2 may fail when
        # opening the image data referenced by the .expt file.
        #
        # Without this try/except:
        #   one bad .expt crashes the whole Slurm array task.
        #
        # With this try/except:
        #   print the bad file pair, record it, and continue to the next file.
        try:
            expt_path = _matching_mfx_expt(refl_path)
            
            if expt_path is None:
                print("WARNING: skipping MFX file pair with missing .expt")
                print(f"  refl: {refl_path}")
                skipped_files.append((str(refl_path), "missing", "missing .expt"))
                continue
                
            # Load reflection table metadata.
            #
            # A single integrated .refl file can contain many reflection rows.
            # In the first test file, idx-data_36001_integrated.refl had 415 rows.
            # Example row information looked like:
            #   row 0: panel=31, xyzcal.px=(777.30, 384.75, 0.0),
            #          intensity.sum.value=16.07, ...
            reflections = flex.reflection_table.from_file(str(refl_path))

            # Check required columns early so errors are easier to understand.
            required_cols = [
                "panel",
                "xyzcal.px",
                "miller_index",
                "intensity.sum.value",
                "intensity.sum.variance",
                "background.sum.value",
                "background.sum.variance",
            ]
            missing = [col for col in required_cols if col not in reflections]

            if missing:
                print("WARNING: skipping MFX file pair with missing columns")
                print(f"  refl: {refl_path}")
                print(f"  expt: {expt_path}")
                print(f"  missing: {missing}")
                skipped_files.append(
                    (str(refl_path), str(expt_path), f"missing columns: {missing}")
                )
                continue

            # Try to record the event/image index stored in the ImageSet JSON, if present.
            # For example, idx-data_36001_integrated.expt had single_file_indices=[36001].
            #
            # In raw-panel-cache mode, this image_index is also used to find:
            #   raw_panels_image_36001.npz
            image_index = _mfx_image_index_from_expt_json(expt_path)

            # These are filled differently depending on whether we read raw pixels
            # from dxtbx/psana2 or from the local raw-panel cache.
            experiments = None
            wavelength = np.nan
            unit_cell_for_cache = None
            raw_cache_file = None

            if raw_cache_dir is None:
                # Load experiment and connect to underlying image source.
                # check_format=True is what lets dxtbx read the real pixels through the
                # imageset/data.loc mechanism.
                experiments = ExperimentListFactory.from_json_file(
                    str(expt_path),
                    check_format=True,
                )

                # New: do not crash if this .expt has an unexpected number of experiments.
                # For this MFX prototype, we expect exactly one experiment per file.
                if len(experiments) != 1:
                    print("WARNING: skipping MFX file pair with unexpected experiment count")
                    print(f"  refl: {refl_path}")
                    print(f"  expt: {expt_path}")
                    print(f"  n_experiments: {len(experiments)}")
                    skipped_files.append(
                        (str(refl_path), str(expt_path), f"n_experiments={len(experiments)}")
                    )
                    continue

                # raw is a tuple of panel images for one event/image:
                #   raw[0], raw[1], ..., raw[31]
                #
                # Mental model:
                #   imageset = the book / file access instructions
                #   raw      = one page/image loaded from that book
                #   panel    = one detector tile/region on that page
                #   pixel    = one number inside that tile
                #
                # experiments[0].imageset is a loader/access object, not pixels yet.
                # get_raw_data(0) loads the actual pixels for image index 0 inside this
                # one-image imageset. For this MFX Jungfrau data, it returns 32 panel
                # flex arrays, each with shape about (514, 1030).
                #
                # New: put get_raw_data(0) inside the try block too.
                # This can fail if dxtbx can read the .expt JSON but cannot open the
                # actual image data referenced by the imageset.
                raw = experiments[0].imageset.get_raw_data(0)

                # retrieve wavelength from the beam object in the experiment. If it fails, set wavelength to NaN.
                try:
                    wavelength = float(experiments[0].beam.get_wavelength())
                except Exception:
                    wavelength = np.nan

            else:
                # Raw-panel-cache mode:
                #   Do not call ExperimentListFactory.from_json_file(..., check_format=True).
                #   Do not call imageset.get_raw_data(0).
                # Instead, load local NumPy panel arrays created by dump_raw_panels_test.py.
                raw, raw_cache_file = _load_mfx_raw_panels_from_npz(
                    raw_cache_dir,
                    image_index,
                )

                # The .expt JSON is still useful metadata. Reading the JSON directly
                # is much lighter than check_format=True + get_raw_data(0).
                wavelength = _mfx_wavelength_from_expt_json(expt_path)
                unit_cell_for_cache = _mfx_unit_cell_from_expt_json(expt_path)

            # If a detector mask was provided, it should have one mask per raw
            # detector panel. For this MFX Jungfrau case, both should have length 32.
            #
            # New: skip instead of crashing if the mask panel count does not match.
            if detector_masks is not None and len(detector_masks) != len(raw):
                print("WARNING: skipping MFX file pair with detector mask mismatch")
                print(f"  refl: {refl_path}")
                print(f"  expt: {expt_path}")
                print(f"  detector mask panels: {len(detector_masks)}")
                print(f"  raw panels: {len(raw)}")
                skipped_files.append(
                    (str(refl_path), str(expt_path), "detector mask panel mismatch")
                )
                continue

        except Exception as e:
            print("WARNING: skipping bad MFX file pair")
            print(f"  refl: {refl_path}")
            print(f"  expt: {expt_path if expt_path is not None else 'unknown'}")
            print(f"  error: {type(e).__name__}: {e}")
            skipped_files.append(
                (
                    str(refl_path),
                    str(expt_path) if expt_path is not None else "unknown",
                    f"{type(e).__name__}: {e}",
                )
            )
            continue

        n_files += 1

        # Count reflection rows, not files. One file may add hundreds of rows.
        n_reflections_seen += len(reflections)

        # image_index and wavelength were already set above.
        # In normal mode, wavelength came from experiments[0].beam.
        # In raw-panel-cache mode, wavelength came from the .expt JSON.

        # Loop over reflections in this event.
        for i in range(len(reflections)):
            panel_id = int(reflections["panel"][i])

            # xyzcal.px is the DIALS/cctbx calculated pixel position from the
            # refined geometry/model. It is not an ML prediction from integrator.
            # We use it as the center of the fixed shoebox crop.
            xyz = reflections["xyzcal.px"][i]
            x_center = float(xyz[0])
            y_center = float(xyz[1])

            if panel_id < 0 or panel_id >= len(raw):
                # Bad panel ID; skip this reflection.
                continue

            panel_image = raw[panel_id]

            # If a detector mask was supplied, select the mask for this same
            # panel. The raw pixel panel and mask panel should have matching
            # shapes, e.g. both (514, 1030).
            panel_mask_np = None
            if detector_masks is not None:
                panel_mask_np = detector_masks[panel_id].as_numpy_array().astype(
                    bool,
                    copy=False,
                )

            # Crop a fixed-size window centered on xyzcal.px.
            # Returns a 2D counts array with shape (args.h, args.w) and a mask.
            # If panel_mask_np is provided, the mask combines:
            #   finite pixel values AND detector-good pixels.
            # If panel_mask_np is None, the mask only checks finite pixels and
            # outside-panel padding.
            shoebox, mask, fixed_bbox = _crop_mfx_fixed_shoebox(
                panel_image=panel_image,
                x_center=x_center,
                y_center=y_center,
                width=args.w,
                height=args.h,
                panel_mask_np=panel_mask_np,
            )

            # Flatten the fixed-size 2D shoebox/mask into one 1D row.
            # Pixel values stay the same; only the shape changes.
            # Example: (25, 25) -> (625,).
            #
            # Streaming update:
            #   append to the current chunk buffer, not to a full-run list.
            chunk_counts.append(shoebox.reshape(-1).astype(counts_dtype, copy=False))
            chunk_masks.append(mask.reshape(-1))
            n_reflections_saved += 1

            # Existing integration bbox is useful metadata, but it may not have
            # the fixed --w/--h shape. Keep it if the column exists.
            integration_bbox = tuple(reflections["bbox"][i]) if "bbox" in reflections else None

            # MFX metadata is built manually, so compute d-spacing here from
            # the experiment crystal/unit cell and the reflection Miller index.
            miller_index = tuple(int(v) for v in reflections["miller_index"][i])
            if raw_cache_dir is None:
                d_spacing = _mfx_d_spacing(experiments[0], miller_index)
            else:
                d_spacing = _mfx_d_spacing_from_unit_cell(unit_cell_for_cache, miller_index)

            # Add one metadata row for this shoebox.
            # background.mean is optional in general, but your checked MFX
            # integrated.refl files contain it, so copy it when present.
            chunk_metadata_rows.append(
                {
                    "source_refl": refl_path.name,
                    "source_expt": expt_path.name,
                    "image_index": image_index,
                    "reflection_id": i,
                    "panel": panel_id,
                    "fixed_bbox": tuple(fixed_bbox),
                    "integration_bbox": integration_bbox,
                    "detector_mask": str(args.detector_mask) if args.detector_mask else None,
                    "raw_cache_file": raw_cache_file.name if raw_cache_file is not None else "",
                    "xyzcal_px": tuple(float(v) for v in reflections["xyzcal.px"][i]),
                    "xyzobs_px_value": tuple(float(v) for v in reflections["xyzobs.px.value"][i])
                    if "xyzobs.px.value" in reflections
                    else None,
                    "miller_index": miller_index,
                    "d": d_spacing,
                    "intensity_sum_value": float(reflections["intensity.sum.value"][i]),
                    "intensity_sum_variance": float(reflections["intensity.sum.variance"][i]),
                    "background_sum_value": float(reflections["background.sum.value"][i]),
                    "background_sum_variance": float(reflections["background.sum.variance"][i]),
                    "background.mean": (
                        float(reflections["background.mean"][i])
                        if "background.mean" in reflections
                        else np.nan
                    ),
                    "wavelength": wavelength,
                }
            )

            # If the current chunk is full, write it to disk and clear memory.
            if len(chunk_counts) >= write_chunk_size:
                flush_chunk()

        # imageset is the loader/cache object. clear_cache() tells it to forget
        # loaded image pixels after we finish this event, so memory does not grow
        # while processing many .refl/.expt pairs.
        if experiments is not None:
            try:
                experiments[0].imageset.clear_cache()
            except Exception as e:
                print(f"WARNING: could not clear image cache for {expt_path}: {type(e).__name__}: {e}")

    # Write the final partial chunk, if any.
    flush_chunk()

    if not chunk_dirs:
        raise SystemExit("No MFX shoeboxes were extracted; check input files and bbox/panel data.")

    counts_path = out_dir / args.counts_fname
    masks_path = out_dir / args.masks_fname
    metadata_path = out_dir / "metadata.npy"

    # Merge all chunk folders into final counts.npy, masks.npy, metadata.npy.
    # merge_mfx_chunks writes counts/masks through memmap so the final merge does
    # not need to hold all pixel arrays in memory at once.
    n_saved = merge_mfx_chunks(
        chunk_dirs=chunk_dirs,
        out_dir=out_dir,
        counts_fname=args.counts_fname,
        masks_fname=args.masks_fname,
    )

    stats = None

    # concentration.npy is created only when we do NOT use: --no-stats
    if not args.no_stats:
        stats = _raw_stats_from_memmap(counts_path, masks_path, chunk=args.stats_chunk)

        _save_concentration_from_memmap(
            counts_path=counts_path,
            out_dir=out_dir,
            chunk=args.stats_chunk,
        )
        print("wrote concentration.npy")

    # Write a simple dataset.yaml so the output folder has the expected dataset
    # description. The metadata format may still need to be aligned with Luis's
    # final data-loader expectations.
    write_dataset_yaml(
        out_dir,
        geometry={"d": args.d, "h": args.h, "w": args.w},
        n_reflections=n_saved,
        polychromatic=False,
        anscombe=False,
        files={
            "counts": args.counts_fname,
            "masks": args.masks_fname,
            "reference": "metadata.npy",
        },
        crystal=None,
        stats=stats,
        refl_file=None,
    )

    print(f"processed {n_files} MFX integrated file(s)")
    print(f"seen {n_reflections_seen} reflection(s)")
    print(f"saved {n_reflections_saved} fixed-size shoebox(es)")
    print(f"wrote {counts_path}")
    print(f"wrote {masks_path}")
    print(f"wrote {metadata_path}")
    print(f"wrote dataset.yaml under {out_dir}")

    print(f"skipped {len(skipped_files)} bad MFX file pair(s)")
    for refl_name, expt_name, reason in skipped_files[:50]:
        print("SKIPPED:")
        print(f"  refl: {refl_name}")
        print(f"  expt: {expt_name}")
        print(f"  reason: {reason}")
    if len(skipped_files) > 50:
        print(f"... {len(skipped_files) - 50} more skipped file pair(s) not printed")


def run_mfx_combined(args):
    """Create an integrator-style shoebox dataset from combined MFX files.

    Expected input:
        args.data_dir should point to a folder containing one combined .expt
        and one combined .refl, for example:

            combined_10/combined_10.expt
            combined_10/combined_10.refl

    Run with:
        --mfx-combined
        --data-dir /path/to/combined_10
        --expt combined_10.expt
        --refl combined_10.refl

    Main idea:
        - dials.combine_experiments creates one combined experiment list and
          one combined reflection table.
        - The combined .refl has an "id" column.
        - reflections["id"] tells which experiment each reflection belongs to.
        - experiments[experiment_id] gives the matching image loader.
        - The original image/event number is preserved in single_file_indices.

    Example mapping:
        reflections with id == 0 -> experiments[0] -> original image 1418
        reflections with id == 1 -> experiments[1] -> original image 2203

    This keeps the chunked writing approach from run_mfx:
        - chunk_counts
        - chunk_masks
        - chunk_metadata_rows
        - flush_chunk()
        - merge_mfx_chunks()

    So the redesign changes how inputs are read, but keeps memory-safe output.

    Timing update:
        - This version measures time spent loading combined files, selecting
          reflection subsets, reading raw images, cropping/building metadata,
          flushing chunks, merging chunks, and writing dataset.yaml/stats.

    Skip-bad-experiment update:
        - If one experiment inside the combined .expt fails during raw-data
          access, detector-mask validation, image-index lookup, or wavelength
          lookup, record the failure and continue with the next experiment.
        - This prevents one bad image/event from stopping the full combined run.
    """
    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory
    from integrator.io import write_dataset_yaml

    # Total timer for the whole combined-file MFX approach.
    t_total_start = time.perf_counter()

    # Timing counters.
    t_path_setup = 0.0
    t_load_mask = 0.0
    t_load_refl = 0.0
    t_load_expt = 0.0
    t_get_ids = 0.0
    t_select_subset = 0.0
    t_get_raw = 0.0
    t_image_index_json = 0.0
    t_crop_loop = 0.0
    t_clear_cache = 0.0
    t_flush = 0.0
    t_merge = 0.0
    t_stats = 0.0
    t_write_yaml = 0.0

    # MFX/cctbx.xfel output files are still images, so depth should be 1.
    if args.d != 1:
        raise ValueError(f"--d must be 1 for MFX still-image extraction (got {args.d})")

    # Fixed-size windows need a clear center pixel.
    if args.w % 2 == 0 or args.h % 2 == 0:
        raise ValueError(f"--w and --h must be odd (got w={args.w}, h={args.h})")

    # Combined mode needs one folder, one .refl, and one .expt.
    _require(args, ["data_dir", "refl", "expt"])

    t0 = time.perf_counter()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir or "out_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    refl_path = data_dir / args.refl
    expt_path = data_dir / args.expt

    if not refl_path.exists():
        raise SystemExit(f"combined reflection file not found: {refl_path}")
    if not expt_path.exists():
        raise SystemExit(f"combined experiment file not found: {expt_path}")

    t_path_setup += time.perf_counter() - t0

    # Storage dtype. Use float32 by default because Jungfrau/cctbx raw pixels can
    # be floating point and may include negative values.
    counts_dtype = np.dtype(args.counts_dtype or "float32")

    # Optional real detector mask for MFX data.
    #
    # Pam's/cctbx mask file loads as a tuple of 32 flex.bool arrays, one per
    # Jungfrau panel. Each panel mask has the same shape as the corresponding
    # raw detector panel, e.g. (514, 1030). In the inspected file, about 98.7%
    # of pixels were True, so we treat True as good/usable and False as
    # bad/hot/masked.
    #
    # If no detector mask is supplied, the MFX path falls back to the simpler
    # finite/padding mask: True for finite pixels inside the panel, False for
    # outside-panel padding or NaN/inf.
    detector_masks = None
    if args.detector_mask is not None:
        t0 = time.perf_counter()

        from libtbx import easy_pickle

        detector_masks = easy_pickle.load(args.detector_mask)
        print(f"loaded detector mask with {len(detector_masks)} panel mask(s): {args.detector_mask}")

        t_load_mask += time.perf_counter() - t0

    # Load combined reflection table once.
    t0 = time.perf_counter()

    reflections = flex.reflection_table.from_file(str(refl_path))
    t_load_refl += time.perf_counter() - t0

    # Combined reflection table must have id so we can connect:
    #   reflections["id"] -> experiments[id]
    required_cols = [
        "id",
        "panel",
        "xyzcal.px",
        "miller_index",
        "intensity.sum.value",
        "intensity.sum.variance",
        "background.sum.value",
        "background.sum.variance",
    ]
    missing = [col for col in required_cols if col not in reflections]
    if missing:
        raise SystemExit(f"{refl_path} is missing required column(s): {missing}")

    # Load combined experiment list once.
    t0 = time.perf_counter()
    experiments = ExperimentListFactory.from_json_file(
        str(expt_path),
        check_format=True,
    )
    t_load_expt += time.perf_counter() - t0

    print(f"loaded combined experiments: {len(experiments)}")
    print(f"loaded combined reflections: {len(reflections)}")

    # Get actual experiment ids present in the reflection table.
    # Usually this is 0, 1, 2, ..., N-1, but using unique ids is safer.
    t0 = time.perf_counter()
    ids_np = reflections["id"].as_numpy_array()
    experiment_ids = sorted(int(v) for v in np.unique(ids_np))
    t_get_ids += time.perf_counter() - t0

    print(f"experiment ids in combined refl: {experiment_ids[:10]}")
    if len(experiment_ids) > 10:
        print(f"... total experiment ids: {len(experiment_ids)}")

    # Chunk buffers.
    # These replace the old full-run lists:
    #   counts = []
    #   masks = []
    #   metadata_rows = []
    #
    # Each buffer stores only up to args.write_chunk_size shoeboxes. When the
    # chunk is full, flush_chunk() writes it to disk and clears memory.
    chunk_counts = []
    chunk_masks = []
    chunk_metadata_rows = []

    # List of chunk folders written to disk. Used later by merge_mfx_chunks().
    chunk_dirs = []

    # Chunk counter: chunk_00000, chunk_00001, ...
    chunk_id = 0

    # Global row counter across the final merged dataset.
    # This keeps refl_ids unique across chunks.
    global_row = 0

    # Number of shoeboxes to hold in memory before writing one chunk.
    write_chunk_size = int(getattr(args, "write_chunk_size", 10_000))

    def flush_chunk():
        """Write current chunk buffers to disk and clear them from memory."""
        nonlocal chunk_id, global_row, t_flush
        nonlocal chunk_counts, chunk_masks, chunk_metadata_rows

        # Nothing to write.
        if not chunk_counts:
            return

        t0 = time.perf_counter()

        # Create folder like chunk_00000.
        chunk_dir = out_dir / f"chunk_{chunk_id:05d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        # Stack chunk lists into arrays.
        # counts_chunk shape: (n_chunk, d*h*w), usually (n_chunk, 625)
        # masks_chunk shape:  (n_chunk, d*h*w), usually (n_chunk, 625)
        counts_chunk = np.stack(chunk_counts)
        masks_chunk = np.stack(chunk_masks).astype(np.bool_)

        # Convert this chunk's row metadata into Luis-style dict.
        # global_start tells the metadata helper where this chunk begins in the
        # final merged dataset, so refl_ids stay unique after merging.
        metadata_chunk = _mfx_metadata_rows_to_luis_dict(
            chunk_metadata_rows,
            args.test_fraction,
            global_start=global_row,
        )

        # This helps long MFX jobs avoid "Too many open files" during np.save.
        import gc
        gc.collect()

        # Save this chunk to disk.
        np.save(chunk_dir / args.counts_fname, counts_chunk)
        np.save(chunk_dir / args.masks_fname, masks_chunk)
        np.save(chunk_dir / "metadata.npy", metadata_chunk)

        # Remember this chunk for final merge.
        chunk_dirs.append(chunk_dir)

        n_chunk = counts_chunk.shape[0]
        print(f"wrote chunk {chunk_id}: {n_chunk} shoeboxes")

        # Advance global row counter.
        global_row += n_chunk

        # Move to next chunk ID.
        chunk_id += 1

        # Clear memory for next chunk.
        chunk_counts = []
        chunk_masks = []
        chunk_metadata_rows = []

        t_flush += time.perf_counter() - t0

    n_experiments_attempted = 0
    n_experiments_seen = 0
    n_reflections_seen = 0
    n_reflections_saved = 0

    # Keep a record of experiments that fail inside the combined file.
    # This is the combined-mode equivalent of skip-bad-file logic in run_mfx().
    skipped_experiments = []

    # Main combined-file loop.
    #
    # Old MFX mode:
    #   for each separate .refl/.expt pair:
    #       load one .refl
    #       load one .expt
    #       crop shoeboxes
    #
    # New combined MFX mode:
    #   load combined .refl once
    #   load combined .expt once
    #   for each experiment id:
    #       select reflections where reflections["id"] == experiment_id
    #       use experiments[experiment_id]
    #       crop shoeboxes
    for experiment_id in experiment_ids:
        n_experiments_attempted += 1

        if experiment_id < 0 or experiment_id >= len(experiments):
            skipped_experiments.append(
                {
                    "experiment_id": experiment_id,
                    "image_index": -1,
                    "n_reflections": 0,
                    "error_type": "BAD_EXPERIMENT_ID",
                    "error_message": (
                        f"reflection table has id={experiment_id}, but combined .expt "
                        f"contains only {len(experiments)} experiment(s)"
                    ),
                }
            )
            print(
                f"WARNING: skipping combined experiment id {experiment_id}: "
                f"outside experiment list range"
            )
            continue

        # Select the rows in the combined reflection table that belong to this
        # experiment/image.
        t0 = time.perf_counter()
        select_mask_np = ids_np == experiment_id

        # New: keep the original row numbers from the full combined .refl.
        # reflection_id below is the row inside this experiment subset, while
        # combined_refl_row is the row in the full combined reflection table.
        # This makes future write-back to the combined .refl safer.
        subset_indices_np = np.nonzero(select_mask_np)[0]

        select_mask = flex.bool(select_mask_np.tolist())
        refl_subset = reflections.select(select_mask)
        t_select_subset += time.perf_counter() - t0

        if len(refl_subset) == 0:
            continue

        experiment = experiments[experiment_id]

        # raw is a tuple of panel images for one event/image:
        #   raw[0], raw[1], ..., raw[31]
        #
        # experiments[experiment_id].imageset is the loader/access object.
        # get_raw_data(0) loads the actual pixels for image index 0 inside this
        # one-image imageset. The original event/image index is preserved in
        # single_file_indices inside the combined .expt file.
        #
        # New skip logic:
        #   If raw-data loading, detector-mask checking, or image-index lookup
        #   fails for this experiment, skip only this experiment and continue.
        try:
            t0 = time.perf_counter()
            raw = experiment.imageset.get_raw_data(0)
            t_get_raw += time.perf_counter() - t0

            # If a detector mask was provided, it should have one mask per raw
            # detector panel. For this MFX Jungfrau case, both should have length 32.
            if detector_masks is not None and len(detector_masks) != len(raw):
                raise RuntimeError(
                    f"detector mask panel count ({len(detector_masks)}) does not "
                    f"match raw panel count ({len(raw)}) for experiment id {experiment_id}"
                )

            # Record original detector image/event number, e.g. 1418.
            t0 = time.perf_counter()
            image_index = _mfx_image_index_from_combined_expt_json(
                expt_path,
                experiment_id,
            )
            t_image_index_json += time.perf_counter() - t0

            # Retrieve wavelength from the beam object in the experiment. If it fails,
            # set wavelength to NaN.
            try:
                wavelength = float(experiment.beam.get_wavelength())
            except Exception:
                wavelength = np.nan

        except Exception as e:
            skipped_experiments.append(
                {
                    "experiment_id": experiment_id,
                    "image_index": -1,
                    "n_reflections": len(refl_subset),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
            )
            print(
                f"WARNING: skipping combined experiment id {experiment_id} "
                f"with {len(refl_subset)} reflection(s): {type(e).__name__}: {e}"
            )

            # Try to release any cached data before moving on.
            try:
                experiment.imageset.clear_cache()
            except Exception:
                pass

            continue

        n_experiments_seen += 1
        n_reflections_seen += len(refl_subset)

        # Loop over reflections for this one experiment/image.
        t0 = time.perf_counter()

        for i in range(len(refl_subset)):
            panel_id = int(refl_subset["panel"][i])

            # xyzcal.px is the DIALS/cctbx calculated pixel position from the
            # refined geometry/model. It is not an ML prediction from integrator.
            # We use it as the center of the fixed shoebox crop.
            xyz = refl_subset["xyzcal.px"][i]
            x_center = float(xyz[0])
            y_center = float(xyz[1])

            if panel_id < 0 or panel_id >= len(raw):
                # Bad panel ID; skip this reflection.
                continue

            panel_image = raw[panel_id]

            # If a detector mask was supplied, select the mask for this same
            # panel. The raw pixel panel and mask panel should have matching
            # shapes, e.g. both (514, 1030).
            panel_mask_np = None
            if detector_masks is not None:
                panel_mask_np = detector_masks[panel_id].as_numpy_array().astype(
                    bool,
                    copy=False,
                )

            # Crop a fixed-size window centered on xyzcal.px.
            # Returns a 2D counts array with shape (args.h, args.w) and a mask.
            # If panel_mask_np is provided, the mask combines:
            #   finite pixel values AND detector-good pixels.
            # If panel_mask_np is None, the mask only checks finite pixels and
            # outside-panel padding.
            shoebox, mask, fixed_bbox = _crop_mfx_fixed_shoebox(
                panel_image=panel_image,
                x_center=x_center,
                y_center=y_center,
                width=args.w,
                height=args.h,
                panel_mask_np=panel_mask_np,
            )

            # Flatten the fixed-size 2D shoebox/mask into one 1D row.
            # Pixel values stay the same; only the shape changes.
            # Example: (25, 25) -> (625,).
            #
            # Streaming update:
            #   append to the current chunk buffer, not to a full-run list.
            chunk_counts.append(shoebox.reshape(-1).astype(counts_dtype, copy=False))
            chunk_masks.append(mask.reshape(-1))
            n_reflections_saved += 1

            # Existing integration bbox is useful metadata, but it may not have
            # the fixed --w/--h shape. Keep it if the column exists.
            integration_bbox = tuple(refl_subset["bbox"][i]) if "bbox" in refl_subset else None

            # MFX metadata is built manually, so compute d-spacing here from
            # the experiment crystal/unit cell and the reflection Miller index.
            miller_index = tuple(int(v) for v in refl_subset["miller_index"][i])
            d_spacing = _mfx_d_spacing(experiment, miller_index)

            # Add one metadata row for this shoebox.
            # background.mean is optional in general, but your checked MFX
            # integrated.refl files contain it, so copy it when present.
            #
            # For combined mode:
            #   source_refl/source_expt are the combined files.
            #   experiment_id tells which combined experiment was used.
            #   image_index preserves the original detector image/event number.
            chunk_metadata_rows.append(
                {
                    "source_refl": refl_path.name,
                    "source_expt": expt_path.name,
                    "experiment_id": experiment_id,
                    "image_index": image_index,
                    "reflection_id": int(i),
                    "combined_refl_row": int(subset_indices_np[i]),
                    "panel": panel_id,
                    "fixed_bbox": tuple(fixed_bbox),
                    "integration_bbox": integration_bbox,
                    "detector_mask": str(args.detector_mask) if args.detector_mask else None,
                    "xyzcal_px": tuple(float(v) for v in refl_subset["xyzcal.px"][i]),
                    "xyzobs_px_value": tuple(float(v) for v in refl_subset["xyzobs.px.value"][i])
                    if "xyzobs.px.value" in refl_subset
                    else None,
                    "miller_index": miller_index,
                    "d": d_spacing,
                    "intensity_sum_value": float(refl_subset["intensity.sum.value"][i]),
                    "intensity_sum_variance": float(refl_subset["intensity.sum.variance"][i]),
                    "background_sum_value": float(refl_subset["background.sum.value"][i]),
                    "background_sum_variance": float(refl_subset["background.sum.variance"][i]),
                    "background.mean": (
                        float(refl_subset["background.mean"][i])
                        if "background.mean" in refl_subset
                        else np.nan
                    ),
                    "wavelength": wavelength,
                }
            )

            # If the current chunk is full, write it to disk and clear memory.
            if len(chunk_counts) >= write_chunk_size:
                flush_chunk()

        t_crop_loop += time.perf_counter() - t0

        # imageset is the loader/cache object. clear_cache() tells it to forget
        # loaded image pixels after we finish this event, so memory does not grow
        # while processing many experiments inside the combined .expt file.
        t0 = time.perf_counter()
        experiment.imageset.clear_cache()
        t_clear_cache += time.perf_counter() - t0

    # Write the final partial chunk, if any.
    flush_chunk()

    if not chunk_dirs:
        raise SystemExit("No MFX shoeboxes were extracted; check input files and bbox/panel data.")

    counts_path = out_dir / args.counts_fname
    masks_path = out_dir / args.masks_fname
    metadata_path = out_dir / "metadata.npy"

    # Merge all chunk folders into final counts.npy, masks.npy, metadata.npy.
    # merge_mfx_chunks writes counts/masks through memmap so the final merge does
    # not need to hold all pixel arrays in memory at once.
    t0 = time.perf_counter()
    n_saved = merge_mfx_chunks(
        chunk_dirs=chunk_dirs,
        out_dir=out_dir,
        counts_fname=args.counts_fname,
        masks_fname=args.masks_fname,
    )
    t_merge += time.perf_counter() - t0

    stats = None

    # concentration.npy is created only when we do NOT use: --no-stats
    t0 = time.perf_counter()
    if not args.no_stats:
        stats = _raw_stats_from_memmap(counts_path, masks_path, chunk=args.stats_chunk)

        _save_concentration_from_memmap(
            counts_path=counts_path,
            out_dir=out_dir,
            chunk=args.stats_chunk,
        )
        print("wrote concentration.npy")
    t_stats += time.perf_counter() - t0

    # Write a simple dataset.yaml so the output folder has the expected dataset
    # description. The metadata format may still need to be aligned with Luis's
    # final data-loader expectations.
    t0 = time.perf_counter()
    write_dataset_yaml(
        out_dir,
        geometry={"d": args.d, "h": args.h, "w": args.w},
        n_reflections=n_saved,
        polychromatic=False,
        anscombe=False,
        files={
            "counts": args.counts_fname,
            "masks": args.masks_fname,
            "reference": "metadata.npy",
        },
        crystal=None,
        stats=stats,
        refl_file=None,
    )
    t_write_yaml += time.perf_counter() - t0

    # Write a small text report of skipped combined experiments.
    # This makes it easy to inspect whether failures were raw-data access,
    # detector-mask mismatch, bad experiment ids, or image-index lookup issues.
    if skipped_experiments:
        skipped_path = out_dir / "skipped_combined_experiments.txt"
        with skipped_path.open("w") as f:
            for row in skipped_experiments:
                f.write(
                    f"experiment_id={row['experiment_id']} "
                    f"image_index={row['image_index']} "
                    f"n_reflections={row['n_reflections']} "
                    f"error_type={row['error_type']} "
                    f"error_message={row['error_message']}\n"
                )
        print(f"wrote {skipped_path}")

    t_total = time.perf_counter() - t_total_start

    accounted = (
        t_path_setup
        + t_load_mask
        + t_load_refl
        + t_load_expt
        + t_get_ids
        + t_select_subset
        + t_get_raw
        + t_image_index_json
        + t_crop_loop
        + t_clear_cache
        + t_flush
        + t_merge
        + t_stats
        + t_write_yaml
    )

    print("")
    print("MFX combined-file timing summary:")
    print(f"  total wall time:        {t_total:.2f} sec ({t_total / 60:.2f} min)")
    print(f"  path/setup checks:      {t_path_setup:.2f} sec")
    print(f"  load detector mask:     {t_load_mask:.2f} sec")
    print(f"  load combined .refl:    {t_load_refl:.2f} sec")
    print(f"  load combined .expt:    {t_load_expt:.2f} sec")
    print(f"  get unique ids:         {t_get_ids:.2f} sec")
    print(f"  select refl subsets:    {t_select_subset:.2f} sec")
    print(f"  get_raw_data:           {t_get_raw:.2f} sec")
    print(f"  read image index json:  {t_image_index_json:.2f} sec")
    print(f"  crop/metadata loop:     {t_crop_loop:.2f} sec")
    print(f"  clear image cache:      {t_clear_cache:.2f} sec")
    print(f"  flush chunks:           {t_flush:.2f} sec")
    print(f"  merge chunks:           {t_merge:.2f} sec")
    print(f"  stats/concentration:    {t_stats:.2f} sec")
    print(f"  write dataset.yaml:     {t_write_yaml:.2f} sec")
    print(f"  unaccounted/other:      {t_total - accounted:.2f} sec")
    print("")
    print("Note:")
    print("  crop/metadata loop includes any flush_chunk() calls that happen inside")
    print("  the reflection loop, so crop/metadata and flush chunks overlap slightly.")
    print("")

    print(f"attempted {n_experiments_attempted} combined experiment id(s)")
    print(f"processed {n_experiments_seen} combined experiment(s)")
    print(f"skipped {len(skipped_experiments)} combined experiment(s)")
    print(f"seen {n_reflections_seen} reflection(s)")
    print(f"saved {n_reflections_saved} fixed-size shoebox(es)")
    print(f"wrote {counts_path}")
    print(f"wrote {masks_path}")
    print(f"wrote {metadata_path}")
    print(f"wrote dataset.yaml under {out_dir}")


def _load_mfx_raw_panels_from_npz(raw_cache_dir: Path, image_index):
    """Load cached MFX raw detector panels for one image/event.

    Expected cache filename:
        raw_panels_image_01418.npz

    Expected arrays inside the .npz:
        panel_00, panel_01, ..., panel_31

    This is the cache/local-copy path:
        old slow path: .expt -> ExperimentListFactory -> get_raw_data(0)
        new fast path: raw_panels_image_XXXXX.npz -> NumPy arrays
    """
    if image_index is None:
        raise ValueError("cannot load raw-panel cache because image_index is None")

    cache_path = raw_cache_dir / f"raw_panels_image_{int(image_index):05d}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"raw-panel cache file not found: {cache_path}")

    with np.load(cache_path) as data:
        panel_keys = sorted(k for k in data.files if k.startswith("panel_"))
        if not panel_keys:
            raise ValueError(f"no panel_XX arrays found in {cache_path}")

        # Copy arrays out before closing the np.load context.
        raw = tuple(data[k].astype(np.float32, copy=False) for k in panel_keys)

    return raw, cache_path


def _mfx_wavelength_from_expt_json(expt_path: Path):
    """Read beam wavelength from an MFX .expt JSON without loading image data."""
    import json

    try:
        with open(expt_path) as f:
            data = json.load(f)

        exp = data["experiment"][0]
        beam_id = int(exp.get("beam", 0))
        return float(data["beam"][beam_id]["wavelength"])
    except Exception:
        return np.nan


def _angle_degrees(u, v):
    """Return angle between two vectors in degrees."""
    import math

    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom == 0:
        return np.nan

    cosang = float(np.dot(u, v) / denom)
    cosang = max(-1.0, min(1.0, cosang))
    return float(math.degrees(math.acos(cosang)))


def _mfx_unit_cell_from_expt_json(expt_path: Path):
    """Read crystal unit cell from an MFX .expt JSON without loading image data.

    The .expt file stores real-space basis vectors. We convert those vectors to
    the six unit-cell parameters needed by cctbx. This avoids the expensive
    ExperimentListFactory.from_json_file(..., check_format=True) call in
    raw-panel-cache mode.
    """
    import json

    try:
        from cctbx import uctbx

        with open(expt_path) as f:
            data = json.load(f)

        exp = data["experiment"][0]
        crystal_id = int(exp.get("crystal", 0))
        crystal = data["crystal"][crystal_id]

        a_vec = np.asarray(crystal["real_space_a"], dtype=float)
        b_vec = np.asarray(crystal["real_space_b"], dtype=float)
        c_vec = np.asarray(crystal["real_space_c"], dtype=float)

        a = float(np.linalg.norm(a_vec))
        b = float(np.linalg.norm(b_vec))
        c = float(np.linalg.norm(c_vec))
        alpha = _angle_degrees(b_vec, c_vec)
        beta = _angle_degrees(a_vec, c_vec)
        gamma = _angle_degrees(a_vec, b_vec)

        return uctbx.unit_cell((a, b, c, alpha, beta, gamma))
    except Exception:
        return None


def _mfx_d_spacing_from_unit_cell(unit_cell, miller_index):
    """Compute d-spacing from a cached-mode unit cell, or NaN if unavailable."""
    if unit_cell is None:
        return np.nan

    try:
        hkl = tuple(int(v) for v in miller_index)
        return float(unit_cell.d(hkl))
    except Exception:
        return np.nan


#Thao: Added a helper function to find the matching .expt file for a given MFX integrated reflection table path.
def _matching_mfx_expt(refl_path: Path): 
    # Given an MFX integrated reflection table path, return the matching .expt path.
    # Example:
    #   idx-data_36001_integrated.refl -> idx-data_36001_integrated.expt
    expt_path = refl_path.with_suffix(".expt")
    if not expt_path.exists():
        return None
        
    return expt_path


# Thao: added helper for combined MFX .expt files.
# In combined files, each experiment has a new internal id: 0, 1, 2, ...
# Each experiment points to an imageset, and that imageset preserves the
# original single_file_indices value, e.g. 1418, 2203, 2660.
def _mfx_image_index_from_combined_expt_json(expt_path: Path, experiment_id: int):
    """Return original single_file_indices for one experiment in a combined .expt.

    Example:
        combined experiment id 0 -> original image/event 1418
        combined experiment id 1 -> original image/event 2203

    This is only metadata for tracking/debugging. Pixel loading still happens
    through ExperimentListFactory and experiments[experiment_id].imageset.
    """
    import json

    with open(expt_path) as f:
        data = json.load(f)

    try:
        exp = data["experiment"][experiment_id]
        imageset_id = int(exp["imageset"])
        return int(data["imageset"][imageset_id]["single_file_indices"][0])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


#Thao: added a helper function to extract the first single_file_indices value from an MFX .expt file. 
# This is only metadata for tracking/debugging. dxtbx can still load the pixels through ExperimentListFactory even if this value is absent.
def _mfx_image_index_from_expt_json(expt_path: Path):
    """Return the first single_file_indices value from an MFX .expt file.

    This is only metadata for tracking/debugging. dxtbx can still load the
    pixels through ExperimentListFactory even if this value is absent.
    """
    import json

    # A DIALS .expt file is stored in a JSON-like text format. We only read it
    # here to record metadata such as the event/image index; pixel loading still
    # happens through ExperimentListFactory above.
    with open(expt_path) as f:
        data = json.load(f)

    # Example .expt structure:
    #   {
    #     "imageset": [
    #       {
    #         "images": ["/.../data.loc"],
    #         "single_file_indices": [36001]
    #       }
    #     ]
    #   }
    try:
        # Example return value: 36001 from {"single_file_indices": [36001]}.
        return int(data["imageset"][0]["single_file_indices"][0])
    except (KeyError, IndexError, TypeError):
        return None


def _mfx_d_spacing(experiment, miller_index):
    """Compute d-spacing for one MFX reflection from the experiment crystal."""
    
    #Example expt file:
    # {
    #   "experiment": [
    #     {
    #       "crystal": 0,
    #       "detector": 0,
    #       "beam": 0,
    #       "imageset": 0
    #     }
    #   ],
    #
    #   "crystal": [
    #     {
    #       "real_space_a": [...],
    #       "real_space_b": [...],
    #       "real_space_c": [...],
    #       "space_group_hall_symbol": "..."
    #     }
    #   ],
        

    unit_cell = experiment.crystal.get_unit_cell()

    #converts miller_index into a clean tuple of three integers: (h,k,l)
    hkl = tuple(int(v) for v in miller_index)
    return float(unit_cell.d(hkl))


def _crop_mfx_fixed_shoebox(
    panel_image,
    x_center,
    y_center,
    width,
    height,
    panel_mask_np=None,
):
    """Crop/pad one fixed-size MFX shoebox from one detector panel.

    Parameters
    ----------
    panel_image
        scitbx/DIALS flex 2D array for one detector panel. Use
        panel_image.all() to get shape because this is not a NumPy array.
    
    x_center, y_center
        Predicted reflection center in pixel coordinates from xyzcal.px.
    
    width, height
        Desired fixed output size from --w and --h.
    
    panel_mask_np
        Optional NumPy boolean detector mask for the same panel as panel_image.
        Shape should match panel_image.as_numpy_array().shape.

        Convention used here:
            True  = good/usable detector pixel
            False = bad/hot/masked detector pixel

        If panel_mask_np is provided, the output shoebox mask is:
            finite detector pixel && good detector-mask pixel

        If panel_mask_np is None, the output shoebox mask is only:
            finite detector pixel inside the panel

    Returns
    -------
    shoebox : np.ndarray
        2D pixel array with shape (height, width). Pixels outside the panel are
        zero-padded.
    mask : np.ndarray
        Boolean array with shape (height, width).

        True means this output pixel is valid for training/integration.
        False means this pixel is outside-panel padding, NaN/inf, or marked bad
        by the detector mask.
    fixed_bbox : tuple[int, int, int, int, int, int]
        The full requested bbox in DIALS order: (x0, x1, y0, y1, z0, z1).
    """

    """
            Full detector panel image
        example shape: 514 x 1030
        ↓

        Predicted reflection center
        from xyzcal.px: x_center, y_center
        ↓

        Request fixed 25 x 25 crop around center
        ↓

        Check if requested crop is fully inside detector panel
        ↓

        If fully inside:
        copy real 25 x 25 detector pixels
        mask = True for finite/good pixels

        If near detector edge:
        requested 25 x 25 box partly falls outside panel
        only copy available real detector pixels
        missing outside-panel pixels stay zero
        mask = False for missing/outside pixels
        ↓

        Output fixed shoebox
        counts[i] shape: 25 x 25 or flattened 625
        mask[i] shape: 25 x 25 or flattened 625
        ↓

        Encoder
        reads the 25 x 25 shoebox
        ↓

        Feature vector
        32 learned numbers
        ↓

        Posterior / surrogate distributions applied here:

        qp branch:
            learned_basis_profile
            predicts spot/profile shape

        qi branch:
            Gamma / LogNormal / FoldedNormal
            predicts reflection intensity

        qbg branch:
            Gamma / LogNormal / FoldedNormal
            predicts background
        ↓

        Combine qp + qi + qbg
        ↓

        Predicted pixels / predicted shoebox
        ↓

        Observation likelihood applied here:
        Normal / Poisson / StudentT
        compares predicted pixels with real input pixels
        ↓

        Loss
    
    """
    # panel_image can be either:
    #   1. a DIALS/scitbx flex array from get_raw_data(0), or
    #   2. a NumPy array loaded from raw_panels_image_XXXXX.npz.
    #
    # Support both so the same crop code works for the normal psana/dxtbx path
    # and the raw-panel-cache path.
    if isinstance(panel_image, np.ndarray):
        panel_np = panel_image
        panel_h, panel_w = panel_np.shape
    else:
        # flex arrays use .all() for shape. For this data: (514, 1030).
        panel_h, panel_w = panel_image.all()

        # Convert once to NumPy so ordinary slicing/assignment works.
        panel_np = panel_image.as_numpy_array()

    # If a detector mask is supplied, make sure it lines up with this panel.
    # This catches mistakes such as using a mask from a different detector size.
    if panel_mask_np is not None and panel_mask_np.shape != panel_np.shape:
        raise ValueError(
            "panel mask shape does not match panel image shape: "
            f"mask={panel_mask_np.shape}, image={panel_np.shape}"
        )

    # Example: for a 25x25 shoebox, nx=12 and ny=12.
    nx = width // 2
    ny = height // 2

    # Center the fixed window on the predicted centroid.
    xc = int(np.floor(x_center))
    yc = int(np.floor(y_center))

    x0 = xc - nx
    x1 = x0 + width
    y0 = yc - ny
    y1 = y0 + height

    fixed_bbox = (x0, x1, y0, y1, 0, 1)

    # Clip source coordinates to the actual panel bounds.
    # This avoids negative or out-of-range NumPy indexing.
    # Example requested y range: -3:22 -> clipped source y range: 0:22.
    xs0 = max(0, x0)
    xs1 = min(panel_w, x1)
    ys0 = max(0, y0)
    ys1 = min(panel_h, y1)

    # Allocate full fixed-size output. Areas outside the detector stay zero and
    # mask False.
    shoebox = np.zeros((height, width), dtype=panel_np.dtype)
    mask = np.zeros((height, width), dtype=bool)

    # If the requested box has no overlap with the panel at all, return all-zero data
    # and an all-False mask.
    if xs0 >= xs1 or ys0 >= ys1:
        return shoebox, mask, fixed_bbox
    
    # Destination offsets inside the fixed-size shoebox.
    #
    # x0/y0 are the requested crop start coordinates.
    # xs0/ys0 are the clipped crop start coordinates inside the real detector panel.
    #
    # Normal case:
    #   x0 = 100, xs0 = 100
    #   xd0 = 100 - 100 = 0
    #   Real pixels start at column 0 in the 25x25 shoebox.
    #
    # Edge case:
    #   x0 = -5, xs0 = 0
    #   xd0 = 0 - (-5) = 5
    #   The first 5 columns are outside-panel padding.
    #   Real pixels start at column 5 in the 25x25 shoebox.

    xd0 = xs0 - x0
    yd0 = ys0 - y0

    # panel_np is the full detector panel image as a NumPy array.
    # Example panel shape: (y, x) = (514, 1030).
    #
    # patch is the cropped region of the panel that fits within the fixed bbox.
    # Example: panel_np[372:397, 765:790] -> patch.shape == (25, 25).
    patch = panel_np[ys0:ys1, xs0:xs1]

    # Place the available real pixels into the fixed-size shoebox. Any area that
    # was outside the panel remains zero padding.
    #
    # Edge example:
    #   shoebox[3:25, 0:25] = patch
    #   rows 0:3 stay zero padding; rows 3:25 contain real detector pixels.


    #paste the available real detector pixels into the correct location inside the fixed 25×25 shoebox.
    shoebox[yd0 : yd0 + patch.shape[0], xd0 : xd0 + patch.shape[1]] = patch

    # Basic validity mask: True only for normal finite numbers. This rejects
    # NaN, +inf, and -inf values.
    finite_patch = np.isfinite(patch)

    if panel_mask_np is None:
        # No real detector mask was provided. The mask only means:
        #   finite pixel inside the detector panel.
        mask_patch = finite_patch
    else:
        # Crop the detector mask using exactly the same source coordinates as
        # the detector pixel patch. Because the slices are identical, patch and
        # detector_mask_patch have the same shape.
        detector_mask_patch = panel_mask_np[ys0:ys1, xs0:xs1]

        # Combine the two masks element-by-element:
        #   True  only if the pixel is finite AND detector mask says good.
        #   False if the pixel is NaN/inf OR detector mask says bad.
        mask_patch = finite_patch & detector_mask_patch

    # Place the mask patch into the same position as the shoebox pixel patch.
    # Outside-panel padding remains False.
    mask[yd0 : yd0 + patch.shape[0], xd0 : xd0 + patch.shape[1]] = mask_patch
    
    # returns:
    # shoebox: np.ndarray, shape (height, width), dtype same as detector pixels
    # mask:    np.ndarray, shape (height, width), dtype bool
    # fixed_bbox: tuple of ints, (x0, x1, y0, y1, 0, 1)
    return shoebox, mask, fixed_bbox

def _stats_from_memmap(
    counts_path: Path,
    masks_path: Path,
    chunk: int = 10_000,
) -> dict:
    
    """Return (mean, var) of masked counts and their Anscombe transform.

    Streams the on-disk memmap in chunks; values are stored in dataset.yaml.
    """

    #r: means read-only. NumPy reads chunks from disk as needed.
    counts = np.load(counts_path, mmap_mode="r")
    masks = np.load(masks_path, mmap_mode="r")
    n, _ = counts.shape

    sum_c = sumsq_c = sum_a = sumsq_a = 0.0
    nel = 0
    for i in range(0, n, chunk):
        c = counts[i : i + chunk].astype(np.float64)
        m = masks[i : i + chunk]
        c = c * m

        #c >= -0.375
        #we have valid MFX counts include values like:-7, -115
        # so for MFX Anscombe becomes invalid and produces nan.
        a = 2.0 * np.sqrt(c + 0.375)
        sum_c += c.sum()
        sumsq_c += (c * c).sum()
        sum_a += a.sum()
        sumsq_a += (a * a).sum()
        nel += c.size
    
    #raw mean
    #raw variance
    #anscombe mean
    #anscombe variance

    mean_c = sum_c / nel
    var_c = sumsq_c / nel - mean_c * mean_c
    mean_a = sum_a / nel
    var_a = sumsq_a / nel - mean_a * mean_a

    return {
        "raw": [float(mean_c), float(var_c)],
        "anscombe": [float(mean_a), float(var_a)],
    }


def _raw_stats_from_memmap(
    counts_path: Path,
    masks_path: Path,
    chunk: int = 10_000,
) -> dict:
    """
    one image -> many reflections/shoeboxes: counts.npy = shoeboxes from many images. 

    Return raw mean/variance of over the whole counts.npy dataset

    Ex: For 1500-file dataset: all shoeboxes from all 1500 files

    MFX data can contain negative valid pixel values, so Anscombe stats are
    skipped because sqrt(c + 0.375) can become invalid.
    """
    counts = np.load(counts_path, mmap_mode="r")
    masks = np.load(masks_path, mmap_mode="r")

    # n = number of shoeboxes. _ = pixels per shoebox, usually 625.
    n, _ = counts.shape

    sum_c = 0.0
    sumsq_c = 0.0
    nel = 0

    for i in range(0, n, chunk):
        c = counts[i : i + chunk].astype(np.float64)
        m = masks[i : i + chunk]
        c = c * m

        sum_c += c.sum()
        sumsq_c += (c * c).sum()
        nel += c.size

    mean_c = sum_c / nel
    var_c = sumsq_c / nel - mean_c * mean_c

    return {
        "raw": [float(mean_c), float(var_c)],
    }


def merge_mfx_chunks(chunk_dirs, out_dir, counts_fname="counts.npy", masks_fname="masks.npy"):
    """Merge MFX chunk folders into final counts.npy, masks.npy, metadata.npy.

    Each chunk folder contains:
        counts.npy
        masks.npy
        metadata.npy

    Final output folder will contain:
        counts.npy
        masks.npy
        metadata.npy

    This function writes final counts/masks through memmap so it does not need
    to hold all shoebox pixels in RAM at once during the merge.
    """
    chunk_dirs = list(chunk_dirs)
    if not chunk_dirs:
        raise RuntimeError("No chunks to merge.")

    # First pass: inspect chunk shapes and dtypes.
    # Example chunk counts shape: (10000, 625).
    total_rows = 0
    n_pixels = None
    counts_dtype = None
    metadata_list = []

    for chunk_dir in chunk_dirs:
        counts_chunk = np.load(chunk_dir / counts_fname, mmap_mode="r")
        masks_chunk = np.load(chunk_dir / masks_fname, mmap_mode="r")

        if n_pixels is None:
            n_pixels = counts_chunk.shape[1]
            counts_dtype = counts_chunk.dtype
        elif counts_chunk.shape[1] != n_pixels:
            raise ValueError(
                f"chunk {chunk_dir} has {counts_chunk.shape[1]} pixels per row, "
                f"expected {n_pixels}"
            )

        if masks_chunk.shape != counts_chunk.shape:
            raise ValueError(
                f"mask/counts shape mismatch in {chunk_dir}: "
                f"counts={counts_chunk.shape}, masks={masks_chunk.shape}"
            )

        total_rows += counts_chunk.shape[0]
        metadata = np.load(chunk_dir / "metadata.npy", allow_pickle=True).item()
        metadata_list.append(metadata)

    # Create final output arrays on disk.
    counts_out = open_memmap(
        out_dir / counts_fname,
        mode="w+",
        dtype=counts_dtype,
        shape=(total_rows, n_pixels),
    )
    masks_out = open_memmap(
        out_dir / masks_fname,
        mode="w+",
        dtype=np.bool_,
        shape=(total_rows, n_pixels),
    )

    # Second pass: copy chunk arrays into the final arrays.
    row0 = 0
    for chunk_dir in chunk_dirs:
        counts_chunk = np.load(chunk_dir / counts_fname, mmap_mode="r")
        masks_chunk = np.load(chunk_dir / masks_fname, mmap_mode="r")

        row1 = row0 + counts_chunk.shape[0]
        counts_out[row0:row1] = counts_chunk
        masks_out[row0:row1] = masks_chunk
        row0 = row1

    counts_out.flush()
    masks_out.flush()
    del counts_out, masks_out

    # Merge metadata key-by-key.
    # Metadata arrays are much smaller than counts/masks, so this part is fine
    # to concatenate in memory for now.
    merged_meta = {}
    keys = metadata_list[0].keys()
    for key in keys:
        merged_meta[key] = np.concatenate([m[key] for m in metadata_list], axis=0)

    np.save(out_dir / "metadata.npy", merged_meta)

    return total_rows


def _mfx_metadata_rows_to_luis_dict(metadata_rows, test_fraction, global_start=0):
    """Convert MFX per-reflection metadata rows into Luis-style metadata.npy.

    Luis's metadata.npy is a zero-dimensional object array containing one dict.
    Each key maps to a NumPy array with one value per reflection.

    Example:
        metadata["panel"].shape == (N,)
        metadata["H"].shape == (N,)
        metadata["K"].shape == (N,)
        metadata["L"].shape == (N,)

    global_start:
        Starting row index for this chunk in the full final dataset.
        Example:
            chunk 0 starts at 0
            chunk 1 starts at 10000
            chunk 2 starts at 20000
    """
    n = len(metadata_rows)

    meta = {}

    # Simple scalar columns.
    # Use row.get(..., np.nan) so optional columns like background.mean do not
    # crash if they are absent in some future MFX reflection table.
    scalar_keys = [
        "reflection_id",
        "combined_refl_row",
        "experiment_id",
        "panel",
        "d",
        "intensity_sum_value",
        "intensity_sum_variance",
        "background_sum_value",
        "background_sum_variance",
        "background.mean",
    ]

    for key in scalar_keys:
        meta[key] = np.array(
            [row.get(key, np.nan) for row in metadata_rows],
            dtype=np.float32,
        )

    # Match Luis-style names for intensity/background columns.
    meta["intensity.sum.value"] = meta.pop("intensity_sum_value")
    meta["intensity.sum.variance"] = meta.pop("intensity_sum_variance")
    meta["background.sum.value"] = meta.pop("background_sum_value")
    meta["background.sum.variance"] = meta.pop("background_sum_variance")

    # Miller indices: Luis uses H, K, L.
    miller = np.array(
        [row["miller_index"] for row in metadata_rows],
        dtype=np.float32,
    )
    meta["H"] = miller[:, 0]
    meta["K"] = miller[:, 1]
    meta["L"] = miller[:, 2]

    # Fixed bbox: use the fixed shoebox bbox, because that aligns with counts/masks.
    bbox = np.array(
        [row["fixed_bbox"] for row in metadata_rows],
        dtype=np.float32,
    )
    for j in range(6):
        meta[f"bbox.{j}"] = bbox[:, j]

    # xyzcal.px
    xyzcal = np.array(
        [row["xyzcal_px"] for row in metadata_rows],
        dtype=np.float32,
    )
    for j in range(3):
        meta[f"xyzcal.px.{j}"] = xyzcal[:, j]

    # xyzobs.px.value, if available.
    xyzobs = np.array(
        [
            row["xyzobs_px_value"]
            if row["xyzobs_px_value"] is not None
            else (np.nan, np.nan, np.nan)
            for row in metadata_rows
        ],
        dtype=np.float32,
    )
    for j in range(3):
        meta[f"xyzobs.px.value.{j}"] = xyzobs[:, j]

    # this is_test follows Luis's convention: randomly flag a fraction of reflections as test.
    # For example, if test_fraction=0.1, then about 10% of reflections will have is_test=True.
    # global_start makes the split deterministic across chunks, matching the same
    # RNG sequence as if all rows were processed together.
    rng = np.random.default_rng(42)
    is_test_all = rng.random(global_start + n)
    meta["is_test"] = (is_test_all[global_start:] < test_fraction).astype(np.float32)

    meta["wavelength"] = np.array(
        [row.get("wavelength", np.nan) for row in metadata_rows],
        dtype=np.float32,
    )

    meta["image_num"] = np.array(
        [
            row["image_index"] if row["image_index"] is not None else -1
            for row in metadata_rows
        ],
        dtype=np.float32,
    )

    # Global row ids across the final merged dataset.
    # This keeps refl_ids unique across chunks.
    meta["refl_ids"] = np.arange(global_start, global_start + n, dtype=np.float32)

    meta["source_refl"] = np.array(
        [row.get("source_refl", "") for row in metadata_rows],
        dtype=object,
    )

    meta["source_expt"] = np.array(
        [row.get("source_expt", "") for row in metadata_rows],
        dtype=object,
    )

    meta["detector_mask"] = np.array(
        [row.get("detector_mask", "") for row in metadata_rows],
        dtype=object,
    )

    meta["raw_cache_file"] = np.array(
        [row.get("raw_cache_file", "") for row in metadata_rows],
        dtype=object,
    )

    return meta



def _save_concentration_from_memmap(
    counts_path: Path,
    out_dir: Path,
    chunk: int = 10_000,
    out_fname: str = "concentration.npy",
):
    """Per-reflection mean over voxels, computed by streaming the memmap."""
    counts = np.load(counts_path, mmap_mode="r")
    n = counts.shape[0]
    conc = np.empty(n, dtype=np.float32)
    for i in range(0, n, chunk):
        c = counts[i : i + chunk].astype(np.float32)
        conc[i : i + chunk] = c.mean(axis=1)

    if torch is None:
        raise RuntimeError(
            "torch is required to save concentration.npy; "
            "rerun with --no-stats or use an environment with torch."
        )

    save_data(torch.from_numpy(conc), out_dir / out_fname)


def _write_spec(
    args, out_dir: Path, counts_path: Path, *, polychromatic, crystal, stats
):
    """Write the consolidated dataset.yaml (geometry, files, stats, crystal)."""
    from integrator.io import write_dataset_yaml

    n = int(np.load(counts_path, mmap_mode="r").shape[0])
    ext = "pt" if args.shoebox_format == "pt" else "npy"
    write_dataset_yaml(
        out_dir,
        geometry={"d": args.d, "h": args.h, "w": args.w},
        n_reflections=n,
        polychromatic=polychromatic,
        anscombe=not args.no_stats,
        files={
            "counts": f"counts.{ext}",
            "masks": f"masks.{ext}",
            "reference": "metadata.npy",
        },
        crystal=crystal,
        stats=stats,
        refl_file=str(out_dir / args.refl_fname),
    )
    print(f"wrote dataset.yaml under {out_dir}")


def _convert_npy_memmap_to_pt(npy_path: Path) -> Path:
    """Convert a .npy file into a .pt tensor.

    Reads the file fully into RAM in one shot (torch.save does not stream).
    Returns the new .pt path.
    """
    arr = np.load(npy_path)
    pt_path = npy_path.with_suffix(".pt")
    torch.save(torch.from_numpy(arr), pt_path)
    del arr
    npy_path.unlink()
    return pt_path


def _apply_overlap_mask(
    masks_path: Path,
    bboxes,
    centroids,
    image_ids,
    dz,
    dy,
    dx,
    nproc,
    chunk,
):
    """Mask neighbor-owned pixels in an already-written masks memmap."""
    from integrator.cli.utils.overlap import compute_overlap_mask

    n = len(bboxes)
    overlap = compute_overlap_mask(
        bboxes,
        centroids,
        dz,
        dy,
        dx,
        nproc=nproc,
        image_ids=image_ids,
    )
    overlap_flat = overlap.reshape(n, dz * dy * dx)

    masks_mm = np.load(masks_path, mmap_mode="r+")
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        masks_mm[sl] &= ~overlap_flat[sl]
    masks_mm.flush()
    del masks_mm

    ov_frac = overlap_flat.mean(-1)
    any_ov = ov_frac > 0
    print("  overlap masking:")
    print(f"    mean overlap per refl:   {ov_frac.mean() * 100:.2f}%")
    print(
        f"    refl with any overlap:   {int(any_ov.sum()):,} / {n:,} "
        f"({any_ov.mean() * 100:.1f}%)"
    )
    print(f"    refl with >30% overlap:  {int((ov_frac > 0.30).sum()):,}")


def _get_bounding_boxes(x, y, z, nx, ny, nz):
    """Return full centered bounding boxes.

    Clipping/padding is handled later during extraction.
    """
    from dials.array_family import flex

    bbox = flex.int6(len(x))
    for j, (_x, _y, _z) in enumerate(zip(x, y, z, strict=True)):
        bbox[j] = (
            _x - nx,
            _x + nx + 1,
            _y - ny,
            _y + ny + 1,
            _z - nz,
            _z + nz + 1,
        )
    return bbox


def _get_blocks(block_ids) -> list:
    """Split a sorted block-id array into contiguous index ranges."""
    blocks = []
    start = 0
    for i in range(1, len(block_ids)):
        if block_ids[i] != block_ids[start]:
            blocks.append(np.arange(start, i))
            start = i
    blocks.append(np.arange(start, len(block_ids)))
    return blocks


def process_block(
    block_indices,
    bboxes_full,  # full boxes, may go out of bounds
    refl_ids,
    expt_path,
    dz,
    dy,
    dx,
):
    """Worker: extract one block of reflections from a contiguous z range."""
    import numpy as np
    from dxtbx.model.experiment_list import ExperimentListFactory

    experiments = ExperimentListFactory.from_json_file(expt_path)
    imageset = experiments[0].imageset

    # detector size (single-panel assumption)
    det = imageset.get_detector()[0]
    dx_det, dy_det = det.get_image_size()

    block_boxes = bboxes_full[block_indices]
    z0_block = int(block_boxes[:, 4].min())
    z1_block = int(block_boxes[:, 5].max())

    scan = imageset.get_scan()
    frame0, frame1 = scan.get_array_range()

    z_load0 = max(frame0, z0_block)
    z_load1 = min(frame1, z1_block)

    images = {}
    detmasks = {}
    for z in range(z_load0, z_load1):
        raw = imageset.get_raw_data(z)[0]
        images[z] = raw.as_numpy_array()
        m = imageset.get_mask(z)[0]
        detmasks[z] = m.as_numpy_array().astype(bool)

    n = len(block_indices)
    if images:
        any_z = next(iter(images))
        dtype = images[any_z].dtype
    else:
        dtype = np.float32

    shoeboxes = np.zeros((n, dz, dy, dx), dtype=dtype)
    mask = np.zeros((n, dz, dy, dx), dtype=bool)

    for i, idx in enumerate(block_indices):
        x0f, x1f, y0f, y1f, z0f, z1f = bboxes_full[idx]

        for zz in range(z0f, z1f):
            if zz not in images:
                continue

            # clip source range to detector bounds
            xs0 = max(0, x0f)
            xs1 = min(dx_det, x1f)
            ys0 = max(0, y0f)
            ys1 = min(dy_det, y1f)
            if xs0 >= xs1 or ys0 >= ys1:
                continue

            # destination offsets (clipped source lands inside the full box)
            xd0 = xs0 - x0f
            yd0 = ys0 - y0f
            zd = zz - z0f

            img = images[zz]
            dm = detmasks[zz]
            patch = img[ys0:ys1, xs0:xs1]
            dm_patch = dm[ys0:ys1, xs0:xs1]
            valid = (patch >= 0) & dm_patch

            shoeboxes[
                i, zd, yd0 : yd0 + patch.shape[0], xd0 : xd0 + patch.shape[1]
            ] = patch
            mask[
                i, zd, yd0 : yd0 + patch.shape[0], xd0 : xd0 + patch.shape[1]
            ] = valid

    imageset.clear_cache()

    return {
        "shoeboxes": shoeboxes.reshape(n, dz * dy * dx),
        "mask": mask.reshape(n, dz * dy * dx),
        "refl_ids": refl_ids[block_indices],
    }


def run_all_blocks(
    blocks, bboxes, refl_ids, expt_path, dz, dy, dx, max_workers
):
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_block,
                block,
                bboxes,
                refl_ids,
                expt_path,
                dz,
                dy,
                dx,
            )
            for block in blocks
        ]
        for f in as_completed(futures):
            results.append(f.result())
    return results


def run_dials(args):
    """Extract fixed windows from a monochromatic rotation sequence."""
    _require(args, ["data_dir", "refl", "expt"])
    for name, val in (("w", args.w), ("h", args.h), ("d", args.d)):
        if val % 2 == 0:
            raise ValueError(f"--{name} must be odd (got {val})")

    nx, ny, nz = args.w // 2, args.h // 2, args.d // 2
    dz, dy, dx = args.d, args.h, args.w
    counts_dtype_str = args.counts_dtype or "uint16"

    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory

    data_dir = Path(args.data_dir)
    refl_path_in = data_dir / args.refl
    expt_path_in = data_dir / args.expt
    out_dir = Path(args.out_dir or "out_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {refl_path_in}")
    reflections = flex.reflection_table.from_file(str(refl_path_in))
    print(f"  {len(reflections)} reflections")

    print(f"loading {expt_path_in}")
    experiments = ExperimentListFactory.from_json_file(
        str(expt_path_in), check_format=False
    )
    if len(experiments) != 1:
        raise SystemExit(
            f"rotation mode expects exactly 1 experiment, got "
            f"{len(experiments)} (use --laue for stills)"
        )
    if "panel" not in reflections:
        raise SystemExit("refl table has no 'panel' column")

    x, y, z = reflections["xyzcal.px"].parts()
    x = flex.floor(x).iround()
    y = flex.floor(y).iround()
    z = flex.floor(z).iround()
    reflections["bbox"] = _get_bounding_boxes(x, y, z, nx, ny, nz)

    n_refl = len(reflections)
    reflections["refl_ids"] = flex.int(np.arange(n_refl))
    rng = np.random.default_rng(42)
    is_test = rng.random(n_refl) < args.test_fraction
    reflections["is_test"] = flex.bool(is_test.tolist())
    print(
        f"  is_test: {is_test.sum()} / {n_refl} ({100 * is_test.mean():.1f}%)"
    )

    # block by z centroid so each worker loads a contiguous frame range once
    reflections["z_px"] = reflections["xyzcal.px"].parts()[2]
    reflections.sort("z_px")

    bbox_sorted = reflections["bbox"]
    bboxes = np.stack([b.as_numpy_array() for b in bbox_sorted.parts()]).T
    refl_ids = reflections["refl_ids"].as_numpy_array()

    z0 = bboxes[:, 4]
    z1 = bboxes[:, 5]
    zc = (z0 + z1) // 2
    block_ids = zc // args.block_size
    reflections["block_ids"] = flex.int(block_ids)

    # restore original (refl_id) order before saving the refl table and
    # before computing overlap, so both align with memmap rows
    perm = flex.sort_permutation(reflections["refl_ids"])
    refl_path_out = out_dir / args.refl_fname
    reflections.reorder(perm)
    reflections.as_file(str(refl_path_out))
    print(f"wrote refl with bbox/refl_ids -> {refl_path_out}")

    blocks = _get_blocks(block_ids)
    max_workers = min(args.max_workers, os.cpu_count() or 1)
    print(f"running {len(blocks)} blocks across {max_workers} workers")

    results = run_all_blocks(
        blocks,
        bboxes,
        refl_ids,
        str(expt_path_in),
        dz,
        dy,
        dx,
        max_workers,
    )

    # aggregate into memmaps
    N = len(refl_ids)
    counts_path = out_dir / args.counts_fname
    masks_path = out_dir / args.masks_fname
    counts_dtype = np.dtype(counts_dtype_str)
    shoeboxes_all = open_memmap(
        counts_path, mode="w+", dtype=counts_dtype, shape=(N, dz * dy * dx)
    )
    mask_all = open_memmap(
        masks_path, mode="w+", dtype=np.bool_, shape=(N, dz * dy * dx)
    )

    dtype_max = (
        np.iinfo(counts_dtype).max
        if np.issubdtype(counts_dtype, np.integer)
        else None
    )
    n_clipped = 0
    for res in results:
        ids = res["refl_ids"]
        sbox = res["shoeboxes"]
        if dtype_max is not None:
            over = sbox > dtype_max
            if over.any():
                n_clipped += int(over.sum())
                sbox = np.clip(sbox, 0, dtype_max)
        shoeboxes_all[ids] = sbox.astype(counts_dtype, copy=False)
        mask_all[ids] = res["mask"]
    shoeboxes_all.flush()
    mask_all.flush()
    del shoeboxes_all, mask_all

    if n_clipped > 0:
        print(
            f"WARNING: {n_clipped} pixel(s) exceeded {counts_dtype} max "
            f"({np.iinfo(counts_dtype).max}); clipped. Consider --counts-dtype "
            "int32 if overloads matter."
        )
    print(f"extracted {N} shoeboxes -> {counts_path}, {masks_path}")

    # overlap masking, grouped per start frame (bbox z0)
    if not args.no_mask_overlap:
        bb = np.stack(
            [b.as_numpy_array() for b in reflections["bbox"].parts()]
        ).T  # (N, 6), original order == memmap row order
        xyz = reflections["xyzcal.px"]
        centroids = np.stack(
            [p.as_numpy_array() for p in xyz.parts()], axis=-1
        )  # (N, 3)
        _apply_overlap_mask(
            masks_path=masks_path,
            bboxes=bb,
            centroids=centroids,
            image_ids=None,
            dz=dz,
            dy=dy,
            dx=dx,
            nproc=max_workers,
            chunk=args.stats_chunk,
        )

    # metadata.npy (ensure is_test is captured alongside the defaults)
    from integrator.io import DEFAULT_REFL_COLS

    cols = list(DEFAULT_REFL_COLS)
    if "is_test" not in cols:
        cols.append("is_test")
    refl_as_pt(
        refl=str(refl_path_out),
        column_names=cols,
        out_dir=out_dir,
        out_fname="metadata.npy",
    )
    print(f"wrote metadata.npy under {out_dir}")

    stats = None
    if not args.no_stats:
        stats = _stats_from_memmap(
            counts_path, masks_path, chunk=args.stats_chunk
        )
        _save_concentration_from_memmap(
            counts_path=counts_path,
            out_dir=out_dir,
            chunk=args.stats_chunk,
        )
        print("wrote concentration.npy")
    _write_spec(
        args,
        out_dir,
        counts_path,
        polychromatic=False,
        crystal=None,
        stats=stats,
    )

    if args.shoebox_format == "pt":
        nbytes = counts_path.stat().st_size + masks_path.stat().st_size
        print(
            f"converting counts/masks .npy -> .pt "
            f"(loads {nbytes / 1e9:.1f} GB into RAM)"
        )
        new_counts = _convert_npy_memmap_to_pt(counts_path)
        new_masks = _convert_npy_memmap_to_pt(masks_path)
        print(f"  -> {new_counts}, {new_masks}")


def _beam_center_px(expt_path: Path) -> tuple[float, float]:
    """Extract beam center in pixels from .expt JSON without loading images."""
    import json

    with open(expt_path) as f:
        data = json.load(f)

    s0 = np.array(data["beam"][0]["direction"])
    panel = data["detector"][0]["panels"][0]
    origin = np.array(panel["origin"])
    fast = np.array(panel["fast_axis"])
    slow = np.array(panel["slow_axis"])
    pix = panel["pixel_size"]

    normal = np.cross(fast, slow)
    t = -origin.dot(normal) / s0.dot(normal)
    bc_mm = t * s0 - origin
    cx = bc_mm.dot(fast) / pix[0]
    cy = bc_mm.dot(slow) / pix[1]
    return (cx, cy)


def _path_to_image_num(path: str) -> int:
    """Parse the trailing integer in a laue-dials image filename.

    e.g. HEWL_NaI_3_2_0001.mccd -> 0   (1-indexed on disk -> 0-indexed here)
    """
    m = _TRAILING_INT_RE.search(path)
    if m is None:
        raise ValueError(f"could not parse image number from filename: {path}")
    return int(m.group(1)) - 1


def _shift_bbox_xy(_x, _y, nx, ny, dx_det, dy_det):
    """Shift the (x, y) range so the full window stays on the detector.

    Falls back to clipping only when the requested window is wider than the
    detector itself.
    """
    fw_x = 2 * nx + 1
    fw_y = 2 * ny + 1

    x0_full = _x - nx
    x1_full = _x + nx + 1
    if x0_full < 0:
        x0 = 0
        x1 = min(dx_det, x0 + fw_x)
        if x1 - x0 < fw_x:
            x1 = min(dx_det, x0_full + fw_x)
    elif x1_full >= dx_det:
        x1 = dx_det
        x0 = max(0, x1 - fw_x)
        if x1 - x0 < fw_x:
            x0 = max(0, x1_full - fw_x)
    else:
        x0 = x0_full
        x1 = x1_full

    y0_full = _y - ny
    y1_full = _y + ny + 1
    if y0_full < 0:
        y0 = 0
        y1 = min(dy_det, y0 + fw_y)
        if y1 - y0 < fw_y:
            y1 = min(dy_det, y0_full + fw_y)
    elif y1_full >= dy_det:
        y1 = dy_det
        y0 = max(0, y1 - fw_y)
        if y1 - y0 < fw_y:
            y0 = max(0, y1_full - fw_y)
    else:
        y0 = y0_full
        y1 = y1_full

    return x0, x1, y0, y1


def _process_image_chunk(
    image_records,
    expt_path,
    dz,
    dy,
    dx,
    counts_path,
    masks_path,
    counts_dtype_str,
):
    """Worker for laue extraction.

    Opens the pre-allocated `counts.npy` / `masks.npy` memmaps in r+ mode and
    writes its refls directly.

    image_records: list of dicts, one per image to process. Each dict has:
        - "expt_idx": int, position into the .expt's experiment list
        - "panels":   (n,) int array
        - "bboxes":   (n, 6) int array, z range = (0, 1)
        - "refl_ids": (n,) int array

    Returns a small summary dict.
    """
    import numpy as np
    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory

    experiments = ExperimentListFactory.from_json_file(
        expt_path, check_format=True
    )
    counts_dtype = np.dtype(counts_dtype_str)
    dtype_max = (
        np.iinfo(counts_dtype).max
        if np.issubdtype(counts_dtype, np.integer)
        else None
    )

    counts_mm = np.load(counts_path, mmap_mode="r+")
    masks_mm = np.load(masks_path, mmap_mode="r+")

    n_done = 0
    n_clipped = 0
    for rec in image_records:
        expt_idx = rec["expt_idx"]
        panels = rec["panels"]
        bboxes = rec["bboxes"]
        refl_ids = rec["refl_ids"]
        n = len(refl_ids)

        subset = flex.reflection_table()
        subset["panel"] = flex.size_t(panels.astype(np.int64))
        bbox_col = flex.int6(n)
        for j in range(n):
            bbox_col[j] = (
                int(bboxes[j, 0]),
                int(bboxes[j, 1]),
                int(bboxes[j, 2]),
                int(bboxes[j, 3]),
                int(bboxes[j, 4]),
                int(bboxes[j, 5]),
            )
        subset["bbox"] = bbox_col
        subset["shoebox"] = flex.shoebox(
            subset["panel"], subset["bbox"], allocate=True
        )

        imageset = experiments[expt_idx].imageset
        subset.extract_shoeboxes(imageset)

        counts = np.zeros((n, dz, dy, dx), dtype=np.int32)
        masks = np.zeros((n, dz, dy, dx), dtype=bool)
        for i, sb in enumerate(subset["shoebox"]):
            counts[i] = sb.data.as_numpy_array()
            masks[i] = (sb.mask.as_numpy_array() & 1).astype(bool)

        counts_flat = counts.reshape(n, -1)
        masks_flat = masks.reshape(n, -1)

        if dtype_max is not None:
            over = counts_flat > dtype_max
            if over.any():
                n_clipped += int(over.sum())
                counts_flat = np.clip(counts_flat, 0, dtype_max)

        # Direct write to memmap
        counts_mm[refl_ids] = counts_flat.astype(counts_dtype, copy=False)
        masks_mm[refl_ids] = masks_flat
        n_done += n

    counts_mm.flush()
    masks_mm.flush()
    return {"n_done": n_done, "n_clipped": n_clipped}


def run_laue(args):
    """Extract fixed-size windows from laue-dials single-frame stills."""
    _require(args, ["data_dir", "refl", "expt", "max_images"])

    if args.w % 2 == 0 or args.h % 2 == 0:
        raise ValueError(
            f"--w and --h must be odd (got w={args.w}, h={args.h})"
        )
    if args.d != 1:
        raise ValueError(f"--d must be 1 for laue extraction (got {args.d})")

    counts_dtype_str = args.counts_dtype or "int32"

    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory

    nx = args.w // 2
    ny = args.h // 2
    dz, dy, dx = args.d, args.h, args.w

    data_dir = Path(args.data_dir)
    refl_path_in = data_dir / args.refl
    expt_path_in = data_dir / args.expt
    out_dir = Path(args.out_dir or "out_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {refl_path_in}")
    reflections = flex.reflection_table.from_file(str(refl_path_in))
    print(f"  {len(reflections)} reflections")

    print(f"loading {expt_path_in}")
    experiments = ExperimentListFactory.from_json_file(
        str(expt_path_in),
        check_format=False,
    )
    print(f"  {len(experiments)} experiments")

    #  expt_idx -> image_num map (laue-dials filename convention)
    expt_to_img = np.empty(len(experiments), dtype=np.int64)
    for i in range(len(experiments)):
        expt_to_img[i] = _path_to_image_num(
            experiments[i].imageset.get_path(0)
        )
    if len(np.unique(expt_to_img)) != len(expt_to_img):
        raise ValueError(
            "duplicate image_num across experiments; check filenames"
        )
    print(
        f"  image_num range: {expt_to_img.min()}-{expt_to_img.max()} "
        f"({len(expt_to_img)} images)"
    )

    # detector size (single-panel, taken from experiment 0)
    det0 = experiments[0].detector[0]
    dx_det, dy_det = det0.get_image_size()

    #  per-refl image_num via id (stills convention)
    expt_idx_per_refl = np.array(reflections["id"]).astype(np.int64)
    if expt_idx_per_refl.min() < 0 or expt_idx_per_refl.max() >= len(
        experiments
    ):
        raise ValueError(
            f"refl['id'] out of bounds: range=[{expt_idx_per_refl.min()}, "
            f"{expt_idx_per_refl.max()}], len(experiments)={len(experiments)}"
        )
    image_num_per_refl = expt_to_img[expt_idx_per_refl]

    # filter: --max-images on derived image_num
    keep_np = image_num_per_refl < args.max_images
    keep_mask = flex.bool(keep_np.tolist())
    n_kept = int(keep_np.sum())
    print(
        f"keeping {n_kept} / {len(reflections)} refls "
        f"(image_num < {args.max_images})"
    )
    reflections = reflections.select(keep_mask)
    expt_idx_per_refl = expt_idx_per_refl[keep_np]
    image_num_per_refl = image_num_per_refl[keep_np]

    # wavelength: read from refl table (laue-dials writes it)
    if "wavelength" not in reflections:
        raise ValueError(
            "refl table has no 'wavelength' column; this mode expects "
            "laue-dials .refl tables"
        )

    # bbox column with shift logic; z fixed to (0, 1) per single-frame
    x_int = flex.floor(reflections["xyzcal.px"].parts()[0]).iround()
    y_int = flex.floor(reflections["xyzcal.px"].parts()[1]).iround()

    bbox = flex.int6(len(reflections))
    for j in range(len(reflections)):
        x0, x1, y0, y1 = _shift_bbox_xy(
            x_int[j], y_int[j], nx, ny, dx_det, dy_det
        )
        bbox[j] = (x0, x1, y0, y1, 0, 1)
    reflections["bbox"] = bbox

    # refl_ids + image_num + is_test
    n_refl = len(reflections)
    reflections["refl_ids"] = flex.int(np.arange(n_refl, dtype=np.int32))
    reflections["image_num"] = flex.int(image_num_per_refl.astype(np.int32))
    rng = np.random.default_rng(42)
    is_test = rng.random(n_refl) < args.test_fraction
    reflections["is_test"] = flex.bool(is_test.tolist())
    print(
        f"  is_test: {is_test.sum()} / {n_refl} ({100 * is_test.mean():.1f}%)"
    )

    # d-spacing per refl via per-experiment unit cell
    reflections.compute_d(experiments)
    d_arr = np.array(reflections["d"])
    print(f"  d range: {d_arr.min():.3f} - {d_arr.max():.3f} A")

    # save the reflection table (now also carries d)
    refl_path_out = out_dir / args.refl_fname
    reflections.as_file(str(refl_path_out))
    print(f"wrote refl with bbox/wavelength/refl_ids/d -> {refl_path_out}")

    # crystal metadata: cell + spacegroup + beam center
    # All per-image experiments are copies of the same refined crystal model,
    # so the first one contains the necessary metadata
    crystal0 = experiments[0].crystal
    cell_params = crystal0.get_unit_cell().parameters()
    sg_info = crystal0.get_space_group().info()

    beam_center_px = _beam_center_px(expt_path_in)

    # spectrum support
    wl_arr = np.array(reflections["wavelength"])
    pad = 0.01

    crystal_meta = {
        "cell": [float(x) for x in cell_params],
        "space_group": sg_info.symbol_and_number(),
        "space_group_number": int(sg_info.type().number()),
        "beam_center_px": [
            float(beam_center_px[0]),
            float(beam_center_px[1]),
        ],
        "lambda_min": float(wl_arr.min()) - pad,
        "lambda_max": float(wl_arr.max()) + pad,
    }
    print(
        f"crystal: cell={tuple(round(c, 3) for c in crystal_meta['cell'])}, "
        f"sg={crystal_meta['space_group']}, "
        f"beam_center_px=({beam_center_px[0]:.1f}, {beam_center_px[1]:.1f})"
    )

    # group refls by expt_idx for parallel extraction
    panels_np = np.array(reflections["panel"])
    bboxes_np = np.stack(
        [b.as_numpy_array() for b in reflections["bbox"].parts()],
        axis=-1,
    )  # (N, 6)
    refl_ids_np = np.array(reflections["refl_ids"])

    image_records: list[dict] = []
    for ei in np.unique(expt_idx_per_refl):
        sel = expt_idx_per_refl == ei
        image_records.append(
            {
                "expt_idx": int(ei),
                "panels": panels_np[sel],
                "bboxes": bboxes_np[sel],
                "refl_ids": refl_ids_np[sel],
            }
        )
    print(f"prepared {len(image_records)} image records for extraction")

    #  chunk and run in parallel
    block_size = args.block_size
    chunks = [
        image_records[i : i + block_size]
        for i in range(0, len(image_records), block_size)
    ]

    max_workers = min(args.max_workers, os.cpu_count() or 1)
    print(f"running {len(chunks)} chunks across {max_workers} workers")

    # Pre-allocate output memmaps and close them
    N = len(refl_ids_np)
    counts_path = out_dir / args.counts_fname
    masks_path = out_dir / args.masks_fname
    counts_dtype = np.dtype(counts_dtype_str)

    counts_mm = open_memmap(
        counts_path,
        mode="w+",
        dtype=counts_dtype,
        shape=(N, dz * dy * dx),
    )
    masks_mm = open_memmap(
        masks_path,
        mode="w+",
        dtype=np.bool_,
        shape=(N, dz * dy * dx),
    )
    counts_mm.flush()
    masks_mm.flush()
    del counts_mm, masks_mm  # close so workers can reopen r+

    n_done = 0
    n_clipped_total = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_image_chunk,
                chunk,
                str(expt_path_in),
                dz,
                dy,
                dx,
                str(counts_path),
                str(masks_path),
                counts_dtype_str,
            )
            for chunk in chunks
        ]
        for f in as_completed(futures):
            res = f.result()
            n_done += res["n_done"]
            n_clipped_total += res["n_clipped"]

    if n_clipped_total > 0:
        print(
            f"WARNING: {n_clipped_total} pixel(s) exceeded {counts_dtype} max "
            f"({np.iinfo(counts_dtype).max}); clipped. Consider --counts-dtype "
            "int32 if overloads matter."
        )
    print(f"extracted {n_done} shoeboxes -> {counts_path}, {masks_path}")

    # overlap masking, grouped per image
    if not args.no_mask_overlap:
        xyz = reflections["xyzcal.px"]
        centroids = np.stack(
            [p.as_numpy_array() for p in xyz.parts()],
            axis=-1,
        )  # (N, 3)
        _apply_overlap_mask(
            masks_path=masks_path,
            bboxes=bboxes_np,
            centroids=centroids,
            image_ids=image_num_per_refl,
            dz=dz,
            dy=dy,
            dx=dx,
            nproc=max_workers,
            chunk=args.stats_chunk,
        )

    #  metadata.npy via refl_as_pt
    from integrator.io import DEFAULT_REFL_COLS

    cols = list(DEFAULT_REFL_COLS)
    for must_have in ("wavelength", "d", "image_num", "is_test"):
        if must_have not in cols:
            cols.append(must_have)
    refl_as_pt(
        refl=str(refl_path_out),
        column_names=cols,
        out_dir=out_dir,
        out_fname="metadata.npy",
    )
    print(f"wrote metadata.npy under {out_dir}")

    stats = None
    if not args.no_stats:
        stats = _stats_from_memmap(
            counts_path, masks_path, chunk=args.stats_chunk
        )
        _save_concentration_from_memmap(
            counts_path=counts_path,
            out_dir=out_dir,
            chunk=args.stats_chunk,
        )
        print("wrote concentration.npy")
    _write_spec(
        args,
        out_dir,
        counts_path,
        polychromatic=True,
        crystal=crystal_meta,
        stats=stats,
    )

    if args.shoebox_format == "pt":
        nbytes = counts_path.stat().st_size + masks_path.stat().st_size
        print(
            f"converting counts/masks .npy -> .pt "
            f"(loads {nbytes / 1e9:.1f} GB into RAM)"
        )
        new_counts = _convert_npy_memmap_to_pt(counts_path)
        new_masks = _convert_npy_memmap_to_pt(masks_path)
        print(f"  -> {new_counts}, {new_masks}")


if __name__ == "__main__":
    main()
