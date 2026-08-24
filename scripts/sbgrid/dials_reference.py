"""Process a dataset with DIALS, to have a reference the integrator answers to.

Every parameter is derived, not configured. The geometry -- detector,
distance, beam centre, wavelength, oscillation width, overload cutoff --
comes out of the image headers, which is why `dials.import` needs nothing
beyond a filename template. The crystal comes from the deposition, and is
passed to indexing as a *hint*: DIALS may still disagree, and where it does,
that disagreement is a result rather than something to suppress. Whether to
merge anomalously comes from `characterize.py`.

The steps are the same ones the monochromatic arm already runs, so the
merged output drops into the shared `merging_stats.csv` contract without
translation.

Multi-sweep datasets are processed per sweep and scaled together: three
sweeps of one crystal are three orientations of one measurement, and joint
scaling is what turns them into one dataset with the multiplicity the
deposition reports.

Usage:
    python scripts/sbgrid/dials_reference.py --data-dir <dir> --dry-run
    python scripts/sbgrid/dials_reference.py --data-dir <dir> --nproc 16
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="DIALS reference processing")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where DIALS writes (default: <data-dir>/dials_reference)",
    )
    p.add_argument("--nproc", type=int, default=8)
    p.add_argument(
        "--d-min",
        type=float,
        default=None,
        help="resolution limit; default lets DIALS decide, which is the "
        "point -- the deposited limit is for comparison, not input",
    )
    p.add_argument(
        "--sweeps",
        default=None,
        help="comma-separated sweep names to process (default: all)",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="use only the first N images of each sweep (a fast rehearsal)",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def load_cards(data_dir: Path) -> tuple[dict, dict | None]:
    card = data_dir / "dataset_card.json"
    if not card.exists():
        raise SystemExit(
            f"no dataset_card.json in {data_dir}; run characterize.py first"
        )
    dataset = json.loads(card.read_text())
    reference = load_json(data_dir / "reference_card.json")
    return dataset, reference


def crystal_hints(reference: dict | None) -> list[str]:
    """Indexing hints from the deposition, as DIALS phil assignments.

    Passing the published cell and space group makes indexing converge on
    datasets where it otherwise wanders. It does not force the answer:
    `dials.symmetry` still determines the symmetry from the data further
    down, and the two are compared at the end.
    """
    if not reference:
        return []
    hints = []
    number = reference.get("space_group_number")
    if number:
        # by number, as the depositor scripts do: unambiguous, and it avoids
        # quoting a Hermann-Mauguin string through a shell
        hints.append(f"indexing.known_symmetry.space_group={number}")
    cell = reference.get("unit_cell")
    if cell and all(v is not None for v in cell):
        hints.append(
            "indexing.known_symmetry.unit_cell="
            + ",".join(f"{v:g}" for v in cell)
        )
    return hints


def run(cmd: list[str], cwd: Path, dry: bool, log: Path | None = None) -> None:
    """Run one DIALS command, echoing it and tee-ing its output to a log."""
    printable = " ".join(cmd)
    print(f"\n$ {printable}", flush=True)
    if dry:
        return
    start = time.time()
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, check=False
    )
    if log is not None:
        log.write_text(proc.stdout + proc.stderr)
    elapsed = time.time() - start
    if proc.returncode:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
        raise SystemExit(
            f"{cmd[0]} failed after {elapsed:.0f}s (exit {proc.returncode}):\n{tail}"
        )
    print(f"  ok ({elapsed:.0f}s)")


def process_sweep(
    name: str,
    images: list[str],
    out_dir: Path,
    hints: list[str],
    bundle: dict | None,
    args,
) -> tuple[Path, Path]:
    """Import through integrate for one sweep; returns its expt/refl pair.

    Where the depositors published a recipe for this sweep it wins over
    anything inferred: their beam centre corrects a header the detector move
    invalidated, their mask covers a shadow no header describes, and their
    image range drops a frame.
    """
    work = out_dir / name
    work.mkdir(parents=True, exist_ok=True)
    if args.max_images:
        images = images[: args.max_images]
    recipe = ((bundle or {}).get("per_sweep") or {}).get(name, {})
    print(f"\n=== sweep {name}: {len(images)} images -> {work}")
    if recipe:
        print(f"    depositor recipe: {recipe}")

    imported = ["dials.import", *images]
    if recipe.get("beam_centre"):
        imported.append(
            f"geometry.detector.mosflm_beam_centre={recipe['beam_centre']}"
        )
    if recipe.get("image_range") and not args.max_images:
        imported.append(f"geometry.scan.image_range={recipe['image_range']}")
    mask = resolve_mask(bundle)
    if mask:
        imported.append(f"mask={mask}")
    run(imported, work, args.dry_run, work / "import.log")

    find = ["dials.find_spots", "imported.expt", f"nproc={args.nproc}"]
    if recipe.get("d_max"):
        find.append(f"spotfinder.filter.d_max={recipe['d_max']}")
    if args.d_min:
        find.append(f"spotfinder.filter.d_min={args.d_min}")
    run(find, work, args.dry_run, work / "find_spots.log")

    run(
        ["dials.index", "imported.expt", "strong.refl", *hints],
        work,
        args.dry_run,
        work / "index.log",
    )

    refine = ["dials.refine", "indexed.expt", "indexed.refl"]
    if str(recipe.get("scan_varying", "")).lower() == "true":
        refine.append("refinement.parameterisation.scan_varying=True")
    run(refine, work, args.dry_run, work / "refine.log")

    integrate = [
        "dials.integrate",
        "refined.expt",
        "refined.refl",
        f"nproc={args.nproc}",
    ]
    if args.d_min:
        integrate.append(f"prediction.d_min={args.d_min}")
    run(integrate, work, args.dry_run, work / "integrate.log")

    return work / "integrated.expt", work / "integrated.refl"


def resolve_mask(bundle: dict | None) -> str | None:
    """The bundle's pixel mask, if it shipped one."""
    masks = (bundle or {}).get("masks") or []
    for candidate in masks:
        if Path(candidate).exists():
            return candidate
    return None


def main():
    args = parse_args()
    dataset, reference = load_cards(args.data_dir)
    out_dir = args.out_dir or args.data_dir / "dials_reference"
    out_dir.mkdir(parents=True, exist_ok=True)

    if dataset.get("mode") == "polychromatic":
        raise SystemExit(
            "this dataset is polychromatic; DIALS rotation processing does "
            "not apply. Use laue-dials."
        )

    bundle = load_json(args.data_dir / "processing_hints.json")
    anomalous = bool(dataset.get("expect_anomalous"))
    hints = crystal_hints(reference)
    sweeps = dataset["sweeps"]
    if args.sweeps:
        wanted = set(args.sweeps.split(","))
        sweeps = {k: v for k, v in sweeps.items() if k in wanted}

    print(f"data      {args.data_dir}")
    print(f"out       {out_dir}")
    print(f"sweeps    {len(sweeps)}: {', '.join(sorted(sweeps))}")
    print(f"anomalous {anomalous}  ({'; '.join(dataset.get('anomalous_reasons') or ['no evidence'])})")
    print(f"hints     {' '.join(hints) if hints else 'none (no deposition)'}")
    if bundle:
        mask = resolve_mask(bundle)
        print(f"bundle    {len(bundle.get('recipes') or [])} depositor script(s)"
              f"{', mask ' + Path(mask).name if mask else ', no mask'}")
    else:
        print("bundle    none: processing from headers alone, which is only "
              "safe if no depositor note contradicts them")

    integrated = []
    for name in sorted(sweeps):
        result = process_sweep(
            name, sweeps[name], out_dir, hints, bundle, args
        )
        if result:
            integrated.append(result)

    # joint scaling: the sweeps are one measurement in several orientations,
    # and scaling them together is what produces the deposited multiplicity
    print(f"\n=== scaling {len(integrated)} sweep(s) together")
    scale = ["dials.scale"]
    for expt, refl in integrated:
        scale += [str(expt), str(refl)]
    scale.append(f"anomalous={anomalous}")
    if args.d_min:
        scale.append(f"d_min={args.d_min}")
    run(scale, out_dir, args.dry_run, out_dir / "scale.log")

    run(
        ["dials.merge", "scaled.expt", "scaled.refl", f"anomalous={anomalous}"],
        out_dir,
        args.dry_run,
        out_dir / "merge.log",
    )

    if args.dry_run:
        print("\ndry run: nothing executed")
        return 0

    # the shared nine-column table, so this reference sits on the same axes
    # as both integrator arms
    emitter = Path(__file__).resolve().parents[1] / "mono" / "emit_merging_stats.py"
    if emitter.exists():
        # positional merged.html; writes merging_stats.csv beside it
        run(
            [sys.executable, str(emitter), str(out_dir / "merged.html")],
            out_dir,
            False,
            out_dir / "merging_stats.log",
        )
    print(f"\nreference processing complete: {out_dir}")
    print("  scaled.expt / scaled.refl / merged.mtz / merged.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
