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

# the schema both integrator arms emit
COLUMNS = [
    "bin",
    "d_max",
    "d_min",
    "n_obs",
    "n_unique",
    "cc_half",
    "cc_anom",
    "r_pim",
    "i_over_sigma",
]


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
    p.add_argument(
        "--merge-with",
        default="aimless",
        choices=["aimless", "dials", "both"],
        help="aimless reproduces the depositors' route; dials keeps "
        "everything in one toolchain; both runs each, which is how you "
        "find out whether the merging program matters",
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
    # deliberately no unit_cell hint: the depositor scripts pass the space
    # group alone, and adding the cell would make indexing a different
    # experiment from theirs
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

    # an unmerged MTZ per pass, which is what the depositors hand to AIMLESS
    mtz = work / f"integrated_{name}.mtz"
    run(
        [
            "dials.export",
            "integrated.expt",
            "integrated.refl",
            f"mtz.hklout={mtz.name}",
        ],
        work,
        args.dry_run,
        work / "export.log",
    )

    return work / "integrated.expt", work / "integrated.refl"


def resolve_mask(bundle: dict | None) -> str | None:
    """The bundle's pixel mask, if it shipped one."""
    masks = (bundle or {}).get("masks") or []
    for candidate in masks:
        if Path(candidate).exists():
            return candidate
    return None


CCP4_SETUP = (
    "/n/hekstra_lab_tier0/Lab/garden/ccp4/ccp4-7.1/bin/ccp4.setup-sh"
)


def run_ccp4(script: str, cwd: Path, dry: bool, log: Path) -> None:
    """Run a CCP4 program, which needs its environment sourced first."""
    command = f"source {CCP4_SETUP} >/dev/null 2>&1 && {script}"
    print(f"\n$ {script.splitlines()[0]} ...", flush=True)
    if dry:
        return
    start = time.time()
    proc = subprocess.run(
        ["bash", "-c", command],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    log.write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
        raise SystemExit(f"CCP4 step failed (exit {proc.returncode}):\n{tail}")
    print(f"  ok ({time.time() - start:.0f}s)")


def merge_with_aimless(
    mtzs: list[Path], out_dir: Path, anomalous: bool, dry: bool
) -> None:
    """Sort with POINTLESS and merge with AIMLESS, as the depositors did.

    The passes are separate DIALS integrations of one crystal, so POINTLESS
    puts them on a common indexing and sort order first; AIMLESS then scales
    and merges them into the single dataset whose multiplicity the deposition
    reports.
    """
    inputs = " ".join(
        f"HKLIN {mtz}" for mtz in mtzs
    )
    run_ccp4(
        f"pointless {inputs} HKLOUT sorted.mtz > pointless.log",
        out_dir,
        dry,
        out_dir / "pointless_run.log",
    )
    anom = "ANOMALOUS ON" if anomalous else "ANOMALOUS OFF"
    run_ccp4(
        "aimless HKLIN sorted.mtz HKLOUT merged_aimless.mtz "
        "XMLOUT aimless.xml << EOF > aimless.log\n"
        f"{anom}\nEOF",
        out_dir,
        dry,
        out_dir / "aimless_run.log",
    )


def aimless_stats(xml_path: Path, out_path: Path) -> int:
    """Turn AIMLESS's XML into the shared nine-column table.

    AIMLESS reports per-shell statistics in its XML rather than only in the
    log, which is why XMLOUT is requested: parsing the log's fixed-width
    tables is fragile in a way the XML is not.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(xml_path).getroot()

    def number(node, tag):
        found = node.find(tag)
        if found is None or found.text is None:
            return None
        text = found.text.strip()
        try:
            return float(text)
        except ValueError:
            return None

    rows = []
    shells = root.findall(".//Result/ResolutionShell") or root.findall(
        ".//ResolutionShell"
    )
    for i, shell in enumerate(shells):
        rows.append(
            {
                "bin": i + 1,
                "d_max": number(shell, "MinRes") or number(shell, "ResolutionLow"),
                "d_min": number(shell, "MaxRes") or number(shell, "ResolutionHigh"),
                "n_obs": number(shell, "NumberObservations"),
                "n_unique": number(shell, "NumberReflections"),
                "cc_half": number(shell, "CChalf"),
                "cc_anom": number(shell, "CCanom"),
                "r_pim": number(shell, "RpimOverall")
                or number(shell, "Rpim"),
                "i_over_sigma": number(shell, "MeanIoverSD"),
            }
        )
    if not rows:
        return 0

    import csv

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


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

    mtzs = [
        expt.parent / f"integrated_{expt.parent.name}.mtz"
        for expt, _ in integrated
    ]

    # AIMLESS is the depositors' route: they integrated each pass in DIALS,
    # exported an MTZ each, and merged the three together in AIMLESS
    if args.merge_with in ("aimless", "both"):
        print(f"\n=== POINTLESS + AIMLESS over {len(mtzs)} pass(es)")
        merge_with_aimless(mtzs, out_dir, anomalous, args.dry_run)

    # the DIALS route keeps everything in one toolchain and feeds the shared
    # merging_stats contract without translation
    if args.merge_with in ("dials", "both"):
        print(f"\n=== dials.scale + dials.merge over {len(integrated)} sweep(s)")
        scale = ["dials.scale"]
        for expt, refl in integrated:
            scale += [str(expt), str(refl)]
        scale.append(f"anomalous={anomalous}")
        if args.d_min:
            scale.append(f"d_min={args.d_min}")
        run(scale, out_dir, args.dry_run, out_dir / "scale.log")
        run(
            [
                "dials.merge",
                "scaled.expt",
                "scaled.refl",
                f"anomalous={anomalous}",
                "output.html=merged.html",
                "output.log=merged.log",
                "output.mtz=merged.mtz",
            ],
            out_dir,
            args.dry_run,
            out_dir / "merge_run.log",
        )

    if args.dry_run:
        print("\ndry run: nothing executed")
        return 0

    # the shared nine-column table, so this reference sits on the same axes
    # as both integrator arms
    xml = out_dir / "aimless.xml"
    if xml.exists():
        n = aimless_stats(xml, out_dir / "merging_stats_aimless.csv")
        print(f"\nmerging_stats_aimless.csv: {n} shells")
    merged_html = out_dir / "merged.html"
    if merged_html.exists():
        emitter = (
            Path(__file__).resolve().parents[1] / "mono" / "emit_merging_stats.py"
        )
        if emitter.exists():
            run(
                [sys.executable, str(emitter), str(merged_html)],
                out_dir,
                False,
                out_dir / "merging_stats.log",
            )

    print(f"\nreference processing complete: {out_dir}")
    print("  scaled.expt / scaled.refl / merged.mtz / merged.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
