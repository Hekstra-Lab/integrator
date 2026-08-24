"""Refine a merged dataset and measure its anomalous peaks.

The same step for both arms: the DIALS reference and the integrator's own
merge produce a merged MTZ, and refining each against the same model with the
same parameters is what makes their R-factors and peak heights comparable.

Refinement uses the deposited R-free flags when they are available. Generating
a fresh free set per arm would make R-free incomparable between them and with
the deposition, since each arm would be holding out different reflections.

Usage:
    python scripts/sbgrid/refine.py --mtz merged.mtz --data-dir <dataset dir>
    python scripts/sbgrid/refine.py --mtz merged.mtz --model 7LVC.pdb --no-peaks
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PHENIX_ENV = (
    "/n/hekstra_lab_tier0/Lab/garden/phenix_1_20/phenix-1.20.1-4487/phenix_env.sh"
)
MODEL_URL = "https://files.rcsb.org/download/{pdb}.pdb"
# elements whose anomalous peaks are worth measuring, when present in the model
ANOMALOUS_ELEMENTS = ("MN", "FE", "ZN", "CU", "NI", "CO", "SE", "S", "P", "I", "BR")


def parse_args():
    p = argparse.ArgumentParser(description="Refine and find anomalous peaks")
    p.add_argument("--mtz", type=Path, required=True, help="merged MTZ")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="dataset dir, for the cards and to cache the deposited model",
    )
    p.add_argument("--model", type=Path, default=None, help="starting PDB")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: <mtz parent>/refine",
    )
    p.add_argument("--macro-cycles", type=int, default=3)
    p.add_argument(
        "--no-peaks",
        action="store_true",
        help="skip the anomalous peak search even when the data is anomalous",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run(script: str, cwd: Path, env_setup: str | None, dry: bool, log: Path):
    """Run a command under an environment that has to be sourced first."""
    command = (
        f"source {env_setup} >/dev/null 2>&1 && {script}" if env_setup else script
    )
    print(f"\n$ {script[:160]}", flush=True)
    if dry:
        return
    start = time.time()
    proc = subprocess.run(
        ["bash", "-c", command], cwd=cwd, capture_output=True, text=True, check=False
    )
    log.write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
        raise SystemExit(f"failed (exit {proc.returncode}):\n{tail}")
    print(f"  ok ({time.time() - start:.0f}s)")


def fetch_model(pdb_id: str, out_dir: Path) -> Path:
    path = out_dir / f"{pdb_id.upper()}.pdb"
    if path.exists():
        return path
    with urllib.request.urlopen(  # noqa: S310
        MODEL_URL.format(pdb=pdb_id.upper()), timeout=120
    ) as response:
        path.write_bytes(response.read())
    print(f"fetched {path.name}")
    return path


def anomalous_sites(model: Path) -> dict[str, int]:
    """Count the elements in the model that carry anomalous signal."""
    counts: dict[str, int] = {}
    for line in model.read_text(errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        element = line[76:78].strip().upper()
        if element in ANOMALOUS_ELEMENTS:
            counts[element] = counts.get(element, 0) + 1
    return counts


def free_flags_argument(data_dir: Path | None) -> str:
    """Point phenix at the deposited R-free flags, if the sf file is here.

    Using the depositors' free set is what makes R-free comparable across
    arms and with the publication; a freshly generated set per arm holds out
    different reflections in each.
    """
    if data_dir is None:
        return "xray_data.r_free_flags.generate=True"
    sf = next(iter(sorted(data_dir.glob("*-sf.cif"))), None)
    if sf is None:
        return "xray_data.r_free_flags.generate=True"
    return (
        f"refinement.input.xray_data.r_free_flags.file_name={sf} "
        "refinement.input.xray_data.r_free_flags.label=pdbx_r_free_flag"
    )


def main():
    args = parse_args()
    out_dir = args.out_dir or args.mtz.parent / "refine"
    out_dir.mkdir(parents=True, exist_ok=True)

    card = {}
    if args.data_dir and (args.data_dir / "dataset_card.json").exists():
        card = json.loads((args.data_dir / "dataset_card.json").read_text())

    model = args.model
    if model is None:
        pdb_id = card.get("pdb_id")
        if not pdb_id:
            raise SystemExit("no --model and no pdb_id in the dataset card")
        model = fetch_model(pdb_id, args.data_dir or out_dir)

    sites = anomalous_sites(model)
    anomalous = bool(card.get("expect_anomalous")) and not args.no_peaks
    print(f"mtz       {args.mtz}")
    print(f"model     {model}")
    print(f"anomalous {anomalous}"
          + (f"  sites: {sites}" if sites else "  (no anomalous elements in the model)"))

    refine = (
        f"phenix.refine {args.mtz.resolve()} {Path(model).resolve()} "
        f"refinement.main.number_of_macro_cycles={args.macro_cycles} "
        f"{free_flags_argument(args.data_dir)} "
        "refinement.output.prefix=refined --overwrite"
    )
    if anomalous:
        # ask for the anomalous difference map the peak search reads
        refine += (
            " refinement.electron_density_maps.map_coefficients.map_type=anomalous"
        )
    run(refine, out_dir, PHENIX_ENV, args.dry_run, out_dir / "phenix_refine.log")

    if not anomalous:
        print("\nno anomalous peak search requested")
        return 0

    refined_mtz = out_dir / "refined_001.mtz"
    refined_pdb = out_dir / "refined_001.pdb"
    elements = "[" + ",".join(sorted(sites)) + "]" if sites else "[S]"
    peaks = (
        f"rs.find_peaks {refined_mtz} {refined_pdb} "
        f"-f ANOM -p PHANOM -z 5.0 -o {out_dir / 'peaks.csv'}"
    )
    run(peaks, out_dir, None, args.dry_run, out_dir / "find_peaks.log")
    print(f"\nrefinement and peaks in {out_dir}  (elements {elements})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
