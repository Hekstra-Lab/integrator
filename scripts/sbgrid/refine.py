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
    p.add_argument("--rfree-fraction", type=float, default=0.05)
    p.add_argument(
        "--rfree-from",
        type=Path,
        default=None,
        help="MTZ whose R-free flags to copy, so two arms hold out the same "
        "reflections; pass the arm that was refined first",
    )
    p.add_argument(
        "--rfree-seed",
        type=int,
        default=2026,
        help="fixed so every arm holds out the same reflections",
    )
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


def ensure_free_flags(
    mtz: Path,
    out_dir: Path,
    fraction: float,
    seed: int,
    donor: Path | None = None,
) -> Path:
    """Add an R-free set to the merged MTZ, or reuse one already there.

    One free set is generated per dataset and then copied to every arm, not
    generated per arm: each arm holding out different reflections would make
    R-free incomparable between them, which is the comparison the whole
    exercise exists for. The seed is fixed for the same reason, but a fixed
    seed is not enough on its own -- `add_rfree` draws over whichever
    reflections the arm actually measured, and two arms rarely measure the
    same set -- so the second arm copies rather than redraws.

    `donor` names that first arm's MTZ explicitly. Relying on a shared file
    turning up at a fixed relative path is how the arms silently stop sharing:
    the reference's flags ended up inside its own merged MTZ, where no
    sibling-directory lookup would ever have found them.
    """
    import reciprocalspaceship as rs

    ds = rs.read_mtz(str(mtz))
    existing = [c for c in ds.columns if "free" in c.lower()]
    shared = out_dir.parent / "rfree.mtz"

    # a named donor overrides flags already in the file. dials.merge writes a
    # FreeR_flag of its own, drawn independently per run: two arms merged
    # separately came out agreeing at chance, so trusting the column that is
    # already there is what makes R-free incomparable.
    if donor is not None:
        if not donor.exists():
            raise SystemExit(f"--rfree-from {donor} does not exist")
        reference = rs.read_mtz(str(donor))
        if not [c for c in reference.columns if "free" in c.lower()]:
            raise SystemExit(f"--rfree-from {donor} carries no free flags")
        ds = ds.drop(columns=existing) if existing else ds
        ds = rs.utils.copy_rfree(ds, reference)
        if existing:
            print(f"  replaced {existing[0]} with the free flags from {donor}")
        else:
            print(f"  copied free flags from {donor}")
    elif existing:
        print(f"  free flags already present: {existing[0]}")
        return mtz
    elif shared.exists():
        # a sibling arm already made one; copy it so both hold out the same
        # reflections
        reference = rs.read_mtz(str(shared))
        ds = rs.utils.copy_rfree(ds, reference)
        print(f"  copied free flags from {shared.name}")
    else:
        ds = rs.utils.add_rfree(ds, fraction=fraction, seed=seed)
        ds.write_mtz(str(shared))
        print(f"  generated {fraction:.0%} free flags (seed {seed}) -> {shared.name}")

    flagged = out_dir / f"{mtz.stem}_rfree.mtz"
    ds.write_mtz(str(flagged))
    return flagged


def xray_labels(mtz: Path, anomalous: bool) -> str | None:
    """Which intensity array phenix should refine against.

    dials.merge with anomalous=True writes both the merged mean (IMEAN) and
    the Bijvoet pairs, and phenix refuses to guess between them. For an
    anomalous dataset the pairs are the ones to use: refining against IMEAN
    would discard the Bijvoet differences and produce no anomalous map for
    the peak search to read.
    """
    import reciprocalspaceship as rs

    columns = list(rs.read_mtz(str(mtz)).columns)
    if anomalous and {"I(+)", "I(-)"} <= set(columns):
        return "I(+),SIGI(+),I(-),SIGI(-)"
    if "IMEAN" in columns:
        return "IMEAN,SIGIMEAN"
    return None


def ligand_restraints(
    model: Path, out_dir: Path, dry: bool
) -> tuple[list[Path], Path]:
    """Prepare a deposited model for refinement.

    phenix.refine refuses to start when it cannot type an atom, and a
    deposited model usually gives it two reasons to: ligands with no entry in
    the monomer library (7LVC's NADP, 73 atoms) and stray hydrogens with no
    restraint definition (one on its folate).

    ready_set fixes both -- it writes restraint CIFs for the ligands and an
    updated model with the problem atoms resolved. The updated model is the
    one to refine; the original is what ready_set consumes. Returns both, so
    the caller refines what ready_set produced rather than what it was given.
    """
    if dry:
        return [], model
    updated = out_dir / f"{model.stem}.updated.pdb"
    existing = sorted(out_dir.glob("*.ligands.cif"))
    if existing and updated.exists():
        print(f"  restraints already generated: {[p.name for p in existing]}")
        return existing, updated

    # ready_set takes no --overwrite and writes beside its input, so it runs
    # in the output directory against a copy
    local = out_dir / model.name
    if not local.exists():
        local.write_bytes(model.read_bytes())
    run(
        f"phenix.ready_set {local.name}",
        out_dir,
        PHENIX_ENV,
        False,
        out_dir / "ready_set.log",
    )
    cifs = sorted(out_dir.glob("*.ligands.cif")) or sorted(
        c for c in out_dir.glob("*.cif") if "ready_set" not in c.name
    )
    if cifs:
        print(f"  restraints: {[p.name for p in cifs]}")
    if updated.exists():
        print(f"  refining {updated.name}, not the deposited model")
        return cifs, updated
    return cifs, model


def anomalous_map_columns(mtz: Path) -> tuple[str, str] | None:
    """Find the anomalous difference map's amplitude and phase columns.

    By MTZ column *type*, not by name. Names vary with how the map was
    requested -- phenix writes ANOM/PHANOM by default and
    `anomalous`/`PHanomalous` when the map type is named explicitly -- but
    the types do not: a map coefficient pair is an amplitude (type F) and a
    phase (type P) sharing an MTZ dataset. The model's own F-model columns
    are type G, an anomalous amplitude pair, so they fall out on their own.

    Among candidate pairs the one whose name mentions the anomalous map wins;
    otherwise the first is returned, since a refinement asked for one map.
    """
    import gemmi

    columns = gemmi.read_mtz_file(str(mtz)).columns
    amplitudes = [c for c in columns if c.type == "F"]
    phases = [c for c in columns if c.type == "P"]

    pairs = []
    for amplitude in amplitudes:
        for phase in phases:
            if phase.dataset_id != amplitude.dataset_id:
                continue
            # a phase column is conventionally the amplitude's name with a
            # PH prefix, which disambiguates when a dataset holds several
            if amplitude.label in phase.label or len(amplitudes) == 1:
                pairs.append((amplitude.label, phase.label))

    if not pairs:
        return None
    for amplitude, phase in pairs:
        if "anom" in amplitude.lower():
            return amplitude, phase
    return pairs[0]


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

    mtz = args.mtz
    if not args.dry_run:
        mtz = ensure_free_flags(
            mtz, out_dir, args.rfree_fraction, args.rfree_seed, args.rfree_from
        )
    labels = None if args.dry_run else xray_labels(mtz, anomalous)
    if labels:
        print(f"  refining against {labels}")
    cifs, model = ligand_restraints(Path(model), out_dir, args.dry_run)
    restraints = "".join(
        f"refinement.input.monomers.file_name={c.resolve()} " for c in cifs
    )
    refine = (
        f"phenix.refine {mtz.resolve()} {Path(model).resolve()} "
        f"refinement.main.number_of_macro_cycles={args.macro_cycles} "
        + (f'refinement.input.xray_data.labels="{labels}" ' if labels else "")
        + restraints
        + "refinement.output.prefix=refined --overwrite"
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
    columns = anomalous_map_columns(refined_mtz)
    if columns is None:
        print(
            f"\nno anomalous map in {refined_mtz.name}: skipping the peak "
            "search. Refinement produced no ANOM/PHANOM pair, which usually "
            "means it ran against merged means rather than Bijvoet pairs."
        )
        return 0
    amplitude, phase = columns
    print(f"  anomalous map: {amplitude} / {phase}")
    peaks = (
        f"rs.find_peaks {refined_mtz} {refined_pdb} "
        f"-f {amplitude} -p {phase} -z 5.0 -o {out_dir / 'peaks.csv'}"
    )
    run(peaks, out_dir, None, args.dry_run, out_dir / "find_peaks.log")
    print(f"\nrefinement and peaks in {out_dir}")
    print(f"  anomalous scatterers in the model: {sites}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
