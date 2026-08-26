"""Archive a dataset's durable artifacts off netscratch.

netscratch is wiped every few months, so anything worth keeping has to leave.
The destination is inode-limited, which changes what "archive" means: the
large arrays are 3 files and cost nothing, while the small stuff -- scripts,
DIALS output, per-epoch checkpoints -- is over a thousand files per dataset
and is what would exhaust a file-count quota. So the many-small-files tiers
are tarred into one object each, and only the arrays stay loose.

Tiers, by whether they can be recreated:

    essential   scripts, configs, provenance, merging stats, merged MTZs,
                refinements, peaks. Megabytes, and the thing that makes
                everything below reproducible. Always archived.
    checkpoints the final epoch only by default. `save_top_k: -1` keeps every
                epoch, which is 84 files and 5.5 GB per run; the intermediate
                ones are for resuming a run that is already over.
    arrays      counts.npy / masks.npy. Tens of GB, 3 files, and fully
                regenerable from the raw images plus the archived scripts, so
                opt-in rather than default.

Raw images are never archived: SBGrid re-serves them and sbgrid_source.json
records exactly what to ask for.

Usage:
    python scripts/sbgrid/archive.py --dataset <netscratch dir>
    python scripts/sbgrid/archive.py --dataset <dir> --arrays --all-checkpoints
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

DEFAULT_ROOT = Path(
    "/n/holylabs/LABS/hekstra_lab/Users/laldama/integrator_data"
)
# directories worth keeping, relative to the dataset root; missing ones are skipped
ESSENTIAL_DIRS = ("scripts",)
ESSENTIAL_GLOBS = ("*.json", "*.pdb", "*.md")
# within dials_reference: everything except the bulky per-sweep intermediates
REFERENCE_KEEP = (
    "*.csv", "*.log", "*.html", "merged.mtz", "scaled.expt",
    "choice_*/merging_stats.csv", "choice_*/merged.mtz",
    "refine/*.pdb", "refine/*.mtz", "refine/*.csv", "refine/*.log",
    "refine/*.cif", "refine/*.eff",
)


def parse_args():
    p = argparse.ArgumentParser(description="Archive a dataset off netscratch")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--archive-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument(
        "--arrays",
        action="store_true",
        help="also copy counts.npy/masks.npy -- tens of GB, and regenerable",
    )
    p.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="archive every epoch, not just the last",
    )
    p.add_argument(
        "--method",
        choices=("rotation", "laue"),
        default=None,
        help="override the detected method; needed for a raw-only dataset, "
        "which has no dataset_card yet, and useful if detection is ever wrong",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def human(n: int) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}T"


def collect_essential(dataset: Path) -> list[tuple[Path, str]]:
    """(source, name-inside-tar) for everything in the essential tier."""
    items: list[tuple[Path, str]] = []
    for name in ESSENTIAL_DIRS:
        path = dataset / name
        if path.is_dir():
            items.extend(
                (f, str(f.relative_to(dataset)))
                for f in sorted(path.rglob("*"))
                if f.is_file()
            )
    for pattern in ESSENTIAL_GLOBS:
        items.extend(
            (f, str(f.relative_to(dataset)))
            for f in sorted(dataset.glob(pattern))
            if f.is_file()
        )
    reference = dataset / "dials_reference"
    if reference.is_dir():
        for pattern in REFERENCE_KEEP:
            items.extend(
                (f, str(f.relative_to(dataset)))
                for f in sorted(reference.glob(pattern))
                if f.is_file()
            )
    # the integrator arms' merged results and comparison outputs
    for pattern in (
        "integrator/*/run_paths.yaml",
        "integrator/*/config_log.yaml",
        "integrator/dataset/dataset.yaml",
        "integrator/wandb_logs/wandb/*/predictions/*/scaled*/merging_stats.csv",
        "integrator/wandb_logs/wandb/*/predictions/*/scaled*/merged.mtz",
        "integrator/wandb_logs/wandb/*/predictions/*/scaled*/refine*/peaks.csv",
        "integrator/wandb_logs/wandb/*/predictions/*/scaled*/refine*/*.log",
        "integrator/wandb_logs/wandb/*/predictions/*/scaled*/refine*/refined_*.pdb",
    ):
        items.extend(
            (f, str(f.relative_to(dataset)))
            for f in sorted(dataset.glob(pattern))
            if f.is_file()
        )
    seen, unique = set(), []
    for source, name in items:
        if name not in seen:
            seen.add(name)
            unique.append((source, name))
    return unique


def checkpoints(dataset: Path, keep_all: bool) -> list[Path]:
    found: list[Path] = []
    for run in sorted(dataset.glob("integrator/wandb_logs/wandb/*/files/checkpoints")):
        epochs = sorted(run.glob("epoch=*.ckpt"))
        if not epochs:
            continue
        found.extend(epochs if keep_all else [epochs[-1]])
    return found


def write_tar(items, destination: Path, dry: bool) -> int:
    total = sum(source.stat().st_size for source, _ in items)
    if dry:
        return total
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as tar:
        for source, name in items:
            tar.add(source, arcname=name)
    return destination.stat().st_size


def detect_method(dataset: Path, override: str | None) -> str:
    """rotation / laue / blank, but never a guess dressed as a fact.

    The flag lives in dataset_card.json, which `mksbox` writes during
    processing -- so a dataset that has only been downloaded has no card, and
    reading `polychromatic` off an empty dict returns falsy and labels it
    "rotation". That is wrong precisely when the label is least verifiable,
    and a wrong value in a durable index outlives whatever produced it.
    Absent is reported as absent; `--method` states it when it is known.
    """
    if override:
        return override
    for path, key in (
        (dataset / "dataset_card.json", "polychromatic"),
        (dataset / "integrator" / "dataset" / "dataset.yaml", "polychromatic"),
    ):
        if not path.exists():
            continue
        if path.suffix == ".json":
            record = json.loads(path.read_text())
        else:
            import yaml

            record = yaml.safe_load(path.read_text()) or {}
        if key in record:
            return "laue" if record[key] else "rotation"
    return ""


def index_row(dataset: Path, tiers: dict, method: str | None = None) -> dict:
    card, source = {}, {}
    if (dataset / "dataset_card.json").exists():
        card = json.loads((dataset / "dataset_card.json").read_text())
    if (dataset / "sbgrid_source.json").exists():
        source = json.loads((dataset / "sbgrid_source.json").read_text())
    return {
        "id": dataset.name,
        "pdb": card.get("pdb_id", ""),
        "method": detect_method(dataset, method),
        # where the images are while the transition to netscratch is in progress
        "raw_location": str(dataset),
        "n_images": source.get("n_images", ""),
        "raw_bytes": human(source.get("total_bytes", 0)) if source else "",
        "archived": " ".join(sorted(tiers)),
        "archived_on": time.strftime("%Y-%m-%d"),
    }


def update_index(root: Path, row: dict, dry: bool) -> None:
    path = root / "INDEX.md"
    columns = list(row)
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("| ") and not line.startswith("| id ") and "---" not in line:
                cells = [c.strip() for c in line.strip("|").split("|")]
                if cells:
                    rows[cells[0]] = dict(zip(columns, cells, strict=False))
    rows[row["id"]] = row
    lines = [
        "# Archived SBGrid datasets",
        "",
        "Durable record of what was processed, kept here because netscratch is",
        "wiped every few months and takes each dataset's own provenance files",
        "with it. `raw_location` is where the images were last seen; SBGrid",
        "re-serves them from the id, so they are not archived.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    lines += [
        "| " + " | ".join(str(rows[k].get(c, "")) for c in columns) + " |"
        for k in sorted(rows)
    ]
    if dry:
        print(f"\nwould update {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"\nupdated {path}")


def main():
    args = parse_args()
    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        raise SystemExit(f"{dataset} is not a directory")
    target = args.archive_root / dataset.name
    tiers = {}
    arrays: list[Path] = []

    items = collect_essential(dataset)
    print(f"essential: {len(items)} files -> 1 tarball")
    size = write_tar(items, target / f"{dataset.name}_essential.tar.gz", args.dry_run)
    print(f"  {human(size)}")
    tiers["essential"] = size

    ckpts = checkpoints(dataset, args.all_checkpoints)
    if ckpts:
        which = "all epochs" if args.all_checkpoints else "final epoch per run"
        print(f"checkpoints: {len(ckpts)} files ({which}) -> 1 tarball")
        pairs = [(c, f"{c.parents[2].name}/{c.name}") for c in ckpts]
        size = write_tar(pairs, target / f"{dataset.name}_checkpoints.tar.gz",
                         args.dry_run)
        print(f"  {human(size)}")
        tiers["checkpoints"] = size

    if args.arrays:
        arrays = sorted((dataset / "integrator" / "dataset").glob("*.npy"))
        arrays += sorted((dataset / "integrator" / "dataset").glob("*.yaml"))
        total = sum(a.stat().st_size for a in arrays)
        print(f"arrays: {len(arrays)} files, {human(total)} (copied loose -- "
              "few inodes, and tarring tens of GB buys nothing)")
        if not args.dry_run:
            (target / "dataset").mkdir(parents=True, exist_ok=True)
            for a in arrays:
                subprocess.run(
                    ["cp", "-u", str(a), str(target / "dataset" / a.name)],
                    check=True,
                )
        tiers["arrays"] = total

    files_written = len(tiers) + (len(arrays) if args.arrays else 0)
    print(f"\n{dataset.name}: {sum(tiers.values()) and human(sum(tiers.values()))} "
          f"in ~{files_written} files at {target}")
    print(f"  (the same content loose would be "
          f"{len(items) + len(ckpts)} files)")
    update_index(args.archive_root, index_row(dataset, tiers, args.method), args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
