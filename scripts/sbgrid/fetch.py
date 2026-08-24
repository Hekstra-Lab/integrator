"""Download an SBGrid dataset by id, and record where it came from.

SBGrid serves every dataset over anonymous rsync at
`rsync://data.sbgrid.org/10.15785/SBGRID/<id>`, mirrored in Sweden, Uruguay
and China. Datasets are tens of GB, so the destination defaults to scratch
rather than anywhere with a quota.

The download is resumable: rsync skips files it already has, so re-running
after an interruption costs a listing rather than a re-transfer.

Usage:
    python scripts/sbgrid/fetch.py --id 821 --list
    python scripts/sbgrid/fetch.py --id 821
    python scripts/sbgrid/fetch.py --id 821 --mirror uppsala --dest /path
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# SBGrid's public rsync mirrors, nearest-first for this cluster
MIRRORS = {
    "harvard": "rsync://data.sbgrid.org/10.15785/SBGRID",
    "uppsala": "rsync://sbgrid.icm.uu.se/10.15785/SBGRID",
    "montevideo": "rsync://sbgrid.pasteur.edu.uy/10.15785/SBGRID",
    "shanghai": "rsync://sbgrid.ncpss.org/10.15785/SBGRID",
}

DEFAULT_ROOT = os.environ.get(
    "SBGRID_ROOT", "/n/netscratch/hekstra_lab/Lab/laldama/sbgrid"
)

# extensions that mean "diffraction image" across the detectors SBGrid hosts
IMAGE_SUFFIXES = (".cbf", ".h5", ".nxs", ".img", ".mccd", ".osc", ".mar")


def parse_args():
    p = argparse.ArgumentParser(description="Download an SBGrid dataset")
    p.add_argument("--id", required=True, help="SBGrid dataset id, e.g. 821")
    p.add_argument(
        "--dest",
        type=Path,
        default=None,
        help=f"destination root (default: {DEFAULT_ROOT}/<id>)",
    )
    p.add_argument(
        "--mirror",
        default="harvard",
        choices=sorted(MIRRORS),
        help="which SBGrid mirror to pull from",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="list the dataset's files and total size, download nothing",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="download only the first N images (a cheap subset for testing)",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def rsync_url(dataset_id: str, mirror: str) -> str:
    return f"{MIRRORS[mirror]}/{dataset_id}/"


def listing(url: str) -> list[dict]:
    """Return the remote file list as dicts, without transferring anything."""
    proc = subprocess.run(
        ["rsync", "--list-only", "--recursive", url],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode:
        raise SystemExit(f"rsync listing failed:\n{proc.stderr.strip()}")

    entries = []
    for line in proc.stdout.splitlines():
        parts = line.split(maxsplit=4)
        if len(parts) < 5 or parts[0].startswith("d"):
            continue
        entries.append(
            {"size": int(parts[1].replace(",", "")), "name": parts[4]}
        )
    return entries


def summarize(entries: list[dict]) -> dict:
    """Total size, and the image files grouped by extension and by prefix."""
    images = [
        e for e in entries if e["name"].lower().endswith(IMAGE_SUFFIXES)
    ]
    by_suffix: dict[str, int] = {}
    for e in images:
        suffix = Path(e["name"]).suffix.lower()
        by_suffix[suffix] = by_suffix.get(suffix, 0) + 1
    # a "sweep" is one image-name prefix, e.g. 281_helical_1_0001.cbf
    prefixes: dict[str, int] = {}
    for e in images:
        stem = Path(e["name"]).stem
        prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    return {
        "n_files": len(entries),
        "n_images": len(images),
        "total_bytes": sum(e["size"] for e in entries),
        "images_by_suffix": by_suffix,
        "sweeps": prefixes,
    }


def main():
    args = parse_args()
    url = rsync_url(args.id, args.mirror)
    dest = args.dest or Path(DEFAULT_ROOT) / args.id

    print(f"dataset : SBGrid {args.id}")
    print(f"source  : {url}")
    print(f"dest    : {dest}")

    entries = listing(url)
    info = summarize(entries)
    print(
        f"\n{info['n_files']:,} files, {info['n_images']:,} images, "
        f"{info['total_bytes'] / 1e9:.1f} GB"
    )
    for suffix, n in sorted(info["images_by_suffix"].items()):
        print(f"  {suffix:6s} {n:,}")
    if len(info["sweeps"]) <= 12:
        for prefix, n in sorted(info["sweeps"].items()):
            print(f"  sweep {prefix}: {n:,} images")
    else:
        print(f"  {len(info['sweeps'])} image-name prefixes")

    if args.list:
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-av", "--partial", "--info=progress2"]
    if args.max_files:
        # take the first N images by name, plus every non-image file
        images = sorted(
            e["name"]
            for e in entries
            if e["name"].lower().endswith(IMAGE_SUFFIXES)
        )[: args.max_files]
        keep = {e["name"] for e in entries} - set(
            e["name"]
            for e in entries
            if e["name"].lower().endswith(IMAGE_SUFFIXES)
        )
        include = sorted(keep | set(images))
        filters = [f"--include={name}" for name in include]
        cmd += ["--include=*/", *filters, "--exclude=*"]
        print(f"\nsubset: {len(images)} images + {len(keep)} other files")
    cmd += [url, str(dest)]

    print(f"\n$ {' '.join(cmd[:4])} ... {url} {dest}")
    if args.dry_run:
        print("dry run: nothing downloaded")
        return 0

    proc = subprocess.run(cmd, check=False)
    if proc.returncode:
        raise SystemExit(f"rsync failed with exit code {proc.returncode}")

    # a record of provenance next to the data, since the id alone is not
    # enough to reproduce a download once mirrors diverge
    manifest = {
        "sbgrid_id": args.id,
        "source": url,
        "mirror": args.mirror,
        "subset": args.max_files,
        **info,
    }
    (dest / "sbgrid_source.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {dest / 'sbgrid_source.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
