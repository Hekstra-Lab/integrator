"""Fetch an SBGrid processing bundle and read the depositors' own recipe.

This is the part of a dataset that is *not* derivable from the images. SBGrid
lets depositors attach a processing bundle -- masks, and the scripts they
actually ran -- alongside free-text reprocessing instructions. Where those
exist they override anything inferred from headers, because they record what
the headers get wrong.

Dataset 821 is the case in point: the detector was raised between passes, so
the beam centre written into the headers is correct for pass 1 and wrong for
passes 2 and 3. Nothing in the images says so. Processing it automatically
from header geometry alone would index two thirds of the data incorrectly
and report no error.

What is parsed out of the scripts is deliberately narrow -- the parameters
that change the answer, keyed by the image template each script processes.
Everything else is left as text for a person to read.

Usage:
    python scripts/sbgrid/bundle.py --id 821 --out-dir <dataset dir>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

BUNDLE_URL = (
    "https://data.sbgrid.org/upload/thumbnails/{sid}/processing_bundle-{sid}.tar.gz"
)

# DIALS parameters worth lifting out of a depositor script, because each one
# changes the result and none can be recovered from the images
PARAMETERS = {
    "beam_centre": re.compile(
        r"geometry\.detector\.mosflm_beam_centre\s*=\s*([0-9.,]+)"
    ),
    "image_range": re.compile(r"geometry\.scan\.image_range\s*=\s*([0-9,]+)"),
    "mask": re.compile(r"\bmask\s*=\s*(\S+)"),
    "d_max": re.compile(r"spotfinder\.filter\.d_max\s*=\s*([0-9.]+)"),
    "d_min": re.compile(r"spotfinder\.filter\.d_min\s*=\s*([0-9.]+)"),
    "space_group": re.compile(
        r"indexing\.known_symmetry\.space_group\s*=\s*(\S+)"
    ),
    "unit_cell": re.compile(r"indexing\.known_symmetry\.unit_cell\s*=\s*(\S+)"),
    "scan_varying": re.compile(
        r"refinement\.parameterisation\.scan_varying\s*=\s*(\w+)"
    ),
}

# the image template a script processes, e.g. images=/path/281_helical_1_????.cbf
TEMPLATE_RE = re.compile(r"images=\S*?/?([\w.-]+?)_\?+\.\w+")
# which DIALS programs the script actually calls, in order
PROGRAM_RE = re.compile(r"^\s*(dials\.\w+)", re.MULTILINE)


def parse_args():
    p = argparse.ArgumentParser(description="SBGrid processing bundle")
    p.add_argument("--id", required=True, help="SBGrid dataset id")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def download(dataset_id: str, out_dir: Path) -> Path | None:
    """Fetch and extract the bundle, or return None when there is not one."""
    url = BUNDLE_URL.format(sid=dataset_id)
    archive = out_dir / f"processing_bundle-{dataset_id}.tar.gz"
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            archive.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"no processing bundle for dataset {dataset_id}")
            return None
        raise

    target = out_dir / "processing"
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if not m.name.startswith("/")]
        tar.extractall(out_dir, members=members)  # noqa: S202
    print(f"extracted {len(members)} member(s) to {target}")
    return target


def parse_script(path: Path) -> dict:
    """Lift the DIALS parameters and the image template out of one script."""
    text = path.read_text(errors="replace")
    found = {}
    for name, pattern in PARAMETERS.items():
        match = pattern.search(text)
        if match:
            found[name] = match.group(1)
    template = TEMPLATE_RE.search(text)
    return {
        "script": path.name,
        "sweep": template.group(1) if template else None,
        "programs": PROGRAM_RE.findall(text),
        "parameters": found,
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    processing = download(args.id, args.out_dir)
    if processing is None:
        return 0

    scripts = sorted(processing.glob("*.sh"))
    masks = sorted(processing.glob("*.mask")) + sorted(
        processing.glob("*mask*")
    )
    recipes = [parse_script(p) for p in scripts]

    hints = {
        "sbgrid_id": args.id,
        "processing_dir": str(processing),
        "masks": [str(p) for p in dict.fromkeys(masks)],
        # keyed by sweep so per-pass overrides stay attached to their pass:
        # a beam centre that is right for one sweep is wrong for another
        "per_sweep": {
            r["sweep"]: r["parameters"] for r in recipes if r["sweep"]
        },
        "recipes": recipes,
    }
    path = args.out_dir / "processing_hints.json"
    path.write_text(json.dumps(hints, indent=2))

    print(f"\n{len(scripts)} depositor script(s), {len(hints['masks'])} mask(s)")
    for recipe in recipes:
        print(f"\n  {recipe['script']}  ->  sweep {recipe['sweep']}")
        print(f"    programs: {' '.join(recipe['programs'])}")
        for key, value in sorted(recipe["parameters"].items()):
            print(f"    {key}: {value}")

    print(f"\nwrote {path}")
    if any("beam_centre" in r["parameters"] for r in recipes):
        print(
            "\nNOTE: a depositor script overrides the beam centre, so the "
            "value in the image headers is wrong for at least one sweep. "
            "Header-only processing would index it incorrectly and report "
            "no error."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
