"""Decide how a downloaded dataset should be processed, before processing it.

Answers the three questions the pipeline needs and cannot guess:

  geometry     rotation or still, and monochromatic or polychromatic
  sweeps       how many, and which images belong to each
  anomalous    whether to expect a measurable anomalous signal

The first two come from the image headers through dxtbx and are decisive.
The third is only ever a prior here: the deposited phasing method and the
wavelength's distance from an absorption edge both say what the experiment
was *for*, but whether the signal survives is measured after merging, not
declared now. `expect_anomalous` is a recommendation to process with
anomalous=true, never a claim that signal exists.

Writes `dataset_card.json`, which the rest of the chain reads instead of
re-deriving any of this.

Usage:
    dials.python scripts/sbgrid/characterize.py --data-dir <dir>
    dials.python scripts/sbgrid/characterize.py --data-dir <dir> --pdb 7LVC
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

IMAGE_SUFFIXES = (".cbf", ".h5", ".nxs", ".img", ".mccd", ".osc", ".mar")

# K-edge wavelengths in Angstrom for the elements that carry anomalous signal
# in practice. A wavelength sitting on one of these was almost certainly
# chosen for it.
K_EDGES_ANGSTROM = {
    "P": 5.787,
    "S": 5.018,
    "Cl": 4.397,
    "Ca": 3.070,
    "Mn": 1.896,
    "Fe": 1.743,
    "Co": 1.608,
    "Ni": 1.488,
    "Cu": 1.381,
    "Zn": 1.284,
    "Se": 0.980,
    "Br": 0.920,
}
# L-III edges for the heavy atoms used in derivatization
L3_EDGES_ANGSTROM = {"Ta": 1.255, "Pt": 1.072, "Au": 1.040, "Hg": 1.009}

# Proximity is judged in energy, not wavelength: a fixed wavelength window is
# a tiny energy window at long wavelengths and a huge one at short. A peak or
# inflection wavelength is chosen within a few tens of eV of the edge.
EDGE_TOLERANCE_EV = 50.0
HC_KEV_ANGSTROM = 12.39842  # E[keV] = HC / lambda[A]


def parse_args():
    p = argparse.ArgumentParser(description="Characterize a diffraction dataset")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument(
        "--pdb",
        default=None,
        help="PDB id, if the dataset names one; adds deposited evidence",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where dataset_card.json goes (default: --data-dir)",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=3,
        help="images to open per sweep; headers are identical within a sweep",
    )
    return p.parse_args()


def find_sweeps(data_dir: Path) -> dict[str, list[Path]]:
    """Group image files into sweeps by their filename prefix.

    Detectors write `<prefix>_<number>.<ext>`, so the prefix separates sweeps
    of the same crystal. This is how the collection was organized; it is not
    a claim that the sweeps share a crystal orientation.
    """
    images = [
        p
        for p in sorted(data_dir.rglob("*"))
        if p.suffix.lower() in IMAGE_SUFFIXES
    ]
    sweeps: dict[str, list[Path]] = defaultdict(list)
    for path in images:
        match = re.match(r"^(.*?)_?(\d+)$", path.stem)
        prefix = match.group(1) if match else path.stem
        sweeps[prefix].append(path)
    return dict(sweeps)


def read_geometry(paths: list[Path], max_images: int) -> dict:
    """Beam, detector and scan geometry, read through dxtbx."""
    from dxtbx.model.experiment_list import ExperimentListFactory

    sample = [str(p) for p in paths[:max_images]]
    experiments = ExperimentListFactory.from_filenames(sample)
    if not experiments:
        return {"error": "dxtbx could not read these images"}

    experiment = experiments[0]
    beam, detector, scan = experiment.beam, experiment.detector, experiment.scan

    info: dict = {
        "n_images": len(paths),
        "wavelength": float(beam.get_wavelength()),
        "detector": detector[0].get_name() if len(detector) else None,
        "n_panels": len(detector),
        "pixel_size_mm": list(detector[0].get_pixel_size())
        if len(detector)
        else None,
        "image_size_px": list(detector[0].get_image_size())
        if len(detector)
        else None,
        "distance_mm": float(detector[0].get_distance()) if len(detector) else None,
    }

    # a scan with a nonzero oscillation width is a rotation sequence; without
    # one the images are stills, which is what Laue and serial data look like
    if scan is not None:
        width = float(scan.get_oscillation()[1])
        info["oscillation_deg"] = width
        info["is_rotation"] = width > 0.0
    else:
        info["oscillation_deg"] = None
        info["is_rotation"] = False

    # a polychromatic beam carries a wavelength range rather than one value
    spectrum = getattr(beam, "get_spectrum_weights", None)
    info["has_spectrum"] = spectrum is not None and beam.get_spectrum_weights() is not None
    return info


def nearest_edge(wavelength: float) -> dict | None:
    """The absorption edge closest to a wavelength, if it is close at all."""
    energy_ev = HC_KEV_ANGSTROM / wavelength * 1000.0
    best = None
    for element, edge in {**K_EDGES_ANGSTROM, **L3_EDGES_ANGSTROM}.items():
        delta = abs(energy_ev - HC_KEV_ANGSTROM / edge * 1000.0)
        if best is None or delta < best["delta_ev"]:
            best = {
                "element": element,
                "shell": "K" if element in K_EDGES_ANGSTROM else "L3",
                "edge_angstrom": edge,
                "delta_ev": round(delta, 1),
            }
    if best and best["delta_ev"] <= EDGE_TOLERANCE_EV:
        return best
    return None


def classify(geometry: dict, reference: dict | None) -> dict:
    """Turn the evidence into the decisions the pipeline needs."""
    deposited_mono = None
    phasing_anomalous = False
    if reference:
        deposited_mono = reference.get("mono_or_laue")
        phasing_anomalous = bool(
            (reference.get("anomalous") or {}).get("phasing_is_anomalous")
        )

    # the deposition states monochromatic vs Laue outright; the headers only
    # tell us rotation vs still, which is a different question
    if deposited_mono in ("M", "L"):
        mode = "monochromatic" if deposited_mono == "M" else "polychromatic"
        mode_evidence = f"PDB pdbx_monochromatic_or_laue_m_l = {deposited_mono}"
    elif geometry.get("has_spectrum"):
        mode = "polychromatic"
        mode_evidence = "beam carries a spectrum"
    elif geometry.get("is_rotation"):
        mode = "monochromatic"
        mode_evidence = (
            f"rotation scan, {geometry['oscillation_deg']} deg per image"
        )
    else:
        mode = "unknown"
        mode_evidence = (
            "stills with no spectrum: could be serial monochromatic or Laue"
        )

    edge = nearest_edge(geometry["wavelength"]) if geometry.get("wavelength") else None
    reasons = []
    if phasing_anomalous and reference is not None:
        method = (reference.get("anomalous") or {}).get("phasing_method")
        reasons.append(f"deposited phasing method is {method}")
    if edge:
        reasons.append(
            f"wavelength {geometry['wavelength']:.4f} A is "
            f"{edge['delta_ev']:.0f} eV from the {edge['element']} "
            f"{edge['shell']} edge"
        )

    return {
        "mode": mode,
        "mode_evidence": mode_evidence,
        "geometry": "rotation" if geometry.get("is_rotation") else "stills",
        # a recommendation to merge anomalously, not a claim of signal:
        # absence of these is weak evidence, since plenty of usable anomalous
        # data was phased by molecular replacement
        "expect_anomalous": bool(phasing_anomalous or edge),
        "anomalous_reasons": reasons,
        "nearest_edge": edge,
    }


def main():
    args = parse_args()
    out_dir = args.out_dir or args.data_dir
    sweeps = find_sweeps(args.data_dir)
    if not sweeps:
        raise SystemExit(f"no diffraction images under {args.data_dir}")

    print(f"{len(sweeps)} sweep(s) under {args.data_dir}")
    for prefix, paths in sorted(sweeps.items()):
        print(f"  {prefix}: {len(paths)} images")

    first = sorted(sweeps)[0]
    geometry = read_geometry(sweeps[first], args.max_images)
    if "error" in geometry:
        raise SystemExit(geometry["error"])

    reference = None
    if args.pdb:
        card = out_dir / "reference_card.json"
        if card.exists():
            reference = json.loads(card.read_text())
        else:
            print(
                f"\nno reference_card.json in {out_dir}; run "
                "reference_stats.py first for the deposited evidence"
            )

    decisions = classify(geometry, reference)

    card = {
        "data_dir": str(args.data_dir),
        "pdb_id": args.pdb,
        "sweeps": {k: [str(p) for p in v] for k, v in sorted(sweeps.items())},
        "n_sweeps": len(sweeps),
        "geometry": geometry,
        **decisions,
    }
    path = out_dir / "dataset_card.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2))

    print(f"\nwavelength      {geometry['wavelength']:.4f} A")
    print(f"detector        {geometry['detector']} "
          f"{geometry['image_size_px']} @ {geometry['distance_mm']:.1f} mm")
    print(f"geometry        {decisions['geometry']} "
          f"({geometry['oscillation_deg']} deg/image)")
    print(f"mode            {decisions['mode']}  [{decisions['mode_evidence']}]")
    print(f"expect anomalous {decisions['expect_anomalous']}")
    for reason in decisions["anomalous_reasons"]:
        print(f"  - {reason}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
