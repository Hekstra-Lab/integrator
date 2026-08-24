"""Pull the deposited crystallographic statistics for a PDB entry.

An SBGrid dataset usually names the PDB entry it produced, which gives a
published baseline to compare against: what the authors got from this exact
data. Those numbers land in the same nine columns the integrator's two arms
emit, so a deposited CC1/2 and a careless CC1/2 can go on one axis.

The deposited statistics are overall values, not per shell, so the table is
written as a single row with `bin = 0` spanning the full resolution range.
The PDB does publish an outer shell in `refine_ls_shell`; it is written as a
second row when present.

Usage:
    python scripts/sbgrid/reference_stats.py --pdb 7LVC --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb}"

# the shared schema both integrator arms emit
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

# entry fields worth keeping beyond the merging statistics. The PDB's field
# names are case-sensitive and inconsistently cased, so they are spelled here
# exactly as the REST API returns them.
CARD_FIELDS = {
    "space_group": ("symmetry", "space_group_name_H_M"),
    "space_group_number": ("symmetry", "Int_Tables_number"),
    "method": ("exptl", 0, "method"),
    "phasing_method": ("refine", 0, "pdbx_method_to_determine_struct"),
    "r_work": ("refine", 0, "ls_R_factor_R_work"),
    "r_free": ("refine", 0, "ls_R_factor_R_free"),
    "temperature_k": ("diffrn", 0, "ambient_temp"),
    "beamline": ("diffrn_source", 0, "type"),
    # the PDB states monochromatic vs Laue outright: "M" or "L"
    "mono_or_laue": ("diffrn_radiation", 0, "pdbx_monochromatic_or_laue_m_l"),
    "diffrn_protocol": ("diffrn_radiation", 0, "pdbx_diffrn_protocol"),
    "wavelength_list": ("diffrn_source", 0, "pdbx_wavelength_list"),
    "wilson_b": ("reflns", 0, "B_iso_Wilson_estimate"),
}


def parse_args():
    p = argparse.ArgumentParser(description="Deposited stats for a PDB entry")
    p.add_argument("--pdb", required=True, help="PDB id, e.g. 7LVC")
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def fetch_entry(pdb: str) -> dict:
    url = ENTRY_URL.format(pdb=pdb.upper())
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        return json.load(response)


def dig(data, path):
    """Walk a mixed dict/list path, returning None at the first miss."""
    for key in path:
        if data is None:
            return None
        try:
            data = data[key]
        except (KeyError, IndexError, TypeError):
            return None
    return data


def merging_rows(entry: dict) -> list[dict]:
    """Deposited merging statistics as rows of the shared schema."""
    rows = []
    for i, shell in enumerate(entry.get("reflns") or []):
        rows.append(
            {
                "bin": i,
                "d_max": shell.get("d_resolution_low"),
                "d_min": shell.get("d_resolution_high"),
                "n_obs": shell.get("number_obs"),
                "n_unique": shell.get("number_all"),
                "cc_half": shell.get("pdbx_CC_half"),
                # the PDB has no CCanom field; anomalous quality is reported,
                # when at all, as a separate completeness/multiplicity
                "cc_anom": None,
                "r_pim": shell.get("pdbx_Rpim_I_all"),
                "i_over_sigma": shell.get("pdbx_netI_over_sigmaI"),
            }
        )
    return rows


def anomalous_evidence(entry: dict) -> dict:
    """What the deposition says about anomalous data, without inferring.

    Anomalous phasing is strong evidence the signal is both present and
    measurable, because the structure was solved from it. Its absence is
    weak evidence of anything: plenty of datasets carry usable anomalous
    signal and were phased by molecular replacement.
    """
    refine = (entry.get("refine") or [{}])[0]
    method = refine.get("pdbx_method_to_determine_struct") or ""
    wavelengths = []
    for source in entry.get("diffrn_source") or []:
        raw = source.get("pdbx_wavelength_list") or source.get(
            "pdbx_wavelength"
        )
        if raw:
            wavelengths += [
                float(w) for w in str(raw).replace(",", " ").split()
            ]
    return {
        "phasing_method": method or None,
        "phasing_is_anomalous": any(
            tag in method.upper() for tag in ("SAD", "MAD", "SIRAS", "MIRAS")
        ),
        "wavelengths": wavelengths,
    }


def main():
    args = parse_args()
    entry = fetch_entry(args.pdb)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = merging_rows(entry)
    if rows:
        import csv

        path = args.out_dir / "reference_merging_stats.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path} ({len(rows)} row(s))")
        for row in rows:
            print(
                f"  {row['d_max']} - {row['d_min']} A  "
                f"CC1/2 {row['cc_half']}  Rpim {row['r_pim']}  "
                f"I/sig {row['i_over_sigma']}"
            )
    else:
        print("no reflns block in the entry: no merging statistics deposited")

    card = {"pdb_id": args.pdb.upper()}
    for name, path in CARD_FIELDS.items():
        card[name] = dig(entry, path)
    cell = dig(entry, ("cell",)) or {}
    card["unit_cell"] = [
        cell.get(k)
        for k in ("length_a", "length_b", "length_c", "angle_alpha",
                  "angle_beta", "angle_gamma")
    ]
    card["anomalous"] = anomalous_evidence(entry)

    path = args.out_dir / "reference_card.json"
    path.write_text(json.dumps(card, indent=2))
    print(f"\nwrote {path}")
    print(f"  space group     {card['space_group']}")
    print(f"  phasing         {card['anomalous']['phasing_method']}")
    print(f"  anomalous       {card['anomalous']['phasing_is_anomalous']}")
    print(f"  R-work/R-free   {card['r_work']} / {card['r_free']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
