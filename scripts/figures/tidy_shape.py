"""Turn a `geom_profile_transfer` JSON into the long-format CSV figtex reads.

The measurement writes a nested JSON with every quartile of every metric for
every model. A figure needs a fraction of that, and the rendering happens on
the laptop, where the Minion Pro and mtpro2 fonts the documents are set in
live, so only the columns that are actually drawn travel.

Output is one row per (method, dataset, reference, metric), which is the shape
`figtex.project.schema("profile_shape", ...)` declares:

    method     the training arm (m9_geom, m9_geomshift, ...)
    dataset    hewl_9b7c
    reference  local | far | ceiling
                 local    the k bright reflections nearest the weak one
                 far      k bright reflections at least `far_px` away
                 ceiling  a second disjoint draw of local bright reflections,
                          which is how close two estimates of the SAME local
                          shape get, so it bounds what any model could reach
    metric     dtheta_deg | fft_cos | dlogratio
    value      the median, with `p25` and `p75` beside it
    n          how many weak reflections that row is over

The orientation index (local - far) / (ceiling - far) is NOT a column: it is
three of these rows divided, and the producer computes it, so the definition
lives in one place.

A second file carries the widths, one row per (method, dataset,
population), because an axis ratio is a property of a set of profiles and
not a comparison against a reference:

    population  weak | local_bright
    axis_ratio  sqrt(l1/l2) of the central second-moment tensor

The same script converts the anisotropy measurement, which is a different
JSON over the SAME models: there the axis ratio is taken on each profile
separately and then summarised, where `profile_width.csv` above is the ratio
of a MEAN profile. Individual profiles are the honest number, because
averaging profiles of genuinely different shapes broadens the mean even when
every member is re-centred first.

    profile_anisotropy.csv
    population  bright | weak | observed
    axis_ratio  the median, with `p25` and `p75` beside it, over `n` profiles

`observed` is the second moment of max(c - b, 0) on the shoebox itself. It
reads 1.089 and is NOT a ground truth for the profile ratio: the clip keeps
every upward background fluctuation over all 441 pixels and the second moment
weights by r^2, so an isotropic pedestal at large radius swamps the core.

Usage: python tidy_shape.py <in.json> <out_shape.csv> [out_width.csv] [dataset]
       python tidy_shape.py --aniso <aniso.json> <out_anisotropy.csv> [dataset]
"""

import csv
import json
import sys

METRICS = {
    "dtheta": "dtheta_deg",
    "fft": "fft_cos",
    "dlogratio": "dlogratio",
}
REFERENCES = ("local", "far", "ceiling")


def rows(doc: dict, dataset: str):
    for method, entry in doc.items():
        if not isinstance(entry, dict) or "dtheta_local" not in entry:
            continue
        for stem, metric in METRICS.items():
            for ref in REFERENCES:
                cell = entry.get(f"{stem}_{ref}")
                if cell is None:
                    continue
                yield {
                    "method": method,
                    "dataset": dataset,
                    "reference": ref,
                    "metric": metric,
                    "value": cell["median"],
                    "p25": cell["p25"],
                    "p75": cell["p75"],
                    "n": entry.get("n_weak"),
                }


WIDTHS = {"median_ratio_weak": "weak",
          "median_ratio_local_bright": "local_bright"}


def width_rows(doc: dict, dataset: str):
    for method, entry in doc.items():
        if not isinstance(entry, dict) or "dtheta_local" not in entry:
            continue
        for key, population in WIDTHS.items():
            if key not in entry:
                continue
            yield {
                "method": method,
                "dataset": dataset,
                "population": population,
                "axis_ratio": entry[key],
                "n": entry.get("n_weak"),
            }


def write(dst: str, out: list[dict]) -> None:
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"{dst}: {len(out)} rows, {len({r['method'] for r in out})} methods")


POPULATIONS = {"bright_profile_ratio": "bright", "weak_profile_ratio": "weak"}


def aniso_rows(doc: dict, dataset: str):
    for method, entry in doc.items():
        if not isinstance(entry, dict):
            continue
        if "ratio_median" in entry:  # the observed shoeboxes
            yield {
                "method": method,
                "dataset": dataset,
                "population": "observed",
                "axis_ratio": entry["ratio_median"],
                "p25": entry["ratio_p25_p75"][0],
                "p75": entry["ratio_p25_p75"][1],
                "n": entry.get("n"),
            }
            continue
        for stem, population in POPULATIONS.items():
            if f"{stem}_median" not in entry:
                continue
            yield {
                "method": method,
                "dataset": dataset,
                "population": population,
                "axis_ratio": entry[f"{stem}_median"],
                "p25": entry[f"{stem}_p25_p75"][0],
                "p75": entry[f"{stem}_p25_p75"][1],
                "n": entry.get(f"n_{population}"),
            }


def main() -> None:
    if len(sys.argv) >= 4 and sys.argv[1] == "--aniso":
        doc = json.load(open(sys.argv[2]))
        dataset = sys.argv[4] if len(sys.argv) > 4 else "hewl_9b7c"
        out = list(aniso_rows(doc, dataset))
        if not out:
            raise SystemExit(f"no anisotropy entries in {sys.argv[2]}")
        write(sys.argv[3], out)
        return
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    wdst = sys.argv[3] if len(sys.argv) > 3 else None
    dataset = sys.argv[4] if len(sys.argv) > 4 else "hewl_9b7c"
    doc = json.load(open(src))
    out = list(rows(doc, dataset))
    if not out:
        raise SystemExit(f"no model entries in {src}")
    write(dst, out)
    if wdst:
        write(wdst, list(width_rows(doc, dataset)))


if __name__ == "__main__":
    main()
