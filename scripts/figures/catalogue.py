"""Every rendered figure, indexed by data type, dataset and experiment.

The question this answers is "which figures exist for PDZ2,
svae against hierarchical?", which `docs/figures/gallery.html` cannot
answer: the gallery groups by KIND -- CCsym, merging statistics, the
learned basis -- and a kind cuts across experiments. A CCsym figure
comparing arms to `laue.integrate` and a CCsym figure comparing two
integrator families are the same kind and different experiments.

**Generated, not typed.** 138 figures hand-listed would be wrong within a
week, the way `docs/scripts.md` would be. Three facts already on disk
determine an entry:

- **kind** from `build_gallery.RULES`, imported rather than restated, so
  the gallery and this file cannot disagree about what a figure is.
- **dataset** from the figure's own name, which the drawing scripts end
  with the dataset key (`ccsym_200ns_pdz2_1116`), falling back to the
  directory.
- **data type** from the dataset, since a dataset is monochromatic or
  polychromatic and never both.

**experiment** is the one axis not on disk, because it is a statement
about why a figure was drawn. It is matched below by kind and dataset,
and anything unmatched is reported as `unassigned` rather than guessed --
a wrong experiment label is worse than a missing one, since the whole
point is to filter on it.

    uv run python scripts/figures/catalogue.py           # write the csv
    uv run python scripts/figures/catalogue.py --check   # fail if stale
    uv run python scripts/figures/catalogue.py --experiment svae_vs_hier
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "docs" / "figures"
OUT = FIGURES / "catalogue.csv"
sys.path.insert(0, str(Path(__file__).resolve().parent))

COLUMNS = ("id", "datatype", "dataset", "experiment", "kind", "path")

# Which data type each dataset is. A dataset is one or the other and never
# both, so this is the whole mapping rather than a heuristic.
DATATYPE = {
    "hewl_9b7c": "mono",
    "hewl_1118": "poly",
    "pdz2_1116": "poly",
    "dhfr_sim": "poly",
}
# Longest first, so `hewl_1118` is not matched by a shorter key inside it.
_DATASET_KEYS = sorted(DATATYPE, key=len, reverse=True)

# The experiment each (kind, dataset) belongs to. A kind alone is not
# enough: the same CCsym figure is the model-against-reference comparison
# on PDZ2 and nothing on 9b7c, which has no CCsym.
#
# `*` matches any dataset. The first matching rule wins, so a rule naming a
# dataset must precede the `*` rule for the same kind.
EXPERIMENTS: tuple[tuple[str, str, str], ...] = (
    # kind contains, dataset, experiment
    ("svae against hierarchical", "*", "svae_vs_hierarchical"),
    ("PDZ2: refinement", "*", "model_vs_reference"),
    ("PDZ2: CCsym", "*", "model_vs_reference"),
    ("careless.stats", "pdz2_1116", "model_vs_reference"),
    ("careless.stats", "*", "merging_quality"),
    ("Merging statistics", "*", "merging_quality"),
    ("Refinement over training", "*", "training_trajectory"),
    ("Anomalous peak heights", "*", "anomalous_signal"),
    ("Training loss", "*", "training_trajectory"),
    ("The Wilson prior", "*", "training_trajectory"),
    ("The learned profile basis", "*", "profile_models"),
    ("Profile shape", "*", "profile_models"),
    ("Input shoeboxes", "*", "profile_models"),
    ("learned spectrum", "*", "training_trajectory"),
    ("Model against the reference integrator", "*", "model_vs_reference"),
    ("Arm comparison", "*", "arm_comparison"),
    ("Architecture diagrams", "*", "architecture"),
)


# A figure that is about more than one dataset, so no single key is right.
ACROSS = "all"


def _arm_datasets() -> dict[str, str]:
    """{arm: dataset} from the tidy tables, for figures named by arm.

    The basis figures are named after the run they came from
    (`basis_components_pdz2_svae`), not the dataset, and an arm name does
    not say which dataset it trained on. The reducers already write both as
    key columns, so the answer is read rather than pattern-matched.
    """
    import csv as _csv

    out: dict[str, str] = {}
    for name in ("basis_cmp/basis_components.csv", "basis/basis_components.csv"):
        path = FIGURES / "data" / name
        if not path.is_file():
            continue
        with open(path, newline="") as fh:
            for rec in _csv.DictReader(fh):
                arm, dataset = rec.get("method"), rec.get("dataset")
                if arm and dataset:
                    out.setdefault(arm, dataset)
    return out


def dataset_of(stem: str, path: Path, arms: dict[str, str]) -> str:
    """The dataset a figure is of: its name, then its arm, then its directory."""
    for key in _DATASET_KEYS:
        if stem.endswith(key) or f"_{key}_" in stem or stem.startswith(key):
            return key
    # named after an arm rather than a dataset; longest first so a name that
    # contains another is not matched by the shorter one
    for arm in sorted(arms, key=len, reverse=True):
        if stem.endswith(arm):
            return arms[arm]
    # the drawing scripts write into a directory per dataset
    by_dir = {"m9": "hewl_9b7c", "pdz2": "pdz2_1116",
              "poly1118": "hewl_1118"}
    for part in path.parts[::-1]:
        if part in by_dir:
            return by_dir[part]
    return ""


def experiment_of(kind: str, dataset: str) -> str:
    for needle, want, name in EXPERIMENTS:
        if needle.lower() in kind.lower() and want in ("*", dataset):
            return name
    return "unassigned"


def rows() -> list[dict]:
    import build_gallery as gallery

    files = gallery.collect(FIGURES)
    arms = _arm_datasets()
    claimed: set[Path] = set()
    out: list[dict] = []
    for title, _note, matches in gallery.RULES:
        for p in files:
            if p in claimed or not matches(p.stem):
                continue
            claimed.add(p)
            dataset = dataset_of(p.stem, p, arms)
            if title.startswith("svae against"):
                # drawn across every dataset at once; naming one would be
                # wrong and dropping it would hide them from every filter
                dataset = ACROSS
            out.append({
                "id": p.stem,
                "datatype": DATATYPE.get(dataset, ACROSS if dataset == ACROSS
                                         else ""),
                "dataset": dataset,
                "experiment": experiment_of(title, dataset),
                "kind": title,
                "path": p.relative_to(ROOT).as_posix(),
            })
    for p in files:
        if p in claimed:
            continue
        dataset = dataset_of(p.stem, p, arms)
        out.append({
            "id": p.stem,
            "datatype": DATATYPE.get(dataset, ""),
            "dataset": dataset,
            "experiment": "unassigned",
            "kind": "unsorted",
            "path": p.relative_to(ROOT).as_posix(),
        })
    return sorted(out, key=lambda r: (r["datatype"], r["dataset"],
                                      r["experiment"], r["kind"], r["id"]))


def render(entries: list[dict]) -> str:
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(COLUMNS), lineterminator="\n")
    w.writeheader()
    w.writerows(entries)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the catalogue is out of date")
    ap.add_argument("--datatype")
    ap.add_argument("--dataset")
    ap.add_argument("--experiment")
    args = ap.parse_args()

    entries = rows()
    if args.datatype or args.dataset or args.experiment:
        for field in ("datatype", "dataset", "experiment"):
            want = getattr(args, field)
            if want:
                entries = [e for e in entries if e[field] == want]
        for e in entries:
            print(f"{e['experiment']:<24} {e['kind']:<44} {e['id']}")
        print(f"\n{len(entries)} figures")
        return 0

    text = render(entries)
    if args.check:
        current = OUT.read_text() if OUT.is_file() else ""
        if current != text:
            print(f"{OUT.relative_to(ROOT)} is out of date; "
                  "run `uv run python scripts/figures/catalogue.py`",
                  file=sys.stderr)
            return 1
        print(f"{OUT.relative_to(ROOT)} is up to date")
        return 0

    OUT.write_text(text)
    counts: dict[tuple[str, str, str], int] = {}
    for e in entries:
        key = (e["datatype"], e["dataset"], e["experiment"])
        counts[key] = counts.get(key, 0) + 1
    print(f"{OUT.relative_to(ROOT)}: {len(entries)} figures")
    for (dt, ds, ex), n in sorted(counts.items()):
        print(f"  {dt or '-':<6} {ds or '-':<12} {ex:<24} {n:>3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
