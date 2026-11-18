"""List every script in this repository and what it does, into docs/scripts.md.

The point is one question: **does a script for this already exist?** It gets
asked before anything new is written, and a hand-maintained list would be
wrong within a week, so this reads the scripts themselves -- a Python
module's docstring, a shell script's first real comment -- and the answer is
only as good as those first lines. A script whose summary reads badly here
has a bad summary, and that is worth fixing at the source rather than here.

    uv run python scripts/inventory.py          # write docs/scripts.md
    uv run python scripts/inventory.py --check  # fail if it is out of date

`--check` is for catching a script added without its line appearing, which is
the way this kind of list dies.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "docs" / "scripts.md"
SUFFIXES = (".py", ".sh", ".slurm", ".sbatch")

# The scripts to reach for first, by the job they do. Everything else in the
# repository is a one-off analysis or a diagnostic kept for the record.
# Keyed by path relative to `scripts/`.
ENTRY_POINTS = {
    "mono/cannon/run_all.sh": "train and score, monochromatic, whole chain",
    "poly/cannon/run_all.sh": "train and score, polychromatic, whole chain",
    "poly/train.slurm": "train one arm: `CONFIG=... sbatch`",
    "poly/run_pipeline.py": "predict, careless, stats, phenix, peaks",
    "poly/pipeline.slurm": "the same, as one batch job",
    "mono/cannon/03_pipeline.sh": "predict, DIALS, phenix, peaks",
    "poly/laue_process.sh": "laue-dials on one sweep",
    "poly/pdz2/arm_pdz2.sbatch": "score one PDZ2 arm end to end",
    "figures/pull_figdata.sh": "cluster to laptop, every table",
    "figures/render_all.sh": "draw every figure, rebuild the gallery",
    "figures/make_tables.py": "write docs/tables/",
    "inventory.py": "this list",
}
# Directories whose contents are a coherent set, in reported order. Anything
# else is grouped under its own path.
SECTIONS = (
    ("mono", "Monochromatic chain"),
    ("poly", "Polychromatic chain"),
    ("poly/cannon", "Polychromatic chain: the SLURM stages"),
    ("poly/pdz2", "PDZ2 / 1116 chain"),
    ("poly/pdz2/deposited", "PDZ2: the deposited laue-dials scripts"),
    ("mono/cannon", "Monochromatic chain: the SLURM stages"),
    ("sbgrid", "SBGrid ingestion"),
    ("sbgrid/cannon", "SBGrid ingestion: the SLURM stages"),
    ("figures", "Figures and tables"),
    ("physics", "Simulation and the nanoBragg port"),
    ("jungfrau_sim", "JUNGFRAU SFX simulation"),
    ("ablation_likelihood", "Likelihood ablation"),
    ("shrinkage_experiment", "Shrinkage experiment"),
    ("ragged", "Ragged windows"),
    ("", "One-off analysis and diagnostics"),
)

# Third-party scripts we must not edit -- a docstring added here would
# diverge from `deposited/upstream.patch`. Described from their README.
OVERRIDES = {
    "poly/pdz2/deposited/expt_concat.py":
        "Frame cuts and concatenation (MODIFIED: drops pass-d frames over 2 px RMSD)",
    "poly/pdz2/deposited/custom_ccsym.py":
        "CCsym: half-dataset correlation of F(h) - F(2-fold h), Spearman",
    "poly/pdz2/deposited/custom_ccsym-overall.py": "CCsym over all shells at once",
    "poly/pdz2/deposited/wdm.py": "Weighted difference map, rs.scaleit then rs.diffmap",
    "poly/pdz2/deposited/plot_HKLs.py": "Plot the Miller indices of a sweep",
}

_DOC = re.compile(r'^\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\'|$)', re.S)


def summary(path: Path) -> str:
    """The script's own first line, cleaned up enough to tabulate."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    if path.suffix == ".py":
        m = _DOC.match(text.lstrip("﻿"))
        if not m:
            return ""
        first = m.group(1).strip().splitlines()
        line = first[0].strip() if first else ""
    else:
        line = ""
        for raw in text.splitlines():
            s = raw.strip()
            if not s or s.startswith("#!") or s.startswith("#SBATCH"):
                continue
            if s.startswith("#"):
                s = s.lstrip("#").strip()
                if s:
                    line = s
                    break
                continue
            break
    line = " ".join(line.split())
    # a table cell, so no pipes and not a paragraph
    return line.replace("|", "/")[:160]


def collect() -> dict[str, list[tuple[str, str]]]:
    found: dict[str, list[tuple[str, str]]] = {}
    for path in sorted(SCRIPTS.rglob("*")):
        if path.suffix not in SUFFIXES or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(SCRIPTS).as_posix()
        if rel in OVERRIDES:
            found.setdefault(
                path.parent.relative_to(SCRIPTS).as_posix().strip("."), []
            ).append((rel, OVERRIDES[rel]))
            continue
        found.setdefault(path.parent.relative_to(SCRIPTS).as_posix().strip("."),
                         []).append((rel, summary(path)))
    return found


def render() -> str:
    found = collect()
    total = sum(len(v) for v in found.values())
    lines = [
        "# Every script in this repository",
        "",
        "**Generated by `scripts/inventory.py`. Do not edit by hand.**",
        "Each line is the script's own first docstring line or first comment,",
        "so a summary that reads badly here is a summary to fix at the source.",
        "",
        "Before writing a script, look for it here. If something close exists,",
        "extend it with an option that defaults to today's behaviour rather",
        "than adding a second script beside it -- see CLAUDE.md, \"One script",
        "per job, and it lives in the repo\".",
        "",
        f"{total} scripts.",
        "",
        "## Start here",
        "",
        "| job | script |",
        "|---|---|",
    ]
    for rel, what in ENTRY_POINTS.items():
        lines.append(f"| {what} | `scripts/{rel}` |")
    lines.append("")

    seen: set[str] = set()
    for key, title in SECTIONS:
        entries = found.get(key)
        if not entries:
            continue
        seen.add(key)
        lines += [f"## {title}", "", "| script | what it does |", "|---|---|"]
        for rel, text in entries:
            star = " *" if rel in ENTRY_POINTS else ""
            lines.append(f"| `{rel}`{star} | {text or '--'} |")
        lines.append("")
    for key in sorted(set(found) - seen):
        lines += [f"## `scripts/{key}`", "", "| script | what it does |",
                  "|---|---|"]
        for rel, text in found[key]:
            lines.append(f"| `{rel}` | {text or '--'} |")
        lines.append("")
    lines += ["`*` marks an entry point from the table at the top.", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if docs/scripts.md is out of date")
    args = ap.parse_args()
    text = render()
    if args.check:
        current = OUT.read_text() if OUT.is_file() else ""
        if current != text:
            print(f"{OUT.relative_to(ROOT)} is out of date; "
                  "run `uv run python scripts/inventory.py`", file=sys.stderr)
            return 1
        print(f"{OUT.relative_to(ROOT)} is up to date")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    n = text.count("\n| `")
    print(f"{OUT.relative_to(ROOT)}: {n} scripts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
