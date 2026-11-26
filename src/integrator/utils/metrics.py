"""The numbers this work reports, defined and parsed in one place.

Every arm is scored through the same chain and the same quantities come out
of it. Those quantities were being parsed in twenty-odd places, each with its
own regular expression, and the expressions had already drifted apart: the
chain emits

    RESULT m9_geomshift_rb_F | Final R-work = 0.1607, R-free = 0.1785

while `results_table.py` looked for `Final` immediately after the arm name
and `tidy_xtal.sh` grepped for an arm without the `_rb`/`_model` infix. Both
matched nothing, and both reported that as an empty table rather than as an
error. This module is the single definition, so a format change breaks one
test instead of silently emptying a figure.

**No metric is invented here.** These are the quantities the chain already
produces: R-work and R-free from phenix, anomalous peak heights from
`rs.find_peaks`, and the per-shell table `dials.merge` prints. Anything new
belongs in a measurement that is argued for first, not in this module.

The emitting half lives here too, so the format cannot drift from its
readers:

    python -m integrator.utils.metrics emit-result \\
        --label m9_geomshift_rb --variant F --r-work 0.1607 --r-free 0.1785
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

__all__ = [
    "ANOMALOUS_RESIDUES",
    "ESTIMATORS",
    "MERGING_SUMMARY_KEYS",
    "MERGING_CSV_COLUMNS",
    "PEAKS_RE",
    "RESULT_RE",
    "SHELL_COLUMNS",
    "VARIANTS",
    "LogTable",
    "anomalous_differences_from_log",
    "d_range_from_label",
    "emit_peaks",
    "model_atoms",
    "nearest_atoms",
    "heaviest_atoms",
    "heavy_atom_candidates",
    "error_model_from_log",
    "find_table",
    "html_tables",
    "log_tables",
    "merging_summary_from_log",
    "scaling_summary_from_log",
    "wilson_from_log",
    "emit_result",
    "parse_peaks_lines",
    "parse_result_lines",
    "peak_heights_from_csv",
    "peaks_from_find_peaks_csv",
    "refinement_from_phenix_log",
    "shell_midpoint",
    "shells_from_any",
    "shells_from_merge_log",
    "shells_from_merged_html",
    "shells_from_merging_stats_csv",
    "site_label",
    "panel_site_label",
    "split_label",
]

# The DIALS logs carry tables that never reach `merging_stats.csv` -- the
# error model, the full merging summary, the Wilson analysis. They are read
# in `dials_log` and re-exported here, so a caller still has one import.
from integrator.utils.dials_log import (  # noqa: E402
    MERGING_SUMMARY_KEYS,
    LogTable,
    anomalous_differences_from_log,
    error_model_from_log,
    find_table,
    log_tables,
    merging_summary_from_log,
    scaling_summary_from_log,
    wilson_from_log,
)

# The two exports scored per run: the Rao-Blackwellized profile fit computed
# at export, and the posterior mean. See the two-estimators rule in the docs.
ESTIMATORS = ("rb", "model")

# A merged MTZ carries both F(+)/SigF(+) and I(+)/SigI(+) and both are
# refined; a gap between them is an export defect, not a result.
VARIANTS = ("F", "I")

# The residues carrying anomalous signal in HEWL 9b7c. Waters sit at the
# noise floor and drag a mean towards it, so they are not scatterers here.
ANOMALOUS_RESIDUES = ("IOD", "CYS", "MET", "SER")

# The per-shell table `dials.merge` prints, in its own column order.
SHELL_COLUMNS = (
    "d_max", "d_min", "n_obs", "n_uniq", "multiplicity", "completeness",
    "i_mean", "i_over_sigma", "r_merge", "r_meas", "r_pim", "r_anom",
    "cc_half", "cc_anom",
)

# `RESULT <label>_<variant> | Final R-work = <rw>, R-free = <rf>`
# The label carries the arm and, usually, the estimator: `m9_geomshift_rb`.
# `_F`/`_I` is matched non-greedily off the end so an arm whose own name ends
# in a capital letter is not mistaken for a variant.
RESULT_RE = re.compile(
    r"^RESULT\s+(?P<label>\S+?)_(?P<variant>[FI])\s*\|\s*"
    r"Final R-work\s*=\s*(?P<r_work>[\d.]+),\s*"
    r"R-free\s*=\s*(?P<r_free>[\d.]+)\s*$",
    re.M,
)

# `PEAKS <label>_<variant> 10.508,10.558,...` -- one peak height per site,
# in the column order of the refinement's `peak_heights.csv`.
PEAKS_RE = re.compile(
    r"^PEAKS\s+(?P<label>\S+?)_(?P<variant>[FI])\s+(?P<peaks>[\d.,eE+-]+)\s*$",
    re.M,
)

# phenix prints a `Start R-work` line before the `Final` one, so a parser
# that takes the first match reports the starting model.
PHENIX_FINAL_RE = re.compile(
    r"Final R-work\s*=\s*([0-9]*\.?[0-9]+)\s*,\s*"
    r"R-free\s*=\s*([0-9]*\.?[0-9]+)"
)
# A stats file writes the same two numbers as `Rwork:  0.1570` /
# `Rfree:  0.1738`. The 9b7c reference is in this form and in no other, so
# dropping the fallback silently drops the reference line from every
# refinement figure.
STATS_RWORK_RE = re.compile(r"Rwork\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
STATS_RFREE_RE = re.compile(r"Rfree\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

# The canonical per-shell CSV both arms emit (`emit_merging_stats.py`).
# These names are the ones used throughout; `dials.merge`'s own headings and
# `merged.html`'s are mapped onto them at the parser.
MERGING_CSV_COLUMNS = (
    "bin", "d_max", "d_min", "n_obs", "n_unique",
    "cc_half", "cc_anom", "r_pim", "i_over_sigma",
)

# Substrings that identify a column in a DIALS `merged.html` table. DIALS
# writes `CC½` and `CCano` with the typographic characters, and phenix-era
# tables write `CC1/2`, so each name has several spellings.
_HTML_COLUMNS = {
    "cc_half": (("cc", "½"), ("cc", "half"), ("cc", "1/2")),
    "cc_anom": (("ccano",), ("ccanom",), ("cc", "anom")),
    "r_pim": (("rpim",), ("r", "pim")),
    "i_over_sigma": (("i/σ",), ("mean i/",), ("i/sig",)),
}


def split_label(label: str) -> tuple[str, str | None]:
    """`m9_geomshift_rb` -> ("m9_geomshift", "rb"); unsuffixed -> (label, None).

    The reference integrator is merged without an estimator suffix, because
    it has only one set of intensities, so the estimator is optional.
    """
    for est in ESTIMATORS:
        if label.endswith(f"_{est}"):
            return label[: -len(est) - 1], est
    return label, None


# --------------------------------------------------------------------------- #
# emitting
# --------------------------------------------------------------------------- #
def emit_result(label: str, variant: str, r_work: float, r_free: float) -> str:
    """The one line the chain prints for a refinement, and the only source
    of its format. `RESULT_RE` reads exactly what this writes."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    return (
        f"RESULT {label}_{variant} | "
        f"Final R-work = {float(r_work):.4f}, R-free = {float(r_free):.4f}"
    )


def emit_peaks(label: str, variant: str, peaks) -> str:
    """The one line the chain prints for a set of anomalous peak heights.

    A string is written through unchanged. The heights come from a CSV the
    refinement wrote, and reformatting a number that is already decimal text
    can only lose digits -- `f"{1234.567:g}"` is `1234.57` -- so the text is
    preserved and only genuine floats are formatted.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    # `repr` of a float is its shortest round-tripping decimal form:
    # `33.605`, where `f"{x:.17g}"` gives `33.604999999999997` and
    # `f"{x:g}"` silently truncates at six significant digits.
    body = ",".join(
        p.strip() if isinstance(p, str) else repr(float(p)) for p in peaks
    )
    return f"PEAKS {label}_{variant} {body}"


# --------------------------------------------------------------------------- #
# parsing the chain's own log lines
# --------------------------------------------------------------------------- #
def parse_result_lines(text: str, last_only: bool = True) -> list[dict]:
    """Every `RESULT` line in `text`, as rows.

    A log can hold several runs of the same arm, because a rerun appends, so
    by default the last line for each (label, variant) wins -- the earlier
    ones are superseded, not alternatives to average.
    """
    rows: dict[tuple[str, str], dict] = {}
    out: list[dict] = []
    for m in RESULT_RE.finditer(text):
        label = m.group("label")
        arm, est = split_label(label)
        row = {
            "label": label,
            "arm": arm,
            "estimator": est,
            "variant": m.group("variant"),
            "r_work": float(m.group("r_work")),
            "r_free": float(m.group("r_free")),
        }
        if last_only:
            rows[(label, row["variant"])] = row
        else:
            out.append(row)
    return list(rows.values()) if last_only else out


def parse_peaks_lines(text: str, last_only: bool = True) -> list[dict]:
    """Every `PEAKS` line in `text`, as rows carrying the height list."""
    rows: dict[tuple[str, str], dict] = {}
    out: list[dict] = []
    for m in PEAKS_RE.finditer(text):
        label = m.group("label")
        arm, est = split_label(label)
        heights = [float(v) for v in m.group("peaks").split(",") if v != ""]
        row = {
            "label": label,
            "arm": arm,
            "estimator": est,
            "variant": m.group("variant"),
            "peaks": heights,
        }
        if last_only:
            rows[(label, row["variant"])] = row
        else:
            out.append(row)
    return list(rows.values()) if last_only else out


# --------------------------------------------------------------------------- #
# parsing the tools' own output
# --------------------------------------------------------------------------- #
def refinement_from_phenix_log(path) -> dict | None:
    """`r_work` and `r_free` of the FINAL cycle in a phenix refine log.

    phenix prints `Start R-work = ...` first; taking the first match in the
    file reports the model it started from.

    Several paths may be given, as the chain passes `<out>/*.log`: they are
    read in the order given and the last `Final` line across all of them
    wins, which is what `grep ... *.log | tail -1` did.
    """
    paths = [path] if isinstance(path, str | Path) else list(path)
    hits, texts = [], []
    for p in paths:
        p = Path(p)
        if not p.is_file():
            continue
        text = p.read_text(errors="replace")
        texts.append(text)
        hits += PHENIX_FINAL_RE.findall(text)
    if hits:
        r_work, r_free = hits[-1]
        return {"r_work": float(r_work), "r_free": float(r_free)}
    # a stats file rather than a refine log
    for text in texts:
        mw, mf = STATS_RWORK_RE.search(text), STATS_RFREE_RE.search(text)
        if mw and mf:
            return {
                "r_work": float(mw.group(1)), "r_free": float(mf.group(1))
            }
    return None


_PHENIX_BIN_RE = re.compile(
    r"^\|\s*(\d+):\s+([\d.]+)\s+-\s+([\d.]+)\s+((?:[-\d.]+\s+){6}[-\d.]+)\s*\|\s*$")
_PHENIX_STEP_RE = re.compile(
    r"^\s*(\S+)\s*:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s+([\d.]+)\s+([\d.]+)"
    r"\s+(\d+)\s+([\d.]+)\s*$")


def refinement_shells_from_phenix_log(path) -> dict[str, list[dict]]:
    """The per-shell tables of the FINAL statistics in a phenix.refine log.

    Two tables, each the last occurrence in the file (phenix prints them for the
    start and the end of refinement):

    `rfactors`: shell, d_max, d_min, completeness, n_work, n_free, r_work,
    r_free, target_work, target_free -- from the table headed `R-factors`.
    `fom`: shell, d_max, d_min, n_work, n_free, fom, phase_error, scale,
    alpha, beta -- from the table headed `figures of merit` (the R-free
    likelihood estimates, Acta Cryst. (1995). A51, 880-887).

    Both tables have ten numbers per row, so they are told apart by the header
    above them, not by their shape.
    """
    text = Path(path).read_text(errors="replace")
    tables: dict[str, list[list[dict]]] = {"rfactors": [], "fom": []}
    current = None
    keys = {"rfactors": ("completeness", "n_work", "n_free", "r_work", "r_free", "target_work", "target_free"),
            "fom": ("n_work", "n_free", "fom", "phase_error", "scale", "alpha", "beta")}
    for line in text.splitlines():
        if "R-factors" in line and "Resolution" in line:
            current = "rfactors"
            tables[current].append([])
            continue
        if "figures of merit" in line:
            current = "fom"
            tables[current].append([])
            continue
        m = _PHENIX_BIN_RE.match(line)
        if m and current is not None:
            vals = [float(v) for v in m.group(4).split()]
            row = {"shell": int(m.group(1)), "d_max": float(m.group(2)), "d_min": float(m.group(3))}
            row.update(dict(zip(keys[current], vals)))
            for k in ("n_work", "n_free"):
                row[k] = int(row[k])
            tables[current][-1].append(row)
        elif current is not None and tables[current] and tables[current][-1] and line.startswith("|---"):
            current = None
    return {k: (v[-1] if v else []) for k, v in tables.items()}


def refinement_steps_from_phenix_log(path) -> list[dict]:
    """The `REFINEMENT STATISTICS STEP BY STEP` table: stage, r_work, r_free,
    bonds, angles, b_min, b_max, b_ave, n_water, shift, in the order printed."""
    text = Path(path).read_text(errors="replace")
    # anchored on the column header, which every version prints, not on the banner
    if "stage r-work r-free" not in text:
        return []
    block = text[text.rindex("stage r-work r-free"):]
    out = []
    names = ("r_work", "r_free", "bonds", "angles", "b_min", "b_max", "b_ave", "n_water", "shift")
    for line in block.splitlines():
        m = _PHENIX_STEP_RE.match(line)
        if m:
            vals = [float(v) for v in m.groups()[1:]]
            row = {"stage": m.group(1), **dict(zip(names, vals))}
            row["n_water"] = int(row["n_water"])
            out.append(row)
        elif out and line.startswith("----"):
            break
    return out


def shell_midpoint(d_max, d_min) -> float:
    """The resolution at the middle of a shell, in 1/d^2.

    Shells are cut on 1/d^2, so the arithmetic mean of d_max and d_min is
    not the middle of the shell and would put every point slightly wrong on
    a resolution axis.
    """
    try:
        s_hi, s_lo = 1.0 / float(d_max) ** 2, 1.0 / float(d_min) ** 2
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")
    return float(1.0 / ((s_hi + s_lo) / 2.0) ** 0.5)


def _shell_label(d_max, d_min) -> str:
    return f"{float(d_max):.2f} - {float(d_min):.2f}"


def d_range_from_label(label) -> tuple[float, float] | None:
    """`"79.63 - 2.97"` -> (79.63, 2.97); None when it is not a range.

    DIALS, careless and our own CSVs all label a shell this way, so a figure
    that has only the label can still put the points at the resolution they
    were measured at rather than at a shell index.
    """
    nums = re.findall(r"[0-9]*\.?[0-9]+", str(label))
    if len(nums) < 2:
        return None
    return float(nums[0]), float(nums[1])


# the name this had while it was private
_d_from_label = d_range_from_label


def shells_from_merging_stats_csv(path) -> list[dict] | None:
    """The canonical per-shell CSV both arms emit, as rows.

    Returns the same keys `shells_from_merge_log` does, so a figure does not
    care which of the three sources a run was scored through.
    """
    rows: list[dict] = []
    try:
        with open(path, newline="") as fh:
            records = list(csv.DictReader(fh))
    except OSError:
        return None
    if not records:
        return None
    if not {"d_max", "d_min"} <= set(records[0]):
        return None
    for rec in records:
        row: dict = {}
        for key, raw in rec.items():
            if key is None:
                continue
            value = (raw or "").strip()
            try:
                row[key] = float(value.rstrip("%*"))
            except ValueError:
                row[key] = value
        row["resolution"] = _shell_label(row["d_max"], row["d_min"])
        row["d"] = shell_midpoint(row["d_max"], row["d_min"])
        rows.append(row)
    return rows


class _TableReader(HTMLParser):
    """Every `<table>` in a document, as a header row and body rows.

    The stdlib parser rather than `pandas.read_html`, which needs `lxml`:
    without it `read_html` raises `ImportError`, this function returned None
    for every real `merged.html`, and the DIALS reference line was missing
    from the merging figures with only a warning to show for it.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._rows: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._rows = []
        elif tag == "tr" and self._rows is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self._rows.append(self._row)
            self._row = None
        elif tag == "table" and self._rows is not None:
            if self._rows:
                self.tables.append(self._rows)
            self._rows = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_tables(text: str) -> list[list[list[str]]]:
    """`[[cells of row] for row in table]` for each table in `text`."""
    reader = _TableReader()
    reader.feed(text)
    reader.close()
    return reader.tables


def shells_from_merged_html(path) -> list[dict] | None:
    """A DIALS `merged.html` resolution-bin table, as the same rows.

    `merged.html` holds two tables: a summary with Overall/Low/High columns
    and the per-bin table. The one with a resolution column and a CC column
    is the per-bin table, whichever order DIALS wrote them in.
    """
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    table = None
    for rows in html_tables(text):
        header = [c.lower() for c in rows[0]]
        if any("resolution" in c for c in header) and any(
            "cc" in c for c in header
        ):
            table = rows
            break
    if table is None:
        return None
    header, body = table[0], table[1:]

    def column_for(*spellings):
        """The index of the first header matching any spelling."""
        for subs in spellings:
            for i, name in enumerate(header):
                s = name.lower()
                if all(sub.lower() in s for sub in subs):
                    return i
        return None

    res_col = column_for(("resolution",))
    if res_col is None:
        return None
    found = {k: column_for(*v) for k, v in _HTML_COLUMNS.items()}

    rows: list[dict] = []
    for rec in body:
        if res_col >= len(rec):
            continue
        label = rec[res_col]
        # the table ends with an `Overall` row, which is not a shell
        if label.strip().lower().startswith("overall"):
            continue
        row: dict = {"resolution": label}
        span = _d_from_label(label)
        if span:
            row["d_max"], row["d_min"] = span
            row["d"] = shell_midpoint(*span)
        for key, col in found.items():
            raw = rec[col] if col is not None and col < len(rec) else ""
            # DIALS marks a significant correlation with a trailing `*` and
            # a completeness with `%`; as text either one sorts as text
            try:
                row[key] = float(raw.strip().rstrip("%*"))
            except (AttributeError, ValueError):
                row[key] = float("nan")
        rows.append(row)
    return rows or None


def shells_from_any(path) -> list[dict] | None:
    """Per-shell rows from whichever of the three forms `path` is."""
    p = Path(path)
    if p.suffix == ".csv":
        return shells_from_merging_stats_csv(p)
    if p.suffix in (".html", ".htm"):
        return shells_from_merged_html(p)
    return shells_from_merge_log(p) or None


# The columns `rs.find_peaks` writes. `name` is the nearest atom's name and
# is the one that matters for a panel title -- a sulfur peak on a methionine
# is SD and on a cysteine is SG -- but the fork writes it empty, so it is
# resolved against the refined model instead (`nearest_atoms` below).
FIND_PEAKS_COLUMNS = (
    "chain", "seqid", "residue", "name", "dist", "peak", "peakz",
    "score", "scorez", "cenx", "ceny", "cenz", "coordx", "coordy", "coordz",
)

# The peak's own position. `cen*` is written wrong by the fork in use --
# `IOD 206` comes out with cenx = ceny = cenz = 27.69 -- so the orthogonal
# coordinates are the ones to read.
PEAK_POSITION_COLUMNS = ("coordx", "coordy", "coordz")


def site_label(residue, seqid, atom: str = "", chain: str = "") -> str:
    """How one anomalous site is named: `IOD 204 I`, `MET 105 SD`.

    The atom name is appended when it is known, because the residue alone
    does not say which atom carries the peak: a methionine sulfur is SD and
    a cysteine sulfur is SG. The chain is prefixed only when it is not the
    only chain, which the caller decides -- passing it always would put `A`
    in front of every title in a single-chain structure.
    """
    parts = [str(residue).strip(), str(seqid).strip()]
    if atom:
        parts.append(str(atom).strip())
    label = " ".join(p for p in parts if p)
    chain = str(chain).strip()
    return f"{chain}/{label}" if chain else label


def panel_site_label(
    residue, seqid, atom: str = "", n_candidates: int = 1
) -> str:
    """How one site's figure panel is titled: `IOD 204`, or `CYS 30 SG`
    only where the residue holds more than one atom that could carry the
    peak.

    The anomalous atom is implied by the residue in this experiment -- an
    iodide has one iodine, a cysteine and a methionine one sulfur each, a
    sulfate one sulfur -- so naming it in a title repeats the residue name.
    `n_candidates` comes from `heavy_atom_candidates`, and the atom is said
    where the residue does not imply it.

    `site_label` is the other rule and keeps the atom unconditionally: it
    names the CSV column, which is read by name rather than by eye, and a
    column that sometimes carries the atom could not be grouped on.
    """
    if atom and n_candidates > 1:
        return site_label(residue, seqid, atom)
    return site_label(residue, seqid)


def model_atoms(path) -> list[dict]:
    """Every atom in a refined model: chain, residue, atom name, element.

    Read with gemmi, which `reciprocalspaceship` already brings in. The
    element comes from the file rather than from the residue name, so a
    sulfate sulfur and a cysteine sulfur are both reported as S without a
    table mapping residues to elements.
    """
    import gemmi

    st = gemmi.read_structure(str(path))
    st.setup_entities()
    out = []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                out.append({
                    "chain": chain.name,
                    "seqid": res.seqid.num,
                    "residue": res.name,
                    "atom": atom.name,
                    "element": atom.element.name,
                    "x": atom.pos.x,
                    "y": atom.pos.y,
                    "z": atom.pos.z,
                })
    return out


def nearest_atoms(path, positions, max_dist: float = 2.0) -> list[dict | None]:
    """The nearest atom in a refined model to each orthogonal position.

    Symmetry images are included: a peak can sit on a symmetry copy of an
    atom whose deposited coordinates are elsewhere in the cell, and a search
    over the asymmetric unit alone would then return an atom several
    angstroms away and name the site wrongly.

    Hydrogens are left out of the search. A phenix model carries riding
    hydrogens, and one of them is routinely the atom nearest a peak -- the
    HEWL 9b7c peak list names `ILE 88 HG12` at 0.03 A -- but a hydrogen
    cannot carry an anomalous peak, so naming a site after one is wrong in
    a way that reaches a figure title.

    Returns one record per position -- chain, seqid, residue, atom, element
    and `dist` -- or None where nothing lies within `max_dist`.
    """
    import gemmi

    st = gemmi.read_structure(str(path))
    st.setup_entities()
    model = st[0]
    search = gemmi.NeighborSearch(model, st.cell, max(max_dist, 5.0))
    search.populate(include_h=False)
    out: list[dict | None] = []
    for x, y, z in positions:
        pos = gemmi.Position(float(x), float(y), float(z))
        mark = search.find_nearest_atom(pos)
        if mark is None:
            out.append(None)
            continue
        dist = mark.pos.dist(pos)
        if dist > max_dist:
            out.append(None)
            continue
        cra = mark.to_cra(model)
        out.append({
            "chain": cra.chain.name,
            "seqid": cra.residue.seqid.num,
            "residue": cra.residue.name,
            "atom": cra.atom.name,
            "element": cra.atom.element.name,
            "dist": dist,
        })
    return out


# Above oxygen. Sulfur and heavier carry the anomalous signal a difference
# map is made from -- $f''$ is about 0.56 e for S and 6.8 e for I against
# 0.03 e for O -- so phosphorus, sulfur, chlorine and the halides count and
# carbon, nitrogen and oxygen do not.
ANOMALOUS_MIN_Z = 9


def heavy_atom_candidates(path, min_z: int = ANOMALOUS_MIN_Z) -> dict:
    """The atoms in each residue that could carry an anomalous peak.

    Keyed the same two ways as `heaviest_atoms`, with and without the chain,
    and holding a sorted list of atom names.

    This is what decides whether a site has to be named with its atom. A
    cysteine, a methionine, an iodide and a sulfate each hold exactly one
    candidate, so `CYS 30` is unambiguous and the SG adds nothing; a residue
    holding two is not, and there the atom has to be said.
    """
    import gemmi

    out: dict = {}
    for atom in model_atoms(path):
        if gemmi.Element(atom["element"]).atomic_number < min_z:
            continue
        for key in (
            (atom["chain"], atom["seqid"], atom["residue"]),
            (atom["seqid"], atom["residue"]),
        ):
            out.setdefault(key, [])
            if atom["atom"] not in out[key]:
                out[key].append(atom["atom"])
    return {k: sorted(v) for k, v in out.items()}


def heaviest_atoms(path) -> dict[tuple, dict]:
    """The heaviest atom of each residue in a refined model, by residue key.

    Keyed by `(chain, seqid, residue)` and also by `(seqid, residue)`, since
    the older `peak_heights.csv` names a site without its chain.

    This is the fallback for a peak height that arrives without a position.
    It is not a nearest-atom measurement: it is the atom of greatest atomic
    number in the residue the peak search named, which is the atom that
    scatters anomalously there -- SG in a cysteine, SD in a methionine, I in
    an iodide. Where the peak's own position is on hand, `nearest_atoms` is
    the one to use, and a row says which of the two named it.
    """
    import gemmi

    best: dict[tuple, dict] = {}
    for atom in model_atoms(path):
        z = gemmi.Element(atom["element"]).atomic_number
        for key in (
            (atom["chain"], atom["seqid"], atom["residue"]),
            (atom["seqid"], atom["residue"]),
        ):
            prev = best.get(key)
            if prev is None or z > prev["z"]:
                best[key] = {**atom, "z": z}
    return best


def peaks_from_find_peaks_csv(path, model=None) -> list[dict]:
    """Sites and peak heights from the table `rs.find_peaks` writes.

    One row per site, with `seqid`, `residue` and `peakz` -- the height in
    standard deviations, which is the number reported.

    `model` is the refined PDB or mmCIF the peak search ran against. Given
    one, the atom carrying each peak is resolved from it and reported as
    `atom` and `element`; the fork of `rs.find_peaks` in use leaves its own
    `name` column empty. The model's assignment is accepted only where its
    chain, residue number and residue name agree with what the peak search
    already reported, so a disagreement leaves `atom` empty and is counted
    in `atom_from_position` rather than overwriting the search's own answer.
    """
    out = []
    with open(path, newline="") as fh:
        for rec in csv.DictReader(fh):
            if not rec.get("peakz"):
                continue
            try:
                peakz = float(rec["peakz"])
            except ValueError:
                continue
            residue = (rec.get("residue") or "").strip()
            seqid = (rec.get("seqid") or "").strip()
            atom = (rec.get("name") or "").strip()
            row = {
                "site": f"{residue}_{seqid}" if residue and seqid else seqid,
                "residue": residue,
                "seqid": int(float(seqid)) if seqid else None,
                "chain": (rec.get("chain") or "").strip(),
                "atom": atom,
                "element": "",
                "peakz": peakz,
                "anomalous": residue in ANOMALOUS_RESIDUES,
                "atom_from_position": False,
            }
            for name, key in zip(
                PEAK_POSITION_COLUMNS, ("x", "y", "z"), strict=True
            ):
                try:
                    row[key] = float(rec[name])
                except (KeyError, TypeError, ValueError):
                    row[key] = None
            row["dist"] = _maybe_float(rec.get("dist"))
            row["label"] = site_label(residue, seqid, atom)
            out.append(row)

    if model is None or not out:
        return out

    want = [r for r in out if None not in (r["x"], r["y"], r["z"])]
    found = nearest_atoms(model, [(r["x"], r["y"], r["z"]) for r in want])
    for row, near in zip(want, found, strict=True):
        if near is None:
            continue
        if row["chain"] and near["chain"] != row["chain"]:
            continue
        if row["seqid"] is not None and near["seqid"] != row["seqid"]:
            continue
        if row["residue"] and near["residue"] != row["residue"]:
            continue
        row["atom"] = near["atom"]
        row["element"] = near["element"]
        row["dist_to_atom"] = near["dist"]
        row["atom_from_position"] = True
        row["label"] = site_label(row["residue"], row["seqid"], near["atom"])
    return out


def _maybe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def peak_heights_from_csv(path, model=None) -> list[dict]:
    """Sites and heights from the two-row `peak_heights.csv`.

    That file is transposed: the header names the sites (`CYS 6,MET 12,...`)
    and the single data row holds their heights. It is the form the earlier
    laue-dials runs used, kept for comparability.

    It carries no chain and no peak position, so `model` -- the refined PDB
    -- fills `atom` and `element` with the heaviest atom of the named
    residue rather than with the atom nearest the peak. `atom_from_position`
    is False on every row it fills, which is how a table built from this
    file is told apart from one built from `peaks.csv`.
    """
    with open(path, newline="") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    if len(lines) < 2:
        return []
    sites = next(csv.reader([lines[0]]))
    values = next(csv.reader([lines[-1]]))
    heaviest = heaviest_atoms(model) if model is not None else {}
    out = []
    for site, val in zip(sites, values, strict=False):
        try:
            height = float(val)
        except ValueError:
            continue
        fields = site.split()
        residue = fields[0] if fields else ""
        seqid = None
        if len(fields) > 1:
            try:
                seqid = int(float(fields[1]))
            except ValueError:
                seqid = None
        near = heaviest.get((seqid, residue)) if seqid is not None else None
        atom = near["atom"] if near else ""
        out.append({
            # `label` is the header field exactly as written, `site` the
            # same without the space. Both are kept because tables already
            # written carry the spaced form and must keep sorting beside
            # tables written now.
            "label": site_label(residue, seqid if seqid is not None else "",
                                atom),
            "site": site.replace(" ", "_"),
            "residue": residue,
            "seqid": seqid,
            "chain": near["chain"] if near else "",
            "atom": atom,
            "element": near["element"] if near else "",
            "peakz": height,
            "anomalous": residue in ANOMALOUS_RESIDUES,
            "atom_from_position": False,
        })
    return out


def shells_from_merge_log(path) -> list[dict]:
    """The per-resolution-shell table `dials.merge` prints.

    `dials.merge` marks a statistically significant correlation with a
    trailing `*` (`1.000*`), and completeness with a `%`, so every field is
    stripped of both before it is read as a number -- without that, `cc_half`
    and `cc_anom` come back as strings and sort as text.
    """
    rows: list[dict] = []
    seen_header = False
    with open(path, errors="replace") as fh:
        for line in fh:
            if "d_max" in line and "cc1/2" in line:
                seen_header = True
                continue
            if not seen_header:
                continue
            fields = line.split()
            if len(fields) != len(SHELL_COLUMNS):
                if rows:
                    break
                continue
            try:
                float(fields[0])
            except ValueError:
                break
            row: dict = {}
            significant = {}
            for name, raw in zip(SHELL_COLUMNS, fields, strict=True):
                significant[name] = raw.endswith("*")
                try:
                    row[name] = float(raw.rstrip("%*"))
                except ValueError:
                    row[name] = raw
            row["cc_half_significant"] = significant["cc_half"]
            row["cc_anom_significant"] = significant["cc_anom"]
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# command line: the chain emits through this, so the format has one definition
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Emit or read the chain's metrics")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("emit-result", help="print one RESULT line")
    e.add_argument("--label", required=True, help="e.g. m9_geomshift_rb")
    e.add_argument("--variant", required=True, choices=VARIANTS)
    e.add_argument("--r-work", type=float, required=True)
    e.add_argument("--r-free", type=float, required=True)

    k = sub.add_parser("emit-peaks", help="print one PEAKS line")
    k.add_argument("--label", required=True)
    k.add_argument("--variant", required=True, choices=VARIANTS)
    k.add_argument("--peaks", required=True, help="comma-separated heights")

    f = sub.add_parser(
        "from-phenix", help="emit both lines from a phenix log and a peaks CSV"
    )
    f.add_argument("--label", required=True,
                   help="without the variant suffix, e.g. m9_geomshift_rb")
    f.add_argument("--variant", required=True, choices=VARIANTS)
    f.add_argument("--log", type=Path, required=True, nargs="+",
                   help="phenix log(s); the chain passes <out>/*.log")
    f.add_argument("--peaks-csv", type=Path, default=None)

    args = p.parse_args(argv)

    if args.cmd == "emit-result":
        print(emit_result(args.label, args.variant, args.r_work, args.r_free))
        return 0
    if args.cmd == "emit-peaks":
        peaks = [v for v in args.peaks.split(",") if v != ""]
        print(emit_peaks(args.label, args.variant, peaks))
        return 0

    got = refinement_from_phenix_log(args.log)
    if got is None:
        print(f"no Final R-work line in {args.log}", file=sys.stderr)
        return 1
    print(emit_result(args.label, args.variant, got["r_work"], got["r_free"]))
    if args.peaks_csv and args.peaks_csv.is_file():
        # the CSV's own text, not a reparsed float: this replaces a shell
        # line that echoed `tail -1` of the file, and the two have to agree
        # digit for digit or every stored peak height shifts
        lines = [
            ln.strip() for ln in args.peaks_csv.read_text().splitlines()
            if ln.strip()
        ]
        if len(lines) >= 2:
            print(emit_peaks(args.label, args.variant, lines[-1].split(",")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
