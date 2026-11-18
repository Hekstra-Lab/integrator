"""The tables DIALS prints into `scale.log` and `merge.log`.

`dials.scale` and `dials.merge` report a good deal that never reaches
`merging_stats.csv`: the scaling model's parameter counts, the error model
`a` and `b`, the uncorrected and corrected variances by intensity range, the
three-column merging summary with every R-factor, the Wilson analysis, the
size of the anomalous differences by shell and the ice-ring check. All of it
is already computed; it was being read by eye.

Two shapes cover almost every table in both logs:

    +--------------+----------------+        a `tabulate` grid, from
    | correction   |   n_parameters |        dials.scale
    |--------------+----------------|
    | scale        |             50 |
    +--------------+----------------+

      -------------------------------------------------
      | Eigenvector | Value   | Vector                |   an indented box,
      -------------------------------------------------   from the cctbx
      | 1           |  17.950 | ( 0.00,  0.00,  1.00) |   Wilson analysis
      -------------------------------------------------

so `log_tables` reads both into the same `LogTable`, and the quantities worth
naming get a named reader on top of it. `integrator.utils.metrics` re-exports
all of this, so a caller still has one import.

The third shape is the merging summary, which is columns of fixed width with
no rules at all, and has its own reader.

    python -m integrator.utils.dials_log scale.log            # what is in it
    python -m integrator.utils.dials_log merge.log --summary  # the R-factors
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "LogTable",
    "MERGING_SUMMARY_KEYS",
    "anomalous_differences_from_log",
    "find_table",
    "error_model_from_log",
    "log_tables",
    "merging_summary_from_log",
    "scaling_summary_from_log",
    "wilson_from_log",
]

# A row of cells: `| a | b |`, with any indent.
_CELL = re.compile(r"^\s*\|(?P<body>.*)\|\s*$")
# A rule: `+-----+-----+`, `|-----+-----|`, or a bare `-----` box edge.
_RULE = re.compile(r"^\s*(?:[+|][-+|]*[-+|]|-{3,})\s*$")
# A cell holding only a unit, which belongs to the header above it.
_UNIT = re.compile(r"^\(.*\)$")
# `label   1.10   2.97   1.10` in the merging summary.
_SUMMARY_ROW = re.compile(
    r"^(?P<label>\S.*?)\s{2,}(?P<nums>-?[\d.]+(?:\s+-?[\d.]+)*)\s*$"
)


@dataclass
class LogTable:
    """One table read out of a DIALS log."""

    headers: list[str]
    rows: list[list[str]]
    title: str | None = None
    units: dict[str, str] = field(default_factory=dict)

    def records(self) -> list[dict[str, str]]:
        """Rows as dicts keyed by header, short rows padded with ``""``."""
        out = []
        for row in self.rows:
            padded = list(row) + [""] * (len(self.headers) - len(row))
            out.append(dict(zip(self.headers, padded, strict=False)))
        return out

    def numeric(self, *columns: str) -> list[dict[str, float | str]]:
        """`records()` with the named columns coerced to float.

        With no columns named, every cell that parses as a number is
        converted and the rest are left as text.
        """
        want = set(columns) if columns else None
        out: list[dict[str, float | str]] = []
        for rec in self.records():
            row: dict[str, float | str] = {}
            for key, raw in rec.items():
                if want is not None and key not in want:
                    row[key] = raw
                    continue
                row[key] = _number(raw, default=raw if want is None else float("nan"))
            out.append(row)
        return out

    def __len__(self) -> int:
        return len(self.rows)


def _number(raw, default=float("nan")):
    """`"1.000*"` -> 1.0. DIALS marks significance with `*`, units with `%`."""
    try:
        return float(str(raw).strip().rstrip("%*").replace(",", ""))
    except (AttributeError, ValueError):
        return default


def _split(line: str) -> list[str]:
    body = _CELL.match(line).group("body")
    return [c.strip() for c in body.split("|")]


def _read(source) -> str:
    """A path or the text itself."""
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8", errors="replace")
    text = str(source)
    if "\n" in text:
        return text
    p = Path(text)
    return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else text


def log_tables(source) -> list[LogTable]:
    """Every rule-delimited table in a DIALS log, in the order printed.

    A caption is taken from a single-cell row inside the box (which is how
    the cctbx Wilson analysis labels its tables) or, failing that, from the
    last line of text before it, which is how dials.scale does.
    """
    lines = _read(source).splitlines()
    tables: list[LogTable] = []
    i = 0
    while i < len(lines):
        if not (_CELL.match(lines[i]) or _RULE.match(lines[i])):
            i += 1
            continue
        start = i
        cells: list[list[str]] = []
        while i < len(lines) and (_CELL.match(lines[i]) or _RULE.match(lines[i])):
            if _CELL.match(lines[i]) and not _RULE.match(lines[i]):
                cells.append(_split(lines[i]))
            i += 1
        if len(cells) < 2:
            continue

        title = None
        # a one-cell row above a wider one is the table's own caption
        if len(cells[0]) == 1 and len(cells) > 2 and len(cells[1]) > 1:
            title = cells[0][0]
            cells = cells[1:]
        if title is None:
            title = _title_above(lines, start)

        tables.extend(_split_block(cells, title))
    return tables


def _split_block(cells: list[list[str]], title: str | None) -> list[LogTable]:
    """One run of cell lines, which may hold more than one table.

    dials.merge prints the screw-axis table and the space-group table with
    no blank line between them, so a row whose width does not match the
    current header starts a new table rather than being a short row of this
    one.
    """
    out: list[LogTable] = []
    headers: list[str] = []
    body: list[list[str]] = []
    units: dict[str, str] = {}

    def flush():
        if headers and body:
            out.append(LogTable(
                headers=list(headers), rows=list(body),
                title=title if not out else None, units=dict(units),
            ))

    for row in cells:
        if not headers:
            headers, body, units = row, [], {}
            continue
        if len(row) == len(headers):
            # a second header line carrying only units: `|  | (a.u) |`
            if not body and all(c == "" or _UNIT.match(c) for c in row):
                units = {h: c.strip("()") for h, c in
                         zip(headers, row, strict=False) if c}
                continue
            body.append(row)
            continue
        if not body and len(row) < len(headers):
            # The header was split on a literal `|` inside a column name:
            # `|  d_max |  d_min |  <|ΔF|/σ(ΔF)> |` is five cells where the
            # data rows are three. Put the surplus back together. A genuine
            # second table instead arrives after this one has rows, which is
            # the case below.
            keep = len(row)
            headers = headers[: keep - 1] + ["|".join(headers[keep - 1:])]
            body.append(row)
            continue
        flush()
        headers, body, units = row, [], {}
    flush()
    return out


def _title_above(lines: list[str], start: int) -> str | None:
    """The caption dials.scale prints on the line before a table."""
    for j in range(start - 1, max(start - 3, -1), -1):
        text = lines[j].strip()
        if not text:
            continue
        if _CELL.match(lines[j]) or _RULE.match(lines[j]):
            return None
        if len(text) > 80 or text.endswith((".", ",")):
            return None
        # `Scoring method: direct` is a status line, not a table name
        if ": " in text:
            return None
        return text.rstrip(":").strip()
    return None


def find_table(source, *words: str, last: bool = False) -> LogTable | None:
    """A table whose caption or headers contain all of `words`.

    The first match, or with `last` the final one: dials.scale prints
    `Refinement steps` once per cycle and only the last one ends at the
    RMSD the run finished with.
    """
    tables = source if isinstance(source, list) else log_tables(source)
    found = [
        t for t in tables
        if all(w.lower() in " ".join([t.title or "", *t.headers]).lower()
               for w in words)
    ]
    if not found:
        return None
    return found[-1] if last else found[0]


# The merging summary, named by what each number is rather than by the label
# DIALS prints. `(I+/-)` is the statistic computed on the anomalous pairs
# kept apart, which is a different number from the same statistic on the
# merged intensities and is why both are reported.
MERGING_SUMMARY_KEYS = {
    "high resolution limit": "d_min",
    "low resolution limit": "d_max",
    "completeness": "completeness",
    "multiplicity": "multiplicity",
    "i/sigma": "i_over_sigma",
    "rmerge(i)": "r_merge",
    "rmerge(i+/-)": "r_merge_anom",
    "rmeas(i)": "r_meas",
    "rmeas(i+/-)": "r_meas_anom",
    "rpim(i)": "r_pim",
    "rpim(i+/-)": "r_pim_anom",
    "cc half": "cc_half",
    "wilson b factor": "wilson_b",
    "anomalous completeness": "anom_completeness",
    "anomalous multiplicity": "anom_multiplicity",
    "anomalous correlation": "cc_anom",
    "anomalous slope": "anom_slope",
    "df/f": "df_over_f",
    "di/s(di)": "di_over_sigma_di",
    "total observations": "n_obs",
    "total unique": "n_unique",
}
_SUMMARY_SCOPES = ("overall", "low", "high")


def merging_summary_from_log(path) -> dict[str, dict[str, float]] | None:
    """The `Summary of merging statistics` block, by scope.

    `{"overall": {...}, "low": {...}, "high": {...}}`, keyed by the names in
    `MERGING_SUMMARY_KEYS`. A row DIALS prints once (the Wilson B factor,
    the anomalous slope, dF/F) appears under `overall` only.

    This is where the R-factors other than `r_pim` live: Rmerge and Rmeas,
    each both merged and on the anomalous pairs kept apart. The per-shell
    table has `r_merge`, `r_meas` and `r_pim` but not the `(I+/-)` forms.
    """
    text = _read(Path(path) if not isinstance(path, str) or "\n" not in str(path) else path)
    marker = text.rfind("Summary of merging statistics")
    if marker < 0:
        return None
    out: dict[str, dict[str, float]] = {s: {} for s in _SUMMARY_SCOPES}
    blanks = 0
    for line in text[marker:].splitlines()[1:]:
        if not line.strip():
            blanks += 1
            if blanks >= 3 and any(out.values()):
                break
            continue
        m = _SUMMARY_ROW.match(line)
        if not m:
            if any(out.values()):
                break
            continue
        key = MERGING_SUMMARY_KEYS.get(m.group("label").strip().lower())
        if key is None:
            continue
        blanks = 0
        for scope, raw in zip(_SUMMARY_SCOPES, m.group("nums").split(), strict=False):
            out[scope][key] = _number(raw)
    return out if any(out.values()) else None


def error_model_from_log(path) -> dict[str, float] | None:
    """The scaling error model's `a` and `b`, and what they imply.

    `sigma'^2 = a^2 (sigma^2 + (b I)^2)`, so `a` inflates every uncertainty
    and `b` sets the asymptotic `I/sigma` a bright reflection can reach.
    The log prints the model once as loaded and again after each refinement
    round; the last is the one the reflection table was written with.
    """
    text = _read(Path(path))
    hits = list(re.finditer(
        r"Parameters:\s*a\s*=\s*([\d.]+),\s*b\s*=\s*([\d.]+)", text))
    if not hits:
        return None
    a, b = float(hits[-1].group(1)), float(hits[-1].group(2))
    out = {"a": a, "b": b}
    limits = re.findall(
        r"estimated I/sigma asymptotic limit:\s*([\d.]+)", text)
    if limits:
        out["i_over_sigma_limit"] = float(limits[-1])
    return out


def scaling_summary_from_log(path) -> dict[str, float] | None:
    """What `dials.scale` did: parameters, reflections and outliers.

    Counts rather than quality: how many reflections were read, how many
    the scaling model had parameters for, and how many were rejected. A
    model comparison that moves these has changed the data, not just the
    scale, so they belong beside the R-factors.
    """
    text = _read(Path(path))
    out: dict[str, float] = {}
    read = re.search(r"Read (\d+) predicted reflections", text)
    if read:
        out["n_read"] = float(read.group(1))
    tables = log_tables(text)
    corrections = find_table(tables, "correction")
    if corrections:
        for rec in corrections.records():
            name = rec.get("correction", "").strip().lower()
            if name:
                out[f"n_param_{name}"] = _number(rec.get("n_parameters"))
        out["n_param_total"] = sum(
            v for k, v in out.items() if k.startswith("n_param_")
        )
    outliers = re.findall(r"(\d+) outliers have been identified", text)
    if outliers:
        out["n_outliers"] = float(outliers[-1])
        out["n_outliers_total"] = float(sum(int(x) for x in outliers))
    partiality = find_table(tables, "partiality")
    if partiality:
        for rec in partiality.records():
            label = rec.get("Partiality (p)", "").strip().lower()
            if label == "all reflections":
                out["n_refl"] = _number(rec.get("n_refl"))
    rmsd = find_table(tables, "rmsd_i", last=True)
    if rmsd and len(rmsd):
        out["rmsd_i_final"] = _number(rmsd.rows[-1][-1])
    uncertain = re.search(
        r"([\d.]+)% of model parameters have significant uncertainty", text)
    if uncertain:
        out["pct_param_uncertain"] = float(uncertain.group(1))
    return out or None


def wilson_from_log(path) -> dict[str, float] | None:
    """The cctbx Wilson analysis `dials.merge` prints.

    The isotropic B and the scale factor come from the maximum-likelihood
    fit to the Wilson plot; `b_cart_*` are the eigenvalues of the
    anisotropic B tensor, largest first, so `b_aniso_range` is how much the
    falloff differs between the best and worst direction.
    """
    text = _read(Path(path))
    out: dict[str, float] = {}
    iso = re.search(
        r"ML estimate of overall B value:\s*\n\s*([\d.]+)\s*A\*\*2", text)
    if iso:
        out["wilson_b"] = float(iso.group(1))
    scale = re.search(
        r"Estimated -log of scale factor:\s*\n\s*(-?[\d.]+)", text)
    if scale:
        out["neg_log_scale"] = float(scale.group(1))
    eigen = find_table(text, "eigenvector")
    if eigen:
        values = sorted(
            (_number(rec.get("Value")) for rec in eigen.records()),
            reverse=True,
        )
        for i, v in enumerate(values, start=1):
            out[f"b_cart_{i}"] = v
        if len(values) >= 2:
            out["b_aniso_range"] = values[0] - values[-1]
    return out or None


def anomalous_differences_from_log(path) -> list[dict] | None:
    """`<|dF|/sigma(dF)>` by resolution shell, from `dials.merge`.

    The mean anomalous difference in units of its own uncertainty. It falls
    below 1 where the differences stop being measurable, which is a
    resolution cut chosen by the data rather than by the metric under test.
    """
    table = find_table(_read(Path(path)), "d_max", "d_min")
    if table is None:
        return None
    key = next(
        (h for h in table.headers if "F" in h and "σ" in h or "sigma" in h.lower()),
        None,
    )
    if key is None:
        return None
    rows = []
    for rec in table.records():
        d_max, d_min = _number(rec.get("d_max")), _number(rec.get("d_min"))
        rows.append({
            "d_max": d_max,
            "d_min": d_min,
            "d": _midpoint(d_max, d_min),
            "dano_over_sigma": _number(rec.get(key)),
        })
    return rows or None


def _midpoint(d_max: float, d_min: float) -> float:
    """The shell's centre in `1/d^2`, where the shells are equal in volume."""
    try:
        s_hi, s_lo = 1.0 / float(d_min) ** 2, 1.0 / float(d_max) ** 2
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")
    return float(1.0 / ((s_hi + s_lo) / 2.0) ** 0.5)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("log", type=Path, nargs="+")
    p.add_argument("--summary", action="store_true",
                   help="the merging summary, the error model and Wilson")
    args = p.parse_args(argv)
    for path in args.log:
        print(f"===== {path}")
        if args.summary:
            for name, value in (
                ("merging", merging_summary_from_log(path)),
                ("error model", error_model_from_log(path)),
                ("scaling", scaling_summary_from_log(path)),
                ("wilson", wilson_from_log(path)),
            ):
                print(f"  {name}: {value}")
            continue
        for t in log_tables(path):
            print(f"  [{t.title or '-'}] {len(t)} rows: {t.headers}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
