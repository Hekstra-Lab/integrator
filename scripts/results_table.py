#!/usr/bin/env python
"""Render an arm-by-method results table from the chain's own log lines.

Every arm of the ablation chain prints two lines its driver script greps for:

    RESULT <arm> Final R-work = 0.1732, R-free = 0.2282
    PEAKS  <arm> 5.172,5.1,6.293,...        # the 13 anomalous sites

An arm can be scored more than once -- once per export (the posterior moments,
the Kabsch sum) -- and the comparison that matters is between those exports on
the same arm, against a classical baseline. This reads one log set per method
and puts them side by side.

Usage:

    uv run python scripts/results_table.py \
        posterior=post.log kabsch=kab.log --baseline laue-dials=laue.log

    ssh cannon "grep -h RESULT ..." |
        uv run python scripts/results_table.py posterior=-

A `--baseline` source is rendered as its own columns repeated down the table,
so each row can be read on its own, and as a delta against the first method.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RESULT_RE = re.compile(
    r"^RESULT\s+(\S+)\s+Final R-work = ([\d.]+), R-free = ([\d.]+)", re.M
)
PEAKS_RE = re.compile(r"^PEAKS\s+(\S+)\s+(\S+)", re.M)

# The replicate floors measured from four runs of one configuration
# (docs/ablation_1118_360.md). Anything inside these is not an effect.
FLOOR_RFREE = 0.0070
FLOOR_PEAK = 0.31


def parse(text: str) -> dict[str, dict]:
    """Pull the arm -> {r_work, r_free, peaks} map out of a chain log."""
    out: dict[str, dict] = {}
    for arm, rw, rf in RESULT_RE.findall(text):
        out.setdefault(arm, {})["r_work"] = float(rw)
        out[arm]["r_free"] = float(rf)
    for arm, peaks in PEAKS_RE.findall(text):
        try:
            vals = [float(v) for v in peaks.split(",")]
        except ValueError:
            continue
        out.setdefault(arm, {})["peaks"] = vals
    return out


def read_source(spec: str) -> tuple[str, dict[str, dict]]:
    """`label=path`, or `label=-` to read stdin."""
    if "=" not in spec:
        raise SystemExit(f"source must be label=path, got {spec!r}")
    label, path = spec.split("=", 1)
    text = sys.stdin.read() if path == "-" else Path(path).read_text()
    return label, parse(text)


def mean_peak(entry: dict) -> float | None:
    p = entry.get("peaks")
    return sum(p) / len(p) if p else None


def fmt(v: float | None, places: int) -> str:
    return "" if v is None else f"{v:.{places}f}"


def box_table(headers: list[str], rows: list[list[str]], rules: bool) -> str:
    """Box-drawing table, one horizontal rule between every row."""
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
        for i, h in enumerate(headers)
    ]
    def line(l: str, m: str, r: str) -> str:
        return l + m.join("─" * (w + 2) for w in widths) + r
    def row(cells: list[str]) -> str:
        return "│ " + " │ ".join(
            c.ljust(w) for c, w in zip(cells, widths)
        ) + " │"

    parts = [line("┌", "┬", "┐"), row(headers), line("├", "┼", "┤")]
    for i, r in enumerate(rows):
        parts.append(row(r))
        if rules and i < len(rows) - 1:
            parts.append(line("├", "┼", "┤"))
    parts.append(line("└", "┴", "┘"))
    return "\n".join(parts)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("sources", nargs="+", help="label=path (path '-' = stdin)")
    p.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="label=path for a classical pipeline; its single row is "
        "repeated as columns so every row stands alone",
    )
    p.add_argument("--sort-by", default=None, help="method label to sort on")
    p.add_argument("--markdown", action="store_true")
    p.add_argument(
        "--no-rules", action="store_true", help="no rule between every row"
    )
    a = p.parse_args()

    methods = [read_source(s) for s in a.sources]
    baselines = [read_source(s) for s in a.baseline]

    arms = sorted({arm for _, d in methods for arm in d})
    if a.sort_by:
        key = dict(methods).get(a.sort_by, {})
        arms.sort(key=lambda x: key.get(x, {}).get("r_free", float("inf")))

    headers = ["arm"]
    headers += [f"R-free {label}" for label, _ in methods]
    headers += [f"R-free {label}" for label, _ in baselines]
    headers += [f"peaks {label}" for label, _ in methods]
    headers += [f"peaks {label}" for label, _ in baselines]

    # a baseline log holds one row whatever it is called
    base_vals = [next(iter(d.values()), {}) for _, d in baselines]

    rows = []
    for arm in arms:
        cells = [arm]
        cells += [fmt(d.get(arm, {}).get("r_free"), 4) for _, d in methods]
        cells += [fmt(b.get("r_free"), 4) for b in base_vals]
        cells += [fmt(mean_peak(d.get(arm, {})), 3) for _, d in methods]
        cells += [fmt(mean_peak(b), 3) for b in base_vals]
        rows.append(cells)

    render = md_table if a.markdown else (
        lambda h, r: box_table(h, r, not a.no_rules)
    )
    print(render(headers, rows))

    if base_vals and methods:
        print(
            f"\nfloors from four runs of one config: R-free {FLOOR_RFREE}, "
            f"peak {FLOOR_PEAK}"
        )
        for label, d in methods:
            beats = [
                arm
                for arm in arms
                if (rf := d.get(arm, {}).get("r_free")) is not None
                and (b := base_vals[0].get("r_free")) is not None
                and rf < b - 2 * FLOOR_RFREE
            ]
            print(
                f"  {label}: {len(beats)}/{len(arms)} arms beat "
                f"{baselines[0][0]} on R-free by more than 2 floors"
                + (f" -- {', '.join(beats)}" if beats else "")
            )


if __name__ == "__main__":
    main()
