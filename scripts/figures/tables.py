"""LaTeX tables in the mydesign dialect: booktabs rules and siunitx columns.

figTeX's own `render("latex")` writes a plain `tabular` with `l`/`r` columns
and `\\textbf` for the best value. The document styles want something more
specific, settled with the figtex side on 2026-11-17:

- booktabs rules only, never a vertical rule, and `@{}` at both outer edges
- numbers in `siunitx` `S` columns so they align on the decimal point, with
  `table-format` read off the data rather than guessed
- every header and text cell braced, because an `S` column parses anything
  unbraced as a number
- units on a second header line, in parentheses and upright, which is what
  DIALS itself prints
- the body at `\\small` (9 pt, the caption size), or `\\footnotesize` (8 pt)
  with `tabcolsep` at 4 pt once the table has enough numeric columns to
  strain the measure. Never smaller, and never `\\resizebox`: scaled type
  stops matching the text around it.
- the best value in a comparison marked with Minion Semibold, not bold, and
  only in a table that compares arms -- never in a per-shell table, where
  there is nothing to win

Registered as a figTeX renderer rather than written into figTeX, so the
table machinery stays one implementation:

    import tables  # registers "mydesign"
    t.save("refinement.tex", fmt="mydesign", caption="...", label="tab:ref")

Single-letter quantities are italic and multi-letter names upright, the same
rule the figure labels follow: $d_{\\min}$, but $\\mathrm{CC}_{1/2}$.
"""

from __future__ import annotations

import re

from figtex.render import renderer
from figtex.render.latex import escape

# Enough numeric columns and the table stops fitting the article measure at
# 9 pt. Counted, not guessed: 8 columns of `1.234` plus a label column is
# about where it crosses.
DENSE_COLUMNS = 8
_NUM = re.compile(r"^[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?$")


def _is_number(text: str) -> bool:
    return bool(_NUM.match(text.strip().rstrip("%*")))


def table_format(cells: list[str]) -> str:
    """`siunitx` `table-format` wide enough for every value in a column.

    `1.3` reserves one integer digit and three decimals; a leading `+-`
    reserves the sign column so a negative value does not shift the decimal
    point out of line with the rest.
    """
    before = after = 0
    signed = False
    for text in cells:
        t = text.strip().rstrip("%*")
        if not _is_number(t):
            continue
        if t.startswith(("-", "+")):
            signed = True
            t = t[1:]
        head, _, tail = t.partition(".")
        before = max(before, len(head))
        after = max(after, len(tail))
    before = max(before, 1)
    fmt = f"{before}.{after}" if after else f"{before}.0"
    return f"+-{fmt}" if signed else fmt


def _column_cells(table, index: int) -> list[str]:
    out = []
    for _, rows in table.groups():
        for row in rows:
            if index < len(row) and not row[index].missing:
                out.append(row[index].text)
    return out


# Registered as "mydesign", not "siunitx": figtex ships its own siunitx
# renderer, and registering a second one under that name shadows
# whichever imported last. That is not a conflict worth winning -- the
# two are different dialects, and this one is the mydesign house style
# (caption above, booktabs only, units on a second header line). It is
# superseded by figtex.sitable once make_tables.py ports onto it.
@renderer("mydesign")
def to_siunitx(
    table,
    float: bool = False,
    placement: str = "htbp",
    caption: str | None = None,
    label: str | None = None,
    notes: list[str] | None = None,
    best_mark: str = r"\sbseries",
    group_style: str = r"\textit",
    size: str | None = None,
) -> str:
    """A booktabs `tabular` with `S` columns, in the mydesign dialect.

    `float=True` wraps it in a `table` with the caption **above** the rules,
    which is where mydesign puts it. `notes` adds a `threeparttable` block
    under the bottom rule.
    """
    shown = table.shown
    cells = {i: _column_cells(table, i) for i in range(len(shown))}
    numeric = {
        i: bool(c) and all(_is_number(x) for x in c) for i, c in cells.items()
    }

    spec = []
    for i, col in enumerate(shown):
        if numeric[i]:
            spec.append(f"S[table-format={table_format(cells[i])}]")
        else:
            spec.append(col.align or "l")
    colspec = "@{}" + " ".join(spec) + "@{}"

    if size is None:
        size = r"\footnotesize" if sum(numeric.values()) >= DENSE_COLUMNS \
            else r"\small"
    dense = size == r"\footnotesize"

    def head(text: str) -> str:
        """A header cell is always braced: an `S` column parses it otherwise."""
        return "{" + escape(text) + "}"

    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    lines.append(" & ".join(head(c.label or c.name) for c in shown) + r" \\")
    if any(c.unit for c in shown):
        lines.append(
            " & ".join(head(f"({c.unit})" if c.unit else "") for c in shown)
            + r" \\"
        )
    lines.append(r"\midrule")

    n = len(shown)
    for g, (group, rows) in enumerate(table.groups()):
        if group is not None:
            if g > 0:
                lines.append(r"\addlinespace")
            g_text = escape(group)
            lines.append(
                rf"\multicolumn{{{n}}}{{@{{}}l}}{{{group_style}{{{g_text}}}}} \\"
            )
        for row in rows:
            out = []
            for i, cell in enumerate(row):
                if cell.missing:
                    out.append("{" + cell.text + "}")
                elif numeric[i]:
                    out.append(
                        f"{{{best_mark} {cell.text}}}" if cell.best
                        else cell.text
                    )
                else:
                    text = escape(cell.text)
                    out.append(
                        f"{{{best_mark} {text}}}" if cell.best else f"{{{text}}}"
                    )
            lines.append(" & ".join(out) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    body = "\n".join(lines)
    if dense:
        body = "\\setlength{\\tabcolsep}{4pt}%\n" + body
    body = f"{size}\n{body}"

    if notes:
        items = "\n".join(rf"\item {escape(t)}" for t in notes)
        body = (
            "\\begin{threeparttable}\n"
            f"{body}\n"
            "\\begin{tablenotes}[flushleft]\\footnotesize\n"
            f"{items}\n"
            "\\end{tablenotes}\n"
            "\\end{threeparttable}"
        )
    if not float:
        return body + "\n"

    caption = caption if caption is not None else table.caption
    label = label if label is not None else table.label
    out = [rf"\begin{{table}}[{placement}]", r"\centering"]
    # mydesign sets the caption above the table
    if caption:
        out.append(rf"\caption{{{caption}}}")
    if label:
        out.append(rf"\label{{{label}}}")
    out += [body, r"\end{table}", ""]
    return "\n".join(out)
