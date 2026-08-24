"""Emit a canonical merging_stats.csv from a DIALS merged.html (mono pipeline).

Shared cross-pipeline contract with the poly pipeline: one wide CSV per merge,
one row per resolution bin, columns

    bin, d_max, d_min, n_obs, n_unique, cc_half, cc_anom, r_pim, i_over_sigma

so plot_merging can read merging_stats.csv (preferred) and place mono and poly
runs side by side on one figure. The mono values come from the per-shell table in
DIALS merged.html; the poly side derives the same columns from careless output.

Usage:
    python emit_merging_stats.py MERGED_HTML [-o OUT_CSV]
    python emit_merging_stats.py --run-dir RUN     # every predictions/epoch_*/dials/merged.html
"""

import argparse
import re
from pathlib import Path

import pandas as pd

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


def _find_col(cols, *subs):
    """First column whose lowercased label contains all substrings, else None."""
    for c in cols:
        cl = str(c).lower()
        if all(s in cl for s in subs):
            return c
    return None


def _num(v):
    """Parse a table cell to float, stripping DIALS significance '*' and commas."""
    if v is None:
        return None
    s = str(v).replace("*", "").replace(",", "").strip()
    if s in ("", "nan", "none", "None", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(x):
    return int(x) if x is not None else None


def stats_from_merged_html(path: Path) -> pd.DataFrame | None:
    """Return the canonical wide merging-stats frame from a DIALS merged.html."""
    tables = pd.read_html(path)
    for t in tables:
        cols = [c[-1] if isinstance(c, tuple) else c for c in t.columns]
        t.columns = cols
        res = _find_col(cols, "resolution")
        c_cano = _find_col(cols, "ccano") or _find_col(cols, "cc", "ano")
        if res is None or c_cano is None:
            continue  # not the per-shell merging-statistics table
        c_obs = _find_col(cols, "n(obs") or _find_col(cols, "nobs") or _find_col(cols, "n obs")
        c_uniq = _find_col(cols, "n(uniq") or _find_col(cols, "unique")
        c_cch = (
            _find_col(cols, "cc½")
            or _find_col(cols, "cc1/2")
            or _find_col(cols, "cchalf")
            or _find_col(cols, "cc", "half")
        )
        c_rpim = _find_col(cols, "rpim") or _find_col(cols, "r-pim") or _find_col(cols, "r_pim")
        c_isig = (
            _find_col(cols, "i/σ")
            or _find_col(cols, "i/sig")
            or _find_col(cols, "mean i/")
        )
        rows = []
        for i, (_, r) in enumerate(t.iterrows(), start=1):
            nums = re.findall(r"[-+]?\d*\.?\d+", str(r[res]))
            d_max = float(nums[0]) if len(nums) >= 2 else None
            d_min = float(nums[1]) if len(nums) >= 2 else None
            rows.append(
                {
                    "bin": i,
                    "d_max": d_max,
                    "d_min": d_min,
                    "n_obs": _to_int(_num(r[c_obs])) if c_obs else None,
                    "n_unique": _to_int(_num(r[c_uniq])) if c_uniq else None,
                    "cc_half": _num(r[c_cch]) if c_cch else None,
                    "cc_anom": _num(r[c_cano]) if c_cano else None,
                    "r_pim": _num(r[c_rpim]) if c_rpim else None,
                    "i_over_sigma": _num(r[c_isig]) if c_isig else None,
                }
            )
        return pd.DataFrame(rows, columns=COLUMNS)
    return None


def emit(merged_html: Path, out_csv: Path | None = None) -> Path | None:
    df = stats_from_merged_html(merged_html)
    if df is None or df.empty:
        print(f"  WARN: no merging-stats table in {merged_html}")
        return None
    out = out_csv or merged_html.parent / "merging_stats.csv"
    df.to_csv(out, index=False)
    print(f"  wrote {out}  ({len(df)} bins)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("merged_html", nargs="?", help="Path to a DIALS merged.html.")
    parser.add_argument("-o", "--out", help="Output CSV (default: merging_stats.csv beside merged.html).")
    parser.add_argument(
        "--run-dir",
        help="Run dir: emit for every predictions/epoch_*/dials/merged.html.",
    )
    args = parser.parse_args()

    if args.run_dir:
        run = Path(args.run_dir)
        htmls = sorted(run.glob("predictions/epoch_*/dials/merged.html"))
        if not htmls:
            raise SystemExit(f"no merged.html under {run}/predictions/epoch_*/dials/")
        for h in htmls:
            emit(h)
    elif args.merged_html:
        emit(Path(args.merged_html), Path(args.out) if args.out else None)
    else:
        parser.error("provide MERGED_HTML or --run-dir")


if __name__ == "__main__":
    main()
