"""The learned basis converging, svae against hierarchical, per dataset.

Comparison 1 drawn as a comparison. `fig_profile_basis.py` draws one arm at
a time -- every component's amplitude and its cosine to its own final value --
which says how one basis settled but puts nothing beside it. This reduces each
arm to the three numbers that describe the basis as a whole and overlays the
two families on one axes per dataset.

- **effective rank**, `exp(H)` of the normalized singular values, so a basis
  using all of its components sits at the component count and one that has
  collapsed onto a few sits far below it. The component count differs by
  dataset (12 on 9b7c and 1118, 16 on PDZ2), so it is drawn as a dotted
  ceiling rather than assumed.
- **principal angle to the final basis**, which is the subspace settling.
- **relative step** per checkpoint, which is how fast it is still moving.

The pairs differ by the integrator family alone -- same profile model, same
data, same basis size -- so a gap here is the family and not the setup.

Reads `docs/figures/data/basis_cmp/<arm>/basis_diagnostics.parquet` from
`dump_basis.py`, with each arm's dataset from `basis_components.csv` and its
family from the `cmp_*.csv` tables, both pulled by
`bash scripts/figures/pull_figdata.sh basis models`.

Run:
    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_basis_convergence.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def figstyle():
    """figTeX, imported at the call: it is not a dependency of this project."""
    import figstyle as _figstyle

    return _figstyle


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA = ROOT / "docs" / "figures" / "data"
OUT = ROOT / "docs" / "figures" / "cmp"
DATASETS = ("hewl_9b7c", "hewl_1118", "pdz2_1116")
TITLES = {"hewl_9b7c": "9b7c (mono)", "hewl_1118": "1118 (poly)",
          "pdz2_1116": "PDZ2 (poly)"}
PANELS = (
    ("eff_rank", "effective rank"),
    ("angle_to_final_deg", r"angle to final basis ($^\circ$)"),
    ("rel_step", "relative step"),
)
FAMILY_SLOT = {"hierarchical": 0, "svae": 3}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--prefix", default="basis_convergence")
    p.add_argument("--split", action="store_true",
                   help="write one row per quantity, <prefix>_<quantity>, "
                        "instead of the 3-row grid; the grid does not fit "
                        "a slide frame")
    p.add_argument("--template", default=None,
                   help="figTeX template to size for (article, paper, book, "
                        "slides, poster); default is figstyle's")
    return p.parse_args()


def arm_table(data_dir: Path) -> pl.DataFrame:
    """One row per arm: method, dataset, family, and its diagnostics path.

    The dataset comes from `basis_components.csv` and the family from the
    `cmp_*.csv` tables, because `basis_diagnostics.parquet` is per-arm and
    carries neither. Both are key columns written by the cluster reducers,
    so nothing here has to infer a label from a directory name.
    """
    comp = data_dir / "basis_cmp" / "basis_components.csv"
    if not comp.is_file():
        raise SystemExit(
            f"no {comp}. Run `bash scripts/figures/pull_figdata.sh basis`"
        )
    arms = (pl.read_csv(comp, infer_schema_length=None)
            .select("method", "dataset").unique())

    family = {}
    for path in sorted(data_dir.glob("cmp_*.csv")):
        df = pl.read_csv(path, infer_schema_length=0)
        for row in df.select("method", "integrator").unique().iter_rows():
            family[row[0]] = row[1]
    if not family:
        raise SystemExit(
            "no cmp_*.csv to take the integrator family from. Run "
            "`bash scripts/figures/pull_figdata.sh models`"
        )

    rows = []
    for row in arms.iter_rows(named=True):
        fam = family.get(row["method"])
        path = data_dir / "basis_cmp" / row["method"] / "basis_diagnostics.parquet"
        if fam is None:
            logger.warning("  %s: no family in cmp_*.csv, skipped", row["method"])
            continue
        if not path.is_file():
            logger.warning("  %s: no %s, skipped", row["method"], path.name)
            continue
        rows.append({**row, "family": fam, "path": str(path)})
    return pl.DataFrame(rows)


def components_of(data_dir: Path, method: str) -> int | None:
    """The basis's component count, for the effective-rank ceiling."""
    comp = pl.read_csv(data_dir / "basis_cmp" / "basis_components.csv",
                       infer_schema_length=None)
    sub = comp.filter(pl.col("method") == method)
    return sub["component"].n_unique() if "component" in sub.columns else None


def draw_grid(arms: pl.DataFrame, present: list[str], quantities,
              colours: dict, data_dir: Path, path: Path) -> None:
    """One row per quantity, one column per dataset."""
    # rows are different quantities and columns datasets with different
    # component counts, so no axis is shared: `panels` would otherwise
    # share x on a grid, and sharing y would put an angle on an effective
    # rank's axis
    ncols, nrows = len(present), len(quantities)
    layout = f"{nrows}x{ncols}" if nrows > 1 else str(ncols)
    P = figstyle().panels(layout, shared=False)
    flat = list(P.flat)
    if len(flat) != nrows * ncols:
        raise SystemExit(f"{figstyle().active()}: panels returned "
                         f"{len(flat)} axes for a {nrows}x{ncols} grid")

    for col, dataset in enumerate(present):
        here = arms.filter(pl.col("dataset") == dataset)
        for rowi, (metric, ylabel) in enumerate(quantities):
            ax = flat[rowi * ncols + col]
            for arm in here.iter_rows(named=True):
                diag = pl.read_parquet(arm["path"])
                if metric not in diag.columns:
                    continue
                sub = diag.drop_nulls(metric).sort("epoch")
                # the first `rel_step` is NaN: there is no previous
                # checkpoint to step from
                y = sub[metric].to_numpy().astype(float)
                keep = ~np.isnan(y)
                ax.plot(sub["epoch"].to_numpy()[keep], y[keep], lw=1.1,
                        color=colours.get(arm["family"]), label=arm["family"])
            if metric == "eff_rank":
                n = components_of(data_dir, here["method"][0])
                if n:
                    ax.axhline(n, ls=":", lw=0.8,
                               color=figstyle().roles().get("muted", "0.5"))
                    ax.set_ylim(top=n * 1.06)
            if metric == "rel_step":
                ax.set_yscale("log")
            if col == 0:
                ax.set_ylabel(ylabel)
            if rowi == 0:
                P.head(ax, col, TITLES.get(dataset, dataset))
            if rowi == nrows - 1:
                ax.set_xlabel("epoch")

    handles = [plt.Line2D([], [], color=colours[f], lw=1.4)
               for f in FAMILY_SLOT]
    # a boxed strip above the panels rather than inside one: inside the
    # first axes it sat on the 9b7c curves, and every panel shares the
    # same two series
    P.top_legend(handles, list(FAMILY_SLOT), frameon=True,
                 handlelength=1.4)
    P.finish()
    figstyle().save(P.fig, path)
    plt.close(P.fig)
    logger.info("Saved %s", path)


def main() -> None:
    args = parse_args()
    arms = arm_table(args.data)
    if not arms.height:
        raise SystemExit("no arms with diagnostics")
    if args.template:
        figstyle().use(args.template)
    else:
        figstyle().use()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cat = figstyle().categorical(8)
    colours = {f: cat[s % len(cat)] for f, s in FAMILY_SLOT.items()}
    present = [d for d in DATASETS if d in arms["dataset"].to_list()]

    if args.split:
        for metric, ylabel in PANELS:
            draw_grid(arms, present, ((metric, ylabel),), colours, args.data,
                      args.out_dir / f"{args.prefix}_{metric}")
    else:
        draw_grid(arms, present, PANELS, colours, args.data,
                  args.out_dir / args.prefix)

    for arm in arms.sort(["dataset", "family"]).iter_rows(named=True):
        diag = pl.read_parquet(arm["path"]).sort("epoch")
        logger.info("  %-12s %-13s %-26s eff_rank %.2f -> %.2f over %d epochs",
                    arm["dataset"], arm["family"], arm["method"],
                    float(diag["eff_rank"][0]), float(diag["eff_rank"][-1]),
                    int(diag["epoch"][-1]))


if __name__ == "__main__":
    main()
