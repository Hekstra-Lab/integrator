"""svae against hierarchical, every metric that exists for both, side by side.

One panel per metric. Each pair differs by the integrator family alone --
same profile model, same data, same basis size -- so a gap is the family
and not the setup. Which arms form a pair is a scientific statement about
what was held fixed, so `PAIRS` names them rather than matching on a
substring.

**Panels are drawn only where both arms have the metric.** The svae arm on
PDZ2 has been through CCsym but not yet `careless.stats` or refinement, and
9b7c and 1118 have no polychromatic downstream at all. A panel with one
family in it compares nothing, so it is left out and the gap is reported in
the log rather than drawn as an absence the eye has to notice.

Reads the tidy tables under `docs/figures/data/`, all of them pulled by
`bash scripts/figures/pull_figdata.sh pdz2 basis models`:

    cmp_*.csv                parameters, seconds per epoch
    basis_cmp/<arm>/…parquet effective rank, angle to final, relative step
    pdz2_ccsym.csv           CCsym by delay
    pdz2_training_loss.csv   the ELBO and its terms

Run:
    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/fig_svae_vs_hierarchical.py
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

# (dataset, svae arm, hierarchical arm). The arm names differ between tables
# -- `compare_models` keys off the run directory (`pdz2_svae`) and the PDZ2
# downstream off the tree (`svae`) -- so both spellings are carried.
PAIRS = (
    ("hewl_9b7c", ("hewl9b7c_svae",), ("hewl9b7c_mono_expmean_lp",)),
    ("hewl_1118", ("learned",), ("softmax_std",)),
    # `tim_*` are the matched-allocation re-measurement of the same two
    # PDZ2 models, and they must be named here or `cmp_timing` is skipped
    # and the panel falls back to the unmatched 8-CPU-against-16 numbers
    ("pdz2_1116", ("pdz2_svae", "svae", "tim_svae"),
     ("pdz2_w21_k7m", "w21_k7m", "tim_hier")),
)
LABELS = {"hewl_9b7c": "9b7c", "hewl_1118": "1118", "pdz2_1116": "PDZ2"}
DELAYS = ("off", "50ns", "100ns", "200ns")
FAMILIES = ("svae", "hierarchical")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--prefix", default="svae_vs_hierarchical")
    return p.parse_args()


def family_of(method: str) -> str | None:
    for _dataset, svae, hier in PAIRS:
        if method in svae:
            return "svae"
        if method in hier:
            return "hierarchical"
    return None


def dataset_of(method: str) -> str | None:
    for dataset, svae, hier in PAIRS:
        if method in svae or method in hier:
            return dataset
    return None


def read(path: Path, **kw) -> pl.DataFrame | None:
    if not path.is_file():
        return None
    return pl.read_csv(path, infer_schema_length=kw.pop("infer", 0), **kw)


def cmp_values(data_dir: Path, quantity: str) -> dict:
    """{(dataset, family): float} for one `compare_models` quantity."""
    out: dict[tuple[str, str], float] = {}
    for path in sorted(data_dir.glob("cmp_*.csv")):
        df = read(path)
        if df is None or "quantity" not in df.columns:
            continue
        for row in df.filter(pl.col("quantity") == quantity).iter_rows(
                named=True):
            fam, ds = family_of(row["method"]), dataset_of(row["method"])
            if fam and ds:
                # cmp_timing is the matched re-measurement of PDZ2 and must
                # win over the unmatched row for the same pair
                key = (ds, fam)
                if path.stem == "cmp_timing" or key not in out:
                    out[key] = float(row["value"])
    return out


def basis_final(data_dir: Path, column: str) -> dict:
    """{(dataset, family): value at the last checkpoint} from the diagnostics."""
    out: dict[tuple[str, str], float] = {}
    base = data_dir / "basis_cmp"
    for path in sorted(base.glob("*/basis_diagnostics.parquet")):
        arm = path.parent.name
        fam, ds = family_of(arm), dataset_of(arm)
        if not fam or not ds:
            continue
        df = pl.read_parquet(path).drop_nulls(column).sort("epoch")
        if df.height:
            out[(ds, fam)] = float(df[column][-1])
    return out


def ccsym_means(data_dir: Path) -> dict:
    """{(delay, family): mean CCsym over shells and repeats} on PDZ2."""
    df = read(data_dir / "pdz2_ccsym.csv")
    if df is None:
        return {}
    df = df.with_columns(pl.col("ccsym").cast(pl.Float64, strict=False))
    out: dict[tuple[str, str], float] = {}
    for row in df.iter_rows(named=True):
        fam = family_of(row["method"])
        if fam is None or row["ccsym"] is None:
            continue
        out.setdefault((row["delay"], fam), []).append(row["ccsym"])
    return {k: float(np.mean(v)) for k, v in out.items()}


def loss_curves(data_dir: Path) -> dict:
    """{family: (epochs, value)} for the validation ELBO on PDZ2."""
    df = read(data_dir / "pdz2_training_loss.csv")
    if df is None:
        return {}
    df = df.with_columns(pl.col("value").cast(pl.Float64, strict=False),
                         pl.col("epoch").cast(pl.Float64, strict=False))
    # the table names the total `loss`, not `elbo`; the ELBO is what it is
    terms = set(df["term"].unique())
    term = next((t for t in ("elbo", "loss") if t in terms), None)
    if term is None:
        return {}
    out: dict[str, tuple] = {}
    for row_family in FAMILIES:
        sub = df.filter(
            (pl.col("term") == term)
            & (pl.col("split") == "val")
            & pl.col("method").map_elements(
                lambda m, f=row_family: family_of(m) == f,
                return_dtype=pl.Boolean)
        ).sort("epoch")
        if sub.height:
            out[row_family] = (sub["epoch"].to_numpy(),
                               sub["value"].to_numpy())
    return out


def grouped(ax, keys, values, colours, ylabel, logy=False):
    """One group per key, one bar per family; drawn only where both exist."""
    keys = [k for k in keys
            if all((k, f) in values for f in FAMILIES)]
    if not keys:
        return False
    x = np.arange(len(keys))
    width = 0.36
    for i, fam in enumerate(FAMILIES):
        ax.bar(x + (i - 0.5) * width, [values[(k, fam)] for k in keys],
               width, color=colours[fam], zorder=2, label=fam)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(k, k) for k in keys])
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    return True


def main() -> None:
    args = parse_args()
    figstyle().use()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pal = figstyle().arm_colors(list(FAMILIES))
    colours = {f: pal[f] for f in FAMILIES}

    params = cmp_values(args.data, "n_params_total")
    rate = cmp_values(args.data, "seconds_per_epoch")
    rank = basis_final(args.data, "eff_rank")
    step = basis_final(args.data, "rel_step")
    ccsym = ccsym_means(args.data)
    curves = loss_curves(args.data)

    datasets = [d for d, _s, _h in PAIRS]
    panels = [
        ("parameters", lambda ax: grouped(ax, datasets, params, colours,
                                          "parameters")),
        ("seconds per epoch", lambda ax: grouped(ax, datasets, rate, colours,
                                                 "seconds per epoch")),
        ("effective rank", lambda ax: grouped(ax, datasets, rank, colours,
                                              "effective rank, last")),
        ("relative step", lambda ax: grouped(ax, datasets, step, colours,
                                             "relative step, last", True)),
        ("CCsym", lambda ax: grouped(ax, list(DELAYS), ccsym, colours,
                                     r"$\mathrm{CC}_\mathrm{sym}$ (mean)")),
        ("val ELBO", lambda ax: _elbo(ax, curves, colours)),
    ]

    drawn = []
    fig, axes = plt.subplots(2, 3, figsize=figstyle().figsize("full",
                                                              aspect=0.62),
                             layout="constrained")
    flat = axes.ravel()
    for ax, (name, draw) in zip(flat, panels, strict=True):
        if draw(ax):
            drawn.append(name)
        else:
            ax.set_visible(False)
            logger.info("  %s: skipped, not present for both families", name)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[f]) for f in FAMILIES]
    fig.legend(handles, list(FAMILIES), frameon=False, ncol=2,
               loc="outside upper center")
    path = args.out_dir / args.prefix
    figstyle().save(fig, path)
    plt.close(fig)
    logger.info("Saved %s (%d of %d panels: %s)", path, len(drawn),
                len(panels), ", ".join(drawn))

    for ds in datasets:
        if (ds, "svae") in params and (ds, "hierarchical") in params:
            logger.info("  %-10s svae %.0f%% of hierarchical's parameters",
                        LABELS.get(ds, ds),
                        100 * params[(ds, "svae")] / params[(ds, "hierarchical")])


def _elbo(ax, curves, colours) -> bool:
    if len(curves) < 2:
        return False
    for fam, (x, y) in curves.items():
        ax.plot(x, y, lw=1.1, color=colours[fam], label=fam)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation ELBO")
    return True


if __name__ == "__main__":
    main()
