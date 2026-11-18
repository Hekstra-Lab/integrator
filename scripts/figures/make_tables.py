"""The LaTeX tables, from the DIALS logs each arm already wrote.

`dials.scale` and `dials.merge` report a good deal that never reaches
`merging_stats.csv`, and it was being read by eye out of the logs: the
error model's fitted `a` and `b`, the Rmerge and Rmeas forms the per-shell
table omits, the Wilson analysis, and the anomalous differences by shell.
`integrator.utils.metrics` parses all of it; this script is what turns it
into tables.

Every table is written twice, as `.tex` for the manuscript and `.md` so the
numbers can be read in the repository without compiling anything. The LaTeX
is the mydesign dialect from `tables.py`: booktabs, siunitx `S` columns,
units on a second header line.

    uv run --with-editable /Users/luis/master/software/figtex \\
        python scripts/figures/make_tables.py

Reads `docs/figures/data/dials_logs/<arm>_<estimator>/{scale,merge}.log`,
which `pull_figdata.sh dials_logs` brings down, and writes `docs/tables/`.

Beside the tables it writes the numbers they were built from, long format,
in `docs/tables/data/`: `dials_log_stats.csv` with one row per statistic
(method, dataset, estimator, section, region, d_max, d_min, statistic,
value) and `statistic_labels.csv` with the header, unit and format each
statistic is set in. A table laid out somewhere else starts from those
rather than from a parsed `.tex`.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tables as sitables  # noqa: F401,E402  -- registers "mydesign"
from figtex.table import Table  # noqa: E402

from integrator.utils import metrics  # noqa: E402

LOGS = ROOT / "docs" / "figures" / "data" / "dials_logs"
OUT = ROOT / "docs" / "tables"
# the numbers behind the tables, long format, for anyone building their own
DATA = OUT / "data"

# The arms in the order they are reported, with the name used in prose.
# `_rb` is the Rao-Blackwellized profile fit computed at export and `_model`
# the integrator's own posterior mean; both are scored for every arm because
# which one wins is not monotone across arms.
ARM_LABELS = {
    "m9_gate_learned": "learned gate",
    "m9_geom": "geometric basis",
    "m9_geomshift": "geometric, local shift",
    "m9_geomresp": "geometric, responsibility",
    "m9_geomrespsteer": "geometric, responsibility + steer",
}
ESTIMATOR_LABELS = {"rb": "profile fit at export", "model": "posterior mean"}

# Column headers. Single-letter quantities italic, multi-letter names
# upright, the same rule the figure labels follow.
H = {
    "arm": "arm",
    "d_max": r"$d_{\max}$",
    "d_min": r"$d_{\min}$",
    "n_obs": r"$N_\mathrm{obs}$",
    "n_unique": r"$N_\mathrm{uniq}$",
    "multiplicity": "multiplicity",
    "completeness": "completeness",
    "i_over_sigma": r"$\langle I/\sigma_I\rangle$",
    "r_merge": r"$R_\mathrm{merge}$",
    "r_meas": r"$R_\mathrm{meas}$",
    "r_pim": r"$R_\mathrm{pim}$",
    "r_merge_anom": r"$R_\mathrm{merge}^{\pm}$",
    "r_meas_anom": r"$R_\mathrm{meas}^{\pm}$",
    "r_pim_anom": r"$R_\mathrm{pim}^{\pm}$",
    "cc_half": r"$\mathrm{CC}_{1/2}$",
    "cc_anom": r"$\mathrm{CC}_\mathrm{anom}$",
    "anom_slope": "anom. slope",
    "df_over_f": r"$\Delta F/F$",
    "di_over_sigma_di": r"$\Delta I/\sigma(\Delta I)$",
    "wilson_b": r"$B_\mathrm{Wilson}$",
    "b_aniso_range": r"$\Delta B_\mathrm{aniso}$",
    "neg_log_scale": r"$-\log k$",
    "a": "$a$",
    "b": "$b$",
    "i_over_sigma_limit": r"$(I/\sigma_I)_\infty$",
    "n_param_total": r"$N_\mathrm{param}$",
    "n_outliers": r"$N_\mathrm{outlier}$",
    "rmsd_i_final": r"$\mathrm{RMSD}_I$",
    "pct_param_uncertain": "uncertain param.",
    "dano_over_sigma": r"$\langle|\Delta F|/\sigma(\Delta F)\rangle$",
    # the rest appear in the long-format export rather than in a table here,
    # and are named so that a table built elsewhere sets them the same way
    "i_mean": r"$\langle I\rangle$",
    "r_anom": r"$R_\mathrm{anom}$",
    "n_uniq": r"$N_\mathrm{uniq}$",
    "n_refl": r"$N_\mathrm{refl}$",
    "n_read": r"$N_\mathrm{refl}$ read",
    "n_outliers_total": r"$N_\mathrm{outlier}$, all rounds",
    "n_param_scale": r"$N_\mathrm{param}$, scale",
    "n_param_decay": r"$N_\mathrm{param}$, decay",
    "n_param_absorption": r"$N_\mathrm{param}$, absorption",
    "anom_completeness": "anom. completeness",
    "anom_multiplicity": "anom. multiplicity",
    # the principal values of the anisotropic B tensor, sorted descending;
    # their spread is `b_aniso_range`
    "b_cart_1": "$B_1$",
    "b_cart_2": "$B_2$",
    "b_cart_3": "$B_3$",
    # `dials.merge` marks a correlation it judges significant with a
    # trailing `*`; the marker travels as its own 0/1 column
    "cc_half_significant": r"$\mathrm{CC}_{1/2}$ significant",
    "cc_anom_significant": r"$\mathrm{CC}_\mathrm{anom}$ significant",
}
UNITS = {
    "d_max": r"\AA", "d_min": r"\AA", "completeness": r"\%",
    "wilson_b": r"\AA$^2$", "b_aniso_range": r"\AA$^2$",
    "pct_param_uncertain": r"\%",
    "b_cart_1": r"\AA$^2$", "b_cart_2": r"\AA$^2$",
    "b_cart_3": r"\AA$^2$",
    "anom_completeness": r"\%",
}
FMT = {
    "d_max": "{:.2f}", "d_min": "{:.2f}", "multiplicity": "{:.1f}",
    "completeness": "{:.1f}", "i_over_sigma": "{:.1f}",
    "r_merge": "{:.3f}", "r_meas": "{:.3f}", "r_pim": "{:.3f}",
    "r_merge_anom": "{:.3f}", "r_meas_anom": "{:.3f}", "r_pim_anom": "{:.3f}",
    "cc_half": "{:.3f}", "cc_anom": "{:.3f}", "anom_slope": "{:.3f}",
    "df_over_f": "{:.3f}", "di_over_sigma_di": "{:.3f}",
    "wilson_b": "{:.2f}", "b_aniso_range": "{:.2f}", "neg_log_scale": "{:.2f}",
    "a": "{:.3f}", "b": "{:.4f}", "i_over_sigma_limit": "{:.1f}",
    "rmsd_i_final": "{:.4f}", "pct_param_uncertain": "{:.1f}",
    "dano_over_sigma": "{:.3f}",
    "n_obs": "{:.0f}", "n_unique": "{:.0f}", "n_param_total": "{:.0f}",
    "n_outliers": "{:.0f}",
    "i_mean": "{:.1f}", "r_anom": "{:.3f}", "n_uniq": "{:.0f}",
    "n_refl": "{:.0f}", "n_read": "{:.0f}", "n_outliers_total": "{:.0f}",
    "n_param_scale": "{:.0f}", "n_param_decay": "{:.0f}",
    "n_param_absorption": "{:.0f}",
    "anom_completeness": "{:.1f}", "anom_multiplicity": "{:.1f}",
    "b_cart_1": "{:.2f}", "b_cart_2": "{:.2f}", "b_cart_3": "{:.2f}",
    "cc_half_significant": "{:.0f}", "cc_anom_significant": "{:.0f}",
}
# Where a larger number is better; everything else marked is a residual,
# where smaller is better. Only comparison tables mark a best value at all.
LARGER_IS_BETTER = {
    "completeness", "multiplicity", "i_over_sigma", "cc_half", "cc_anom",
    "n_obs", "n_unique", "n_uniq", "i_over_sigma_limit",
    "anom_completeness", "anom_multiplicity",
}
# Columns a comparison does not mark. Completeness and multiplicity are the
# same for every arm because they are properties of the collection, so
# marking them marks every row; the Wilson B is descriptive and its best
# value is not its smallest -- `gate: responsibility` returns a NEGATIVE
# Wilson B, which is unphysical, and marking it would read as a win.
NOT_COMPARED = {
    "completeness", "multiplicity", "cc_half", "wilson_b", "b_aniso_range",
    "neg_log_scale", "anom_slope", "df_over_f", "di_over_sigma_di",
    "anom_completeness", "anom_multiplicity", "i_mean",
    "b_cart_1", "b_cart_2", "b_cart_3",
    "cc_half_significant", "cc_anom_significant",
    "n_read", "n_refl",
}


def arms() -> list[tuple[str, str, Path]]:
    """`(arm, estimator, directory)` for every pulled log directory."""
    out = []
    for arm in ARM_LABELS:
        for est in ESTIMATOR_LABELS:
            d = LOGS / f"{arm}_{est}"
            if (d / "merge.log").is_file() or (d / "scale.log").is_file():
                out.append((arm, est, d))
    return out


def configure(t: Table, *, best: bool = False) -> Table:
    """Header, unit and format for every column the table happens to have."""
    for name in t.df.columns:
        if name == t.group_by:
            continue
        settings: dict = {"label": H.get(name, name)}
        if name in UNITS:
            settings["unit"] = UNITS[name]
        if name in FMT:
            settings["fmt"] = FMT[name]
        if (best and name in FMT and name not in NOT_COMPARED
                and name not in ("d_max", "d_min")):
            settings["best"] = "max" if name in LARGER_IS_BETTER else "min"
        t.column(name, **settings)
    return t


def write(t: Table, stem: str, caption: str, **opts) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tex = OUT / f"{stem}.tex"
    tex.write_text(
        t.render("mydesign", float=True, caption=caption,
                 label=f"tab:{stem}", **opts)
    )
    (OUT / f"{stem}.md").write_text(t.render("markdown"))
    print(f"  {tex.relative_to(ROOT)}  ({len(t.df)} rows)")


# the tables -----------------------------------------------------------------

MERGING_COLUMNS = [
    "completeness", "multiplicity", "i_over_sigma", "r_merge", "r_meas",
    "r_pim", "cc_half", "cc_anom", "wilson_b",
]


def table_merging_by_arm() -> None:
    """Every arm at the overall scope: the comparison table."""
    rows = []
    for arm, est, d in arms():
        summary = metrics.merging_summary_from_log(d / "merge.log")
        if not summary:
            continue
        row = {"estimator": ESTIMATOR_LABELS[est], "arm": ARM_LABELS[arm]}
        row.update({k: summary["overall"].get(k) for k in MERGING_COLUMNS})
        rows.append(row)
    if not rows:
        return
    t = Table(rows, group_by="estimator")
    write(
        configure(t, best=True), "merging_by_arm",
        "Merging statistics for each profile model, at the overall scope "
        "reported by \\texttt{dials.merge}. Both estimators are shown: "
        "the Rao-Blackwellized profile fit computed at export and the "
        "integrator's own posterior mean. The best value in each group is "
        "marked.",
        notes=[
            "$\\Delta B_\\mathrm{aniso}$ and the Wilson $B$ come from the "
            "maximum-likelihood fit \\texttt{dials.merge} performs on the "
            "merged intensities.",
        ],
    )


def table_r_factors() -> None:
    """Every R-factor, merged and on the anomalous pairs kept apart.

    The per-shell table carries `r_merge`, `r_meas` and `r_pim`; only the
    summary has the `(I+/-)` forms, which are computed with the Bijvoet
    mates held apart and so are the ones to read for an anomalous
    experiment.
    """
    cols = ["r_merge", "r_merge_anom", "r_meas", "r_meas_anom",
            "r_pim", "r_pim_anom", "cc_anom", "anom_slope", "df_over_f",
            "di_over_sigma_di"]
    rows = []
    for arm, est, d in arms():
        summary = metrics.merging_summary_from_log(d / "merge.log")
        if not summary:
            continue
        row = {"estimator": ESTIMATOR_LABELS[est], "arm": ARM_LABELS[arm]}
        row.update({k: summary["overall"].get(k) for k in cols})
        rows.append(row)
    if not rows:
        return
    t = Table(rows, group_by="estimator")
    write(
        configure(t, best=True), "r_factors",
        "Residuals and anomalous diagnostics for each profile model. A "
        "superscript $\\pm$ marks the form computed with the Bijvoet mates "
        "held apart, which is the one to read for an anomalous experiment "
        "and which the per-shell table does not carry.",
    )


def table_error_model() -> None:
    """What `dials.scale` fitted, and how much it had to reject."""
    rows = []
    for arm, est, d in arms():
        model = metrics.error_model_from_log(d / "scale.log") or {}
        scaling = metrics.scaling_summary_from_log(d / "scale.log") or {}
        if not model and not scaling:
            continue
        rows.append({
            "estimator": ESTIMATOR_LABELS[est],
            "arm": ARM_LABELS[arm],
            "a": model.get("a"),
            "b": model.get("b"),
            "i_over_sigma_limit": model.get("i_over_sigma_limit"),
            "n_param_total": scaling.get("n_param_total"),
            "n_outliers": scaling.get("n_outliers"),
            "rmsd_i_final": scaling.get("rmsd_i_final"),
            "pct_param_uncertain": scaling.get("pct_param_uncertain"),
        })
    if not rows:
        return
    t = Table(rows, group_by="estimator")
    write(
        configure(t), "error_model",
        "The scaling error model each arm was corrected with, "
        "$\\sigma'^2 = a^2(\\sigma^2 + (bI)^2)$, and what the scaling run "
        "cost. $a$ inflates every uncertainty and $b$ sets the asymptotic "
        "$I/\\sigma_I$ a bright reflection can reach, so an arm whose "
        "exported $\\sigma$ is too small is corrected by a larger $a$ "
        "rather than being penalised.",
        notes=[
            "$N_\\mathrm{outlier}$ is the final round's rejection count, "
            "not the sum over rounds.",
        ],
    )


def table_wilson() -> None:
    rows = []
    for arm, est, d in arms():
        w = metrics.wilson_from_log(d / "merge.log")
        if not w:
            continue
        rows.append({
            "estimator": ESTIMATOR_LABELS[est],
            "arm": ARM_LABELS[arm],
            "wilson_b": w.get("wilson_b"),
            "neg_log_scale": w.get("neg_log_scale"),
            "b_aniso_range": w.get("b_aniso_range"),
        })
    if not rows:
        return
    t = Table(rows, group_by="estimator")
    write(
        configure(t), "wilson",
        "The maximum-likelihood Wilson analysis \\texttt{dials.merge} "
        "performs on each merged dataset. $\\Delta B_\\mathrm{aniso}$ is "
        "the spread between the largest and smallest eigenvalue of the "
        "anisotropic $B$ tensor, so it is how much the falloff differs "
        "between the best and worst direction.",
    )


def table_shells(arm: str = "m9_geomshift", est: str = "rb") -> None:
    """The per-shell table for one arm: the dense case."""
    d = LOGS / f"{arm}_{est}"
    shells = metrics.shells_from_merge_log(d / "merge.log")
    if not shells:
        return
    cols = ["d_max", "d_min", "n_obs", "n_uniq", "multiplicity",
            "completeness", "i_over_sigma", "r_merge", "r_meas", "r_pim",
            "cc_half", "cc_anom"]
    rows = [{c: s.get(c) for c in cols if c in s} for s in shells]
    t = Table(rows)
    t.columns["n_uniq"].label = H["n_unique"]
    # a per-shell table is descriptive: nothing in it is a winner
    write(
        configure(t), "merging_shells",
        f"Merging statistics by resolution shell for the "
        f"{ARM_LABELS.get(arm, arm)} arm, {ESTIMATOR_LABELS.get(est, est)}, "
        "as \\texttt{dials.merge} reports them.",
    )


def table_anomalous_differences() -> None:
    """`<|dF|/sigma(dF)>` by shell, arms across the columns.

    The shells are the rows because there are twenty of them and five arms;
    the transpose would not fit the measure.
    """
    by_arm: dict[str, list[dict]] = {}
    shells: list[tuple[float, float]] = []
    for arm, est, d in arms():
        if est != "rb":
            continue
        rows = metrics.anomalous_differences_from_log(d / "merge.log")
        if not rows:
            continue
        by_arm[ARM_LABELS[arm]] = rows
        if not shells:
            shells = [(r["d_max"], r["d_min"]) for r in rows]
    if not by_arm:
        return
    out = []
    for i, (d_max, d_min) in enumerate(shells):
        row: dict = {"d_max": d_max, "d_min": d_min}
        for label, rows in by_arm.items():
            row[label] = rows[i]["dano_over_sigma"] if i < len(rows) else None
        out.append(row)
    t = Table(out)
    for name in t.df.columns:
        if name in ("d_max", "d_min"):
            continue
        t.column(name, fmt="{:.3f}")
    t.column("d_max", label=H["d_max"], unit=UNITS["d_max"], fmt=FMT["d_max"])
    t.column("d_min", label=H["d_min"], unit=UNITS["d_min"], fmt=FMT["d_min"])
    write(
        t, "anomalous_differences",
        "$\\langle|\\Delta F|/\\sigma(\\Delta F)\\rangle$ by resolution "
        "shell, one column per profile model, from the Rao-Blackwellized "
        "export. The mean anomalous difference in units of its own "
        "uncertainty: it falls below 1 where the differences stop being "
        "measurable, which is a resolution cut chosen by the data rather "
        "than by the quantity under test.",
    )



# the numbers, long format ---------------------------------------------------
#
# The tables above are one layout of these rows. A second layout -- a slide,
# a different journal's column budget, someone else's table style -- should
# start from the numbers rather than from a parsed `.tex`, so every statistic
# this script reads is also written out long.

STATS_CSV = "dials_log_stats.csv"
LABELS_CSV = "statistic_labels.csv"

# the long-format columns, in order
STATS_COLUMNS = (
    "method", "dataset", "estimator", "section", "region",
    "d_max", "d_min", "statistic", "value",
)

DATASET = "hewl_9b7c"

# One name per quantity, where two readers spell it differently.
STATISTIC_NAMES = {"n_uniq": "n_unique"}

# Which reader each section comes from, and which log it reads. `region` is
# the scope within that log: `dials.merge` prints its summary three times,
# over everything and over the lowest and highest resolution shell.
SECTIONS = (
    ("merging_summary", "merge.log"),
    ("error_model", "scale.log"),
    ("scaling", "scale.log"),
    ("wilson", "merge.log"),
    ("shells", "merge.log"),
    ("anomalous_differences", "merge.log"),
)


def _rows_for(section: str, d: Path) -> list[dict]:
    """The long-format rows of one section, for one arm and estimator."""
    out: list[dict] = []

    def add(statistic, value, region="", d_max=None, d_min=None):
        if value is None or isinstance(value, str):
            return
        out.append({
            "section": section, "region": region,
            "d_max": d_max, "d_min": d_min,
            # one name per quantity across every section: the per-shell
            # table calls the unique count `n_uniq` and the summary calls
            # it `n_unique`, and two names for one quantity is how a pivot
            # comes out with two half-empty columns
            "statistic": STATISTIC_NAMES.get(statistic, statistic),
            # `dials.merge` marks a significant correlation with a trailing
            # `*`, which the reader returns as a bool; it travels as 0/1 so
            # the column is numeric throughout
            "value": int(value) if isinstance(value, bool) else value,
        })

    if section == "merging_summary":
        summary = metrics.merging_summary_from_log(d / "merge.log") or {}
        for region, stats in summary.items():
            for statistic, value in stats.items():
                add(statistic, value, region=region)
    elif section == "error_model":
        for statistic, value in (
            metrics.error_model_from_log(d / "scale.log") or {}
        ).items():
            add(statistic, value)
    elif section == "scaling":
        for statistic, value in (
            metrics.scaling_summary_from_log(d / "scale.log") or {}
        ).items():
            add(statistic, value)
    elif section == "wilson":
        for statistic, value in (
            metrics.wilson_from_log(d / "merge.log") or {}
        ).items():
            add(statistic, value)
    elif section == "shells":
        for shell in metrics.shells_from_merge_log(d / "merge.log"):
            for statistic, value in shell.items():
                if statistic in ("d_max", "d_min"):
                    continue
                add(statistic, value, region="shell",
                    d_max=shell["d_max"], d_min=shell["d_min"])
    elif section == "anomalous_differences":
        for shell in metrics.anomalous_differences_from_log(d / "merge.log"):
            add("dano_over_sigma", shell.get("dano_over_sigma"),
                region="shell", d_max=shell.get("d_max"),
                d_min=shell.get("d_min"))
    return out


def write_long_format() -> None:
    """Every statistic read out of the logs, one row each."""
    rows = []
    for arm, est, d in arms():
        for section, log in SECTIONS:
            if not (d / log).is_file():
                continue
            for row in _rows_for(section, d):
                rows.append({
                    "method": arm, "dataset": DATASET, "estimator": est,
                    **row,
                })
    if not rows:
        return
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / STATS_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=STATS_COLUMNS, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"  {(DATA / STATS_CSV).relative_to(ROOT)}  ({len(rows)} rows)")


def write_statistic_labels() -> None:
    """How each statistic is set: header, unit, format, which way is better.

    The same four dictionaries the tables above are configured from, written
    out so a table built elsewhere sets a quantity the same way -- single
    letters italic, multi-letter names upright, units on their own header
    line, and a best value marked only where a comparison is meaningful.
    """
    # `arm` and `estimator` are row keys, not statistics
    names = sorted((set(H) | set(FMT) | set(UNITS)) - {"arm", "estimator"})
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / LABELS_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["statistic", "header", "unit", "fmt",
                    "larger_is_better", "compared"])
        for name in names:
            w.writerow([
                name,
                H.get(name, name),
                UNITS.get(name, ""),
                FMT.get(name, ""),
                "1" if name in LARGER_IS_BETTER else "0",
                "0" if name in NOT_COMPARED else "1",
            ])
    print(f"  {(DATA / LABELS_CSV).relative_to(ROOT)}  ({len(names)} rows)")



# the hyperparameter tables --------------------------------------------------

# Which group a varying parameter is taken from first, when the table has to
# be narrowed. Architecture before optimisation: what the model IS matters
# more in a methods table than how it was fitted.
GROUP_ORDER = ("integrator", "surrogate", "loss", "encoder", "optimizer",
               "trainer", "data")
# A paper table wider than this stops being readable at full measure.
MAX_COLUMNS = 8
HPARAMS = ROOT / "docs" / "figures" / "data" / "hparams.csv"
COUNTS = ("cmp_mono.csv", "cmp_1118.csv")


def _hparam_rows() -> list[dict]:
    if not HPARAMS.is_file():
        return []
    import csv as _csv

    with open(HPARAMS, newline="") as fh:
        return list(_csv.DictReader(fh))


def _parameter_counts() -> dict[tuple[str, str], int]:
    """Total parameters per (dataset, method), where it was measured."""
    import csv as _csv

    out = {}
    for name in COUNTS:
        path = HPARAMS.parent / name
        if not path.is_file():
            continue
        with open(path, newline="") as fh:
            for r in _csv.DictReader(fh):
                if r.get("quantity") == "n_params_total":
                    try:
                        out[(r["dataset"], r["method"])] = int(float(r["value"]))
                    except (TypeError, ValueError):
                        pass
    return out


def _columns_for(rows: list[dict]) -> list[str]:
    """The varying parameters worth a column, narrowed to MAX_COLUMNS.

    Two rules do the narrowing, and both are about information rather than
    taste.

    A parameter that varies only because it is ABSENT from some arms is
    dropped: it takes one value wherever it appears, so a column of it is a
    column of one value and a blank, which says less than a sentence.

    A parameter whose value PATTERN across the arms is identical to one
    already chosen is redundant and dropped. The window size shows up as
    `integrator.h`, `integrator.w`, `encoder.*.input_shape` and
    `surrogate.qp.output_dim` at once; they partition the arms identically,
    so one column carries all four.
    """
    varying = [r for r in rows if r.get("varies") in ("1", 1, True)]
    by_name: dict[str, dict[str, str]] = {}
    groups: dict[str, str] = {}
    for r in varying:
        by_name.setdefault(r["parameter"], {})[r["method"]] = r["value"]
        groups[r["parameter"]] = r["group"]

    methods = {r["method"] for r in rows}
    chosen: list[str] = []
    seen_patterns: list[tuple] = []

    def rank(name: str) -> tuple:
        g = groups[name]
        return (GROUP_ORDER.index(g) if g in GROUP_ORDER else len(GROUP_ORDER),
                name)

    for name in sorted(by_name, key=rank):
        values = by_name[name]
        if len(set(values.values())) < 2:
            continue  # varies only by being absent
        if len(values) < len(methods):
            continue
        # Compare the PARTITION a parameter induces, not its values.
        # `integrator.h` is 21 or 35 and `surrogate.qp.output_dim` is 441
        # or 1225; the raw tuples differ while the split of the arms is the
        # same, and only one of the two is worth a column.
        order: dict[str, int] = {}
        pattern = tuple(
            order.setdefault(values.get(m, ""), len(order))
            for m in sorted(methods)
        )
        if pattern in seen_patterns:
            continue
        seen_patterns.append(pattern)
        chosen.append(name)
        if len(chosen) >= MAX_COLUMNS:
            break
    return chosen


def table_hyperparameters() -> None:
    """One row per model, columns the settings that tell them apart.

    Built from each run's own `config_log.yaml`, so it reports what trained
    rather than what a repository config says now. The constants are not
    here by design: a column with one value in every row tells a reader
    nothing, and `hparams.csv` beside this carries every parameter whether
    it varies or not.
    """
    rows = _hparam_rows()
    if not rows:
        print("  hyperparameters: no hparams.csv "
              "(pull_figdata.sh hparams)")
        return
    counts = _parameter_counts()

    for dataset in sorted({r["dataset"] for r in rows if r["dataset"]}):
        here = [r for r in rows if r["dataset"] == dataset]
        columns = _columns_for(here)
        if not columns:
            continue
        values: dict[str, dict[str, str]] = {}
        for r in here:
            if r["parameter"] in columns:
                values.setdefault(r["method"], {})[r["parameter"]] = r["value"]

        # Short column names, lengthened where two would collide:
        # `integrator.name` and `surrogate.qp.name` are both "name", and
        # the first is the most important column in the table.
        tails = [c.rsplit(".", 1)[-1] for c in columns]
        labels = {
            c: (tail if tails.count(tail) == 1
                else ".".join(c.split(".")[-2:]))
            for c, tail in zip(columns, tails, strict=True)
        }
        out = []
        for method in sorted(values):
            row = {"arm": method}
            row.update({labels[c]: values[method].get(c, "")
                        for c in columns})
            n = counts.get((dataset, method))
            if n is not None:
                row["n_params"] = n
            out.append(row)
        if not out:
            continue

        stem = f"hyperparameters_{dataset}"
        t = Table(out)
        n_total = len({r["parameter"] for r in here})
        n_varying = len({r["parameter"] for r in here
                         if r["varies"] in ("1", 1, True)})
        write(
            configure(t), stem,
            f"Model settings for the {dataset.replace('_', ' ')} arms, read "
            "from each run's own configuration log rather than from the "
            f"repository. {n_varying} of {n_total} configured parameters "
            "differ across these arms; the columns are those that take more "
            "than one value and that are not determined by another column "
            "already shown. The full set, varying or not, is in "
            "\texttt{hparams.csv}.",
        )


def main() -> int:
    if not LOGS.is_dir():
        raise SystemExit(
            f"no logs at {LOGS}; run `bash scripts/figures/pull_figdata.sh "
            "dials_logs` first"
        )
    print(f"tables -> {OUT.relative_to(ROOT)}")
    table_merging_by_arm()
    table_r_factors()
    table_error_model()
    table_wilson()
    table_shells()
    table_anomalous_differences()
    table_hyperparameters()
    write_long_format()
    write_statistic_labels()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
