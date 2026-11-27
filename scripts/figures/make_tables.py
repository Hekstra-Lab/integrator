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


SIM_TRUTH_SCORES = ROOT / "docs" / "figures" / "data" / "sim_truth_scores_mos010_pc.csv"
# truth-intensity bins reported in the table, (lo, hi) in photons, as cut by
# `score_against_truth.py` (EDGES there)
SIM_TRUTH_BINS = ((10, 30), (100, 300), (1000, 3000))


def table_sim_truth_scores() -> None:
    """laue.integrate and every trained arm against the simulated truth.

    Rows are (dataset, method); per bin of the truth intensity, the mean
    relative deviation and the standard deviation of z, from the long table
    of `score_against_truth.py --out`. The whole-domain row of that file
    gives N and sd z overall.
    """
    if not SIM_TRUTH_SCORES.is_file():
        return
    import polars as pl

    df = pl.read_csv(SIM_TRUTH_SCORES, infer_schema_length=None)
    rows = []
    for (dataset, method), g in df.group_by(["dataset", "method"], maintain_order=True):
        row = {"dataset": dataset, "method": method}
        tot = g.filter(pl.col("i_true_hi") == float("inf"))
        row["n"] = int(tot["n"][0]) if tot.height else None
        for lo, hi in SIM_TRUTH_BINS:
            b = g.filter((pl.col("i_true_lo") == lo) & (pl.col("i_true_hi") == hi))
            row[f"rel_dev_{lo}"] = float(b["rel_dev_mean"][0]) if b.height else None
        for lo, hi in SIM_TRUTH_BINS:
            b = g.filter((pl.col("i_true_lo") == lo) & (pl.col("i_true_hi") == hi))
            row[f"z_sd_{lo}"] = float(b["z_sd"][0]) if b.height else None
        row["z_sd_all"] = float(tot["z_sd"][0]) if tot.height else None
        rows.append(row)
    t = Table(rows, group_by="dataset")
    t.column("method", label="method")
    t.column("n", label="$N$", fmt="{:.0f}")
    for lo, hi in SIM_TRUTH_BINS:
        t.column(f"rel_dev_{lo}", label=rf"$\langle\Delta I/I_\mathrm{{true}}\rangle_{{[{lo},{hi})}}$", fmt="{:+.3f}")
    for lo, hi in SIM_TRUTH_BINS:
        t.column(f"z_sd_{lo}", label=rf"$\mathrm{{sd}}\,z_{{[{lo},{hi})}}$", fmt="{:.2f}")
    t.column("z_sd_all", label=r"$\mathrm{sd}\,z$, all", fmt="{:.2f}")
    write(
        t, "sim_truth_scores",
        "Integrated intensities of the datsimx mos010 photon-counting images "
        "scored against the per-reflection truth, inside the domain of "
        "\\texttt{score\\_against\\_truth.py}. "
        "$\\Delta I = \\hat I - I_\\mathrm{true}$ in photons and "
        "$z = \\Delta I/\\hat\\sigma$; the subscript is the bin of "
        "$I_\\mathrm{true}$ in photons. Each trained arm is shown with both "
        "estimators: \\texttt{\\_rb}, the profile fit at export, and "
        "\\texttt{\\_posterior}, the posterior mean.",
    )


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


# the mos010 careless -> phenix arms: name in the tables, profile model, export, and
# the method name the truth-side CRPS table (sim_crps_mos010_pc.csv) uses for the same export
SIM_ARMS = {
    "laue_integrate": ("laue.integrate", "ellipse (laue-dials)", "summation / fit", "laue_integrate"),
    "svae_rb": ("svae", "learned basis", r"$i_\mathrm{rb}$", "svae_rb"),
    "svae_bg_rb": (r"svae\_bg", "learned basis", r"$i_\mathrm{rb}$", "svae_bg_rb"),
    "t0_rb": ("T0", "streak mixture, geometry only", r"$i_\mathrm{rb}$", "svae_kmix_base_rb"),
    "p2_rb": ("P2", "T0, wider geometry MLP, half-width start", r"$i_\mathrm{rb}$", "svae_kmix_h128_grid05_rb"),
    "b1_rb": ("B1", "streak mixture + per-reflection shape", r"$i_\mathrm{rb}$", "svae_kmixv2_post_avg_avg"),
    "b1_laplace": ("B1", "streak mixture + per-reflection shape", "Laplace posterior", "svae_kmixv2_post_avg_lc"),
    "geom_laplace": ("geometric", "streak mixture, width law", "collapsed posterior",
                     "svae_kmixv2_post_collapsed_geom_avg_geom_posterior"),
}
SIM_DOWNSTREAM = ROOT / "docs" / "figures" / "data" / "sim_downstream"
SIM_CRPS = ROOT / "docs" / "figures" / "data" / "sim_crps_mos010_pc.csv"
SIM_SHELLS = ((99.0, 3.0), (3.0, 2.2), (2.2, 1.95), (1.95, 1.78))


def _sim_rows():
    import polars as pl

    ref = pl.read_csv(SIM_DOWNSTREAM / "downstream_refine.csv").filter(seed=1)
    sh = pl.read_csv(SIM_DOWNSTREAM / "downstream_refine_shells.csv").filter(seed=1)
    mg = pl.read_csv(SIM_DOWNSTREAM / "downstream_merged.csv", infer_schema_length=None).filter(seed=1)
    pk = pl.read_csv(SIM_DOWNSTREAM / "downstream_peaks.csv").filter(seed=1)
    crps = pl.read_csv(SIM_CRPS) if SIM_CRPS.is_file() else None
    rows = []
    for m, (name, profile, export, crps_name) in SIM_ARMS.items():
        r = ref.filter(method=m)
        if not r.height:
            continue
        s = sh.filter(method=m)
        hi = s.filter(pl.col("d_min") < 1.95)
        g = mg.filter((pl.col("method") == m) & (pl.col("shell").cast(pl.Utf8) == "all"))
        p = pk.filter(method=m)
        row = {"arm": name, "profile": profile, "export": export,
               "r_work": float(r["r_work"][0]), "r_free": float(r["r_free"][0]),
               "gap": float(r["r_free"][0] - r["r_work"][0]),
               "r_free_hi": float((hi["r_free"] * hi["n_free"]).sum() / hi["n_free"].sum()) if hi.height else None,
               "fom_hi": float((hi["fom"] * hi["n_free"]).sum() / hi["n_free"].sum()) if hi.height else None,
               "phase_hi": float((hi["phase_error"] * hi["n_free"]).sum() / hi["n_free"].sum()) if hi.height else None,
               "r_F_true": float(g["r_F"][0]) if g.height else None,
               "cc_dano": float(g["cc_dano"][0]) if g.height else None,
               "peak_mn": float(p.filter(pl.col("residue") == "MN")["peakz"].mean()) if p.height else None}
        if crps is not None:
            c = crps.filter(method=crps_name)
            for b in ("10-30", "100-300", ">3000"):
                v = c.filter(bin=b)["crps_rel"]
                row[f"crps_{b}"] = float(v[0]) if len(v) else None
        shells = {}
        for dmax, dmin in SIM_SHELLS:
            q = s.filter((pl.col("d_min") >= dmin - 1e-6) & (pl.col("d_max") <= dmax + 1e-6))
            shells[f"{dmax:g}-{dmin:g}"] = (float((q["r_free"] * q["n_free"]).sum() / q["n_free"].sum())
                                            if q.height else None)
        row["_shells"] = shells
        rows.append(row)
    return rows


def table_sim_downstream() -> None:
    """R-work and R-free of every mos010 arm, beside the per-reflection truth scores.

    One careless seed per arm, phenix.refine twice against the pinned flags
    (sim_7lvc_rfree.mtz). r_F is sum |F' - F_true| / sum F_true after a fitted
    scale and B; cc_dano the correlation of merged with true Bijvoet
    differences; CRPS / I_true per bin of the true intensity
    (crps_against_truth.py).
    """
    if not (SIM_DOWNSTREAM / "downstream_refine.csv").is_file():
        return
    rows = _sim_rows()
    main_rows = [{k: v for k, v in r.items() if k != "_shells"} for r in rows]
    t = Table(main_rows)
    t.column("arm", label="arm")
    t.column("profile", label="profile model")
    t.column("export", label="export")
    t.column("r_work", label=r"$R_\mathrm{work}$", fmt="{:.4f}", best="min")
    t.column("r_free", label=r"$R_\mathrm{free}$", fmt="{:.4f}", best="min")
    t.column("gap", label=r"$R_\mathrm{free}-R_\mathrm{work}$", fmt="{:.4f}")
    t.column("r_free_hi", label=r"$R_\mathrm{free}$, $d<1.95$", fmt="{:.4f}", best="min")
    t.column("fom_hi", label=r"FOM, $d<1.95$", fmt="{:.3f}", best="max")
    t.column("phase_hi", label=r"phase err., $d<1.95$", unit="deg", fmt="{:.1f}", best="min")
    t.column("r_F_true", label=r"$r_F$ vs truth", fmt="{:.3f}", best="min")
    t.column("cc_dano", label=r"cc $\Delta F$ vs truth", fmt="{:.3f}", best="max")
    t.column("peak_mn", label=r"Mn peak (mean)", unit=r"$\sigma$", fmt="{:.2f}", best="max")
    for b in ("10-30", "100-300", ">3000"):
        if any(r.get(f"crps_{b}") is not None for r in main_rows):
            t.column(f"crps_{b}", label=rf"CRPS$/I$, {b} ph", fmt="{:.3f}", best="min")
    write(
        t, "sim_downstream_refinement",
        "The mos010 photon-counting simulation: each integrator's export merged by careless "
        "(config 4, one seed) and refined twice by phenix against one pinned R-free set. "
        "Refinement columns are phenix's; $r_F$ and cc $\\Delta F$ compare the merged structure "
        "factors with the simulation's true ones after a fitted scale and $B$; CRPS$/I$ scores each "
        "export's per-reflection predictive distribution against the true intensity. "
        "Single merging seed: $R_\\mathrm{free}$ differences below about 0.008 are not resolved.",
    )
    shell_rows = []
    for r in rows:
        sr = {"arm": r["arm"], "export": r["export"]}
        sr.update(r["_shells"])
        shell_rows.append(sr)
    t2 = Table(shell_rows)
    t2.column("arm", label="arm")
    t2.column("export", label="export")
    for dmax, dmin in SIM_SHELLS:
        k = f"{dmax:g}-{dmin:g}"
        lab = rf"$d>{dmin:g}$ \AA" if dmax > 50 else rf"{dmax:g}--{dmin:g} \AA"
        t2.column(k, label=rf"$R_\mathrm{{free}}$, {lab}", fmt="{:.4f}", best="min")
    write(t2, "sim_downstream_rfree_shells",
          "$R_\\mathrm{free}$ of each mos010 arm by resolution range, from phenix's final 20-shell "
          "table, combined with weights equal to the number of free reflections per shell.")


# mos010 against mos025 (mosaic spread 0.10 and 0.25, otherwise the same datsimx row), every
# careless seed each has: (collect_downstream.py directory, dataset key). mos025's arms are the
# epoch-39-59 averaged exports; mos010's svae_bg, T0 and P2 rows are last-epoch exports. Both are
# compared at careless seed 1, the one seed every mos010 arm has.
SIM_MOSAIC = {
    "mos010": (ROOT / "docs" / "figures" / "data" / "sim_downstream", "mos010_pc_default"),
    "mos025": (ROOT / "docs" / "figures" / "data" / "sim_downstream_mos025", "mos025_pc_m0_default"),
}
SIM_MOSAIC_SEED = 1
SIM_MOSAIC_ARMS = {
    "laue_integrate": ("laue.integrate", "summation / fit"),
    "svae_bg_rb": (r"svae\_bg", r"$i_\mathrm{rb}$"),
    "t0_rb": ("T0", r"$i_\mathrm{rb}$"),
    "p2_rb": ("P2", r"$i_\mathrm{rb}$"),
    "b1_rb": ("B1", r"$i_\mathrm{rb}$"),
    "geom_both_rb": ("geom\\_both", r"$i_\mathrm{rb}$"),
    "svae_bg_posterior": (r"svae\_bg", "posterior"),
    "t0_posterior": ("T0", "posterior"),
    "p2_posterior": ("P2", "posterior"),
    "b1_posterior": ("B1", "posterior"),
    "geom_both_posterior": ("geom\\_both", "posterior"),
}


def _sim_mosaic_rows():
    """Per arm and dataset: seed means of the refinement, the merged-vs-truth r_F and careless.stats."""
    import polars as pl

    tabs = {}
    for k, (d, ds) in SIM_MOSAIC.items():
        if not (d / "downstream_refine.csv").is_file():
            continue
        cs = d / "downstream_careless_stats.csv"
        tabs[k] = dict(
            ref=pl.read_csv(d / "downstream_refine.csv").filter(dataset=ds, seed=SIM_MOSAIC_SEED),
            mg=pl.read_csv(d / "downstream_merged.csv", infer_schema_length=None).filter(
                (pl.col("dataset") == ds) & (pl.col("shell").cast(pl.Utf8) == "all") & (pl.col("seed") == SIM_MOSAIC_SEED)),
            cs=(pl.read_csv(cs, infer_schema_length=None).filter(dataset=ds, seed=SIM_MOSAIC_SEED)
                if cs.is_file() else None))

    def careless(c, m, tool, b, statistic=None, correlation=None, group=None):
        if c is None:
            return None
        q = c.filter((pl.col("method") == m) & (pl.col("tool") == tool) & (pl.col("bin") == b))
        if statistic is not None:
            q = q.filter(pl.col("statistic") == statistic)
        if correlation is not None:
            q = q.filter(pl.col("correlation") == correlation)
        if group is not None:
            q = q.filter(pl.col("group").cast(pl.Utf8) == group)
        if not q.height:
            return None
        # mean over careless's repeat groups within a seed, then over seeds
        return float(q.group_by("seed").agg(pl.col("value").mean())["value"].mean())

    laue = {k: t["ref"].filter(method="laue_integrate")["r_free"].mean() for k, t in tabs.items()}
    rows = []
    for m, (arm, export) in SIM_MOSAIC_ARMS.items():
        row = {"arm": arm, "export": export}
        seen = False
        for k, t in tabs.items():
            r = t["ref"].filter(method=m)
            if not r.height:
                continue
            seen = True
            cc = t["cs"].filter((pl.col("method") == m) & (pl.col("tool") == "cchalf")) if t["cs"] is not None else None
            outer = int(cc["bin"].max()) if cc is not None and cc.height else None  # completeness has an extra bin
            g = t["mg"].filter(method=m)
            row.update({
                f"{k}_r_work": float(r["r_work"].mean()),
                f"{k}_r_free": float(r["r_free"].mean()),
                f"{k}_d_laue": float(r["r_free"].mean() - laue[k]) if laue.get(k) is not None else None,
                f"{k}_r_F": float(g["r_F"].mean()) if g.height else None,
                f"{k}_cchalf_in": careless(t["cs"], m, "cchalf", 1, correlation="pearson"),
                f"{k}_cchalf_out": careless(t["cs"], m, "cchalf", outer, correlation="pearson") if outer else None,
                f"{k}_ccpred_out": careless(t["cs"], m, "ccpred", outer, correlation="pearson", group="Test") if outer else None,
                f"{k}_rsplit_out": careless(t["cs"], m, "rsplit", outer) if outer else None,
                f"{k}_isigi_out": careless(t["cs"], m, "isigi", outer) if outer else None,
            })
        if seen:
            rows.append(row)
    return rows, list(tabs)


def table_sim_mosaic() -> None:
    """mos010 against mos025: refinement and careless.stats per arm, averaged over careless seeds.

    The two datasets are one datsimx row at mosaic spread 0.10 and 0.25, through the same
    laue-dials (mos025 with optimize_indexing n_macrocycles=0), 25x25 shoeboxes, careless
    config 4 and two phenix passes against one pinned R-free set.
    """
    rows, sets = _sim_mosaic_rows()
    if not rows:
        return
    lab = {"mos010": "0.10", "mos025": "0.25"}

    def only(*suffixes):
        return [{k: v for k, v in r.items() if k in ("arm", "export") or k.endswith(suffixes)} for r in rows]

    t = Table(only("_r_work", "_r_free", "_d_laue", "_r_F"))
    t.column("arm", label="arm")
    t.column("export", label="export")
    for k in sets:
        t.column(f"{k}_r_work", label=rf"$R_\mathrm{{work}}$, {lab[k]}", fmt="{:.4f}", best="min")
        t.column(f"{k}_r_free", label=rf"$R_\mathrm{{free}}$, {lab[k]}", fmt="{:.4f}", best="min")
        t.column(f"{k}_d_laue", label=rf"$\Delta R_\mathrm{{free}}$ vs laue, {lab[k]}", fmt="{:+.4f}", best="min")
        t.column(f"{k}_r_F", label=rf"$r_F$ vs truth, {lab[k]}", fmt="{:.3f}", best="min")
    write(t, "sim_mosaic_refinement",
          "The same datsimx simulation at mosaic spread 0.10$^\\circ$ (mos010) and 0.25$^\\circ$ (mos025): "
          "each export merged by careless (config 4, seed 1) and refined twice by phenix against one pinned "
          "R-free set; one merging seed, so $R_\\mathrm{free}$ differences below about 0.008 are not resolved. "
          "$\\Delta R_\\mathrm{free}$ is against that dataset's laue.integrate. $r_F$ compares merged with true structure factors "
          "after a fitted scale and $B$. mos025's model exports average the checkpoints of epochs 39--59; "
          "mos010's svae\\_bg, T0 and P2 exports are the last epoch.")
    t2 = Table(only("_cchalf_in", "_cchalf_out", "_ccpred_out", "_rsplit_out", "_isigi_out"))
    t2.column("arm", label="arm")
    t2.column("export", label="export")
    for k in sets:
        t2.column(f"{k}_cchalf_in", label=rf"CC$_{{1/2}}$ inner, {lab[k]}", fmt="{:.3f}", best="max")
        t2.column(f"{k}_cchalf_out", label=rf"CC$_{{1/2}}$ outer, {lab[k]}", fmt="{:.3f}", best="max")
        t2.column(f"{k}_ccpred_out", label=rf"CC$_\mathrm{{pred}}$ test, outer, {lab[k]}", fmt="{:.3f}", best="max")
        t2.column(f"{k}_rsplit_out", label=rf"$R_\mathrm{{split}}$ outer, {lab[k]}", fmt="{:.3f}", best="min")
        t2.column(f"{k}_isigi_out", label=rf"$I/\sigma$ outer, {lab[k]}", fmt="{:.2f}")
    write(t2, "sim_mosaic_careless",
          "careless.stats of the same merges, in careless's innermost and outermost of ten resolution "
          "shells, careless seed 1 (Pearson correlations; each statistic averaged over careless's repeat groups).")


def main() -> int:
    if "--only" in sys.argv and sys.argv[sys.argv.index("--only") + 1] == "sim":
        # the simulation tables read docs/figures/data/sim_downstream and need no DIALS logs
        print(f"tables -> {OUT.relative_to(ROOT)}")
        table_sim_truth_scores()
        table_sim_downstream()
        table_sim_mosaic()
        return 0
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
    table_sim_truth_scores()
    table_sim_downstream()
    table_sim_mosaic()
    write_long_format()
    write_statistic_labels()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
