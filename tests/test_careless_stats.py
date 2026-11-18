"""The polychromatic per-shell statistics, from careless.stats to the figure.

`tests/data/careless/<tool>.csv` carry the column names careless itself
writes, read out of `careless/stats/<tool>.py`: cchalf, ccanom and rsplit
split by cross-validation `repeat`, ccpred by `test` (Train/Test), and each
names its statistic after itself (`CChalf`, `I/sigI`, ...). The numbers in
them are made up; the shapes and the names are not.

What these pin is the join between the three pieces -- what careless writes,
what `_tidy_careless_stats` melts it into, and what the figure draws -- which
is where a renamed column goes quiet rather than raising: the melt keeps
every column it does not recognise as an identifier, so a statistic that
stopped being numeric would simply stop appearing in the figure.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "careless"
TOOLS = ("cchalf", "ccanom", "ccpred", "rsplit", "isigi", "completeness")
# the tools careless gives a `-m/--method` flag: the correlations
CORRELATION_TOOLS = ("cchalf", "ccanom", "ccpred")

sys.path.insert(0, str(ROOT / "scripts" / "poly"))
sys.path.insert(0, str(ROOT / "scripts" / "figures"))

import fig_careless_stats as figure  # noqa: E402
import polars as pl  # noqa: E402
import run_pipeline  # noqa: E402


@pytest.fixture(scope="module")
def tidy(tmp_path_factory):
    """The long table, built by the pipeline's own tidy step."""
    out = tmp_path_factory.mktemp("stats") / "careless_stats.csv"
    run_pipeline._tidy_careless_stats(
        # (tool, path, correlation). The correlation is what `-m` asked
        # careless for; the tools that do not take `-m` carry "".
        [(t, DATA / f"{t}.csv", "spearman" if t in CORRELATION_TOOLS else "")
         for t in TOOLS],
        out,
        {"keys": {"method": "integrator", "dataset": "hewl_1118"}},
        49,
    )
    return pl.read_csv(out, infer_schema_length=None)


# the long table -------------------------------------------------------------


def test_every_tool_reaches_the_long_table(tidy):
    assert set(tidy["tool"].unique().to_list()) == set(TOOLS)


def test_the_key_columns_are_present(tidy):
    assert {"method", "dataset", "epoch", "tool", "statistic",
            "bin", "shell", "group", "value"} <= set(tidy.columns)
    assert tidy["method"].unique().to_list() == ["integrator"]
    assert tidy["dataset"].unique().to_list() == ["hewl_1118"]


def test_each_tool_keeps_its_own_statistic(tidy):
    """careless names the column after the tool; nothing renames it."""
    named = {
        (r["tool"], r["statistic"]) for r in tidy.iter_rows(named=True)
    }
    assert ("cchalf", "CChalf") in named
    assert ("ccanom", "CCanom") in named
    assert ("ccpred", "CCpred") in named
    assert ("rsplit", "Rsplit") in named
    assert ("isigi", "I/sigI") in named


def test_the_identifier_columns_are_not_melted_into_statistics(tidy):
    """`bin` and `Spacegroup` are row labels, not measurements."""
    statistics = set(tidy["statistic"].unique().to_list())
    assert "bin" not in statistics
    assert "Spacegroup" not in statistics
    assert "repeat" not in statistics


def test_the_crossvalidation_repeat_becomes_the_group(tidy):
    repeats = tidy.filter(pl.col("tool") == "cchalf")["group"].unique()
    assert sorted(str(v) for v in repeats.to_list()) == ["0", "1", "2"]


def test_ccpred_groups_on_train_against_test(tidy):
    groups = tidy.filter(pl.col("tool") == "ccpred")["group"].unique()
    assert sorted(str(v) for v in groups.to_list()) == ["Test", "Train"]


def test_a_tool_without_a_grouping_leaves_it_empty(tidy):
    groups = tidy.filter(pl.col("tool") == "isigi")["group"].unique().to_list()
    assert [g for g in groups if g not in (None, "")] == []


def test_the_shell_label_survives_the_melt(tidy):
    shells = tidy.filter(pl.col("tool") == "isigi")["shell"].to_list()
    assert shells[0] == "19.90 - 3.52"


# what the figure makes of it ------------------------------------------------


def test_one_panel_per_tool(tidy):
    panels = figure.panels_of(tidy)
    assert [t for t, _, _ in panels] == list(figure.TOOL_ORDER)


def test_panels_are_ordered_as_reported(tidy):
    """CC-half and I/sigma are the two named in docs/figures/README.md."""
    order = [t for t, _, _ in figure.panels_of(tidy)]
    assert order.index("cchalf") < order.index("isigi")


def test_careless_column_names_get_a_typeset_label():
    assert figure.label_for("cchalf", "CChalf") == r"$\mathrm{CC}_{1/2}$"
    assert figure.label_for("isigi", "I/sigI") == r"$I/\sigma_I$"
    assert figure.label_for("rsplit", "Rsplit") == r"$R_\mathrm{split}$"


def test_multi_letter_names_are_upright_not_italic():
    """`$CC$` sets two italic variables; the name is upright."""
    for tool, stat in (("cchalf", "CChalf"), ("ccanom", "CCanom"),
                       ("ccpred", "CCpred")):
        assert r"\mathrm{CC}" in figure.label_for(tool, stat)


def test_a_statistic_nobody_named_is_still_drawn():
    """A figure must not quietly omit a number the pipeline produced."""
    assert figure.label_for("newtool", "SomeNewStatistic") == \
        "SomeNewStatistic"


def test_the_resolution_axis_is_one_over_d_squared(tidy):
    shells = tidy.filter(pl.col("tool") == "isigi")["shell"].to_list()
    x = figure.shell_axis(shells)
    assert x == sorted(x)  # low resolution first
    # the first shell spans 19.90-3.52; its centre in 1/d^2 is 4.90 A, not
    # the arithmetic mean of 11.71 A
    assert 1.0 / x[0] ** 0.5 == pytest.approx(4.90, abs=0.01)


def test_a_shell_label_that_is_not_a_range_does_not_raise():
    x = figure.shell_axis(["overall", ""])
    assert len(x) == 2
    assert all(v != v for v in x)  # NaN


def test_repeats_split_on_the_group_not_on_the_epoch_ramp(tidy):
    """A cross-validation repeat is not an ordering and is not a ramp."""
    panel = tidy.filter(pl.col("tool") == "cchalf")
    column, lines = figure.series_of(panel)
    assert column == "group"
    assert [n for n, _ in lines] == ["0", "1", "2"]


def test_several_methods_outrank_the_grouping(tidy):
    """Two methods on one axis is the comparison worth drawing."""
    other = tidy.filter(pl.col("tool") == "cchalf").with_columns(
        pl.lit("laue-dials").alias("method")
    )
    both = pl.concat([tidy.filter(pl.col("tool") == "cchalf"), other])
    column, lines = figure.series_of(both)
    assert column == "method"
    assert sorted(n for n, _ in lines) == ["integrator", "laue-dials"]


def test_one_series_when_nothing_varies(tidy):
    panel = tidy.filter(pl.col("tool") == "isigi")
    column, lines = figure.series_of(panel)
    assert column == ""
    assert len(lines) == 1


# the correlation method ------------------------------------------------------


class TestCorrelationMethod:
    """Pearson and Spearman are different numbers and must stay separable.

    They do not rank arms the same way -- Pearson follows the bright end,
    Spearman is insensitive to the scale -- so a table that lost which one
    it held would silently mix them in one panel.
    """

    def test_the_correlation_is_its_own_column(self, tidy):
        assert "correlation" in tidy.columns

    def test_only_the_tools_with_a_method_flag_carry_one(self, tidy):
        carried = (tidy.filter(pl.col("correlation") != "")["tool"]
                   .unique().sort().to_list())
        assert carried == sorted(CORRELATION_TOOLS)

    def test_it_is_not_folded_into_the_arm_or_the_statistic(self, tidy):
        # `method` is the arm and `statistic` names the column within a
        # tool's own CSV; neither may absorb the correlation
        assert tidy["method"].unique().to_list() == ["integrator"]
        assert not any("spearman" in s.lower()
                       for s in tidy["statistic"].unique().to_list())

    def test_two_methods_of_one_tool_coexist(self, tmp_path):
        out = tmp_path / "both.csv"
        run_pipeline._tidy_careless_stats(
            [("cchalf", DATA / "cchalf.csv", "spearman"),
             ("cchalf", DATA / "cchalf.csv", "pearson")],
            out,
            {"keys": {"method": "integrator", "dataset": "hewl_1118"}},
            49,
        )
        df = pl.read_csv(out, infer_schema_length=None)
        assert sorted(df["correlation"].unique().to_list()) == [
            "pearson", "spearman"]
        # the same tool and statistic, told apart only by `correlation`
        one = df.filter((pl.col("tool") == "cchalf")
                        & (pl.col("correlation") == "pearson"))
        assert one.height * 2 == df.height


def test_a_tool_without_a_method_flag_still_gets_a_panel(tidy):
    """rsplit, isigi and completeness take no `-m` and must not vanish.

    Their correlation is an empty field, which a CSV round-trip returns as
    null rather than "". Matching on equality dropped all three from the
    figure while cchalf, ccanom and ccpred still drew, so the figure got
    shorter and nothing raised.
    """
    tools = {t for t, _, _ in figure.panels_of(tidy)}
    assert {"rsplit", "isigi", "completeness"} <= tools


def test_the_correlation_reaches_the_panel_name(tidy):
    """Two correlations of one tool must be distinguishable in the figure."""
    names = [s for t, s, _ in figure.panels_of(tidy) if t == "cchalf"]
    assert all("spearman" in n for n in names), names
