"""The quantities the model-against-reference scatters compare.

DIALS integrates twice -- a profile fit and a flat sum over the mask -- and
the model has one intensity, so `qi_mean` is drawn against both columns. Two
things had to be got right when the second one was added, and both are the
kind of defect that empties a figure without failing:

* a run exported before `intensity.sum.value` travelled in the prediction
  parquet is missing one column, not all three, and must still produce the
  comparisons it does have;
* `scatter_points.csv` files already dumped call their quantities
  `intensity` and `background`, from before there was a second intensity to
  tell apart, and must still draw.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# `plot_compare.py` sits at the repository root rather than in the package,
# because it runs on the cluster too, so it is loaded by path.
spec = importlib.util.spec_from_file_location(
    "plot_compare", ROOT / "plot_compare.py"
)
assert spec is not None and spec.loader is not None
plot_compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot_compare)


def test_both_dials_integrators_are_compared_against_the_one_model_column():
    q = plot_compare.QUANTITIES
    assert q["intensity_prf"]["dials"] == "intensity.prf.value"
    assert q["intensity_sum"]["dials"] == "intensity.sum.value"
    assert q["intensity_prf"]["model"] == q["intensity_sum"]["model"] == (
        "qi_mean"
    )
    assert q["background_mean"] == {
        "model": "qbg_mean",
        "dials": "background.mean",
    }


def test_every_quantity_has_an_axis():
    assert set(plot_compare.QUANTITIES) == set(plot_compare.AXES)


def test_the_two_intensities_are_labelled_apart():
    axes = plot_compare.AXES
    assert axes["intensity_prf"]["dials"] != axes["intensity_sum"]["dials"]
    # both are compared on a log axis; the background is not
    assert axes["intensity_prf"]["scale"] == "log"
    assert axes["intensity_sum"]["scale"] == "log"
    assert axes["background_mean"]["scale"] == "linear"


def test_a_missing_dials_column_drops_one_quantity_not_all_of_them():
    have = ["refl_ids", "d", "qi_mean", "qbg_mean",
            "intensity.prf.value", "background.mean"]
    present = plot_compare.quantities_in(have)
    assert set(present) == {"intensity_prf", "background_mean"}


def test_a_parquet_with_every_column_gives_every_quantity():
    have = ["refl_ids", "d", "qi_mean", "qbg_mean", "intensity.prf.value",
            "intensity.sum.value", "background.mean"]
    assert set(plot_compare.quantities_in(have)) == set(
        plot_compare.QUANTITIES
    )


def test_the_old_quantity_labels_are_read_as_the_new_ones():
    assert plot_compare.canonical_quantity("intensity") == "intensity_prf"
    assert plot_compare.canonical_quantity("background") == "background_mean"
    # a current label passes through untouched
    assert plot_compare.canonical_quantity("intensity_sum") == "intensity_sum"


def test_every_alias_target_is_a_quantity():
    for old, new in plot_compare.QUANTITY_ALIASES.items():
        assert new in plot_compare.QUANTITIES, old


def test_a_dumped_csv_with_the_old_labels_still_draws(tmp_path):
    """The alias is applied on read, not by rewriting the file."""
    polars = pytest.importorskip("polars")
    csv = tmp_path / plot_compare.POINTS_CSV
    csv.write_text(
        "method,dataset,split,epoch,quantity,model,dials,d\n"
        "m9,hewl_9b7c,test,49,intensity,100.0,101.0,2.0\n"
        "m9,hewl_9b7c,test,49,background,1.0,1.1,2.0\n"
    )
    df = plot_compare._read_table(tmp_path, plot_compare.POINTS_CSV)
    assert isinstance(df, polars.DataFrame)
    assert sorted(df["quantity"].to_list()) == [
        "background_mean",
        "intensity_prf",
    ]
    # and every label that comes back is one the drawing code knows
    assert all(q in plot_compare.AXES for q in df["quantity"])
