"""The DIALS log tables, pinned against real `scale.log` and `merge.log`.

`tests/data/metrics/dials_{scale,merge}.log` are copied verbatim from
`dials9b7c/m9_geomshift_model`, DIALS 3.27. The failures these guard against
are quiet: a table that reads as one row short, a header split on a literal
`|` inside a column name, or two tables printed back to back merging into
one. None of those raises; each produces a table that looks plausible.
"""

from pathlib import Path

import pytest

from integrator.utils import dials_log

DATA = Path(__file__).parent / "data" / "metrics"
SCALE = DATA / "dials_scale.log"
MERGE = DATA / "dials_merge.log"


# the shapes ----------------------------------------------------------------


def test_scale_log_tables_are_all_found():
    titles = [t.title for t in dials_log.log_tables(SCALE)]
    assert "The following corrections will be applied to this dataset" in titles
    assert titles.count("Refinement steps") == 4
    assert "Summary of dataset partialities" in titles


def test_a_unit_row_becomes_a_unit_not_a_data_row():
    """`| Step | Nref | RMSD_I |` then `|  |  | (a.u) |`."""
    steps = dials_log.find_table(SCALE, "rmsd_i")
    assert steps.headers == ["Step", "Nref", "RMSD_I"]
    assert steps.units == {"RMSD_I": "a.u"}
    # 29 refinement steps, 0 through 28, and no row of empty cells
    assert len(steps) == 29
    assert steps.rows[0][0] == "0"
    assert steps.rows[-1][0] == "28"


def test_two_tables_printed_back_to_back_do_not_merge():
    """dials.merge prints the screw-axis and space-group tables adjacent."""
    tables = dials_log.log_tables(MERGE)
    screw = dials_log.find_table(tables, "screw axis")
    groups = dials_log.find_table(tables, "space group")
    assert len(screw) == 3
    assert len(groups) == 6
    assert groups.headers == ["Space group", "score"]


def test_a_header_split_on_a_literal_pipe_is_put_back_together():
    """`<|dF|/sigma(dF)>` contains two `|`, which splitting cuts up."""
    table = dials_log.find_table(MERGE, "d_max", "d_min")
    assert table.headers == ["d_max", "d_min", "<|ΔF|/σ(ΔF)>"]
    assert len(table) == 20


def test_a_status_line_is_not_read_as_a_table_caption():
    """`Scoring method: direct` precedes the screw-axis table."""
    screw = dials_log.find_table(MERGE, "screw axis")
    assert screw.title is None


def test_find_table_last_picks_the_final_refinement_cycle():
    first = dials_log.find_table(SCALE, "rmsd_i")
    last = dials_log.find_table(SCALE, "rmsd_i", last=True)
    assert len(first) == 29
    assert len(last) == 2
    assert first is not last


def test_records_and_numeric_agree_on_the_corrections_table():
    table = dials_log.find_table(SCALE, "correction")
    assert [r["correction"] for r in table.records()] == [
        "scale", "decay", "absorption"
    ]
    assert [r["n_parameters"] for r in table.numeric("n_parameters")] == [
        50.0, 38.0, 24.0
    ]


# the named quantities -------------------------------------------------------


def test_merging_summary_names_every_r_factor():
    s = dials_log.merging_summary_from_log(MERGE)
    assert s["overall"]["r_merge"] == 0.056
    assert s["overall"]["r_meas"] == 0.056
    assert s["overall"]["r_pim"] == 0.008
    # the `(I+/-)` forms, which the per-shell table does not carry
    assert s["overall"]["r_merge_anom"] == 0.055
    assert s["overall"]["r_meas_anom"] == 0.056
    assert s["overall"]["r_pim_anom"] == 0.011


def test_merging_summary_splits_the_three_scopes():
    s = dials_log.merging_summary_from_log(MERGE)
    assert s["overall"]["d_min"] == 1.10
    assert s["low"]["d_min"] == 2.97
    assert s["high"]["d_min"] == 1.10
    assert s["low"]["i_over_sigma"] == 125.0
    assert s["high"]["i_over_sigma"] == 2.2
    assert s["high"]["cc_half"] == 0.338


def test_a_row_printed_once_lands_only_under_overall():
    s = dials_log.merging_summary_from_log(MERGE)
    assert s["overall"]["anom_slope"] == 1.009
    assert s["overall"]["df_over_f"] == 0.021
    assert "anom_slope" not in s["low"]
    assert "anom_slope" not in s["high"]


def test_counts_are_read_as_numbers_not_text():
    s = dials_log.merging_summary_from_log(MERGE)
    assert s["overall"]["n_obs"] == 1927402
    assert s["overall"]["n_unique"] == 45700


def test_the_wilson_b_is_in_merge_and_not_in_scale():
    """dials.merge runs the Wilson analysis; dials.scale does not."""
    assert dials_log.merging_summary_from_log(MERGE)["overall"]["wilson_b"] == 16.564
    assert "wilson_b" not in dials_log.merging_summary_from_log(SCALE)["overall"]


def test_error_model_takes_the_last_refinement_not_the_loaded_one():
    """The log prints `a = 1.00000, b = 0.02000` as loaded, first."""
    m = dials_log.error_model_from_log(SCALE)
    assert m["a"] == pytest.approx(1.64410)
    assert m["b"] == pytest.approx(0.02699)
    assert m["i_over_sigma_limit"] == pytest.approx(22.539)


def test_scaling_summary_counts_parameters_and_outliers():
    s = dials_log.scaling_summary_from_log(SCALE)
    assert s["n_read"] == 1930231
    assert s["n_param_scale"] == 50
    assert s["n_param_decay"] == 38
    assert s["n_param_absorption"] == 24
    assert s["n_param_total"] == 112
    # the last round rejected 2747; every round together rejected far more
    assert s["n_outliers"] == 2747
    assert s["n_outliers_total"] > s["n_outliers"]
    assert s["pct_param_uncertain"] == 12.5


def test_the_final_rmsd_comes_from_the_last_cycle():
    """1.5995 ends cycle one; 0.99885 is where the run actually stopped."""
    s = dials_log.scaling_summary_from_log(SCALE)
    assert s["rmsd_i_final"] == pytest.approx(0.99885)


def test_wilson_reads_both_the_isotropic_b_and_the_anisotropy():
    w = dials_log.wilson_from_log(MERGE)
    assert w["wilson_b"] == pytest.approx(16.56)
    assert w["neg_log_scale"] == pytest.approx(-1.89)
    # eigenvalues sorted large to small, so the range is non-negative
    assert w["b_cart_1"] >= w["b_cart_2"] >= w["b_cart_3"]
    assert w["b_aniso_range"] == pytest.approx(
        w["b_cart_1"] - w["b_cart_3"], abs=1e-9
    )


def test_anomalous_differences_fall_through_one_with_resolution():
    rows = dials_log.anomalous_differences_from_log(MERGE)
    assert len(rows) == 20
    assert rows[0]["dano_over_sigma"] == pytest.approx(1.88)
    assert rows[-1]["dano_over_sigma"] == pytest.approx(0.319)
    values = [r["dano_over_sigma"] for r in rows]
    assert values == sorted(values, reverse=True)
    # the shell where the differences stop being measurable
    crossing = next(r for r in rows if r["dano_over_sigma"] < 1.0)
    assert crossing["d_min"] == pytest.approx(1.35)


def test_shell_midpoints_are_taken_in_one_over_d_squared():
    rows = dials_log.anomalous_differences_from_log(MERGE)
    first = rows[0]
    # not the arithmetic mean of 79.43 and 3.31, which would be 41.37
    assert first["d"] == pytest.approx(4.677, abs=1e-3)


# failure behaviour ----------------------------------------------------------


def test_a_log_without_the_block_returns_none(tmp_path):
    empty = tmp_path / "nothing.log"
    empty.write_text("DIALS 3.27\nnothing to report\n")
    assert dials_log.merging_summary_from_log(empty) is None
    assert dials_log.error_model_from_log(empty) is None
    assert dials_log.wilson_from_log(empty) is None
    assert dials_log.anomalous_differences_from_log(empty) is None
    assert dials_log.log_tables(empty) == []


def test_metrics_re_exports_the_readers():
    """One import for a caller: everything goes through metrics."""
    from integrator.utils import metrics

    assert metrics.merging_summary_from_log(MERGE) is not None
    assert metrics.log_tables(SCALE)
