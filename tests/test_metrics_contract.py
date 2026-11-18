"""The format contract between the chain and everything that reads it.

These fixtures are copied verbatim from real chain output, not written by
hand, because the failure being guarded against is exactly a format that
drifted away from what a reader expected. Two readers had already drifted:
`results_table.py` looked for `Final` immediately after the arm name, and
`tidy_xtal.sh` grepped for an arm without the `_rb`/`_model` infix. Both
matched nothing and reported it as an empty table rather than an error, so
there is no runtime signal when this breaks -- only these tests.

The round-trip tests are the ones that matter most: whatever
`integrator.utils.metrics` emits, it must also read.
"""

from pathlib import Path

import pytest

from integrator.utils import metrics

DATA = Path(__file__).parent / "data" / "metrics"


@pytest.fixture
def chain_log():
    return (DATA / "chain_arm.log").read_text()


# --------------------------------------------------------------------------- #
# the emitted format is the parsed format
# --------------------------------------------------------------------------- #
def test_emitted_result_is_the_format_the_chain_writes():
    line = metrics.emit_result("m9_geomshift_rb", "F", 0.1607, 0.1785)
    assert line == (
        "RESULT m9_geomshift_rb_F | Final R-work = 0.1607, R-free = 0.1785"
    )


def test_emitted_peaks_is_the_format_the_chain_writes():
    line = metrics.emit_peaks("m9_geomshift_rb", "I", [10.523, 10.528, 33.605])
    assert line == "PEAKS m9_geomshift_rb_I 10.523,10.528,33.605"


def test_result_round_trips():
    line = metrics.emit_result("m9_geom", "F", 0.1798, 0.2011)
    (row,) = metrics.parse_result_lines(line)
    assert row["arm"] == "m9_geom"
    assert row["variant"] == "F"
    assert row["r_work"] == pytest.approx(0.1798)
    assert row["r_free"] == pytest.approx(0.2011)


def test_peaks_round_trips():
    heights = [10.5, 33.598, 14.176]
    line = metrics.emit_peaks("m9_gate_learned_model", "I", heights)
    (row,) = metrics.parse_peaks_lines(line)
    assert row["arm"] == "m9_gate_learned"
    assert row["estimator"] == "model"
    assert row["peaks"] == pytest.approx(heights)


def test_emit_rejects_an_unknown_variant():
    with pytest.raises(ValueError, match="variant must be one of"):
        metrics.emit_result("m9_geom", "X", 0.1, 0.2)


# --------------------------------------------------------------------------- #
# reading real chain output
# --------------------------------------------------------------------------- #
def test_result_lines_parse_from_a_real_log(chain_log):
    rows = {(r["label"], r["variant"]): r for r in
            metrics.parse_result_lines(chain_log)}
    assert ("m9_geomshift_rb", "F") in rows
    assert ("m9_geomshift_rb", "I") in rows
    assert ("dials_ref", "F") in rows


def test_a_rerun_supersedes_the_earlier_line(chain_log):
    """The fixture holds the arm twice; the second run is the result."""
    rows = {(r["label"], r["variant"]): r for r in
            metrics.parse_result_lines(chain_log)}
    assert rows[("m9_geomshift_rb", "F")]["r_free"] == pytest.approx(0.1785)
    assert rows[("m9_geomshift_rb", "F")]["r_work"] == pytest.approx(0.1607)


def test_every_line_is_kept_when_asked(chain_log):
    rows = metrics.parse_result_lines(chain_log, last_only=False)
    frees = [r["r_free"] for r in rows if r["label"] == "m9_geomshift_rb"
             and r["variant"] == "F"]
    assert frees == pytest.approx([0.1899, 0.1785])


def test_the_estimator_infix_is_split_off_not_swallowed(chain_log):
    """`tidy_xtal.sh` grepped `RESULT <arm>_F` and missed `_rb`/`_model`."""
    rows = {(r["label"], r["variant"]): r for r in
            metrics.parse_result_lines(chain_log)}
    row = rows[("m9_geomshift_rb", "F")]
    assert row["arm"] == "m9_geomshift"
    assert row["estimator"] == "rb"


def test_an_arm_with_no_estimator_suffix_keeps_its_whole_name(chain_log):
    """The reference is merged once, so it carries no estimator."""
    rows = {(r["label"], r["variant"]): r for r in
            metrics.parse_result_lines(chain_log)}
    row = rows[("dials_ref", "F")]
    assert row["arm"] == "dials_ref"
    assert row["estimator"] is None


def test_both_variants_are_read_and_agree(chain_log):
    """F and I refine the same data; a gap is an export defect."""
    rows = {(r["label"], r["variant"]): r for r in
            metrics.parse_result_lines(chain_log)}
    f = rows[("m9_geomshift_rb", "F")]["r_free"]
    i = rows[("m9_geomshift_rb", "I")]["r_free"]
    assert abs(f - i) < 0.005


def test_peaks_lines_parse_thirteen_sites(chain_log):
    rows = {(r["label"], r["variant"]): r for r in
            metrics.parse_peaks_lines(chain_log)}
    peaks = rows[("m9_geomshift_rb", "F")]["peaks"]
    assert len(peaks) == 13
    assert max(peaks) == pytest.approx(33.598)


def test_a_line_the_old_results_table_regex_missed_now_parses():
    """The exact regression: `|` between the arm and `Final`."""
    import re
    old = re.compile(
        r"^RESULT\s+(\S+)\s+Final R-work = ([\d.]+), R-free = ([\d.]+)", re.M
    )
    line = "RESULT m9_geomshift_rb_F | Final R-work = 0.1607, R-free = 0.1785"
    assert old.findall(line) == []          # what was happening
    assert len(metrics.parse_result_lines(line)) == 1   # what should happen


# --------------------------------------------------------------------------- #
# reading the tools' own output
# --------------------------------------------------------------------------- #
def test_phenix_log_gives_the_final_not_the_starting_r():
    got = metrics.refinement_from_phenix_log(DATA / "phenix_refine.log")
    assert got["r_work"] == pytest.approx(0.1607)
    assert got["r_free"] == pytest.approx(0.1785)
    assert got["r_work"] != pytest.approx(0.3298)  # the Start line


def test_phenix_log_without_a_final_line_returns_none(tmp_path):
    p = tmp_path / "aborted.log"
    p.write_text("Start R-work = 0.3298, R-free = 0.3381\nkilled\n")
    assert metrics.refinement_from_phenix_log(p) is None


def test_find_peaks_csv_reads_sites_and_peakz():
    rows = metrics.peaks_from_find_peaks_csv(DATA / "find_peaks.csv")
    by_site = {r["site"]: r for r in rows}
    assert by_site["IOD_204"]["peakz"] == pytest.approx(36.500893)
    assert by_site["CYS_115"]["residue"] == "CYS"
    assert all(r["anomalous"] for r in rows)


def test_peak_heights_csv_is_transposed_not_one_row_per_site():
    rows = metrics.peak_heights_from_csv(DATA / "peak_heights.csv")
    assert len(rows) == 13
    by_site = {r["site"]: r["peakz"] for r in rows}
    assert by_site["IOD_204"] == pytest.approx(33.598)
    assert by_site["CYS_6"] == pytest.approx(10.508)


def test_the_two_peak_sources_agree_on_a_shared_site():
    """`peak_heights.csv` and the PEAKS line carry the same numbers."""
    csv_rows = metrics.peak_heights_from_csv(DATA / "peak_heights.csv")
    line = metrics.emit_peaks(
        "m9_geomshift_rb", "F", [r["peakz"] for r in csv_rows]
    )
    (parsed,) = metrics.parse_peaks_lines(line)
    assert parsed["peaks"] == pytest.approx([r["peakz"] for r in csv_rows])


def test_merge_log_shells_parse_with_the_right_columns():
    shells = metrics.shells_from_merge_log(DATA / "merge.log")
    assert len(shells) == 5
    first = shells[0]
    assert first["d_max"] == pytest.approx(79.63)
    assert first["d_min"] == pytest.approx(2.97)
    assert first["n_obs"] == pytest.approx(134331)
    assert first["i_over_sigma"] == pytest.approx(130.0)
    assert first["r_pim"] == pytest.approx(0.006)


def test_the_significance_star_does_not_become_a_string():
    """`dials.merge` writes `1.000*`; read as text it sorts as text."""
    shells = metrics.shells_from_merge_log(DATA / "merge.log")
    assert shells[0]["cc_half"] == pytest.approx(1.000)
    assert shells[0]["cc_anom"] == pytest.approx(0.343)
    assert isinstance(shells[0]["cc_half"], float)
    assert shells[0]["cc_half_significant"] is True
    assert shells[-1]["cc_half_significant"] is False


def test_completeness_is_read_as_a_number():
    shells = metrics.shells_from_merge_log(DATA / "merge.log")
    assert shells[0]["completeness"] == pytest.approx(100.00)
    assert shells[-1]["completeness"] == pytest.approx(99.88)


def test_shells_stop_at_the_end_of_the_table():
    """The overall-statistics prose after the table is not a shell."""
    shells = metrics.shells_from_merge_log(DATA / "merge.log")
    assert all(isinstance(s["d_max"], float) for s in shells)
    assert [s["d_min"] for s in shells] == pytest.approx(
        [2.97, 2.36, 2.06, 1.87, 1.74]
    )


def test_shell_resolution_ranges_are_contiguous():
    shells = metrics.shells_from_merge_log(DATA / "merge.log")
    for a, b in zip(shells, shells[1:], strict=False):
        assert a["d_min"] == pytest.approx(b["d_max"])


# --------------------------------------------------------------------------- #
# the shell readers and the Python readers must agree
# --------------------------------------------------------------------------- #
def test_the_shell_reader_and_this_module_report_the_same_numbers(chain_log):
    """`tidy_xtal.sh` reads these lines with grep/sed/awk rather than with
    this module, so the two paths are pinned against each other here. They
    feed different figures; a disagreement would show up as two figures
    describing the same run differently."""
    import shutil
    import subprocess

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    log = DATA / "chain_arm.log"
    label = "m9_geomshift_rb"

    def sh(cmd):
        return subprocess.run(
            [bash, "-c", cmd], capture_output=True, text=True, check=True
        ).stdout.strip()

    shell_free_f = sh(
        f"grep -a 'RESULT {label}_F' {log} | tail -1 | sed 's/.*R-free = //'"
    )
    shell_peak = sh(
        f"grep -a 'PEAKS {label}_F' {log} | tail -1 | awk '{{print $3}}'"
        " | tr ',' '\\n' | sort -g | tail -1"
    )

    rows = {(r["label"], r["variant"]): r for r in
            metrics.parse_result_lines(chain_log)}
    peaks = {(r["label"], r["variant"]): r for r in
             metrics.parse_peaks_lines(chain_log)}

    assert float(shell_free_f) == pytest.approx(
        rows[(label, "F")]["r_free"]
    )
    assert float(shell_peak) == pytest.approx(
        max(peaks[(label, "F")]["peaks"])
    )


def test_results_table_reads_the_current_chain_format(chain_log):
    """The regression that motivated this module: an empty table."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "results_table",
        Path(__file__).parents[1] / "scripts" / "results_table.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    got = mod.parse(chain_log)
    assert got, "results_table.parse returned an empty table"
    assert got["m9_geomshift_rb_F"]["r_free"] == pytest.approx(0.1785)
    assert max(got["m9_geomshift_rb_F"]["peaks"]) == pytest.approx(33.598)


# --------------------------------------------------------------------------- #
# the module's emitter replaces shell in the chain: it must be byte-identical
# --------------------------------------------------------------------------- #
def test_emitter_matches_the_shell_lines_it_replaces():
    """`refine_june.sh` emitted these with grep/sed and echo:

        grep -ahoE "Final R-work = [0-9.]+, R-free = [0-9.]+" *.log | tail -1 \\
          | sed "s/^/RESULT $LABEL | /"
        echo "PEAKS $LABEL $(tail -1 peak_heights.csv)"

    Replacing that with this module is only safe if the bytes are the same,
    because every stored number downstream was produced by the old form.
    """
    import shutil
    import subprocess

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    label, log = "m9_geomshift_rb_F", DATA / "phenix_refine.log"
    csv_path = DATA / "peak_heights.csv"

    shell_result = subprocess.run(
        [bash, "-c",
         f'grep -ahoE "Final R-work = [0-9.]+, R-free = [0-9.]+" {log}'
         f' | tail -1 | sed "s|^|RESULT {label} \\| |"'],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    shell_peaks = subprocess.run(
        [bash, "-c", f'echo "PEAKS {label} $(tail -1 {csv_path})"'],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    got = metrics.refinement_from_phenix_log(log)
    mine_result = metrics.emit_result(
        "m9_geomshift_rb", "F", got["r_work"], got["r_free"]
    )
    rows = [ln.strip() for ln in csv_path.read_text().splitlines() if ln.strip()]
    mine_peaks = metrics.emit_peaks(
        "m9_geomshift_rb", "F", rows[-1].split(",")
    )

    assert mine_result == shell_result
    assert mine_peaks == shell_peaks


def test_peak_text_is_not_reformatted():
    """`f"{1234.567:g}"` is `1234.57`; a height must not lose digits."""
    line = metrics.emit_peaks("arm", "F", ["10.508", "11.44", "1234.567"])
    assert line.endswith("10.508,11.44,1234.567")


# --------------------------------------------------------------------------- #
# the three per-shell forms a run can be scored through
# --------------------------------------------------------------------------- #
def test_merging_stats_csv_reads_the_canonical_columns():
    rows = metrics.shells_from_merging_stats_csv(DATA / "merging_stats.csv")
    assert len(rows) == 3
    assert rows[0]["d_max"] == pytest.approx(79.63)
    assert rows[0]["cc_anom"] == pytest.approx(0.327)
    assert rows[0]["i_over_sigma"] == pytest.approx(125.5)


def test_merged_html_maps_dials_headings_onto_the_same_names():
    """DIALS writes `CC½`, `CCano`, `Rpim`, `Mean I/σ(I)`."""
    rows = metrics.shells_from_merged_html(DATA / "merged.html")
    assert rows[0]["cc_half"] == pytest.approx(1.000)
    assert rows[0]["cc_anom"] == pytest.approx(0.366)
    assert rows[0]["r_pim"] == pytest.approx(0.006)
    assert rows[0]["i_over_sigma"] == pytest.approx(128.1)


def test_merged_html_drops_the_overall_row():
    """The last row of the table is a total, not a resolution shell."""
    rows = metrics.shells_from_merged_html(DATA / "merged.html")
    assert len(rows) == 3
    assert not any(
        str(r["resolution"]).lower().startswith("overall") for r in rows
    )


def test_merged_html_picks_the_bin_table_not_the_summary():
    rows = metrics.shells_from_merged_html(DATA / "merged.html")
    assert all("d_max" in r for r in rows)


def test_html_significance_star_is_stripped():
    rows = metrics.shells_from_merged_html(DATA / "merged.html")
    assert isinstance(rows[0]["cc_half"], float)
    assert rows[0]["cc_half"] == pytest.approx(1.0)


def test_shell_labels_parse_back_into_d_max_and_d_min():
    rows = metrics.shells_from_merged_html(DATA / "merged.html")
    assert rows[0]["d_max"] == pytest.approx(79.63)
    assert rows[0]["d_min"] == pytest.approx(2.97)


def test_shell_midpoint_is_cut_on_one_over_d_squared():
    """Shells are cut on 1/d^2, so the midpoint is not (d_max+d_min)/2."""
    mid = metrics.shell_midpoint(79.63, 2.97)
    naive = (79.63 + 2.97) / 2
    assert mid == pytest.approx(4.19730, abs=1e-4)
    assert abs(mid - naive) > 30


def test_shell_midpoint_survives_a_bad_value():
    import math
    assert math.isnan(metrics.shell_midpoint(0.0, 2.97))
    assert math.isnan(metrics.shell_midpoint("x", 2.97))


def test_shells_from_any_dispatches_on_the_suffix():
    for name, n in (("merging_stats.csv", 3), ("merged.html", 3),
                    ("merge.log", 5)):
        rows = metrics.shells_from_any(DATA / name)
        assert rows and len(rows) == n, name
        assert "cc_half" in rows[0], name


def test_the_three_forms_agree_on_a_shared_shell():
    """One run scored through any of the three must read the same."""
    csv_first = metrics.shells_from_merging_stats_csv(
        DATA / "merging_stats.csv"
    )[0]
    html_first = metrics.shells_from_merged_html(DATA / "merged.html")[0]
    log_first = metrics.shells_from_merge_log(DATA / "merge.log")[0]
    for row in (csv_first, html_first, log_first):
        assert row["d_max"] == pytest.approx(79.63)
        assert row["d_min"] == pytest.approx(2.97)
        assert row["cc_half"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# the stats-file form of the R factors
# --------------------------------------------------------------------------- #
def test_a_stats_file_gives_the_same_two_numbers():
    """`Rwork:  0.1570` / `Rfree:  0.1738`, with no `Final R-work` line."""
    got = metrics.refinement_from_phenix_log(DATA / "stats_file.log")
    assert got["r_work"] == pytest.approx(0.1570)
    assert got["r_free"] == pytest.approx(0.1738)


def test_a_refine_log_wins_over_the_stats_fallback(tmp_path):
    p = tmp_path / "both.log"
    p.write_text(
        "Rwork:  0.9999\nRfree:  0.9999\n"
        "Final R-work = 0.1607, R-free = 0.1785\n"
    )
    got = metrics.refinement_from_phenix_log(p)
    assert got["r_work"] == pytest.approx(0.1607)


def test_several_logs_are_read_in_order_and_the_last_final_wins(tmp_path):
    """The chain passes `<out>/*.log`, which is several files."""
    a, b = tmp_path / "a.log", tmp_path / "b.log"
    a.write_text("Final R-work = 0.2000, R-free = 0.2200\n")
    b.write_text("Final R-work = 0.1607, R-free = 0.1785\n")
    got = metrics.refinement_from_phenix_log([a, b])
    assert got["r_free"] == pytest.approx(0.1785)


def test_a_missing_log_in_the_list_is_skipped(tmp_path):
    good = tmp_path / "good.log"
    good.write_text("Final R-work = 0.1607, R-free = 0.1785\n")
    got = metrics.refinement_from_phenix_log([tmp_path / "gone.log", good])
    assert got["r_free"] == pytest.approx(0.1785)


# --------------------------------------------------------------------------- #
# the site label already written into tables must not change
# --------------------------------------------------------------------------- #
def test_peak_site_keeps_both_the_spaced_and_underscored_form():
    """Tables written before carry `CYS 6`; figures name files `CYS_6`."""
    rows = metrics.peak_heights_from_csv(DATA / "peak_heights.csv")
    assert rows[0]["label"] == "CYS 6"
    assert rows[0]["site"] == "CYS_6"
    assert rows[0]["residue"] == "CYS"
