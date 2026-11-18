"""The atom carrying an anomalous peak, resolved from the refined model.

`rs.find_peaks` has a `name` column for the nearest atom and the fork in use
writes it empty, so every peak arrived named by its residue alone -- and the
residue does not say which atom holds the peak, since a sulfur peak is SD on
a methionine and SG on a cysteine. These pin the two ways the atom is
recovered and the guard that keeps a wrong one out:

* with the peak's own position, the nearest atom in the model, symmetry
  images included -- without the images a peak on a symmetry copy picks up an
  atom tens of angstroms away and the site is named wrongly;
* without a position (`peak_heights.csv` carries none), the heaviest atom of
  the residue the peak search named, which is a different thing and says so
  through `atom_from_position`.

A figure title and the CSV column name a site by two different rules, which
is deliberate: the column always carries the atom so it can be grouped on,
and a title carries it only where the residue does not imply it.

The fixture model is a handful of residues of HEWL at their refined
positions, in the 9b7c cell and space group, with `peaks.csv` rows written
against it.
"""

from pathlib import Path

import pytest

from integrator.utils import metrics

DATA = Path(__file__).parent / "data" / "peaks"
MODEL = DATA / "model.pdb"
PEAKS = DATA / "peaks.csv"

# the peak search's own answer for CYS 30, and the symmetry image of the SG
# it sits on -- `y+1/2,-x+1/2,z+1/4` applied to (10.134, 38.199, 2.309)
SG_IMAGE = (77.749, 29.416, 11.784)


def test_site_label_appends_the_atom_because_the_residue_does_not_say():
    assert metrics.site_label("IOD", 204, "I") == "IOD 204 I"
    assert metrics.site_label("MET", 105, "SD") == "MET 105 SD"
    # the same residue, two atoms: this is the distinction the label carries
    assert metrics.site_label("CYS", 30, "SG") != metrics.site_label(
        "CYS", 30, "CB"
    )


def test_site_label_omits_an_unknown_atom_rather_than_inventing_one():
    assert metrics.site_label("CYS", 6) == "CYS 6"
    assert metrics.site_label("CYS", 6, "") == "CYS 6"


def test_site_label_prefixes_the_chain_only_when_it_is_given():
    assert metrics.site_label("IOD", 204, "I") == "IOD 204 I"
    assert metrics.site_label("IOD", 204, "I", "A") == "A/IOD 204 I"


def test_model_atoms_reads_the_element_from_the_file_not_the_residue_name():
    atoms = metrics.model_atoms(MODEL)
    by_name = {(a["residue"], a["seqid"], a["atom"]): a for a in atoms}
    assert by_name[("CYS", 30, "SG")]["element"] == "S"
    assert by_name[("CYS", 30, "CB")]["element"] == "C"
    assert by_name[("IOD", 204, "I")]["element"] == "I"
    assert by_name[("SO4", 301, "S")]["element"] == "S"


def test_nearest_atoms_names_the_atom_not_just_the_residue():
    (near,) = metrics.nearest_atoms(MODEL, [(12.137, 34.405, 10.129)])
    assert near["residue"] == "MET"
    assert near["seqid"] == 105
    assert near["atom"] == "SD"
    assert near["element"] == "S"
    assert near["dist"] == pytest.approx(0.035, abs=0.01)


def test_nearest_atoms_searches_symmetry_images():
    """A peak on a symmetry copy still names the atom it sits on.

    Without the images the nearest atom in the asymmetric unit is a water
    52 A away, so the site would be reported as HOH rather than CYS 30 SG.
    """
    (near,) = metrics.nearest_atoms(MODEL, [SG_IMAGE])
    assert (near["residue"], near["seqid"], near["atom"]) == ("CYS", 30, "SG")
    assert near["dist"] == pytest.approx(0.0, abs=1e-3)


def test_nearest_atoms_skips_hydrogens():
    """A hydrogen cannot carry an anomalous peak.

    The fixture has a riding HG on CYS 30 that is nearer the peak than the
    SG is, which is the ordinary case in a phenix model -- the real HEWL
    9b7c peak list named a 10.96 sigma site `ILE 88 HG12` before hydrogens
    were excluded. The sulfur is what the site is named after.
    """
    atoms = {a["atom"] for a in metrics.model_atoms(MODEL)}
    assert "HG" in atoms, "the fixture carries a riding hydrogen"
    (near,) = metrics.nearest_atoms(MODEL, [(10.116, 38.223, 2.309)])
    assert near["atom"] == "SG"
    assert near["element"] == "S"


def test_nearest_atoms_returns_none_beyond_max_dist():
    far = [(40.0, 5.0, 18.0)]
    assert metrics.nearest_atoms(MODEL, far, max_dist=0.5) == [None]


def test_peaks_without_a_model_carry_no_atom():
    rows = metrics.peaks_from_find_peaks_csv(PEAKS)
    assert rows, "the fixture has peaks"
    assert all(r["atom"] == "" for r in rows)
    assert all(r["atom_from_position"] is False for r in rows)
    # the heights are still read
    assert rows[0]["peakz"] == pytest.approx(36.500893)


def test_peaks_with_a_model_name_the_atom_and_the_element():
    rows = {
        r["site"]: r
        for r in metrics.peaks_from_find_peaks_csv(PEAKS, model=MODEL)
    }
    assert rows["IOD_204"]["atom"] == "I"
    assert rows["CYS_30"]["atom"] == "SG"
    assert rows["MET_105"]["atom"] == "SD"
    assert rows["SO4_301"]["atom"] == "S"
    assert rows["CYS_30"]["element"] == "S"
    assert rows["MET_105"]["element"] == "S"
    assert rows["CYS_30"]["label"] == "CYS 30 SG"
    assert rows["MET_105"]["label"] == "MET 105 SD"


def test_peaks_report_which_source_named_the_atom():
    rows = metrics.peaks_from_find_peaks_csv(PEAKS, model=MODEL)
    named = [r for r in rows if r["atom"]]
    assert named, "the model names at least one atom"
    assert all(r["atom_from_position"] for r in named)
    assert all(r["dist_to_atom"] >= 0 for r in named)


def test_a_model_disagreeing_with_the_peak_search_is_not_accepted():
    """The last fixture row claims CYS 6 at the sulfate.

    Nothing in the model is a CYS 6, so the atom is left empty rather than
    being filled with the sulfate sulfur the position happens to be nearest
    -- the peak search's own residue assignment is not overwritten.
    """
    rows = {
        r["site"]: r
        for r in metrics.peaks_from_find_peaks_csv(PEAKS, model=MODEL)
    }
    assert rows["CYS_6"]["atom"] == ""
    assert rows["CYS_6"]["atom_from_position"] is False
    assert rows["CYS_6"]["label"] == "CYS 6"


def test_the_peak_position_is_read_from_coord_not_cen():
    """`cen*` is written wrong by the fork in use.

    The fixture keeps its `IOD 204` row as the fork wrote it, with
    `cenx == cenz`; reading those would put the peak 9 A from the iodide.
    """
    (row,) = [
        r
        for r in metrics.peaks_from_find_peaks_csv(PEAKS)
        if r["site"] == "IOD_204"
    ]
    assert (row["x"], row["y"], row["z"]) == (8.167, 31.257, 16.831)


def test_heaviest_atoms_is_the_fallback_for_a_site_with_no_position():
    heaviest = metrics.heaviest_atoms(MODEL)
    assert heaviest[(30, "CYS")]["atom"] == "SG"
    assert heaviest[(105, "MET")]["atom"] == "SD"
    assert heaviest[(204, "IOD")]["atom"] == "I"
    assert heaviest[(301, "SO4")]["atom"] == "S"
    assert heaviest[(401, "HOH")]["atom"] == "O"
    # keyed with and without the chain, because `peak_heights.csv` has none
    assert heaviest[("A", 30, "CYS")]["atom"] == "SG"


def test_peak_heights_with_a_model_say_the_atom_came_from_the_residue():
    """`peak_heights.csv` has no positions, and the rows say so.

    The atom is then the heaviest in the named residue rather than the one
    nearest the peak, which is a weaker statement; `atom_from_position` is
    what tells a table built from this file apart from one built from
    `peaks.csv`.
    """
    rows = metrics.peak_heights_from_csv(
        Path(__file__).parent / "data" / "metrics" / "peak_heights.csv",
        model=MODEL,
    )
    by_seqid = {r["seqid"]: r for r in rows}
    assert by_seqid[30]["atom"] == "SG"
    assert by_seqid[30]["label"] == "CYS 30 SG"
    assert by_seqid[30]["atom_from_position"] is False
    # a residue the fixture model does not contain stays unnamed
    assert by_seqid[6]["atom"] == ""


def test_peak_heights_without_a_model_are_unchanged():
    rows = metrics.peak_heights_from_csv(
        Path(__file__).parent / "data" / "metrics" / "peak_heights.csv"
    )
    assert [r["site"] for r in rows][:3] == ["CYS_6", "MET_12", "CYS_30"]
    assert rows[0]["peakz"] == pytest.approx(10.508)
    assert all(r["atom"] == "" for r in rows)


def test_the_unique_count_has_two_names_in_the_dials_logs():
    """Why the long-format export normalizes the statistic names.

    The per-shell table calls the unique-reflection count `n_uniq` and the
    merging summary calls it `n_unique`. Exporting both spellings gives a
    pivot two half-empty columns, so `make_tables.py` maps them onto one --
    and this is the disagreement that makes that necessary.
    """
    assert "n_unique" in set(metrics.MERGING_SUMMARY_KEYS.values())
    assert "n_uniq" in metrics.SHELL_COLUMNS
    assert "n_unique" not in metrics.SHELL_COLUMNS


def test_a_panel_title_omits_an_atom_the_residue_already_implies():
    """Every residue in this experiment holds one candidate heavy atom.

    An iodide has one iodine, a cysteine and a methionine one sulfur each, a
    sulfate one sulfur -- so `CYS 30 SG` says `sulfur` twice and the title is
    `CYS 30`.
    """
    assert metrics.panel_site_label("IOD", 204, "I", 1) == "IOD 204"
    assert metrics.panel_site_label("CYS", 30, "SG", 1) == "CYS 30"
    assert metrics.panel_site_label("MET", 105, "SD", 1) == "MET 105"


def test_a_panel_title_names_the_atom_where_the_residue_is_ambiguous():
    assert metrics.panel_site_label("CYS", 30, "SG", 2) == "CYS 30 SG"


def test_a_panel_title_falls_back_when_no_atom_was_resolved():
    assert metrics.panel_site_label("CYS", 127, "", 2) == "CYS 127"
    assert metrics.panel_site_label("CYS", 127) == "CYS 127"


def test_the_csv_column_keeps_the_atom_the_title_drops():
    """The two rules differ on purpose, so neither is a bug in the other."""
    assert metrics.site_label("CYS", 30, "SG") == "CYS 30 SG"
    assert metrics.panel_site_label("CYS", 30, "SG", 1) == "CYS 30"


def test_every_anomalous_residue_holds_exactly_one_candidate():
    """Which is what makes the title rule come out as residue and number.

    If this ever fails for a new dataset -- a residue with two sulfurs, a
    cluster compound -- the titles for that residue start naming the atom
    rather than becoming ambiguous, which is the point of counting.
    """
    candidates = metrics.heavy_atom_candidates(MODEL)
    assert candidates[(204, "IOD")] == ["I"]
    assert candidates[(30, "CYS")] == ["SG"]
    assert candidates[(105, "MET")] == ["SD"]
    assert candidates[(301, "SO4")] == ["S"]


def test_light_atoms_are_not_candidates():
    """Carbon, nitrogen, oxygen and hydrogen carry no measurable signal.

    So a water and an isoleucine hold no candidate at all, and the sulfate's
    four oxygens do not make its sulfur ambiguous.
    """
    candidates = metrics.heavy_atom_candidates(MODEL)
    assert (401, "HOH") not in candidates
    assert len(candidates[(301, "SO4")]) == 1
    # the cysteine's own carbons and its riding hydrogen are excluded
    assert candidates[(30, "CYS")] == ["SG"]
