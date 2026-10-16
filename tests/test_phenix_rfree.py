"""One R-free set for every refinement, instead of a fresh draw per MTZ.

`r_free_flags.generate = True` holds out a different 10% of each MTZ phenix is
handed, and `refinement.main.random_seed` moves that draw as well, so two
refinements of the same data differ by which reflections were held out.  On
weak Laue data that is worth about 0.008 in R-free, the size of the model
effects being measured.  These tests cover the pieces that remove it: the flag
set itself, the command-line form that reads it, and the eff templates that
carry it -- including that the generating templates still generate, so the old
numbers stay reproducible.
"""

import importlib.util
from pathlib import Path

import gemmi
import numpy as np
import pytest

from integrator.utils import phenix

REFERENCE_DATA = Path(__file__).resolve().parents[1] / "reference_data"
CELL = (78.3121, 78.3121, 37.2343, 90.0, 90.0, 90.0)
SG = "P 43 21 2"


def _load_maker():
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_rfree_flags.py"
    spec = importlib.util.spec_from_file_location("make_rfree_flags", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flags(dmin: float = 1.6, fraction: float = 0.1, seed: int = 2026):
    maker = _load_maker()
    mtz = maker.complete_flags(
        gemmi.UnitCell(*CELL), gemmi.SpaceGroup(SG), dmin, fraction, seed
    )
    return maker, mtz, np.array(mtz, copy=False)


def test_the_flag_set_is_complete_over_the_asu():
    """No reflection may be unflagged: phenix would have to invent one."""
    _, _, data = _flags(dmin=1.6)
    have = {tuple(row) for row in data[:, :3].astype(int)}
    coarser = gemmi.make_miller_array(
        gemmi.UnitCell(*CELL), gemmi.SpaceGroup(SG), 1.8, 0.0, True
    )
    missing = [tuple(h) for h in coarser.astype(int) if tuple(h) not in have]
    assert missing == []


def test_the_test_set_is_the_requested_fraction_and_binary():
    _, _, data = _flags(dmin=1.8, fraction=0.1)
    flags = data[:, 3].astype(int)
    assert set(np.unique(flags).tolist()) == {0, 1}
    assert flags.mean() == pytest.approx(0.1, abs=0.002)


def test_the_same_seed_gives_the_same_set():
    """The file is the chain; it has to be reproducible from the command."""
    _, _, a = _flags(dmin=1.8, seed=7)
    _, _, b = _flags(dmin=1.8, seed=7)
    _, _, c = _flags(dmin=1.8, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a[:, 3], c[:, 3])


def test_the_flag_column_carries_the_label_the_effs_ask_for():
    _, mtz, _ = _flags(dmin=2.0)
    assert phenix.FLAG_LABEL in mtz.column_labels()
    assert mtz.spacegroup.hm == gemmi.SpaceGroup(SG).hm


def test_overrides_turn_generation_off_and_name_the_file(tmp_path):
    """`generate = False` has to be explicit: the eff it overrides says True."""
    flags = tmp_path / "flags.mtz"
    args = phenix.rfree_overrides(flags)
    assert "r_free_flags.generate=False" in args
    # absolutized, since phenix runs in the output directory
    assert f'r_free_flags.file_name="{flags.resolve()}"' in args
    assert f'r_free_flags.label="{phenix.FLAG_LABEL}"' in args
    assert (
        f"r_free_flags.test_flag_value={phenix.TEST_FLAG_VALUE}" in args
    )


def test_overrides_default_to_the_set_the_mono_chain_uses():
    assert phenix.DEFAULT_RFREE in phenix.rfree_overrides()
    assert phenix.DEFAULT_RFREE.endswith("rfree.mtz")


@pytest.mark.parametrize("pass_no", [1, 2])
def test_the_fixed_free_effs_read_a_file_and_generate_nothing(pass_no):
    text = (REFERENCE_DATA / f"laue_refine_pass{pass_no}_fixedfree.eff").read_text()
    assert "$FREEFILE" in text
    assert f'name = "{phenix.FLAG_LABEL}"' in text
    assert "generate = False" in text
    assert "generate = True" not in text


def test_the_generating_effs_still_generate():
    """The old numbers came through these; they must not change under me."""
    assert "generate = True" in (REFERENCE_DATA / "laue_refine_pass1.eff").read_text()
    assert "$FREEFILE" not in (REFERENCE_DATA / "laue_refine_pass1.eff").read_text()


def test_fixed_free_pass_two_reads_the_merged_columns():
    """Pass 2 re-reads the original MTZ, which has no `F-obs` columns.

    Phenix writes `refined_data.mtz` only to export flags it generated itself,
    so with a fixed set there is no file carrying phenix's own column names.
    """
    text = (REFERENCE_DATA / "laue_refine_pass2_fixedfree.eff").read_text()
    assert 'name = "F(+),SigF(+),F(-),SigF(-)"' in text
    assert "F-obs(+)" not in text
    assert "R-free-flags(+)" not in text


def test_rendering_leaves_no_placeholder_behind(tmp_path):
    out = phenix.render_eff(
        REFERENCE_DATA / "laue_refine_pass1_fixedfree.eff",
        tmp_path / "refine.eff",
        MTZFILE=tmp_path / "merged.mtz",
        FREEFILE=tmp_path / "rfree.mtz",
        MODELFILE=tmp_path / "model.pdb",
    )
    text = out.read_text()
    assert "$MTZFILE" not in text and "$FREEFILE" not in text
    assert "$MODELFILE" not in text
    assert str((tmp_path / "rfree.mtz").resolve()) in text


def _pipeline_module():
    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "poly" / "run_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location("poly_run_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dry_phenix(tmp_path, rfree):
    pipeline = _pipeline_module()
    cfg = {
        "phenix": {
            "env": "integrator-cuda-dev",
            "eff1": str(REFERENCE_DATA / "laue_refine_pass1_fixedfree.eff"),
            "eff2": str(REFERENCE_DATA / "laue_refine_pass2_fixedfree.eff"),
            "pdb": str(tmp_path / "shaken.pdb"),
            "prefix": "refined",
        }
    }
    if rfree is not None:
        cfg["phenix"]["rfree"] = str(rfree)
    pipeline.step_phenix(cfg, tmp_path / "merged.mtz", tmp_path / "work", dry=True)


def test_the_pipeline_passes_the_fixed_set_to_both_passes(tmp_path, capsys):
    """Through `step_phenix` itself, not by reading the config."""
    flags = tmp_path / "rfree.mtz"
    _dry_phenix(tmp_path, flags)
    passes = [
        line for line in capsys.readouterr().out.splitlines() if "phenix.refine" in line
    ]
    assert len(passes) == 2
    for cmd in passes:
        assert f'r_free_flags.file_name="{flags.resolve()}"' in cmd
        assert "r_free_flags.generate=False" in cmd
    # pass 2 re-reads the merged data: phenix writes no refined_data.mtz when
    # it is handed a flag set
    assert "refined_data.mtz" not in passes[1]
    assert str((tmp_path / "merged.mtz").resolve()) in passes[1]


def test_without_the_key_the_pipeline_is_unchanged(tmp_path, capsys):
    """A run that did not ask for a fixed set must behave exactly as before."""
    _dry_phenix(tmp_path, None)
    passes = [
        line for line in capsys.readouterr().out.splitlines() if "phenix.refine" in line
    ]
    assert len(passes) == 2
    assert all("r_free_flags.generate" not in cmd for cmd in passes)
    assert "refined_data.mtz" in passes[1]
