"""The reductions behind the figures, pinned against the formats they read.

These are parsers, and the failures they guard against are silent: a bin
index read at twice its value puts every CCsym point in the wrong
resolution shell without raising, and a basis reshaped on the wrong
geometry draws a plausible picture of nothing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

FIGURES = Path(__file__).resolve().parents[1] / "scripts" / "figures"


def _load(name: str):
    """Import a figure script by path; they are scripts, not a package."""
    spec = importlib.util.spec_from_file_location(name, FIGURES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tidy_pdz2 = _load("tidy_pdz2")
dump_basis = _load("dump_basis")


# `custom_ccsym.py` writes a blank second line before the data, and its
# `bin` column is doubled: ten bins come out numbered 0, 2, ..., 18.
CCSYM_CSV = """,bin,repeat,delay,CCsym
,,,,
0,0,0,off,0.14695633618883785
1,0,1,off,0.11916025172299667
2,2,0,off,0.0921
3,18,2,200ns,0.0612
"""


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """One arm's directory, laid out as the deposited chain leaves it."""
    where = tmp_path / "tree_x" / tidy_pdz2.CARELESS
    where.mkdir(parents=True)
    (where / "e35_ccsym.csv").write_text(CCSYM_CSV)
    monkeypatch.setattr(tidy_pdz2, "ARMS", {"x": str(tmp_path / "tree_x")})
    # the edges come from calling the deposited script against an MTZ, which
    # a test has no business doing; the undoubling is what is under test
    monkeypatch.setattr(
        tidy_pdz2, "shell_edges",
        lambda _tree: {0: (28.96, 3.64), 1: (3.64, 3.05), 9: (1.80, 1.70)},
    )
    return {"x": str(tmp_path / "tree_x")}


class TestCCsym:
    def test_the_doubled_bin_index_is_halved(self, tree):
        rows = tidy_pdz2.ccsym(tree)
        # bins are reported one-based, so the saved 0, 2 and 18 become 1, 2
        # and 10 -- not 1, 3 and 19
        assert sorted({r["bin"] for r in rows}) == [1, 2, 10]

    def test_the_blank_row_is_dropped(self, tree):
        assert len(tidy_pdz2.ccsym(tree)) == 4

    def test_each_bin_carries_its_own_shell(self, tree):
        rows = tidy_pdz2.ccsym(tree)
        first = next(r for r in rows if r["bin"] == 1)
        last = next(r for r in rows if r["bin"] == 10)
        assert (first["d_max"], first["d_min"]) == (28.96, 3.64)
        assert (last["d_max"], last["d_min"]) == (1.80, 1.70)

    def test_a_bin_with_no_edge_is_empty_not_wrong(self, tree, monkeypatch):
        monkeypatch.setattr(tidy_pdz2, "shell_edges", lambda _t: {})
        rows = tidy_pdz2.ccsym(tree)
        assert all(r["d_max"] == "" and r["d_min"] == "" for r in rows)

    def test_the_delay_and_keys_survive(self, tree):
        rows = tidy_pdz2.ccsym(tree)
        assert {r["delay"] for r in rows} == {"off", "200ns"}
        assert {r["method"] for r in rows} == {"x"}
        assert {r["dataset"] for r in rows} == {tidy_pdz2.DATASET}


class TestCheckpointSelection:
    def test_the_epoch_is_read_from_either_spelling(self):
        assert dump_basis.epoch_of("epoch=0004.ckpt") == 4
        assert dump_basis.epoch_of("epoch_0019.ckpt") == 19

    def test_last_has_no_epoch_and_is_excluded(self, tmp_path):
        ckpts = tmp_path / "files" / "checkpoints"
        ckpts.mkdir(parents=True)
        for name in ("epoch=0009.ckpt", "epoch=0004.ckpt", "last.ckpt",
                     "last-v1.ckpt"):
            (ckpts / name).touch()
        got = dump_basis.checkpoints(str(tmp_path))
        # `last.ckpt` duplicates an epoch already in the list and carries no
        # epoch of its own, so including it would double-count one
        assert [Path(p).name for p in got] == [
            "epoch=0004.ckpt", "epoch=0009.ckpt"
        ]


class TestComponentRows:
    def test_the_final_epoch_has_cosine_one_with_itself(self):
        rng = np.random.default_rng(0)
        weights = [rng.normal(size=(12, 3)) for _ in range(4)]
        rows = dump_basis.component_rows("arm", [0, 1, 2, 3], weights)
        final = [r for r in rows if r["epoch"] == 3.0]
        assert len(final) == 3
        assert all(r["cos_to_final"] == pytest.approx(1.0) for r in final)

    def test_a_dead_component_reads_as_zero_norm(self):
        w = np.ones((6, 2))
        w[:, 1] = 0.0
        rows = dump_basis.component_rows("arm", [0], [w])
        norms = {r["component"]: r["norm"] for r in rows}
        assert norms[1] == pytest.approx(0.0)
        assert norms[0] > 0.0

    def test_no_weights_is_no_rows(self):
        assert dump_basis.component_rows("arm", [], []) == []


class TestBasisGeometry:
    """`as_images` has to handle a decoder wider than its own shoebox."""

    def setup_method(self):
        self.mod = _load("fig_profile_basis")

    def test_the_configured_geometry_is_used_when_it_fits(self):
        weight = np.arange(3 * 5 * 5 * 2, dtype=float).reshape(3 * 5 * 5, 2)
        images = self.mod.as_images(weight, (3, 5, 5))
        assert images.shape == (2, 5, 5)
        # depth is summed, not sliced: a component with its mass on another
        # slice must not vanish
        assert images[0].sum() == pytest.approx(weight[:, 0].sum())

    def test_a_larger_square_canvas_is_recovered(self):
        # the geometric surrogate decodes onto a canvas and then shifts and
        # crops, so 31x31 rows arrive under a 21x21 shoebox geometry
        weight = np.zeros((961, 4))
        images = self.mod.as_images(weight, (1, 21, 21))
        assert images.shape == (4, 31, 31)

    def test_an_unfactorable_basis_draws_as_a_strip(self):
        weight = np.zeros((7, 2))
        images = self.mod.as_images(weight, (1, 21, 21))
        assert images.shape == (2, 1, 7)


class TestCCsymFigureOrdering:
    def setup_method(self):
        self.mod = _load("fig_ccsym")

    def test_delays_are_in_time_order_not_alphabetical(self):
        got = self.mod.delay_order(["200ns", "off", "100ns", "50ns"])
        assert got == ["off", "50ns", "100ns", "200ns"]

    def test_an_unknown_delay_is_kept_and_appended(self):
        got = self.mod.delay_order(["1us", "off"])
        assert got == ["off", "1us"]

    def test_the_reference_is_drawn_last(self):
        got = self.mod.arm_order(["laue", "combo", "geom_both"])
        assert got[-1] == "laue"

    def test_no_reference_present_is_fine(self):
        assert self.mod.arm_order(["combo", "a"]) == ["a", "combo"]
