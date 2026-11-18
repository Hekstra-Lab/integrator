"""The hyperparameter reader, pinned against the shapes real configs have.

The failures this guards against are quiet ones: an anchored encoder
reported four times instead of once, a `varies` flag computed per run
instead of across the set, or an output directory reported as though it
were a hyperparameter.
"""

from __future__ import annotations

import pytest

from integrator.utils import hparams


@pytest.fixture
def hier():
    """A config in the shape the hierarchical integrator's has."""
    return {
        "integrator": {"name": "hierarchical",
                       "args": {"d": 1, "h": 21, "w": 21}},
        "encoders": [
            {"name": "profile_encoder",
             "args": {"conv1_kernel_size": [3, 3], "dropout": 0.0}},
            # four anchored copies, as the PDZ2 config writes them
            {"name": "intensity_encoder", "args": {"encoder_out": 64}},
            {"name": "intensity_encoder", "args": {"encoder_out": 64}},
            {"name": "intensity_encoder", "args": {"encoder_out": 64}},
            {"name": "intensity_encoder", "args": {"encoder_out": 64}},
        ],
        "surrogates": {"qp": {"name": "learned_basis_profile"}},
        "optimizer": {"lr": 0.001},
        "trainer": {"max_epochs": 100, "default_root_dir": "/scratch/run"},
        "output": {"out_dir": "/scratch/out"},
    }


@pytest.fixture
def svae(hier):
    cfg = {k: (v.copy() if isinstance(v, dict) else list(v))
           for k, v in hier.items()}
    cfg["integrator"] = {"name": "svae", "args": {"d": 1, "h": 21, "w": 21}}
    cfg["encoders"] = hier["encoders"][:2]  # one intensity encoder
    return cfg


class TestFlatten:
    def test_args_is_not_a_level(self, hier):
        flat = hparams.flatten(hier["integrator"], "integrator")
        # `args` is a container; a reader wants integrator.d, not
        # integrator.args.d
        assert "integrator.d" in flat
        assert not any(".args." in k for k in flat)

    def test_repeated_encoders_collapse_to_one_row(self, hier):
        flat = hparams.flatten(hier["encoders"], "encoders")
        names = [k for k in flat if k.endswith(".encoder_out")]
        assert names == ["encoders.intensity_encoder.encoder_out"]
        assert flat["encoders.intensity_encoder.n"] == 4

    def test_a_single_encoder_records_no_count(self, svae):
        flat = hparams.flatten(svae["encoders"], "encoders")
        assert "encoders.intensity_encoder.n" not in flat

    def test_entries_are_named_not_indexed(self, hier):
        flat = hparams.flatten(hier["encoders"], "encoders")
        assert any("profile_encoder" in k for k in flat)
        assert not any(k.startswith("encoders.0") for k in flat)


class TestRows:
    def test_output_paths_are_not_hyperparameters(self, hier):
        rows = hparams.hparam_rows(hier, "a")
        names = {r["parameter"] for r in rows}
        assert not any("out_dir" in n or "root_dir" in n for n in names)

    def test_the_group_replaces_the_section(self, hier):
        rows = hparams.hparam_rows(hier, "a")
        names = {r["parameter"] for r in rows}
        # `encoders` the section is reported as the group `encoder`
        assert "encoder.profile_encoder.dropout" in names
        assert not any(n.startswith("encoders.") for n in names)

    def test_units_are_attached_where_known_and_blank_otherwise(self, hier):
        rows = {r["parameter"]: r for r in hparams.hparam_rows(hier, "a")}
        assert rows["trainer.max_epochs"]["unit"] == "epochs"
        assert rows["integrator.h"]["unit"] == "px"
        assert rows["optimizer.lr"]["unit"] == ""

    def test_a_list_value_is_stored_faithfully(self, hier):
        rows = {r["parameter"]: r for r in hparams.hparam_rows(hier, "a")}
        # how to SHOW a kernel size is a presentation decision; the table
        # keeps what the config said
        assert rows["encoder.profile_encoder.conv1_kernel_size"]["value"] \
            == "3, 3"

    def test_booleans_are_lowercase(self):
        rows = hparams.hparam_rows({"loss": {"flag": True}}, "a")
        assert rows[0]["value"] == "true"


class TestVarying:
    def test_only_the_parameters_that_differ_are_flagged(self, hier, svae):
        rows = (hparams.hparam_rows(hier, "hier")
                + hparams.hparam_rows(svae, "svae"))
        hparams.mark_varying(rows)
        varying = sorted({r["parameter"] for r in rows if r["varies"]})
        # the pair is built to differ by the integrator family alone
        assert varying == ["encoder.intensity_encoder.n", "integrator.name"]

    def test_a_parameter_absent_from_one_arm_counts_as_varying(self, hier):
        other = {"integrator": {"name": "hierarchical"},
                 "optimizer": {"lr": 0.001}}
        rows = (hparams.hparam_rows(hier, "a")
                + hparams.hparam_rows(other, "b"))
        hparams.mark_varying(rows)
        by = {r["parameter"]: r["varies"] for r in rows}
        # `b` has no trainer section at all; hiding that would make two
        # differently configured models look identical
        assert by["trainer.max_epochs"] == 1
        assert by["optimizer.lr"] == 0

    def test_one_arm_alone_varies_in_nothing(self, hier):
        rows = hparams.hparam_rows(hier, "a")
        hparams.mark_varying(rows)
        assert all(r["varies"] == 0 for r in rows)


class TestLabels:
    def test_one_label_per_parameter(self, hier, svae):
        rows = (hparams.hparam_rows(hier, "hier")
                + hparams.hparam_rows(svae, "svae"))
        labels = hparams.hparam_labels(rows)
        names = [r["parameter"] for r in labels]
        assert len(names) == len(set(names))

    def test_a_parameter_with_no_curated_header_still_gets_one(self, hier):
        labels = {r["parameter"]: r
                  for r in hparams.hparam_labels(hparams.hparam_rows(hier, "a"))}
        assert labels["encoder.profile_encoder.dropout"]["header"] == "dropout"
        assert labels["integrator.name"]["header"] == "integrator"


def test_a_missing_config_log_is_empty_not_an_error(tmp_path):
    assert hparams.read_config_log(tmp_path) == {}
