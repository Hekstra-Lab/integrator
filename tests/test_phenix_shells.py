"""The per-shell and step-by-step tables of a phenix.refine log, parsed against a copied snippet."""

from pathlib import Path

from integrator.utils import metrics

LOG = Path(__file__).parent / "fixtures" / "phenix_refine_tables.txt"


def test_rfactor_shells():
    t = metrics.refinement_shells_from_phenix_log(LOG)["rfactors"]
    assert len(t) == 20
    assert t[0] == {"shell": 1, "d_max": 33.52, "d_min": 4.82, "completeness": 0.98, "n_work": 1228, "n_free": 177,
                    "r_work": 0.0889, "r_free": 0.1060, "target_work": 5.7034, "target_free": 5.7571}
    assert t[-1]["d_min"] == 1.78 and t[-1]["r_free"] == 0.2795


def test_fom_shells_are_told_apart_by_their_header():
    t = metrics.refinement_shells_from_phenix_log(LOG)["fom"]
    assert t[0]["fom"] == 0.95 and t[0]["phase_error"] == 7.33 and t[0]["beta"] == 8998.78
    assert "r_work" not in t[0]


def test_step_table():
    s = metrics.refinement_steps_from_phenix_log(LOG)
    assert s[0]["stage"] == "0" and s[0]["r_work"] == 0.3621
    assert s[-1]["stage"] == "end" and s[-1]["r_free"] == 0.1248 and s[-1]["bonds"] == 0.007
