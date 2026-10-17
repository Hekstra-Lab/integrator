"""The simulated Laue dataset has to be right, or it answers a wrong question.

A simulator nobody checks is worse than no simulator: it produces confident
numbers fast.  These tests assert the properties the generator exists to
reproduce -- the ones that make Laue scaling hard and that a monochromatic
intuition gets wrong -- rather than its internals.

They also pin the two deliberate mismatches with our own model, because if
those ever stop holding the dataset has started flattering the arm under test:
absorption lives in the crystal frame where no arm's lab-frame scale can see
it, and the background is a tilted plane where the merge infers a constant.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "simulate_laue_merging.py"


def _module():
    spec = importlib.util.spec_from_file_location("sim_laue", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SIM = _module()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """A tiny dataset, generated through the CLI the way it is really used."""
    out = tmp_path_factory.mktemp("sim_laue")
    subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(out), "--images", "40",
         "--d-min", "6.0", "--shoebox", "1", "9", "9", "--seed", "3"],
        check=True, capture_output=True, text=True,
    )
    md = torch.load(out / "metadata.pt", weights_only=False)
    return {
        "dir": out,
        "md": {k: v.numpy() for k, v in md.items()},
        "counts": np.load(out / "counts.npy"),
        "spec": yaml.safe_load((out / "dataset.yaml").read_text()),
    }


def test_friedel_mates_are_kept_separate_but_pooled_together(dataset):
    """The defect that would silently delete the anomalous signal.

    `gemmi.ReciprocalAsu` is defined by the Laue class, so `to_asu` sends h and
    -h to one index.  Using it for `asu_id` would merge every Friedel pair
    before the model ever saw them, and the dataset would have no anomalous
    signal to find while looking perfectly well-formed.
    """
    md = dataset["md"]
    n_sep = len(np.unique(md["asu_id"]))
    n_pool = len(np.unique(md["nonanom_id"]))
    assert n_sep > n_pool, f"{n_sep} separate vs {n_pool} pooled"
    # each pooled class holds one class (centric) or two (acentric), never more
    pairs = {}
    for a, p in zip(md["asu_id"], md["nonanom_id"]):
        pairs.setdefault(int(p), set()).add(int(a))
    sizes = {len(v) for v in pairs.values()}
    assert sizes <= {1, 2}, sizes
    assert 2 in sizes, "no Friedel pair survived, so there is nothing to merge"


def test_a_friedel_pair_is_excited_at_different_wavelengths(dataset):
    """Why scale error does not cancel in a Laue Bijvoet difference.

    On a rotation series both mates ride nearly the same scale.  On stills `h`
    is excited where the Ewald condition puts it, so `-h` needs a different
    wavelength -- and the spectrum spans 18x.
    """
    md = dataset["md"]
    by_class: dict[int, dict[int, list]] = {}
    for a, p, lam in zip(md["asu_id"], md["nonanom_id"], md["wavelength"]):
        by_class.setdefault(int(p), {}).setdefault(int(a), []).append(float(lam))
    gaps = [
        abs(np.mean(v[0]) - np.mean(v[1]))
        for v in (list(d.values()) for d in by_class.values())
        if len(v) == 2
    ]
    assert gaps, "no pair had both mates"
    # the mates' mean wavelengths differ by a real part of the band
    assert np.median(gaps) > 0.01, f"median gap {np.median(gaps):.4f} A"


def test_the_background_share_is_what_was_asked_for(dataset):
    """At 96.6% background the merge amplifies a background error ~28x, so the
    regime has to be the real one rather than a convenient one."""
    md, counts = dataset["md"], dataset["counts"]
    n_pix = counts.shape[1]
    share = float((md["true_bg"] * n_pix).sum() / counts.sum())
    assert share == pytest.approx(dataset["spec"]["truth"]["background_share"],
                                  abs=0.01)


def test_the_spectrum_is_partly_absorbable_by_resolution_and_radius(dataset):
    """The gauge that makes G(lambda) unidentifiable on a still.

    lambda = 2 d sin(theta(r)), so a free surface in (d, r) absorbs most of a
    realistic spectrum -- 73% measured on hewl_1118.  A simulator without this
    would make the spectrum look learnable when it is not.
    """
    md = dataset["md"]
    radius = np.hypot(md["xyzcal.px.0"] - SIM.DETECTOR_PIXELS / 2.0,
                      md["xyzcal.px.1"] - SIM.DETECTOR_PIXELS / 2.0)
    cols = [np.ones(len(radius))]
    for col in (md["d"], radius):
        u = 2.0 * (col - col.min()) / max(np.ptp(col), 1e-12) - 1.0
        cols += [u ** k for k in range(1, 7)]
    design = np.stack(cols, axis=1)

    def absorbed(target):
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        return 1.0 - (target - design @ coef).var() / max(target.var(), 1e-30)

    assert 0.4 < absorbed(md["true_spectrum"]) < 0.9, absorbed(md["true_spectrum"])
    # and the deliberate mismatch: absorption is crystal-frame, so a lab-frame
    # surface cannot see it. If this ever rises, the dataset has gone soft.
    assert absorbed(md["true_absorption"]) < 0.25, absorbed(md["true_absorption"])


def test_the_true_scale_reproduces_the_counts(dataset):
    """The generator's own consistency: truth times intensity is the signal.

    Without this a scoring run measures the simulator's bookkeeping rather than
    the model.
    """
    md, counts = dataset["md"], dataset["counts"]
    n_pix = counts.shape[1]
    signal = counts.sum(axis=1) - md["true_bg"] * n_pix
    expected = md["true_scale"] * md["true_intensity"]
    ok = expected > 50.0            # Poisson noise swamps the ratio below this
    ratio = signal[ok] / expected[ok]
    assert np.median(ratio) == pytest.approx(1.0, abs=0.05), np.median(ratio)


def test_centrics_carry_no_anomalous_difference(dataset):
    """I(+) = I(-) by symmetry, so a centric must not be given one."""
    md = dataset["md"]
    centric = md["centric"].astype(bool)
    if not centric.any():
        pytest.skip("no centric reflection in this corner of reciprocal space")
    by_pool: dict[int, set] = {}
    for p, i, c in zip(md["nonanom_id"], md["true_intensity"], centric):
        if c:
            by_pool.setdefault(int(p), set()).add(round(float(i), 6))
    assert all(len(v) == 1 for v in by_pool.values())


def test_the_loader_contract_is_satisfied(dataset):
    """Everything the real poly loader requires, so no arm needs a special path.

    `POLY_DS_COLS` is deliberately NOT the assertion: `IntegratorDataset`
    filters it with `if k in reference`, so most of it is optional and a
    simulated set is not obliged to invent `n_harmonics` or a `bbox`. What is
    required is what the loader raises on and what the merging model reads.
    """
    required = (
        # the loader raises without these
        "image_num", "nonanom_id",
        # the merge and its scale read these
        "asu_id", "d", "wavelength", "group_label", "refl_ids",
        "xyzcal.px.0", "xyzcal.px.1", "s1.0", "s1.1", "s1.2", "centric",
        # the ground truth this dataset exists to carry
        "true_scale", "true_bg", "true_intensity", "true_spectrum",
        "true_polarisation", "true_absorption",
    )
    md = dataset["md"]
    missing = [c for c in required if c not in md]
    assert not missing, missing
    for name in ("counts.npy", "masks.npy", "metadata.pt", "dataset.yaml",
                 "asu_id_to_hkl.pt"):
        assert (dataset["dir"] / name).exists(), name
    n = len(md["asu_id"])
    assert all(len(v) == n for v in md.values())
    assert dataset["counts"].shape[0] == n
    # the ids must index their own tables, which the writer relies on
    assert md["asu_id"].min() == 0
    table = torch.load(dataset["dir"] / "asu_id_to_hkl.pt", weights_only=False)
    assert len(table) == md["asu_id"].max() + 1
