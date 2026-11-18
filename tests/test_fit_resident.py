"""`integrator.cli.fit --resident` against the streaming path it replaces.

The resident path skips the data module entirely: counts and masks go to the
device once and the metadata is read straight from the reference file. That
makes file order the row order, which is a correctness assumption and not only
a speed one -- if the rows come back in a different order, or a different set,
the parquet is wrong rather than slow, and every downstream join on `refl_ids`
inherits it.

So the test of record is that the two paths write the same parquet: same
columns, same rows, same order. The predict dataloader does not shuffle and
`min_valid_pixels` is read from the same config on both paths, so the row sets
and the row order are exact, and the metadata columns -- which are copied, not
computed -- must match bitwise.

The computed columns do not, and asking them to was the first version of this
test. The chunk size differs between the paths (`batch_size` streaming,
`--chunk` resident), so float32 accumulates in a different order, and
`Ihat = sum w p (c - b) / sum w p^2` is a cancellation: on this dataset every
shoebox is pure background, so `c - b` cancels to the last bits and the
disagreement scales with the counts in the box rather than with `Ihat`. At a
mean of 6 counts over 121 pixels that is about 1e-3 absolute, which the first
version's `rtol=1e-5, atol=1e-5` failed on CUDA while passing on CPU, where both
paths pick the same kernels. On hewl_1118 the same effect is 0.594 counts at
worst against a median intensity of 37.6 and a median sigma of 18.2.

What a real defect would do instead is break the correlation -- a row
misalignment, a different filter or a transposed chunk all show up there at
order one -- so the computed columns are held to a correlation bound and to an
absolute bound scaled to the column's own spread.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch
import yaml

from integrator.cli.fit import _RESIDENT_KEYS, main
from integrator.fit.resident import ResidentShoeboxes
from integrator.io import write_dataset_yaml
from integrator.utils import apply_dataset_defaults, resolve_config

N = 96
H = W = 11

# solved per reflection, so float32 order of accumulation applies
_COMPUTED = (
    "qi_mean",
    "qi_var",
    "qbg_mean",
    "qbg_var",
    "i_rb",
    "var_rb",
    "n_eff",
)


def _write_dataset(tmp_path):
    """A small polychromatic dataset with `.npy` counts, as the real one has."""
    g = torch.Generator().manual_seed(0)
    data_dir = tmp_path / "poly"
    data_dir.mkdir()
    v = H * W

    counts = torch.poisson(torch.full((N, v), 6.0), generator=g)
    masks = torch.ones(N, v, dtype=torch.bool)
    np.save(data_dir / "counts.npy", counts.numpy().astype(np.int32))
    np.save(data_dir / "masks.npy", masks.numpy())

    total = counts.sum(1)
    bg = torch.rand(N, generator=g) * 8.0 + 1.0
    ref = {
        "refl_ids": torch.arange(N, dtype=torch.float32),
        "is_test": torch.zeros(N),
        "d": torch.linspace(1.6, 30.0, N),
        "wavelength": torch.linspace(0.98, 1.22, N),
        "H": torch.randint(-20, 20, (N,), generator=g).float(),
        "K": torch.randint(-20, 20, (N,), generator=g).float(),
        "L": torch.randint(-20, 20, (N,), generator=g).float(),
        "image_num": (torch.arange(N) % 8).float(),
        "intensity.prf.value": total,
        "intensity.prf.variance": torch.ones(N),
        "intensity.sum.value": total,
        "intensity.sum.variance": total.clamp(min=1.0),
        "background.mean": bg,
        "background.sum.value": bg * v,
        "background.sum.variance": bg * v,
        "xyzcal.px.0": torch.rand(N, generator=g) * 1000.0,
        "xyzcal.px.1": torch.rand(N, generator=g) * 1000.0,
        "xyzcal.px.2": torch.rand(N, generator=g) * 10.0,
    }
    torch.save(ref, data_dir / "metadata.pt")

    ans = 2.0 * (counts + 0.375).sqrt()
    write_dataset_yaml(
        data_dir,
        geometry={"d": 1, "h": H, "w": W},
        n_reflections=N,
        polychromatic=True,
        anscombe=False,
        files={
            "counts": "counts.npy",
            "masks": "masks.npy",
            "reference": "metadata.pt",
        },
        crystal={
            "lambda_min": 0.95,
            "lambda_max": 1.25,
            "beam_center_px": [W / 2.0, H / 2.0],
        },
        stats={
            "raw": [float(counts.mean()), float(counts.var())],
            "anscombe": [float(ans.mean()), float(ans.var())],
        },
        refl_file=str(data_dir / "reflections.refl"),
    )
    return data_dir


def _write_config(tmp_path, data_dir, batch_size=16):
    """The config the fit CLI reads, resolved as `config_log.yaml` already is."""
    cfg = {
        "mode": "polychromatic",
        "integrator": {
            "name": "hierarchical",
            "args": {"encoder_out": 16, "mc_samples": 2},
        },
        "data_loader": {
            "args": {
                "data_dir": str(data_dir),
                "batch_size": batch_size,
                "validation_split": 0.25,
                "num_workers": 0,
            }
        },
    }
    cfg = resolve_config(apply_dataset_defaults(cfg))
    # the direct fit needs a profile that is a function of geometry and a few
    # globals; this is the `physical` arm's block, at this dataset's size
    cfg.setdefault("surrogates", {})["qp"] = {
        "name": "physical_profile",
        "args": {
            "input_dim": 16,
            "output_dim": H * W,
            "profile_shape": [1, H, W],
            "beam_center": [W / 2.0, H / 2.0],
            "lambda_min": 0.95,
            "lambda_max": 1.25,
        },
    }
    path = tmp_path / "config_log.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def test_resident_and_streaming_write_the_same_parquet(tmp_path):
    data_dir = _write_dataset(tmp_path)
    cfg = _write_config(tmp_path, data_dir)

    stream, resident = tmp_path / "stream", tmp_path / "resident"
    common = ["--config", str(cfg), "--fit-batches", "2"]
    main([*common, "--out-dir", str(stream)])
    main([*common, "--out-dir", str(resident), "--resident", "--chunk", "32"])

    a = pl.read_parquet(stream / "pred.parquet")
    b = pl.read_parquet(resident / "pred.parquet")

    assert a.height == b.height == N
    assert a.columns == b.columns

    # the rows themselves: same reflections, same order, on both paths
    ids_a, ids_b = a["refl_ids"].to_numpy(), b["refl_ids"].to_numpy()
    assert np.array_equal(ids_a, ids_b)
    assert np.array_equal(ids_a, np.sort(ids_a))

    for col in a.columns:
        x = a[col].to_numpy().astype(np.float64)
        y = b[col].to_numpy().astype(np.float64)
        if col not in _COMPUTED:
            assert np.array_equal(x, y), f"copied column {col} is not identical"
            continue
        # a misaligned or differently filtered pass breaks this at order one
        assert np.corrcoef(x, y)[0, 1] > 1.0 - 1e-9, f"column {col} decorrelates"
        scale = max(float(np.sqrt(np.mean(x**2))), 1.0)
        assert np.abs(x - y).max() <= 1e-3 * scale, (
            f"column {col} differs by {np.abs(x - y).max():.3e}, more than "
            f"float32 accumulation order explains at a spread of {scale:.3g}"
        )


def test_resident_fits_the_same_globals(tmp_path):
    data_dir = _write_dataset(tmp_path)
    cfg = _write_config(tmp_path, data_dir)

    import json

    stream, resident = tmp_path / "stream", tmp_path / "resident"
    common = ["--config", str(cfg), "--fit-batches", "2"]
    main([*common, "--out-dir", str(stream)])
    main([*common, "--out-dir", str(resident), "--resident", "--chunk", "32"])

    a = json.loads((stream / "globals.json").read_text())
    b = json.loads((resident / "globals.json").read_text())
    for name in ("a", "b", "c", "z"):
        assert a[name] == pytest.approx(b[name], rel=1e-4), name


def test_the_head_slice_holds_fit_batches_times_batch_size(tmp_path):
    """The global fit must see the same number of reflections on both paths."""
    data_dir = _write_dataset(tmp_path)
    cfg = _write_config(tmp_path, data_dir, batch_size=16)

    out = tmp_path / "resident"
    main(
        [
            "--config", str(cfg), "--out-dir", str(out),
            "--fit-batches", "3", "--resident", "--chunk", "32", "-v",
        ]
    )
    # 3 batches of 16 is the streaming default's sample size
    assert (out / "globals.json").exists()
    assert pl.read_parquet(out / "pred.parquet").height == N


def test_recenter_runs_through_the_cli_and_writes_the_shifts(tmp_path):
    """`--recenter` through the real CLI, not by reading the code."""
    data_dir = _write_dataset(tmp_path)
    cfg = _write_config(tmp_path, data_dir)

    plain, moved = tmp_path / "plain", tmp_path / "moved"
    common = ["--config", str(cfg), "--fit-batches", "2"]
    main([*common, "--out-dir", str(plain)])
    main(
        [
            *common, "--out-dir", str(moved), "--recenter",
            "--recenter-iter", "2", "--max-shift", "2.0",
        ]
    )

    a = pl.read_parquet(plain / "pred.parquet")
    b = pl.read_parquet(moved / "pred.parquet")
    # the flag is additive: the plain schema is untouched and the shifts appear
    assert "mu_radial" not in a.columns
    assert {"mu_radial", "mu_tangential"} <= set(b.columns)
    assert set(a.columns) <= set(b.columns)
    assert a.height == b.height == N
    for col in ("mu_radial", "mu_tangential"):
        v = b[col].to_numpy()
        assert np.isfinite(v).all()
        assert np.abs(v).max() <= 2.0 + 1e-6, "--max-shift is not enforced"
    assert np.isfinite(b["qi_mean"].to_numpy()).all()


def test_recenter_with_resident_and_image_bins(tmp_path):
    """The two flags compose, and binning the shrinkage by image works."""
    data_dir = _write_dataset(tmp_path)
    cfg = _write_config(tmp_path, data_dir)

    out = tmp_path / "both"
    main(
        [
            "--config", str(cfg), "--out-dir", str(out), "--fit-batches", "2",
            "--resident", "--chunk", "32", "--recenter", "--recenter-iter", "2",
            "--recenter-bins", "4",
        ]
    )
    d = pl.read_parquet(out / "pred.parquet")
    assert d.height == N
    assert np.isfinite(d["mu_radial"].to_numpy()).all()
    assert np.isfinite(d["qi_var"].to_numpy()).all()
    assert (d["qi_var"].to_numpy() > 0).all()


def test_metadata_keys_selects_columns_and_skips_absent_ones(tmp_path):
    data_dir = _write_dataset(tmp_path)

    default = ResidentShoeboxes(data_dir, "cpu", chunk=32)
    assert "intensity.sum.value" not in default.metadata

    wide = ResidentShoeboxes(
        data_dir, "cpu", chunk=32, metadata_keys=_RESIDENT_KEYS
    )
    for key in _RESIDENT_KEYS:
        assert key in wide.metadata, key
    for key in ("d", "wavelength", "xyzcal.px.0"):
        assert torch.allclose(wide.metadata[key], default.metadata[key])

    asked = ResidentShoeboxes(
        data_dir, "cpu", chunk=32, metadata_keys=("d", "not_a_column")
    )
    assert set(asked.metadata) == {"d"}
