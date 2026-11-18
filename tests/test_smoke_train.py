"""End-to-end smoke test of the deploy path on simulated data.

Synthesizes a tiny mono and poly dataset (counts/masks/metadata + dataset.yaml)
entirely in a tempdir -- no stored fixtures -- then runs the real train.py
sequence (prepare priors -> data module -> inject labels -> integrator ->
fit -> checkpoint -> predict). Guards against regressions in the full pipeline
that the unit tests miss.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from integrator.io import write_dataset_yaml
from integrator.utils import (
    apply_dataset_defaults,
    construct_data_loader,
    construct_integrator,
    inject_binning_labels,
    prepare_per_bin_priors,
    resolve_config,
)

N = 64


def _simulate_dataset(tmp_path, *, polychromatic: bool):
    """Write a minimal accepted dataset (counts/masks/metadata + dataset.yaml)."""
    g = torch.Generator().manual_seed(0)
    data_dir = tmp_path / ("poly" if polychromatic else "mono")
    data_dir.mkdir()

    d, h, w = (1, 11, 11) if polychromatic else (3, 11, 11)
    V = d * h * w
    counts = torch.poisson(torch.full((N, V), 5.0), generator=g)
    masks = torch.ones(N, V, dtype=torch.bool)
    ans = 2.0 * (counts + 0.375).sqrt()

    total = counts.sum(1)
    # background needs real spread: the bg-prior Gamma MLE divides by its variance
    bg_mean = torch.rand(N, generator=g) * 8.0 + 1.0
    ref = {
        "refl_ids": torch.arange(N, dtype=torch.int32),
        "is_test": torch.zeros(N, dtype=torch.bool),
        "d": torch.linspace(1.5, 30.0, N),
        "H": torch.randint(-20, 20, (N,), generator=g),
        "K": torch.randint(-20, 20, (N,), generator=g),
        "L": torch.randint(-20, 20, (N,), generator=g),
        "intensity.prf.value": total,
        "intensity.prf.variance": torch.ones(N),  # >=0 so none are filtered
        "intensity.sum.value": total,
        "intensity.sum.variance": total.clamp(min=1.0),
        "background.mean": bg_mean,
        "background.sum.value": bg_mean * V,
        "background.sum.variance": bg_mean * V,
        "xyzcal.px.0": torch.rand(N, generator=g) * 1000.0,
        "xyzcal.px.1": torch.rand(N, generator=g) * 1000.0,
        "xyzcal.px.2": torch.rand(N, generator=g) * 10.0,
    }

    crystal = None
    if polychromatic:
        ref["wavelength"] = torch.linspace(0.98, 1.22, N)
        ref["image_num"] = torch.arange(N) % 8
        crystal = {
            "lambda_min": 0.95,
            "lambda_max": 1.25,
            "beam_center_px": [w / 2.0, h / 2.0],
        }

    torch.save(counts, data_dir / "counts.pt")
    torch.save(masks, data_dir / "masks.pt")
    torch.save(ref, data_dir / "metadata.pt")
    write_dataset_yaml(
        data_dir,
        geometry={"d": d, "h": h, "w": w},
        n_reflections=N,
        polychromatic=polychromatic,
        anscombe=False,  # -> transform derives to 'standardization'
        files={
            "counts": "counts.pt",
            "masks": "masks.pt",
            "reference": "metadata.pt",
        },
        crystal=crystal,
        stats={
            "raw": [float(counts.mean()), float(counts.var())],
            "anscombe": [float(ans.mean()), float(ans.var())],
        },
        refl_file=str(data_dir / "reflections.refl"),
    )
    return data_dir


def _minimal_cfg(data_dir, *, polychromatic: bool):
    """A minimal config like configs/{mono,poly}/*_minimal.yaml; geometry from manifest."""
    return {
        "mode": "polychromatic" if polychromatic else "monochromatic",
        "integrator": {
            "name": "hierarchical",
            "args": {"encoder_out": 16, "mc_samples": 2},
        },
        "data_loader": {
            "args": {
                "data_dir": str(data_dir),
                "batch_size": 16,
                "validation_split": 0.25,
                "num_workers": 0,
            }
        },
    }


def _run_smoke(tmp_path, *, polychromatic: bool):
    data_dir = _simulate_dataset(tmp_path, polychromatic=polychromatic)
    cfg = _minimal_cfg(data_dir, polychromatic=polychromatic)

    # mirror train.py's orchestration
    cfg = apply_dataset_defaults(cfg)
    cfg = resolve_config(cfg)
    prepare_per_bin_priors(cfg)
    data_loader = construct_data_loader(cfg)
    data_loader.setup()
    inject_binning_labels(data_loader, cfg)
    model = construct_integrator(cfg)

    ckpt_dir = tmp_path / "ckpts"
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_train_batches=2,
        limit_val_batches=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=str(ckpt_dir), every_n_epochs=1, save_top_k=-1
            )
        ],
    )
    trainer.fit(
        model,
        train_dataloaders=data_loader.train_dataloader(),
        val_dataloaders=data_loader.val_dataloader(),
    )

    ckpts = list(ckpt_dir.glob("*.ckpt"))
    assert ckpts, "training produced no checkpoint"

    preds = trainer.predict(model, dataloaders=data_loader.val_dataloader())
    assert preds, "predict returned no batches"
    assert torch.isfinite(preds[0]["qi_mean"]).all()


def test_smoke_train_mono(tmp_path):
    _run_smoke(tmp_path, polychromatic=False)


def test_smoke_train_poly(tmp_path):
    _run_smoke(tmp_path, polychromatic=True)
