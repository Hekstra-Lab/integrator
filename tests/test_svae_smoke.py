"""End-to-end smoke test for the SVAE integrators on simulated mono data.

Reuses the dataset synthesizer from test_smoke_train and runs the real
factory path (resolve -> priors -> data module -> integrator -> fit ->
predict) for both 'svae' (conjugate background) and 'svae_hybrid'
(variational background). Guards the conjugate Poisson-Gamma forward pass +
the loss/integrator pairing through full construction.
"""

from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from test_smoke_train import _simulate_dataset

from integrator.utils import (
    apply_dataset_defaults,
    construct_data_loader,
    construct_integrator,
    inject_binning_labels,
    prepare_per_bin_priors,
    resolve_config,
)


def _svae_cfg(data_dir, name):
    return {
        "mode": "monochromatic",
        "integrator": {
            "name": name,
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


@pytest.mark.parametrize("name", ["svae", "svae_hybrid"])
def test_svae_smoke(tmp_path, name):
    data_dir = _simulate_dataset(tmp_path, polychromatic=False)
    cfg = _svae_cfg(data_dir, name)

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

    preds = trainer.predict(model, dataloaders=data_loader.val_dataloader())
    assert preds, "predict returned no batches"
    assert torch.isfinite(preds[0]["qi_mean"]).all()
    assert torch.isfinite(preds[0]["qbg_mean"]).all()
