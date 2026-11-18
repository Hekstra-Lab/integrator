"""End-to-end smoke test for the four incremental-ablation configs.

Loads each shipped `configs/mono/ablation_v*.yaml`, points it at a tiny
synthesized mono dataset, and runs the real train.py orchestration
(resolve -> priors -> data module -> integrator -> fit -> predict). Guards
the rebuilt components -- the 2-encoder integrator, the global-prior loss +
its Gamma MLE, and the Dirichlet profile -- through full construction, plus
the configs themselves.

Ablation chain (one change per step):
    v1: 2-enc  + global prior + Dirichlet
    v2: 3-enc  + global prior + Dirichlet     (v1 -> v2: split intensity/background)
    v3: 5-enc  + global prior + Dirichlet     (v2 -> v3: split each Gamma parameter)
    v4: 5-enc  + Wilson prior + Dirichlet      (v3 -> v4: prior)
    v5: 5-enc  + Wilson prior + learned basis  (v4 -> v5: profile)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
import yaml
from test_smoke_train import _simulate_dataset

from integrator.utils import (
    apply_dataset_defaults,
    construct_data_loader,
    construct_integrator,
    inject_binning_labels,
    prepare_global_priors,
    prepare_per_bin_priors,
    resolve_config,
)

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "mono"

_ABLATION_CONFIGS = {
    "v1": "ablation_v1_2enc_global_dirichlet.yaml",
    "v2": "ablation_v2_3enc_global_dirichlet.yaml",
    "v3": "ablation_v3_5enc_global_dirichlet.yaml",
    "v4": "ablation_v4_5enc_wilson_dirichlet.yaml",
    "v5": "ablation_v5_5enc_wilson_learnedbasis.yaml",
}

_EXPECTED_N_ENCODERS = {"v1": 2, "v2": 3, "v3": 5, "v4": 5, "v5": 5}


def _load_ablation_cfg(name: str, data_dir) -> dict:
    # Keep the configs' own (micro) encoder_out so the explicit `encoders:`
    # widths stay consistent with the surrogate in_features; only shrink the
    # MC sample count and point at the tiny synthesized dataset.
    cfg = yaml.safe_load((_CONFIG_DIR / name).read_text())
    cfg["integrator"]["args"]["mc_samples"] = 2
    cfg["data_loader"]["args"] = {
        "data_dir": str(data_dir),
        "batch_size": 16,
        "validation_split": 0.25,
        "num_workers": 0,
    }
    return cfg


def _construct(cfg):
    """Resolve + prepare priors + build the data module and integrator."""
    cfg = apply_dataset_defaults(cfg)
    cfg = resolve_config(cfg)
    prepare_per_bin_priors(cfg)
    prepare_global_priors(cfg)

    # global-prior configs must have their MLE-fit priors injected
    if cfg["loss"]["name"] == "global_prior":
        loss_args = cfg["loss"]["args"]
        assert loss_args["i_concentration"] > 0
        assert loss_args["i_rate"] > 0
        assert loss_args["bg_concentration"] > 0
        assert loss_args["bg_rate"] > 0

    data_loader = construct_data_loader(cfg)
    data_loader.setup()
    inject_binning_labels(data_loader, cfg)
    model = construct_integrator(cfg)
    return model, data_loader


def _n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _run_factory_path(cfg):
    """Run train.py's orchestration on a config; return (model, preds)."""
    model, data_loader = _construct(cfg)

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
    )
    trainer.fit(
        model,
        train_dataloaders=data_loader.train_dataloader(),
        val_dataloaders=data_loader.val_dataloader(),
    )
    preds = trainer.predict(model, dataloaders=data_loader.val_dataloader())
    assert preds, "predict returned no batches"
    return model, preds


@pytest.mark.parametrize("version", list(_ABLATION_CONFIGS))
def test_ablation_smoke(tmp_path, version):
    data_dir = _simulate_dataset(tmp_path, polychromatic=False)
    cfg = _load_ablation_cfg(_ABLATION_CONFIGS[version], data_dir)

    model, preds = _run_factory_path(cfg)

    assert len(model.encoders) == _EXPECTED_N_ENCODERS[version]
    assert torch.isfinite(preds[0]["qi_mean"]).all()
    assert torch.isfinite(preds[0]["qbg_mean"]).all()


def test_global_prior_with_learned_basis(tmp_path):
    """The global-prior loss must pair with the learned-basis profile (V3b).

    V1-V3 exercise global_prior + Dirichlet and V5 exercises Wilson + learned
    basis; this guards the remaining cross-cell, global_prior + learned basis,
    whose profile KL goes through the Gaussian-latent branch of
    compute_profile_kl rather than the Dirichlet branch.
    """
    from integrator.model.loss import GlobalPriorLoss

    data_dir = _simulate_dataset(tmp_path, polychromatic=False)
    cfg = _load_ablation_cfg(
        "ablation_v3b_5enc_global_learnedbasis.yaml", data_dir
    )
    # surface the learned-basis profile mean so we can assert it is finite
    cfg["integrator"]["args"]["additional_predict_keys"] = ["qp_mean"]

    model, preds = _run_factory_path(cfg)

    assert isinstance(model.loss, GlobalPriorLoss)
    # learned-basis profile has a decoder W (Dirichlet does not)
    assert hasattr(model.surrogates["qp"], "decoder")
    assert len(model.encoders) == 5
    assert torch.isfinite(preds[0]["qi_mean"]).all()
    assert torch.isfinite(preds[0]["qbg_mean"]).all()
    # the learned basis emits a profile mean; confirm it is produced and finite
    assert torch.isfinite(preds[0]["qp_mean"]).all()


def test_v1b_is_parameter_matched_to_v3(tmp_path):
    """V1b (2-enc, widened) must match V3 (5-enc) in total params.

    This is the parameter-count control: V1b keeps V1's 2-encoder shared
    representation but widens the single intensity encoder so its total
    parameter count equals the 5-encoder V3 (whose 4 intensity encoders are the
    only architectural difference from V1). Any V3-over-V1b gain is then
    attributable to encoder decoupling, not to parameter count.
    """
    data_dir = _simulate_dataset(tmp_path, polychromatic=False)

    v1, _ = _construct(_load_ablation_cfg(_ABLATION_CONFIGS["v1"], data_dir))
    v1b, _ = _construct(
        _load_ablation_cfg(
            "ablation_v1b_2enc_global_dirichlet_parammatched.yaml", data_dir
        )
    )
    v3, _ = _construct(_load_ablation_cfg(_ABLATION_CONFIGS["v3"], data_dir))

    # V1b keeps the 2-encoder architecture but matches V3's parameter count.
    assert len(v1b.encoders) == 2
    assert len(v3.encoders) == 5
    assert abs(_n_params(v1b) - _n_params(v3)) <= 50, (
        _n_params(v1b),
        _n_params(v3),
    )
    # The un-widened V1 is far smaller -- the confound the control removes.
    assert _n_params(v1) < 0.6 * _n_params(v3)


def test_v1b_smoke(tmp_path):
    """The widened V1b still trains and predicts finite values."""
    data_dir = _simulate_dataset(tmp_path, polychromatic=False)
    cfg = _load_ablation_cfg(
        "ablation_v1b_2enc_global_dirichlet_parammatched.yaml", data_dir
    )
    model, preds = _run_factory_path(cfg)
    assert len(model.encoders) == 2
    assert torch.isfinite(preds[0]["qi_mean"]).all()
    assert torch.isfinite(preds[0]["qbg_mean"]).all()


def test_ablation_chain_changes_one_thing_at_a_time():
    """Each consecutive config must differ from the previous in exactly one axis."""

    def axes(name):
        cfg = yaml.safe_load((_CONFIG_DIR / name).read_text())
        return {
            "integrator": cfg["integrator"]["name"],
            "loss": cfg["loss"]["name"],
            "profile": cfg["surrogates"]["qp"]["name"],
        }

    a = {v: axes(n) for v, n in _ABLATION_CONFIGS.items()}

    def diff(x, y):
        return [k for k in x if x[k] != y[k]]

    assert diff(a["v1"], a["v2"]) == ["integrator"]  # encoders: 2 -> 3
    assert diff(a["v2"], a["v3"]) == ["integrator"]  # encoders: 3 -> 5
    assert diff(a["v3"], a["v4"]) == ["loss"]  # global -> Wilson prior
    assert diff(a["v4"], a["v5"]) == ["profile"]  # Dirichlet -> learned basis
