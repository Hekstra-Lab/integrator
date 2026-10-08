"""Factory-path smoke test for the nanoBragg profile surrogate.

Needs a dataset cut by scripts/physics/make_sim_shoeboxes.py; point
INTEGRATOR_NANOBRAGG_DATA at it, otherwise the test is skipped.
"""

import copy
import os
from pathlib import Path

import pytest
import torch
import yaml

from integrator.utils import (
    apply_dataset_defaults,
    construct_data_loader,
    construct_integrator,
    inject_binning_labels,
    prepare_global_priors,
    prepare_per_bin_priors,
    resolve_config,
)

DATA = os.environ.get("INTEGRATOR_NANOBRAGG_DATA")
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "poly" / "dhfr_sim_nanobragg.yaml"

pytestmark = pytest.mark.skipif(not DATA or not Path(DATA, "dataset.yaml").exists(),
                                reason="set INTEGRATOR_NANOBRAGG_DATA to a make_sim_shoeboxes dataset")


def _cfg():
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg = copy.deepcopy(cfg)
    cfg["data_loader"]["args"].update(data_dir=DATA, batch_size=16, num_workers=0)
    cfg["integrator"]["args"]["mc_samples"] = 4
    cfg["surrogates"]["qp"]["args"].update(n_domains=8, n_lambda=7)
    cfg = resolve_config(apply_dataset_defaults(cfg))
    prepare_per_bin_priors(cfg)
    prepare_global_priors(cfg)
    return cfg


def test_factory_builds_and_trains_one_step():
    cfg = _cfg()
    assert cfg["surrogates"]["qp"]["args"]["data_dir"] == DATA
    dm = construct_data_loader(cfg)
    dm.setup()
    inject_binning_labels(dm, cfg)
    model = construct_integrator(cfg)
    qp = model.surrogates["qp"]
    assert type(qp).__name__ == "NanoBraggProfileSurrogate"

    batch = next(iter(dm.train_dataloader()))
    model.train()
    loss = model.training_step(batch, 0)
    loss = loss["loss"] if isinstance(loss, dict) else loss
    assert torch.isfinite(loss)
    loss.backward()
    for name in ("log_n_cells", "log_mosaic", "log_spectrum"):
        g = getattr(qp, name).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name
    assert qp.head.weight.grad is not None and qp.head.weight.grad.abs().sum() > 0

    out = model(*_unpack(batch))
    zp = out["qp"].zp
    assert torch.allclose(zp.sum(-1), torch.ones_like(zp.sum(-1)), atol=1e-4)


def _unpack(batch):
    counts, shoebox, mask, metadata = batch
    return counts, shoebox, mask, metadata
