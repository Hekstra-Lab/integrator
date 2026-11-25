"""`trainer.ema_decay`: an EMA of the parameters, buffers following the live model, saved as the checkpoint's state_dict."""

from __future__ import annotations

import pathlib

import pytest
import torch

from integrator.callbacks.ema import ParameterEMA
from integrator.utils.factory_utils import construct_integrator, construct_trainer

from test_gradient_health import _batch

CFG = pathlib.Path(__file__).resolve().parents[1] / "configs" / "sim_truth"


def _cfg(name, **trainer):
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load((CFG / f"{name}.yaml").read_text())
    cfg["trainer"].update(dict(accelerator="cpu", max_epochs=2, check_val_every_n_epoch=1, logger=False,
                              enable_checkpointing=False), **trainer)
    return cfg


class _Loader(torch.utils.data.DataLoader):
    def __init__(self, n):
        super().__init__([_batch() for _ in range(n)], batch_size=None, collate_fn=lambda x: x)


def test_off_by_default():
    t = construct_trainer(_cfg("mos010_default_kmixv2_post"))
    assert not any(isinstance(c, ParameterEMA) for c in t.callbacks)
    with pytest.raises(ValueError, match="ema_decay"):
        construct_trainer(_cfg("mos010_default_kmixv2_post", ema_decay=1.5))


class _Record(__import__("pytorch_lightning").Callback):
    def __init__(self):
        self.steps = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.steps.append({n: p.detach().clone() for n, p in pl_module.named_parameters()})


def _fit(name, rec=None, **trainer):
    cfg = _cfg(name, **trainer)
    torch.manual_seed(0)
    model = construct_integrator(cfg)
    t = construct_trainer(cfg, callbacks=[rec] if rec else None)
    t.fit(model, _Loader(3), _Loader(1))
    return model, t


@pytest.mark.parametrize("name", ["mos010_default_kmixv2_post", "mos010_default_kmixv2_post_eb"])
def test_final_weights_are_the_ema_of_the_trajectory_and_buffers_are_live(name, tmp_path):
    rec = _Record()
    plain, _ = _fit(name, rec)
    ema_model, t = _fit(name, ema_decay=0.5)
    avg = None
    for st in rec.steps:  # the first update copies, then avg <- 0.5 avg + 0.5 theta
        avg = dict(st) if avg is None else {k: 0.5 * avg[k] + 0.5 * v for k, v in st.items()}
    for n, p in ema_model.named_parameters():
        if p.is_floating_point():
            torch.testing.assert_close(p.detach(), avg[n], rtol=1e-4, atol=1e-5, msg=n)
    assert any(not torch.allclose(p, plain.state_dict()[n]) for n, p in ema_model.named_parameters())
    for (n, b), (_, b0) in zip(ema_model.named_buffers(), plain.named_buffers()):
        assert torch.equal(b, b0), n  # buffers follow the live model, never averaged
    path = tmp_path / "x.ckpt"
    t.save_checkpoint(path)
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    for k, v in ema_model.state_dict().items():
        assert torch.equal(sd[k], v), k
