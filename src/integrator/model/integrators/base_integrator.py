from abc import abstractmethod
from typing import Any, Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor

DEFAULT_PREDICT_KEYS = [
    "qi_mean",
    "qi_var",
    "refl_ids",
    "qbg_mean",
    "qbg_var",
    "qbg_scale",
    "intensity.prf.value",
    "intensity.prf.variance",
    "intensity.sum.value",
    "intensity.sum.variance",
    "background.mean",
    "xyzcal.px.0",
    "xyzcal.px.1",
    "xyzcal.px.2",
    "d",
    "H",
    "K",
    "L",
]


def _log_forward_out(
    self,
    forward_out: dict,
    step: Literal["train", "val"],
):
    # if step != "val":
    #     return
    self.log(
        f"{step}: mean(qi.mean)",
        forward_out["qi_mean"].mean(),
        on_step=False,
        on_epoch=True,
    )
    self.log(
        f"{step}: min(qi.mean)",
        forward_out["qi_mean"].min(),
        on_step=False,
        on_epoch=True,
    )
    self.log(
        f"{step}: max(qi.mean)",
        forward_out["qi_mean"].max(),
        on_step=False,
        on_epoch=True,
    )
    self.log(
        f"{step}: max(qi.variance)",
        forward_out["qi_var"].max(),
        on_step=False,
        on_epoch=True,
    )
    self.log(
        f"{step}: min(qi.variance)",
        forward_out["qi_var"].min(),
        on_step=False,
        on_epoch=True,
    )
    self.log(
        f"{step}: mean(qi.variance)",
        forward_out["qi_var"].mean(),
        on_step=False,
        on_epoch=True,
    )
    self.log(
        f"{step}: mean(qbg.mean)",
        forward_out["qbg_mean"].mean(),
        on_step=False,
        on_epoch=True,
    )
    self.log(
        f"{step}: min(qbg.mean)",
        forward_out["qbg_mean"].min(),
        on_step=False,
        on_epoch=True,
    )
    self.log(
        f"{step}: max(qbg.mean)",
        forward_out["qbg_mean"].max(),
        on_step=False,
        on_epoch=True,
    )
    self.log(
        f"{step}: mean(qbg.variance)",
        forward_out["qbg_var"].mean(),
        on_step=False,
        on_epoch=True,
    )
    self.log(
        f"{step}: max(qbg.variance)",
        forward_out["qbg_var"].max(),
        on_step=False,
        on_epoch=True,
    )
    self.log(
        f"{step}: min(qbg.variance)",
        forward_out["qbg_var"].min(),
        on_step=False,
        on_epoch=True,
    )


def _log_loss(
    self,
    kl,
    nll,
    total_loss,
    step: Literal["train", "val"],
):
    self.log(
        f"{step} elbo",
        total_loss,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
    )
    self.log(f"{step} kl", kl, on_step=False, on_epoch=True)
    self.log(f"{step} nll", nll, on_step=False, on_epoch=True)


class BaseIntegrator(pl.LightningModule):
    REQUIRED_ENCODERS: dict[str, type] = {}
    ARGS: type

    def __init__(
        self,
        cfg,
        loss: nn.Module,
        encoders: dict[str, nn.Module],
        surrogates: dict[str, nn.Module],
    ):
        super().__init__()
        self.cfg = cfg

        # hyperparams
        self.lr = cfg.lr
        self.weight_decay = cfg.weight_decay
        self.mc_samples = cfg.mc_samples
        self.renyi_scale = cfg.renyi_scale
        if cfg.data_dim == "2d":
            self.shoebox_shape = (cfg.h, cfg.w)
        else:
            self.shoebox_shape = (cfg.d, cfg.h, cfg.w)

        # predict step keys
        self.predict_keys = (
            DEFAULT_PREDICT_KEYS if cfg.predict_keys == "default" else cfg.predict_keys
        )

        self.encoders = nn.ModuleDict(encoders)
        self.surrogates = nn.ModuleDict(surrogates)
        self.loss = loss

        self.automatic_optimization = True

    def forward(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        metadata: dict,
    ) -> dict[str, Any]:
        out = self._forward_impl(counts, shoebox, mask, metadata)
        return out

    @abstractmethod
    def _forward_impl(
        self,
        counts: torch.Tensor,
        shoebox: torch.Tensor,
        mask: torch.Tensor,
        metadata: dict,
    ) -> dict[str, Any]: ...

    def _step(self, batch, step: Literal["train", "val"]):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]

        loss_dict = self.loss(
            rate=forward_out["rates"],
            counts=forward_out["counts"],
            qp=outputs["qp"],
            qi=outputs["qi"],
            qbg=outputs["qbg"],
            mask=forward_out["mask"],
        )

        total_loss = loss_dict["loss"]

        _log_loss(
            self,
            kl=loss_dict["kl_mean"],
            nll=loss_dict["neg_ll_mean"],
            total_loss=total_loss,
            step=step,
        )

        return {
            "loss": total_loss,
            "forward_out": forward_out,
            "loss_components": {
                "loss": total_loss.detach(),
                "nll": loss_dict["neg_ll_mean"].detach(),
                "kl": loss_dict["kl_mean"].detach(),
                "kl_prf": loss_dict["kl_prf_mean"].detach(),
                "kl_i": loss_dict["kl_i_mean"].detach(),
                "kl_bg": loss_dict["kl_bg_mean"].detach(),
            },
        }

    def training_step(self, batch, _batch_idx):
        return self._step(batch, step="train")

    def validation_step(self, batch, _batch_idx):
        return self._step(batch, step="val")

    def predict_step(self, batch, _batch_idx):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)

        return {
            k: v for k, v in outputs["forward_out"].items() if k in self.predict_keys
        }

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
