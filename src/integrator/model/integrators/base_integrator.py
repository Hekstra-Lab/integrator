from abc import abstractmethod
from typing import Any, Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor

DEFAULT_PREDICT_KEYS = [
    "refl_ids",
    "is_test",
    "qi_mean",
    "qi_var",
    "qi_params",
    "qbg_mean",
    "qbg_var",
    "qbg_scale",
    "qbg_params",
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


def _log_loss(
    self,
    kl,
    nll,
    total_loss,
    step: Literal["train", "val"],
    kl_components: dict[str, Tensor] | None = None,
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
    self.log("epoch", float(self.current_epoch), on_step=False, on_epoch=True)
    if kl_components is not None:
        for name, value in kl_components.items():
            self.log(
                f"{step} kl_{name}",
                value,
                on_step=False,
                on_epoch=True,
            )


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
        self.decoder_weight_decay = cfg.decoder_weight_decay
        self.mc_samples = cfg.mc_samples
        self.renyi_scale = cfg.renyi_scale
        if cfg.data_dim == "2d":
            self.shoebox_shape = (cfg.h, cfg.w)
        else:
            self.shoebox_shape = (cfg.d, cfg.h, cfg.w)

        # predict step keys
        self.predict_keys = (
            DEFAULT_PREDICT_KEYS
            if cfg.predict_keys == "default"
            else cfg.predict_keys
        )

        self.encoders = nn.ModuleDict(encoders)
        self.surrogates = nn.ModuleDict(surrogates)
        self.loss = loss

        self.automatic_optimization = True

    def setup(self, stage: str) -> None:
        """Infer dataset_size for losses that need it (e.g. HierarchicalLoss)."""
        if stage == "fit" and hasattr(self.loss, "dataset_size"):
            dm = self.trainer.datamodule
            if dm is not None and hasattr(dm, "train_dataset"):
                self.loss.dataset_size = len(dm.train_dataset)

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
            kl_components={
                k.removesuffix("_mean"): v
                for k, v in loss_dict.items()
                if k in ("kl_prf_mean", "kl_i_mean", "kl_bg_mean", "kl_hyper_mean")
            },
        )

        # Log hyperprior diagnostics if present (HierarchicalLoss)
        for key in (
            "kl_global",
            "hp_alpha_mean",
            "hp_beta_mean",
            "hp_alpha_std",
            "hp_beta_std",
            "hp_prior_mean",
        ):
            if key in loss_dict:
                self.log(
                    f"{step} {key}",
                    loss_dict[key],
                    on_step=False,
                    on_epoch=True,
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
            k: v
            for k, v in outputs["forward_out"].items()
            if k in self.predict_keys
        }

    def on_validation_epoch_end(self) -> None:
        """Log val - train generalization gaps for each ELBO component.

        For every logged metric that exists as both 'train X' and 'val X',
        emits 'gap X' = val - train. Runs at the end of validation, when
        both train (accumulated over the current train epoch) and val
        values are available in trainer.callback_metrics.
        """
        metrics = self.trainer.callback_metrics
        for key in list(metrics.keys()):
            if not key.startswith("val "):
                continue
            suffix = key[len("val "):]
            train_key = f"train {suffix}"
            if train_key not in metrics:
                continue
            gap = metrics[key] - metrics[train_key]
            self.log(
                f"gap {suffix}",
                gap,
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(self):
        if self.decoder_weight_decay is None:
            return torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        # Decoder-specific weight decay: only targets the learned profile
        # basis W (nn.Linear.weight of qp.decoder). All other parameters
        # keep the base weight_decay.
        decoder_params: list[nn.Parameter] = []
        other_params: list[nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("surrogates.qp.decoder.weight"):
                decoder_params.append(param)
            else:
                other_params.append(param)
        if not decoder_params:
            raise RuntimeError(
                "decoder_weight_decay is set but no "
                "'surrogates.qp.decoder.weight' parameter was found. "
                "Only the learned_basis_profile surrogate exposes this; "
                "set decoder_weight_decay=null for other surrogates."
            )
        return torch.optim.Adam(
            [
                {"params": other_params, "weight_decay": self.weight_decay},
                {
                    "params": decoder_params,
                    "weight_decay": self.decoder_weight_decay,
                },
            ],
            lr=self.lr,
        )
