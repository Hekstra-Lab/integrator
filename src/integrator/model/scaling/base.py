"""Standalone Lightning base for the scaling/merging models."""

import logging
import math
from abc import abstractmethod
from typing import Any, Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor

from integrator.configs import OptimizerConfig
from integrator.model.scaling.config import MergingIntegratorCfg

_logger = logging.getLogger(__name__)

# default keys to write out during prediction
DEFAULT_PREDICT_KEYS = [
    "refl_ids",
    "is_test",
    "qi_mean",
    "qi_var",
    "qbg_mean",
    "qbg_var",
    "background.mean",
    "d",
    "miller_idx_unfriedelized",
    "group_label",
    "H",
    "K",
    "L",
]


class ScalingLightningModule(pl.LightningModule):
    """Lightning base for the merging integrators (no `BaseIntegrator` parent)."""

    REQUIRED_ENCODERS: dict[str, tuple[str, type]] = {}
    DEFAULT_SURROGATES: dict[str, dict] = {}
    CFG_CLASS: type = MergingIntegratorCfg

    def __init__(
        self,
        cfg: MergingIntegratorCfg,
        loss: nn.Module,
        encoders: dict[str, nn.Module],
        surrogates: dict[str, nn.Module],
        optimizer: OptimizerConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg

        opt = optimizer or OptimizerConfig()
        self.optimizer_cfg = opt
        self.lr = opt.lr
        self.weight_decay = opt.weight_decay
        self.decoder_weight_decay = opt.decoder_weight_decay
        self.lr_schedule = opt.lr_schedule
        self.warmup_epochs = opt.warmup_epochs
        self.warmup_steps = opt.warmup_steps
        self.lr_min = opt.lr_min
        # Decoupled scale-field LR lives on the merger's own cfg.
        self.scaling_lr = getattr(cfg, "scaling_lr", None)

        self.mc_samples = cfg.mc_samples
        if cfg.data_dim == "2d":
            self.shoebox_shape = (cfg.h, cfg.w)
        else:
            self.shoebox_shape = (cfg.d, cfg.h, cfg.w)

        self.predict_keys = (
            DEFAULT_PREDICT_KEYS
            if cfg.predict_keys == "default"
            else cfg.predict_keys
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
        return self._forward_impl(counts, shoebox, mask, metadata)

    @abstractmethod
    def _forward_impl(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        metadata: dict,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def _step(self, batch, step: Literal["train", "val"]): ...

    def training_step(self, batch, _batch_idx):
        return self._step(batch, step="train")

    def validation_step(self, batch, _batch_idx):
        return self._step(batch, step="val")

    def predict_step(self, batch, _batch_idx):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]
        self._warn_unknown_predict_keys(forward_out)
        return {k: v for k, v in forward_out.items() if k in self.predict_keys}

    def _warn_unknown_predict_keys(self, forward_out: dict) -> None:
        if getattr(self, "_predict_keys_checked", False):
            return
        self._predict_keys_checked = True
        missing = [k for k in self.predict_keys if k not in forward_out]
        if missing:
            _logger.warning(
                "predict_keys not produced by this integrator (ignored): %s. "
                "Available outputs: %s",
                sorted(missing),
                sorted(forward_out),
            )

    def _profile_basis_penalty(self) -> tuple[Tensor, dict[str, Tensor]]:
        """Spatial-smoothness penalty on the learned qp decoder weight W."""
        zero = torch.zeros((), device=self.device)
        qp = self.surrogates["qp"] if "qp" in self.surrogates else None
        decoder = getattr(qp, "decoder", None)
        W = getattr(decoder, "weight", None)
        smooth_w = float(getattr(qp, "smoothness_weight", 0.0))
        if W is None or W.dim() != 2:
            return zero, {}

        shape = self.shoebox_shape
        D, H, W_spatial = shape if len(shape) == 3 else (1, *shape)
        K, d = W.shape
        if K != D * H * W_spatial:
            return zero, {}

        total = zero
        components: dict[str, Tensor] = {}
        if smooth_w > 0:
            vol = W.T.reshape(d, D, H, W_spatial)
            gx = vol[..., 1:] - vol[..., :-1]
            gy = vol[..., 1:, :] - vol[..., :-1, :]
            sq_sum = (gx.pow(2)).sum() + (gy.pow(2)).sum()
            n_terms = gx.numel() + gy.numel()
            if D > 1:
                gz = vol[..., 1:, :, :] - vol[..., :-1, :, :]
                sq_sum = sq_sum + gz.pow(2).sum()
                n_terms = n_terms + gz.numel()
            smooth = sq_sum / max(n_terms, 1)
            components["profile_smoothness"] = smooth.detach()
            total = total + smooth_w * smooth
        return total, components

    def on_after_backward(self) -> None:
        for sname in ("qi", "qbg"):
            if sname not in self.surrogates:
                continue
            for pname, p in self.surrogates[sname].named_parameters():
                if p.grad is not None:
                    self.log(
                        f"grad/{sname}.{pname}",
                        p.grad.norm(),
                        on_step=False,
                        on_epoch=True,
                    )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Adam; with `scaling_lr` set, the scale field gets its own group.

        The per-frame scale is the slowest-identified field, so a decoupled
        `scaling_lr` lets it equilibrate without raising the encoder LR.
        `scaling_lr=None` falls back to the standard (optionally decoder-split)
        optimizer. Any LambdaLR warmup scales all groups uniformly.
        """
        decoder_split = self.decoder_weight_decay is not None

        if self.scaling_lr is None and not decoder_split:
            return torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        scale_params: list[nn.Parameter] = []
        decoder_params: list[nn.Parameter] = []
        other_params: list[nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if self.scaling_lr is not None and name.startswith("scale_fn."):
                scale_params.append(param)
            elif decoder_split and name.endswith(
                "surrogates.qp.decoder.weight"
            ):
                decoder_params.append(param)
            else:
                other_params.append(param)

        groups: list[dict] = [
            {"params": other_params, "weight_decay": self.weight_decay}
        ]
        if decoder_params:
            groups.append(
                {
                    "params": decoder_params,
                    "weight_decay": self.decoder_weight_decay,
                }
            )
        if scale_params:
            groups.append(
                {
                    "params": scale_params,
                    "weight_decay": self.weight_decay,
                    "lr": self.scaling_lr,
                }
            )
        return torch.optim.Adam(groups, lr=self.lr)

    def _cosine_warmup_lambda(self, max_epochs: int):
        warmup = self.warmup_epochs
        lr_min_ratio = self.lr_min / max(self.lr, 1e-12)

        def lr_lambda(epoch: int) -> float:
            if warmup > 0 and epoch < warmup:
                return float(epoch + 1) / float(warmup)
            tail = max(max_epochs - warmup, 1)
            progress = min((epoch - warmup) / tail, 1.0)
            cos_term = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min_ratio + (1.0 - lr_min_ratio) * cos_term

        return lr_lambda

    def _step_linear_warmup_lambda(self):
        warmup = max(int(self.warmup_steps), 0)

        def lr_lambda(step: int) -> float:
            if warmup == 0 or step >= warmup:
                return 1.0
            return float(step + 1) / float(warmup)

        return lr_lambda

    def configure_optimizers(self) -> Any:
        optimizer = self._build_optimizer()
        if self.lr_schedule is None:
            return optimizer

        if self.lr_schedule == "cosine_warmup":
            max_epochs = self.trainer.max_epochs
            if max_epochs is None or max_epochs <= 0:
                raise RuntimeError(
                    "cosine_warmup requires trainer.max_epochs to be set; "
                    f"got {max_epochs}."
                )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=self._cosine_warmup_lambda(max_epochs)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        if self.lr_schedule == "step_linear_warmup":
            if self.warmup_steps <= 0:
                raise ValueError(
                    "step_linear_warmup requires warmup_steps > 0; "
                    f"got {self.warmup_steps}."
                )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=self._step_linear_warmup_lambda()
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        raise ValueError(
            f"Unknown lr_schedule {self.lr_schedule!r}. "
            "Supported: 'cosine_warmup', 'step_linear_warmup', or null."
        )
