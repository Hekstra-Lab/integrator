import math
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
        self.qp_smoothness_weight = cfg.qp_smoothness_weight
        self.qp_orthogonality_weight = cfg.qp_orthogonality_weight
        self.qp_sparsity_weight = cfg.qp_sparsity_weight
        self.lr_schedule = cfg.lr_schedule
        self.warmup_epochs = cfg.warmup_epochs
        self.lr_min = cfg.lr_min
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

        # Auxiliary regularizers on the learned profile decoder (no-op for
        # fixed bases). ELBO logging above stays pure; penalty is added to
        # the backpropagated loss only.
        penalty, penalty_components = self._profile_basis_penalty()
        for name, value in penalty_components.items():
            self.log(
                f"{step} {name}",
                value,
                on_step=False,
                on_epoch=True,
            )
        total_loss = total_loss + penalty

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

    def _profile_basis_penalty(self) -> tuple[Tensor, dict[str, Tensor]]:
        """Regularization penalties on the learned qp decoder weight W.

        - Smoothness: mean squared spatial gradient across (D, H, W) per
          column, penalizing high-frequency structure in W.
        - Orthogonality: mean squared off-diagonal entry of the column-
          normalized Gram matrix, penalizing redundant / collinear columns.
        - Sparsity: mean |W|, penalizing non-zero entries. Encourages
          localization (most pixels near zero, only peak-region pixels
          carry weight). Complements smoothness (smoothness asks adjacent
          pixels to be similar; sparsity asks most pixels to be zero).

        Returns (total_penalty, components_dict). Zero and empty when
        disabled or when qp has no decoder.weight (e.g. fixed basis).
        """
        zero = torch.zeros((), device=self.device)
        qp = self.surrogates["qp"] if "qp" in self.surrogates else None
        decoder = getattr(qp, "decoder", None)
        W = getattr(decoder, "weight", None)
        smooth_w = self.qp_smoothness_weight
        ortho_w = self.qp_orthogonality_weight
        sparse_w = self.qp_sparsity_weight
        if W is None or W.dim() != 2:
            return zero, {}
        if (
            (smooth_w is None or smooth_w == 0)
            and (ortho_w is None or ortho_w == 0)
            and (sparse_w is None or sparse_w == 0)
        ):
            return zero, {}

        shape = self.shoebox_shape
        D, H, W_spatial = shape if len(shape) == 3 else (1, *shape)
        K, d = W.shape
        if K != D * H * W_spatial:
            return zero, {}

        total = zero
        components: dict[str, Tensor] = {}

        if smooth_w is not None and smooth_w > 0:
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

        if ortho_w is not None and ortho_w > 0:
            W_n = W / W.norm(dim=0, keepdim=True).clamp(min=1e-8)
            gram = W_n.T @ W_n
            # diag of gram is 1 (unit cols); penalty is purely off-diagonal
            off_sq = (gram.pow(2)).sum() - float(d)
            denom = max(d * (d - 1), 1)
            ortho = off_sq / denom
            components["profile_orthogonality"] = ortho.detach()
            total = total + ortho_w * ortho

        if sparse_w is not None and sparse_w > 0:
            sparsity = W.abs().mean()
            components["profile_sparsity"] = sparsity.detach()
            total = total + sparse_w * sparsity

        return total, components

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

    def _build_optimizer(self) -> torch.optim.Optimizer:
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

    def _cosine_warmup_lambda(self, max_epochs: int):
        """Linear warmup for `warmup_epochs`, then cosine decay to lr_min.

        Returns a multiplier in [lr_min/lr, 1.0] to be applied by LambdaLR.
        """
        warmup = self.warmup_epochs
        lr_min_ratio = self.lr_min / max(self.lr, 1e-12)

        def lr_lambda(epoch: int) -> float:
            if warmup > 0 and epoch < warmup:
                # Linear ramp 0 → 1 over the first `warmup` epochs. Start at
                # 1/warmup on epoch 0 (not 0) so the optimizer takes a real
                # step immediately; stepping at lr=0 is a no-op.
                return float(epoch + 1) / float(warmup)
            # Cosine decay from epoch == warmup (value 1) down to lr_min_ratio
            # at epoch == max_epochs.
            tail = max(max_epochs - warmup, 1)
            progress = min((epoch - warmup) / tail, 1.0)
            cos_term = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min_ratio + (1.0 - lr_min_ratio) * cos_term

        return lr_lambda

    def configure_optimizers(self) -> Any:
        optimizer = self._build_optimizer()
        if self.lr_schedule is None:
            return optimizer
        if self.lr_schedule != "cosine_warmup":
            raise ValueError(
                f"Unknown lr_schedule {self.lr_schedule!r}. "
                "Supported: 'cosine_warmup' or null."
            )

        max_epochs = self.trainer.max_epochs
        if max_epochs is None or max_epochs <= 0:
            raise RuntimeError(
                "cosine_warmup requires trainer.max_epochs to be set; "
                f"got {max_epochs}."
            )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=self._cosine_warmup_lambda(max_epochs),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
