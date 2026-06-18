import logging
import math
from abc import abstractmethod
from typing import Any, Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor

from integrator.configs import IntegratorCfg, OptimizerConfig

_logger = logging.getLogger(__name__)

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
    # live per-step loss so the bar moves within an epoch (train only)
    if step == "train":
        self.log(
            "loss", total_loss, on_step=True, on_epoch=False, prog_bar=True
        )
    self.log(
        f"{step} elbo",
        total_loss,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
    )
    self.log(f"{step} kl", kl, on_step=False, on_epoch=True, prog_bar=True)
    self.log(f"{step} nll", nll, on_step=False, on_epoch=True, prog_bar=True)
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
    """Variational integrator base class.

    Encodes shoeboxes, samples the ELBO surrogates, and drives the Lightning
    train/validation/predict loop and optimizer/scheduler setup. Subclasses
    implement `_forward_impl` to assemble the per-pixel rate.
    """

    REQUIRED_ENCODERS: dict[str, tuple[str, type]] = {}

    DEFAULT_SURROGATES: dict[str, dict] = {
        "qp": {
            "name": "learned_basis_profile",
            "args": {"latent_dim": 12, "init_std": 0.5, "prior_scale": 3.0},
        },
        "qbg": {
            "name": "gamma",
            "args": {
                "reparameterization": "mean_fano",
                "eps": 1.0e-6,
                "k_min": 0.01,
            },
        },
        "qi": {
            "name": "gamma",
            "args": {
                "reparameterization": "mean_fano",
                "eps": 1.0e-6,
                "k_min": 0.01,
            },
        },
    }

    ARGS: type

    def __init__(
        self,
        cfg: IntegratorCfg,
        loss: nn.Module,
        encoders: dict[str, nn.Module],
        surrogates: dict[str, nn.Module],
        optimizer: OptimizerConfig | None = None,
    ):
        """Build the integrator from its config and submodules.

        Args:
            cfg: Architecture/inference config (shoebox shape, encoder width, MC samples, predict keys).
            loss: ELBO module returning the NLL, KL, and per-component KL terms.
            encoders: Named encoder modules mapping shoeboxes to embeddings.
            surrogates: Named variational surrogates (`qi`, `qbg`, `qp`) mapping embeddings to posterior distributions.
            optimizer: Optimizer/schedule settings; defaults are used when omitted.
        """
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
        """Run the integrator forward pass, returning the surrogates and assembled outputs."""
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

        group_labels = metadata["group_label"].long()

        loss_dict = self.loss(
            rate=forward_out["rates"],
            counts=forward_out["counts"],
            qp=outputs["qp"],
            qi=outputs["qi"],
            qbg=outputs["qbg"],
            mask=forward_out["mask"],
            group_labels=group_labels,
            metadata=metadata,
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
                if k in ("kl_prf_mean", "kl_i_mean", "kl_bg_mean")
            },
        )

        penalty, penalty_components = self._profile_basis_penalty()
        for name, value in penalty_components.items():
            self.log(
                f"{step} {name}",
                value,
                on_step=False,
                on_epoch=True,
            )
        total_loss = total_loss + penalty

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
        """Run one training step and log the ELBO and its components."""
        return self._step(batch, step="train")

    def validation_step(self, batch, _batch_idx):
        """Run one validation step and log the ELBO and its components."""
        return self._step(batch, step="val")

    def predict_step(self, batch, _batch_idx):
        """Run inference and return the `predict_keys` subset of the forward outputs."""
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]
        self._warn_unknown_predict_keys(forward_out)
        return {k: v for k, v in forward_out.items() if k in self.predict_keys}

    def _warn_unknown_predict_keys(self, forward_out: dict) -> None:
        """Warn once if `predict_keys` requests columns the model never produces."""
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
        """Regularization penalties on the learned qp decoder weight W.

        - Smoothness: mean squared spatial gradient across (D, H, W) per
          column, penalizing high-frequency structure in W.
        Returns (total_penalty, components_dict). Zero and empty when
        disabled or when qp has no decoder.weight (e.g. fixed basis).
        """
        zero = torch.zeros((), device=self.device)
        qp = self.surrogates["qp"] if "qp" in self.surrogates else None
        decoder = getattr(qp, "decoder", None)
        W = getattr(decoder, "weight", None)
        # The smoothness penalty weight is owned by the profile surrogate.
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

        return total, components

    def on_after_backward(self) -> None:
        """Log gradient norms for intensity/background surrogate heads."""
        for sname in ("qi", "qbg"):
            if sname not in self.surrogates:
                continue
            surr = self.surrogates[sname]
            for pname, p in surr.named_parameters():
                if p.grad is not None:
                    self.log(
                        f"grad/{sname}.{pname}",
                        p.grad.norm(),
                        on_step=False,
                        on_epoch=True,
                    )

    def on_validation_epoch_end(self) -> None:
        """Log val - train generalization gaps for each ELBO component."""
        metrics = self.trainer.callback_metrics
        for key in list(metrics.keys()):
            if not key.startswith("val "):
                continue
            suffix = key[len("val ") :]
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

        decoder_params: list[nn.Parameter] = []
        other_params: list[nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # any surrogate decoder weight (e.g. surrogates.qp.decoder.weight)
            if name.startswith("surrogates.") and name.endswith(
                ".decoder.weight"
            ):
                decoder_params.append(param)
            else:
                other_params.append(param)
        if not decoder_params:
            raise RuntimeError(
                "decoder_weight_decay is set but no surrogate "
                "'.decoder.weight' parameter was found; "
                "set decoder_weight_decay=null for surrogates without a decoder."
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
                # Linear ramp 0 -> 1 over the first `warmup` epochs. Start at
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

    def _step_linear_warmup_lambda(self):
        """Linear ramp 0 -> 1 over the first `warmup_steps` optimizer steps."""
        warmup = max(int(self.warmup_steps), 0)

        def lr_lambda(step: int) -> float:
            if warmup == 0:
                return 1.0
            if step >= warmup:
                return 1.0
            # Start at 1/warmup on step 0 (not 0) so the optimizer takes a
            # nonzero step immediately.
            return float(step + 1) / float(warmup)

        return lr_lambda

    def configure_optimizers(self) -> Any:
        """Build the Adam optimizer and the optional warmup/cosine or step-warmup scheduler."""
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

        if self.lr_schedule == "step_linear_warmup":
            if self.warmup_steps <= 0:
                raise ValueError(
                    "step_linear_warmup requires warmup_steps > 0; "
                    f"got {self.warmup_steps}."
                )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=self._step_linear_warmup_lambda(),
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
