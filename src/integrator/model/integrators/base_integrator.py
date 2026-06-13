import logging
import math
from abc import abstractmethod
from typing import Any, Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor

from integrator.configs import IntegratorCfg
from integrator.model.integrators.integrator_utils import ScatterLoggerMixin

logger = logging.getLogger(__name__)

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


class BaseIntegrator(ScatterLoggerMixin, pl.LightningModule):
    REQUIRED_ENCODERS: dict[str, type] = {}
    ARGS: type

    def __init__(
        self,
        cfg: IntegratorCfg,
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
        self.lr_schedule = cfg.lr_schedule
        self.warmup_epochs = cfg.warmup_epochs
        self.warmup_steps = cfg.warmup_steps
        self.lr_min = cfg.lr_min
        self.mc_samples = cfg.mc_samples
        self.coset_mode = cfg.coset_mode
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

        # End-of-epoch model-vs-DIALS scatters (intensity + background), available
        # to every integrator and gated by the log_*_scatter cfg flags.
        self._init_scatter_logger(cfg)

        # Optional: transplant (and freeze) the background modules from another
        # run's checkpoint. The background is the field the merging models keep
        # collapsing (it loses the race to the intensity); freezing a proven bg
        # from a good integration run removes it from the optimization so the
        # merge can only fit the peak.
        self._frozen_eval_modules: list[nn.Module] = []
        bg_ckpt = getattr(cfg, "bg_init_from_checkpoint", None)
        if bg_ckpt:
            self._init_bg_from_checkpoint(
                bg_ckpt, freeze=getattr(cfg, "bg_freeze", True)
            )

    def _init_bg_from_checkpoint(
        self, ckpt_path: str, freeze: bool = True
    ) -> None:
        """Load (and optionally freeze) the background modules from a checkpoint.

        Copies the `k_bg`/`r_bg` background encoders and the `qbg` surrogate from
        another run's checkpoint (their architectures must match this model's).
        When `freeze`, their parameters are frozen and the modules are kept in
        eval mode (no dropout) even while the rest of the model trains -- see the
        `train` override. Raises if a targeted module is absent from the
        checkpoint (a silent partial load would be worse than failing).
        """
        from pathlib import Path

        if not Path(ckpt_path).exists():
            # At prediction the bg comes from the loaded checkpoint anyway, so a
            # missing source is non-fatal -- warn and skip.
            logger.warning(
                "bg_init_from_checkpoint: %s not found; skipping bg transplant.",
                ckpt_path,
            )
            return
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        targets = [
            ("encoders.k_bg", self.encoders["k_bg"] if "k_bg" in self.encoders else None),
            ("encoders.r_bg", self.encoders["r_bg"] if "r_bg" in self.encoders else None),
            ("surrogates.qbg", self.surrogates["qbg"] if "qbg" in self.surrogates else None),
        ]
        loaded = []
        for name, module in targets:
            if module is None:
                continue
            prefix = name + "."
            sub = {
                k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)
            }
            if not sub:
                raise KeyError(
                    f"bg_init_from_checkpoint: no '{name}.*' params in "
                    f"{ckpt_path} -- architecture/name mismatch."
                )
            module.load_state_dict(sub, strict=True)
            loaded.append(name)
            if freeze:
                for p in module.parameters():
                    p.requires_grad_(False)
                module.eval()
                self._frozen_eval_modules.append(module)
        logger.info(
            "bg_init_from_checkpoint: loaded %s from %s (freeze=%s)",
            loaded,
            ckpt_path,
            freeze,
        )

    def train(self, mode: bool = True):
        """Keep frozen background modules in eval mode (no dropout) when the rest
        of the model is switched to train."""
        super().train(mode)
        for module in getattr(self, "_frozen_eval_modules", []):
            module.eval()
        return self

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
            coset_mode=self.coset_mode,
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

        if "coset_aux_mean" in loss_dict:
            self.log(
                f"{step} coset_aux",
                loss_dict["coset_aux_mean"],
                on_step=False,
                on_epoch=True,
            )

        # Track predicted intensity on coset (background-only) reflections.
        # I|coset should sit near the background floor (especially in supervised
        # mode); the gap to I|lattice is a direct false-positive / background-
        # calibration readout. Logged in either coset_mode for comparison.
        if "is_coset" in metadata:
            with torch.no_grad():
                qi_mean = outputs["qi"].mean.detach()
                coset = metadata["is_coset"].bool().to(qi_mean.device)
                if coset.any():
                    self.log(
                        f"{step} I_coset",
                        qi_mean[coset].mean(),
                        on_step=False,
                        on_epoch=True,
                    )
                if (~coset).any():
                    self.log(
                        f"{step} I_lattice",
                        qi_mean[~coset].mean(),
                        on_step=False,
                        on_epoch=True,
                    )

        # penalties on profile basis
        penalty, penalty_components = self._profile_basis_penalty()
        for name, value in penalty_components.items():
            self.log(
                f"{step} {name}",
                value,
                on_step=False,
                on_epoch=True,
            )
        total_loss = total_loss + penalty

        if step == "train":
            self._collect_scatters(outputs, metadata, mask)

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
        Returns (total_penalty, components_dict). Zero and empty when
        disabled or when qp has no decoder.weight (e.g. fixed basis).
        """
        zero = torch.zeros((), device=self.device)
        qp = self.surrogates["qp"] if "qp" in self.surrogates else None
        decoder = getattr(qp, "decoder", None)
        W = getattr(decoder, "weight", None)
        smooth_w = self.qp_smoothness_weight
        ortho_w = self.qp_orthogonality_weight
        if W is None or W.dim() != 2:
            return zero, {}
        if (smooth_w is None or smooth_w == 0) and (
            ortho_w is None or ortho_w == 0
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
            W_n = W / (W.norm(dim=0, keepdim=True) + 1e-8)
            gram = W_n.T @ W_n
            off_sq = (gram.pow(2)).sum() - float(d)
            denom = max(d * (d - 1), 1)
            ortho = off_sq / denom
            components["profile_orthogonality"] = ortho.detach()
            total = total + ortho_w * ortho

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
