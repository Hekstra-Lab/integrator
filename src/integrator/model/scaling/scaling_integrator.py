from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

from integrator import configs
from integrator.model.integrators.base_integrator import (
    BaseIntegrator,
    _log_loss,
)
from integrator.model.integrators.hierarchical_integrator import (
    _add_group_outputs,
    _get_normalized_position,
    _sample_profile,
)
from integrator.model.integrators.integrator_utils import (
    IntegratorBaseOutputs,
    _assemble_outputs,
)
from integrator.model.scaling.chebyshev_scale import ChebyshevScale
from integrator.model.scaling.hkl_table import HKLLookupTable


class ScalingIntegrator(BaseIntegrator):
    """Integrator with per-HKL structure factor lookup table.

    Replaces the per-observation intensity surrogate (qi) with a shared
    Gamma q(F^2_hkl) looked up from an embedding table.  All observations
    of the same Miller index share the same variational distribution for
    the structure factor.

    rate = s(frame) * lp * F^2_hkl * profile + background

    where s(frame) is a smooth Chebyshev scale capturing beam decay and
    other image-to-image variations, lp is the known Lorentz-polarization
    correction from metadata, and F^2 has a Wilson prior (no LP — set
    ``lp_correction: false`` in the loss config).

    Expects ``metadata["asu_id"]`` and ``metadata["lp"]``.

    Uses manual optimization: Adam for encoders/surrogates/loss/scale,
    SparseAdam for the HKL embedding table.
    """

    _MANUAL_OPTIMIZATION = True

    REQUIRED_ENCODERS = {
        "profile": configs.ProfileEncoderArgs,
        "k_bg": configs.IntensityEncoderArgs,
        "r_bg": configs.IntensityEncoderArgs,
    }

    def __init__(
        self,
        cfg: configs.IntegratorCfg,
        loss: nn.Module,
        encoders: dict[str, nn.Module],
        surrogates: dict[str, nn.Module],
    ):
        super().__init__(cfg, loss, encoders, surrogates)
        self.automatic_optimization = False

        if cfg.n_hkl is None:
            raise ValueError(
                "ScalingIntegrator requires n_hkl in config."
            )

        self.hkl_table = HKLLookupTable(
            n_hkl=cfg.n_hkl,
            init_mu=cfg.scaling_init_mu,
            init_fano=cfg.scaling_init_fano,
            eps=cfg.scaling_eps,
            k_min=cfg.scaling_k_min,
        )
        self.scale_fn = ChebyshevScale(
            degree=cfg.scale_degree,
            frame_min=cfg.scale_frame_min,
            frame_max=cfg.scale_frame_max,
        )
        self.scaling_lr = (
            cfg.scaling_lr if cfg.scaling_lr is not None else cfg.lr
        )
        self._clip_val = getattr(cfg, "gradient_clip_val", 1.0)
        self._clip_algo = getattr(cfg, "gradient_clip_algorithm", "norm")

    def _forward_impl(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        metadata: dict,
    ) -> dict[str, Any]:
        counts = torch.clamp(counts, min=0)

        b = shoebox.shape[0]
        shoebox_masked = shoebox * mask
        shoebox_reshaped = shoebox_masked.reshape(b, 1, *self.shoebox_shape)

        # Profile
        position = _get_normalized_position(metadata, shoebox.device)
        x_profile = self.encoders["profile"](
            shoebox_reshaped, position=position
        )

        # Background
        x_k_bg = self.encoders["k_bg"](shoebox_reshaped)
        x_r_bg = self.encoders["r_bg"](shoebox_reshaped)
        qbg = self.surrogates["qbg"](x_k_bg, x_r_bg)

        # Profile surrogate
        prf_labels = metadata.get(
            "profile_group_label", metadata.get("group_label")
        )
        prf_labels = prf_labels.long() if prf_labels is not None else None
        qp = self.surrogates["qp"](
            x_profile,
            mc_samples=self.mc_samples,
            group_labels=prf_labels,
            metadata=metadata,
        )

        # Structure factor from HKL table
        asu_ids = metadata["asu_id"].long().to(shoebox.device)
        qi, F_sq = self.hkl_table(asu_ids, self.mc_samples)
        # F_sq: (S, B) -> (B, S, 1)
        F_sq = F_sq.permute(1, 0).unsqueeze(-1)

        # MC samples for profile and background
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = _sample_profile(qp, self.mc_samples)

        # Per-observation scale: s(frame) * lp
        device = shoebox.device
        frame = metadata["xyzcal.px.2"].to(device).float()
        s = self.scale_fn(frame)
        lp = metadata["lp"].to(device).float().clamp(min=1e-8)
        scale = (s * lp).view(b, 1, 1)

        rate = scale * F_sq * zp + zbg

        if "is_coset" in metadata:
            coset = metadata["is_coset"].bool().view(-1, 1, 1)
            rate = torch.where(coset, zbg, rate)

        out = IntegratorBaseOutputs(
            rates=rate,
            counts=counts,
            mask=mask,
            qbg=qbg,
            qp=qp,
            qi=qi,
            zp=zp,
            zbg=zbg,
            metadata=metadata,
        )
        out = _assemble_outputs(out)
        out["asu_id"] = asu_ids
        _add_group_outputs(out, metadata, self.loss)

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
        }

    # -- Manual optimization: two optimizers per step ----------------

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
        main_opt, sparse_opt = self.optimizers()
        result = self._step(batch, step="train")
        loss = result["loss"]

        main_opt.zero_grad()
        sparse_opt.zero_grad()
        self.manual_backward(loss)
        if self._clip_val is not None and self._clip_val > 0:
            self.clip_gradients(
                main_opt,
                gradient_clip_val=self._clip_val,
                gradient_clip_algorithm=self._clip_algo,
            )
        main_opt.step()
        sparse_opt.step()

        # LR scheduler (if configured)
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            if isinstance(schedulers, list):
                for s in schedulers:
                    s.step()
            else:
                schedulers.step()

        return result

    def validation_step(self, batch, _batch_idx):
        return self._step(batch, step="val")

    def configure_optimizers(self):
        table_params = list(self.hkl_table.parameters())
        table_ids = {id(p) for p in table_params}
        other_params = [
            p
            for p in self.parameters()
            if p.requires_grad and id(p) not in table_ids
        ]

        main_opt = torch.optim.Adam(
            other_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        sparse_opt = torch.optim.SparseAdam(
            table_params,
            lr=self.scaling_lr,
        )

        if self.lr_schedule is None:
            return [main_opt, sparse_opt]

        if self.lr_schedule == "cosine_warmup":
            max_epochs = self.trainer.max_epochs
            if max_epochs is None or max_epochs <= 0:
                raise RuntimeError(
                    "cosine_warmup requires trainer.max_epochs."
                )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                main_opt,
                lr_lambda=self._cosine_warmup_lambda(max_epochs),
            )
            return [main_opt, sparse_opt], [
                {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
            ]

        if self.lr_schedule == "step_linear_warmup":
            if self.warmup_steps <= 0:
                raise ValueError(
                    "step_linear_warmup requires warmup_steps > 0."
                )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                main_opt,
                lr_lambda=self._step_linear_warmup_lambda(),
            )
            return [main_opt, sparse_opt], [
                {"scheduler": scheduler, "interval": "step", "frequency": 1}
            ]

        raise ValueError(f"Unknown lr_schedule {self.lr_schedule!r}.")
