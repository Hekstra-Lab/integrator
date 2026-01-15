from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

from integrator.configs.integrator import IntegratorCfg

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
    "H",
    "K",
    "L",
]


@dataclass
class IntegratorBaseOutputs:
    rates: Tensor
    counts: Tensor
    mask: Tensor
    qbg: Any
    qp: Any
    qi: Any
    zp: Tensor
    metadata: dict[str, torch.Tensor]
    concentration: Tensor | None = None


@dataclass
class IntegratorModelArgs:
    cfg: IntegratorCfg
    loss: nn.Module
    surrogates: dict[str, nn.Module]
    encoders: dict[str, nn.Module]


def _log_forward_out(
    self,
    forward_out: dict,
    step: Literal["train", "val"],
):
    on_ = {
        "on_epoch": True,
        "on_step": False,
    }
    self.log(f"{step}: mean(qi.mean)", forward_out["qi_mean"].mean(), **on_)
    self.log(f"{step}: min(qi.mean)", forward_out["qi_mean"].min(), **on_)
    self.log(f"{step}: max(qi.mean)", forward_out["qi_mean"].max(), **on_)
    self.log(f"{step}: max(qi.variance)", forward_out["qi_var"].max(), **on_)
    self.log(f"{step}: min(qi.variance)", forward_out["qi_var"].min(), **on_)
    self.log(f"{step}: mean(qi.variance)", forward_out["qi_var"].mean(), **on_)
    self.log(f"{step}: mean(qbg.mean)", forward_out["qbg_mean"].mean(), **on_)
    self.log(f"{step}: min(qbg.mean)", forward_out["qbg_mean"].min(), **on_)
    self.log(f"{step}: max(qbg.mean)", forward_out["qbg_mean"].max(), **on_)
    self.log(
        f"{step}: mean(qbg.variance)", forward_out["qbg_var"].mean(), **on_
    )
    self.log(f"{step}: max(qbg.variance)", forward_out["qbg_var"].max(), **on_)
    self.log(f"{step}: min(qbg.variance)", forward_out["qbg_var"].min(), **on_)


def _log_loss(
    self,
    kl,
    nll,
    total_loss,
    step: Literal["train", "val"],
):
    self.log(
        "train/loss",
        total_loss,
        on_step=False,
        on_epoch=False,
        prog_bar=True,
    )
    self.log(f"{step} kl", kl, on_step=False, on_epoch=True)
    self.log(f"{step} nll", nll, on_step=False, on_epoch=True)


def calculate_intensities(counts, qbg, qp, mask, cfg):
    with torch.no_grad():
        counts = counts * mask  # [B,P]
        zbg = qbg.rsample([cfg.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.mean.unsqueeze(1)

        vi = zbg + 1e-6
        intensity = torch.tensor([0.0])

        # kabsch sum
        for _ in range(cfg.max_iterations):
            num = (counts.unsqueeze(1) - zbg) * zp * mask.unsqueeze(1) / vi
            denom = zp.pow(2) / vi
            intensity = num.sum(-1) / denom.sum(-1)  # [batch_size, mc_samples]
            vi = (intensity.unsqueeze(-1) * zp) + zbg
            vi = vi.mean(-1, keepdim=True)
        kabsch_sum_mean = intensity.mean(-1)
        kabsch_sum_var = intensity.var(-1)

        # profile masking
        zp = zp * mask.unsqueeze(1)  # profiles
        thresholds = torch.quantile(
            zp,
            0.99,
            dim=-1,
            keepdim=True,
        )  # threshold values
        profile_mask = zp > thresholds

        masked_counts = counts.unsqueeze(1) * profile_mask

        profile_masking_I = (masked_counts - zbg * profile_mask).sum(-1)

        profile_masking_mean = profile_masking_I.mean(-1)

        profile_masking_var = profile_masking_I.var(-1)

        intensities = {
            "profile_masking_mean": profile_masking_mean,
            "profile_masking_var": profile_masking_var,
            "kabsch_sum_mean": kabsch_sum_mean,
            "kabsch_sum_var": kabsch_sum_var,
        }

        return intensities


def _assemble_outputs(
    out: IntegratorBaseOutputs,
) -> dict[str, Any]:
    base = {
        "rates": out.rates,
        "counts": out.counts,
        "mask": out.mask,
        "zp": out.zp,
        "qbg_mean": out.qbg.mean,
        "qbg_var": out.qbg.variance,
        "qp_mean": out.qp.mean,
        "qi_mean": out.qi.mean,
        "qi_var": out.qi.variance,
        "profile": out.qp.mean,
        "concentration": out.concentration,
    }

    if out.metadata is None:
        return base

    base.update(out.metadata)

    return base
