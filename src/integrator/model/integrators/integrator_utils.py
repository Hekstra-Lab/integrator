from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

from integrator.configs.integrator import IntegratorCfg

DEFAULT_PREDICT_KEYS = [
    # Reflection identifiers
    "refl_ids",
    "is_test",
    # Model predictions
    "qi_mean",
    "qi_var",
    "qi_params",
    "qbg_mean",
    "qbg_var",
    "qbg_params",
    # From DIALS refl table
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
    zbg: Tensor
    metadata: dict[str, torch.Tensor]
    concentration: Tensor | None = None
    compute_pred_var: bool = False


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
        "train elbo",
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


def predictive_intensity_variance(
    out: IntegratorBaseOutputs,
    n_iterations: int = 3,
    eps: float = 1e-6,
) -> Tensor:
    """Posterior predictive variance of the Kabsch profile-fitted intensity.

    For each MC sample *s*, run the iterative Kabsch weighted profile-fitting
    estimator (matching ``calculate_intensities``):

        vi^s       = I_hat^s * zp^s + zbg^s          (pixel variance estimate)
        I_hat^s    = sum(w_i * (c_i - zbg^s)) / sum(w_i * zp^s)
                     where  w_i = zp^s_i / vi^s_i

    The iteration starts with vi = zbg (background-only variance) and refines
    for *n_iterations* cycles. The variance of I_hat across the S MC samples
    marginalises over profile and background uncertainty, giving a more
    complete variance estimate than qi_var alone.

    Parameters
    ----------
    out : IntegratorBaseOutputs
        Forward-pass outputs containing counts, mask, zp, and zbg.
    n_iterations : int
        Number of Kabsch refinement iterations (default 3).
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    Tensor, shape [batch]
        Per-reflection predictive variance.
    """
    # counts: [B, P], mask: [B, P]
    # zp:  [B, S, P]   (profile samples)
    # zbg: [B, S, 1]   (background samples, broadcast over pixels)
    counts = out.counts
    mask = out.mask
    zp = out.zp
    zbg = out.zbg

    with torch.no_grad():
        m = mask.unsqueeze(1)  # [B, 1, P]
        c = (counts * mask).unsqueeze(1)  # [B, 1, P]
        zp_m = zp * m  # [B, S, P]

        # Initialise pixel variance with background-only estimate
        vi = zbg.clamp(min=eps)  # [B, S, 1] broadcast over pixels

        # Iterative Kabsch refinement
        I_hat = torch.zeros(
            counts.shape[0], zp.shape[1], device=counts.device
        )  # [B, S]

        for _ in range(n_iterations):
            w = zp_m / vi  # [B, S, P]
            num = (w * (c - zbg)).sum(dim=-1)  # [B, S]
            denom = (w * zp_m).sum(dim=-1)  # [B, S]

            I_hat = num / denom.clamp(min=eps)  # [B, S]

            # Update pixel variance: vi = I * p + bg, averaged over pixels
            vi = I_hat.unsqueeze(-1) * zp_m + zbg  # [B, S, P]
            vi = vi.mean(dim=-1, keepdim=True).clamp(min=eps)  # [B, S, 1]

        # Guard: if all mask pixels are zero, variance is zero
        n_valid = mask.sum(dim=-1)  # [B]
        pred_var = I_hat.var(dim=1)  # [B]
        pred_var = torch.where(n_valid > 0, pred_var, torch.zeros_like(pred_var))

        return pred_var


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

    if out.compute_pred_var:
        base["qi_pred_var"] = predictive_intensity_variance(out)

    if out.metadata is None:
        return base

    # Storing the surrogate distribution parameters
    distribution_params = {
        "qbg_params": {
            name: getattr(out.qbg, name) for name in out.qbg.arg_constraints
        },
        "qi_params": {
            name: getattr(out.qi, name) for name in out.qi.arg_constraints
        },
    }

    # Update base dictionary
    base.update(out.metadata)
    base.update(distribution_params)

    return base
