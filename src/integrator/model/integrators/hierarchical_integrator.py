"""Hierarchical Integrator: groups reflections and learns per-group intensity priors.

Uses the standard encoder architecture but adds:
  1. A GroupEncoder that pools local features by radial bin → q(log τ_k)
  2. Conditioning of the qi surrogate on the sampled log τ_k
"""

from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

from integrator import configs
from integrator.model.encoders.group_encoder import GroupEncoder
from integrator.model.integrators.base_integrator import BaseIntegrator, _log_loss
from integrator.model.integrators.integrator_utils import (
    IntegratorBaseOutputs,
    IntegratorModelArgs,
    _assemble_outputs,
)


class HierarchicalIntegrator(BaseIntegrator):
    """Integrator with per-group learned intensity priors.

    Encoder keys (must match YAML order):
        profile   - shoebox encoder -> qp
        intensity - intensity encoder -> x_intensity

    Surrogate keys:
        qp  - profile surrogate (unchanged)
        qi  - intensity surrogate, in_features = encoder_out + 1
        qbg - background surrogate (unchanged)

    Metadata must contain ``group_label`` (integer tensor [B]).
    The data loader aliases ``radial_bin`` -> ``group_label`` automatically.
    """

    REQUIRED_ENCODERS = {
        "profile": configs.ShoeboxEncoderArgs,
        "intensity": configs.IntensityEncoderArgs,
    }

    ARGS = IntegratorModelArgs

    def __init__(self, cfg, loss, encoders, surrogates):
        super().__init__(cfg=cfg, loss=loss, encoders=encoders, surrogates=surrogates)

        # Initialize head_mu bias to the prior mean so training starts near prior
        log_tau_init = getattr(loss, "log_tau_mu", -6.9)

        self.group_encoder = GroupEncoder(
            encoder_out=cfg.encoder_out,
            hidden_dim=cfg.group_hidden_dim,
            log_tau_init=log_tau_init,
        )

        # Embed log(τ) through a small MLP before conditioning qi,
        # so the surrogate head sees a learned, bounded representation
        # instead of a raw scalar that can be extreme.
        self.tau_embed_dim = 16
        self.tau_embed = nn.Sequential(
            nn.Linear(1, self.tau_embed_dim),
            nn.SiLU(),
            nn.Linear(self.tau_embed_dim, self.tau_embed_dim),
        )

    def _forward_impl(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        metadata: dict,
    ) -> dict[str, Any]:
        counts = torch.clamp(counts, min=0)

        b = shoebox.shape[0]
        shoebox_reshaped = shoebox.reshape(b, 1, *self.shoebox_shape)

        # Local encoder features
        x_profile = self.encoders["profile"](shoebox_reshaped)
        x_intensity = self.encoders["intensity"](shoebox_reshaped)

        # Group encoder: pool by radial bin → sample τ_k in log-space
        group_labels = metadata["group_label"].long()
        mu, logvar, tau_per_refl, log_tau_per_refl = self.group_encoder(
            x_intensity, group_labels
        )

        # Condition qi on τ_k: embed log(τ) through learned MLP so the
        # surrogate sees a bounded representation, not a raw extreme scalar.
        tau_features = self.tau_embed(log_tau_per_refl)  # (B, tau_embed_dim)
        x_intensity_cond = torch.cat([x_intensity, tau_features], dim=-1)

        # Surrogate modules
        qbg = self.surrogates["qbg"](x_intensity)
        qi = self.surrogates["qi"](x_intensity_cond)
        qp = self.surrogates["qp"](x_profile)

        # Monte Carlo samples
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qi.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        # Poisson rate
        rate = zI * zp + zbg

        out = IntegratorBaseOutputs(
            rates=rate,
            counts=counts,
            mask=mask,
            qbg=qbg,
            qp=qp,
            qi=qi,
            zp=zp,
            zbg=zbg,
            concentration=qp.concentration,
            metadata=metadata,
            compute_pred_var=self.cfg.compute_pred_var,
        )
        out = _assemble_outputs(out)

        # Store q(log τ_k) parameters and per-reflection τ for SBC / prediction
        _, inv = torch.unique(group_labels, return_inverse=True)
        out["tau_per_refl"] = tau_per_refl.squeeze(-1)  # [B]
        out["q_log_tau_mu"] = mu[inv]                    # [B]
        out["q_log_tau_logvar"] = logvar[inv]             # [B]
        out["group_label"] = group_labels                 # [B]

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
            "mu": mu,
            "logvar": logvar,
            "tau_per_refl": tau_per_refl,
            "group_labels": group_labels,
        }

    def _step(self, batch, step: Literal["train", "val"]):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]

        # Pass current epoch so the loss can compute KL warmup beta
        self.loss.current_epoch = self.current_epoch

        loss_dict = self.loss(
            rate=forward_out["rates"],
            counts=forward_out["counts"],
            qp=outputs["qp"],
            qi=outputs["qi"],
            qbg=outputs["qbg"],
            mask=forward_out["mask"],
            mu=outputs["mu"],
            logvar=outputs["logvar"],
            tau_per_refl=outputs["tau_per_refl"],
            group_labels=outputs["group_labels"],
        )

        total_loss = loss_dict["loss"]

        _log_loss(
            self,
            kl=loss_dict["kl_mean"],
            nll=loss_dict["neg_ll_mean"],
            total_loss=total_loss,
            step=step,
        )

        # Log hierarchical diagnostics
        for key in ("kl_global", "tau_mean", "tau_std", "beta_kl_warmup"):
            if key in loss_dict:
                self.log(
                    f"{step} {key}",
                    loss_dict[key],
                    on_step=False,
                    on_epoch=True,
                )

        # Log surrogate concentration ranges to diagnose NaN
        qi = outputs["qi"]
        qbg = outputs["qbg"]
        if hasattr(qi, "concentration"):
            self.log(f"{step} qi_k_max", qi.concentration.max().detach(),
                     on_step=False, on_epoch=True, reduce_fx="max")
        if hasattr(qbg, "concentration"):
            self.log(f"{step} qbg_k_max", qbg.concentration.max().detach(),
                     on_step=False, on_epoch=True, reduce_fx="max")

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
