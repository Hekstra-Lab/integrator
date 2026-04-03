"""Hierarchical Integrators with fixed per-group intensity priors.

Identical to IntegratorModelA / IntegratorModelB / IntegratorModelD except:
  1. _step passes group_labels from metadata to the loss
  2. forward_out includes group_label and tau_per_refl for SBC/prediction
"""

from typing import Any, Literal

import torch
from torch import Tensor

from integrator import configs
from integrator.model.integrators.base_integrator import BaseIntegrator, _log_loss
from integrator.model.integrators.integrator_utils import (
    IntegratorBaseOutputs,
    IntegratorModelArgs,
    _assemble_outputs,
)
from integrator.model.integrators.integrator import IntegratorModelD


def _add_group_outputs(out: dict, metadata: dict, loss) -> None:
    """Add group_label and tau_per_refl to forward outputs."""
    group_labels = metadata["group_label"].long()
    out["group_label"] = group_labels
    if hasattr(loss, "tau_per_group"):
        out["tau_per_refl"] = loss.tau_per_group[group_labels]


def _hierarchical_step(self, batch, step: Literal["train", "val"]):
    """Shared _step for hierarchical integrators — passes group_labels to loss."""
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


class HierarchicalIntegrator(BaseIntegrator):
    """ModelA variant with fixed per-group intensity priors.

    Metadata must contain ``group_label`` (integer tensor [B]).
    """

    REQUIRED_ENCODERS = {
        "profile": configs.ShoeboxEncoderArgs,
        "intensity": configs.IntensityEncoderArgs,
    }

    ARGS = IntegratorModelArgs

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

        x_profile = self.encoders["profile"](shoebox_reshaped)
        x_intensity = self.encoders["intensity"](shoebox_reshaped)

        qbg = self.surrogates["qbg"](x_intensity)
        qi = self.surrogates["qi"](x_intensity)
        qp = self.surrogates["qp"](x_profile)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qi.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

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
        _add_group_outputs(out, metadata, self.loss)

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
        }

    _step = _hierarchical_step


class HierarchicalIntegratorD(IntegratorModelD):
    """ModelD variant with fixed per-group priors.

    Like ModelD, uses five fully decoupled encoders (profile, k_i, r_i, k_bg, r_bg).
    Metadata must contain ``group_label`` (integer tensor [B]).
    """

    def _forward_impl(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        metadata: dict,
    ) -> dict[str, Any]:
        result = super()._forward_impl(counts, shoebox, mask, metadata)
        _add_group_outputs(result["forward_out"], metadata, self.loss)
        return result

    _step = _hierarchical_step


class HierarchicalIntegratorB(BaseIntegrator):
    """ModelB variant with fixed per-group intensity priors.

    Like ModelB, uses separate encoders for the two Gamma parameters (k, r).
    Metadata must contain ``group_label`` (integer tensor [B]).
    """

    REQUIRED_ENCODERS = {
        "profile": configs.ShoeboxEncoderArgs,
        "k": configs.IntensityEncoderArgs,
        "r": configs.IntensityEncoderArgs,
    }

    ARGS = IntegratorModelArgs

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

        x_profile = self.encoders["profile"](shoebox_reshaped)
        x_k = self.encoders["k"](shoebox_reshaped)
        x_r = self.encoders["r"](shoebox_reshaped)

        qbg = self.surrogates["qbg"](x_k, x_r)
        qi = self.surrogates["qi"](x_k, x_r)
        qp = self.surrogates["qp"](x_profile)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qi.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

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
        _add_group_outputs(out, metadata, self.loss)

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
        }

    _step = _hierarchical_step
