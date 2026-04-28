from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from integrator.model.integrators.base_integrator import _log_loss
from integrator.model.integrators.integrator import (
    Integrator,
    _sample_profile,
)
from integrator.model.integrators.integrator_utils import (
    IntegratorBaseOutputs,
    _assemble_outputs,
)


def _add_group_outputs(out: dict, metadata: dict, loss) -> None:
    """Add group_label, tau_per_refl, and alpha_per_refl to forward outputs."""
    group_labels = metadata["group_label"].long()
    out["group_label"] = group_labels
    if hasattr(loss, "tau_per_group"):
        out["tau_per_refl"] = loss.tau_per_group[group_labels]
    if hasattr(loss, "log_alpha_per_group"):
        out["alpha_per_refl"] = F.softplus(
            loss.log_alpha_per_group[group_labels]
        )
    elif getattr(loss, "i_concentration_per_group", None) is not None:
        out["alpha_per_refl"] = loss.i_concentration_per_group[group_labels]


def _hierarchical_step(self, batch, step: Literal["train", "val"]):
    """Shared _step for hierarchical integrators; passes group_labels to loss."""
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

    # get total steps
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

    # Auxiliary regularizers on the learned profile decoder
    penalty, penalty_components = self._profile_basis_penalty()
    for name, value in penalty_components.items():
        self.log(
            f"{step} {name}",
            value,
            on_step=False,
            on_epoch=True,
        )
    total_loss = total_loss + penalty

    loss_components = {
        "loss": total_loss.detach(),
        "nll": loss_dict["neg_ll_mean"].detach(),
        "kl": loss_dict["kl_mean"].detach(),
        "kl_prf": loss_dict["kl_prf_mean"].detach(),
        "kl_i": loss_dict["kl_i_mean"].detach(),
        "kl_bg": loss_dict["kl_bg_mean"].detach(),
    }
    if "kl_hyper" in loss_dict:
        loss_components["kl_hyper"] = loss_dict["kl_hyper"].detach()

    return {
        "loss": total_loss,
        "forward_out": forward_out,
        "loss_components": loss_components,
    }


class HierarchicalIntegrator(Integrator):
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
        x_k_i = self.encoders["k_i"](shoebox_reshaped)
        x_r_i = self.encoders["r_i"](shoebox_reshaped)
        x_k_bg = self.encoders["k_bg"](shoebox_reshaped)
        x_r_bg = self.encoders["r_bg"](shoebox_reshaped)

        qbg = self.surrogates["qbg"](x_k_bg, x_r_bg)
        qi = self.surrogates["qi"](x_k_i, x_r_i)

        prf_labels = metadata.get(
            "profile_group_label", metadata.get("group_label")
        )
        prf_labels = prf_labels.long() if prf_labels is not None else None
        qp = self.surrogates["qp"](
            x_profile, mc_samples=self.mc_samples, group_labels=prf_labels
        )

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = _sample_profile(qp, self.mc_samples)
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
            metadata=metadata,
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


class HierarchicalIntegrator3Enc(Integrator):
    """Hierarchical integrator with 3 encoders: profile, intensity_i, intensity_bg.

    Each intensity surrogate receives the same encoder output for both args,
    so mu and fano heads learn independent projections from shared features.
    """

    from integrator import configs

    REQUIRED_ENCODERS = {
        "profile": configs.ShoeboxEncoderArgs,
        "intensity_i": configs.IntensityEncoderArgs,
        "intensity_bg": configs.IntensityEncoderArgs,
    }

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
        x_i = self.encoders["intensity_i"](shoebox_reshaped)
        x_bg = self.encoders["intensity_bg"](shoebox_reshaped)

        qbg = self.surrogates["qbg"](x_bg, x_bg)
        qi = self.surrogates["qi"](x_i, x_i)

        prf_labels = metadata.get(
            "profile_group_label", metadata.get("group_label")
        )
        prf_labels = prf_labels.long() if prf_labels is not None else None
        qp = self.surrogates["qp"](
            x_profile, mc_samples=self.mc_samples, group_labels=prf_labels
        )

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = _sample_profile(qp, self.mc_samples)
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
            metadata=metadata,
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
