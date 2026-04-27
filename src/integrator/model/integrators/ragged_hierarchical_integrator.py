"""Ragged-input hierarchical integrator (Model C only).

Consumes the dict produced by `pad_collate_ragged`, not a 4-tuple.
Uses five fully decoupled encoders (profile, k_i, r_i, k_bg, r_bg).
Requires `metadata["group_label"]` and `metadata["d"]` for Wilson loss.
"""

from typing import Any

import torch
from torch import Tensor

from integrator import configs
from integrator.model.integrators.base_integrator import BaseIntegrator, _log_loss
from integrator.model.integrators.integrator import _sample_profile
from integrator.model.integrators.integrator_utils import (
    IntegratorBaseOutputs,
    IntegratorModelArgs,
    _assemble_outputs,
)


def _ragged_hierarchical_step(self, batch, step):
    """_step variant that unpacks the ragged batch dict."""
    outputs = self(batch)
    forward_out = outputs["forward_out"]

    metadata = outputs["metadata"]
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
            if k in ("kl_prf_mean", "kl_i_mean", "kl_bg_mean", "kl_hyper_mean")
        },
    )

    penalty, penalty_components = self._profile_basis_penalty()
    for name, value in penalty_components.items():
        self.log(f"{step} {name}", value, on_step=False, on_epoch=True)
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


class RaggedHierarchicalIntegrator(BaseIntegrator):
    """Ragged-input Model C — fully decoupled encoders for qi vs qbg Gamma.

    Mirrors HierarchicalIntegratorC: uses *five* encoders so each
    Gamma surrogate sees its own dedicated features:
      - profile : drives qp (LogisticNormal profile)
      - k_i, r_i : drive qi   (intensity Gamma)
      - k_bg, r_bg : drive qbg (background Gamma)
    """

    REQUIRED_ENCODERS = {
        "profile": configs.RaggedShoeboxEncoderArgs,
        "k_i":     configs.RaggedIntensityEncoderArgs,
        "r_i":     configs.RaggedIntensityEncoderArgs,
        "k_bg":    configs.RaggedIntensityEncoderArgs,
        "r_bg":    configs.RaggedIntensityEncoderArgs,
    }

    ARGS = IntegratorModelArgs

    def forward(self, batch: dict) -> dict[str, Any]:  # type: ignore[override]
        return self._forward_impl(batch)

    def _get_metadata(self, batch: dict) -> dict:
        """Extract metadata dict from ragged batch."""
        if "metadata" in batch:
            meta = dict(batch["metadata"])
        else:
            meta = {}
            for key in ("group_label", "d", "profile_group_label"):
                if key in batch:
                    meta[key] = batch[key]

        if "refl_ids" in batch and "refl_ids" not in meta:
            meta["refl_ids"] = batch["refl_ids"]

        missing = [k for k in ("group_label", "d") if k not in meta]
        if missing:
            raise KeyError(
                "RaggedHierarchicalIntegrator needs "
                f"{missing} in the batch. Either provide a `metadata` dict in "
                "the batch, attach these via your data module, or add them "
                "as top-level keys."
            )
        return meta

    def _forward_impl(self, batch: dict) -> dict[str, Any]:
        raw_3d = batch["counts"].float()
        enc_in_3d = batch.get("standardized_data", raw_3d).float()
        mask_3d = batch["mask"]
        shapes = batch["shapes"]

        B, Dmax, Hmax, Wmax = raw_3d.shape
        K = Dmax * Hmax * Wmax

        counts_flat = raw_3d.clamp(min=0).reshape(B, K)
        mask_flat = mask_3d.reshape(B, K)

        x = enc_in_3d.unsqueeze(1)  # (B, 1, Dmax, Hmax, Wmax)

        x_profile = self.encoders["profile"](x, mask_3d)
        x_k_i     = self.encoders["k_i"](x, mask_3d)
        x_r_i     = self.encoders["r_i"](x, mask_3d)
        x_k_bg    = self.encoders["k_bg"](x, mask_3d)
        x_r_bg    = self.encoders["r_bg"](x, mask_3d)

        qbg = self.surrogates["qbg"](x_k_bg, x_r_bg)
        qi = self.surrogates["qi"](x_k_i, x_r_i)

        qp = self.surrogates["qp"](
            x_profile,
            shapes=shapes,
            mask=mask_3d,
            mc_samples=self.mc_samples,
        )

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = _sample_profile(qp, self.mc_samples)
        zI = qi.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        rate = zI * zp + zbg

        metadata = self._get_metadata(batch)

        out = IntegratorBaseOutputs(
            rates=rate,
            counts=counts_flat,
            mask=mask_flat,
            qbg=qbg,
            qp=qp,
            qi=qi,
            zp=zp,
            zbg=zbg,
            metadata=metadata,
        )
        out = _assemble_outputs(out)

        group_labels = metadata["group_label"].long()
        out["group_label"] = group_labels
        if hasattr(self.loss, "tau_per_group"):
            out["tau_per_refl"] = self.loss.tau_per_group[group_labels]
        if hasattr(self.loss, "log_alpha_per_group"):
            import torch.nn.functional as F
            out["alpha_per_refl"] = F.softplus(
                self.loss.log_alpha_per_group[group_labels]
            )
        elif getattr(self.loss, "i_concentration_per_group", None) is not None:
            out["alpha_per_refl"] = self.loss.i_concentration_per_group[group_labels]

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
            "metadata": metadata,
        }

    _step = _ragged_hierarchical_step

    def predict_step(self, batch, _batch_idx):  # type: ignore[override]
        outputs = self(batch)
        forward_out = outputs["forward_out"]
        return {
            k: v
            for k, v in forward_out.items()
            if k in self.predict_keys
        }
