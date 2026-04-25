"""Ragged-input analogue of HierarchicalIntegratorB.

Differences from `HierarchicalIntegratorB`:
  1. Input is the dict produced by `pad_collate_ragged`, not a 4-tuple.
     The dict carries:
        data:     (B, Dmax, Hmax, Wmax)   float — padded raw pixel counts
        mask:     (B, Dmax, Hmax, Wmax)   bool  — real-and-valid voxels
        shapes:   (B, 3) int              — per-reflection (D, H, W)
        bboxes:   (B, 6) int              — DIALS detector bbox per refl
        refl_ids: (B,)   int              — global IDs, for joining metadata
  2. Encoders are the ragged variants in
     `integrator.model.encoders.ragged_encoders` — they take `(x, mask)`
     and do masked global pool internally. The padded 3D tensor stays
     3D all the way through encoders.
  3. `qp` is `RaggedLogisticNormalSurrogate` which takes `(x_profile, shapes, mask)`
     and returns `ProfileSurrogateOutput` with **flat** voxel shapes
     (K = Dmax*Hmax*Wmax). This matches what WilsonLoss expects.
  4. Before calling the loss, data and mask are flattened to (B, K) so
     `WilsonLoss.forward` works unmodified on the padded-flat tensors.
     Padded voxels contribute 0 to the NLL because the mask is 0 there.

Metadata expected in the batch (via `batch["metadata"]` dict or per-dataset
`self.metadata_fn` hook):
    group_label: (B,) int — group index per reflection
    d:           (B,)     — resolution (A) per reflection, for Wilson s^2
If these aren't populated yet (dataloader work), the integrator raises a
clear error with a hint for how to add them.
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
    """_step variant that unpacks the ragged batch dict.

    Mirrors `_hierarchical_step` in `hierarchical_integrator.py` but sources
    `counts`, `shoebox_3d`, `mask_3d`, and `metadata` from the pad_collate_ragged
    dict instead of a 4-tuple.
    """
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

    # Profile-decoder regularizers (TV, entropy, orthogonality etc.).
    # The ragged profile surrogate doesn't expose the same W matrix as
    # `FixedBasisProfileSurrogate`, so the parent's penalty may noop; we
    # call it defensively so behavior matches when/if it's implemented.
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


class RaggedHierarchicalIntegratorB(BaseIntegrator):
    """Ragged-input Model B (two decoupled encoders for k and r of qi/qbg Gamma).

    Like HierarchicalIntegratorB, but consumes a ragged batch dict. Requires
    `metadata["group_label"]` and `metadata["d"]` (populated upstream by the
    data module or by a registered `metadata_fn` hook) for the Wilson loss.
    """

    REQUIRED_ENCODERS = {
        "profile": configs.RaggedShoeboxEncoderArgs,
        "k": configs.RaggedIntensityEncoderArgs,
        "r": configs.RaggedIntensityEncoderArgs,
    }

    ARGS = IntegratorModelArgs

    # NOTE: we intentionally override `forward` with a different signature
    # (a single batch dict) than BaseIntegrator's 4-tuple. This is required
    # for the ragged collate output; the type-checker will flag it.
    def forward(self, batch: dict) -> dict[str, Any]:  # type: ignore[override]
        return self._forward_impl(batch)

    def _get_metadata(self, batch: dict) -> dict:
        """Extract metadata dict. The ragged collate puts the per-reflection
        scalars under `batch["metadata"]` and the refl_ids at the top level;
        downstream code expects refl_ids inside metadata, so merge them here."""
        # Pull the inner metadata dict if present
        if "metadata" in batch:
            meta = dict(batch["metadata"])
        else:
            meta = {}
            for key in ("group_label", "d", "profile_group_label"):
                if key in batch:
                    meta[key] = batch[key]

        # refl_ids live at the top level of the batch — promote into metadata
        # so `_assemble_outputs` exposes them in the predict_step output.
        if "refl_ids" in batch and "refl_ids" not in meta:
            meta["refl_ids"] = batch["refl_ids"]

        missing = [k for k in ("group_label", "d") if k not in meta]
        if missing:
            raise KeyError(
                "RaggedHierarchicalIntegratorB needs "
                f"{missing} in the batch. Either provide a `metadata` dict in "
                "the batch, attach these via your data module, or add them "
                "as top-level keys. For WilsonLoss, `d` is per-reflection "
                "resolution in Angstroms (B,) and `group_label` is the "
                "per-reflection group index (B,)."
            )
        return meta

    def _forward_impl(self, batch: dict) -> dict[str, Any]:
        """Run the ragged forward.

        batch fields used:
          counts:            (B, Dmax, Hmax, Wmax) float — raw pixel data for Poisson NLL
          standardized_data: (B, Dmax, Hmax, Wmax) float — anscombe+z-scored for encoders
          mask:              (B, Dmax, Hmax, Wmax) bool  — DIALS mask ∧ real-region
          shapes:            (B, 3) int                   — per-reflection (D, H, W)
          refl_ids:          (B,) int                     — passed through
          metadata:          dict                         — must contain d, group_label
        """
        raw_3d = batch["counts"].float()
        # Fallback: if the batch predates the standardization update, reuse raw
        # counts as the encoder input so the pipeline still runs (encoder will
        # see unnormalized values — acceptable for smoke tests, not training).
        enc_in_3d = batch.get("standardized_data", raw_3d).float()
        mask_3d = batch["mask"]
        shapes = batch["shapes"]

        B, Dmax, Hmax, Wmax = raw_3d.shape
        K = Dmax * Hmax * Wmax

        # Clamp raw counts to non-negative (Poisson target can be 0 at padded
        # voxels; the mask zeros their contribution downstream).
        counts_flat = raw_3d.clamp(min=0).reshape(B, K)
        mask_flat = mask_3d.reshape(B, K)

        # Add a channel dim for convs. Encoders see the STANDARDIZED input.
        x = enc_in_3d.unsqueeze(1)  # (B, 1, Dmax, Hmax, Wmax)

        # Three encoders; each returns (B, encoder_out)
        x_profile = self.encoders["profile"](x, mask_3d)
        x_k = self.encoders["k"](x, mask_3d)
        x_r = self.encoders["r"](x, mask_3d)

        # qbg and qi produce per-reflection scalar Gamma params — no ragged
        # handling needed here, the existing gammaB surrogate works as-is.
        qbg = self.surrogates["qbg"](x_k, x_r)
        qi = self.surrogates["qi"](x_k, x_r)

        # qp is the ragged variant. Returns ProfileSurrogateOutput with
        # zp: (mc, B, K), mean_profile: (B, K), mu_h: (B, d), std_h: (B, d).
        qp = self.surrogates["qp"](
            x_profile,
            shapes=shapes,
            mask=mask_3d,
            mc_samples=self.mc_samples,
        )

        # Samples — same shapes as fixed-size path, just K may vary per batch.
        # Layout matches HierarchicalIntegratorB._forward_impl exactly.
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)  # (B, mc, 1)
        zp = _sample_profile(qp, self.mc_samples)                           # (B, mc, K)
        zI = qi.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)   # (B, mc, 1)

        rate = zI * zp + zbg                                                 # (B, mc, K)

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

        # Group-conditioned extras (tau_per_refl, alpha_per_refl) — mirrors
        # the fixed-size hierarchical integrators' behavior.
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

    # Override the _step that the fixed-size integrators use — our forward
    # takes a dict, not a 4-tuple.
    _step = _ragged_hierarchical_step

    def predict_step(self, batch, _batch_idx):  # type: ignore[override]
        """Run the forward and return a subset of outputs keyed by predict_keys.

        BaseIntegrator's predict_step assumes the batch is a 4-tuple; we
        override it to consume the dict from `pad_collate_ragged` instead.
        """
        outputs = self(batch)
        forward_out = outputs["forward_out"]
        return {
            k: v
            for k, v in forward_out.items()
            if k in self.predict_keys
        }
