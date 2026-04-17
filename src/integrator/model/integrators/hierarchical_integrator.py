from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from integrator import configs
from integrator.model.integrators.base_integrator import (
    BaseIntegrator,
    _log_loss,
)
from integrator.model.integrators.integrator import (
    IntegratorModelA,
    IntegratorModelC,
    _sample_profile,
)
from integrator.model.integrators.integrator_utils import (
    IntegratorBaseOutputs,
    IntegratorModelArgs,
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

    # Auxiliary regularizers on the learned profile decoder (no-op for
    # fixed bases). ELBO logging above stays pure; penalty is added to
    # the backpropagated loss only.
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


class HierarchicalIntegratorA(IntegratorModelA):
    """ModelA variant with fixed per-group priors.

    Like ModelA, uses two encoders (profile + shared intensity).
    Metadata must contain `group_label` (integer tensor [B]).
    """

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


class HierarchicalIntegratorB(BaseIntegrator):
    """ModelB variant with fixed per-group intensity priors.

    Like ModelB, uses separate encoders for the two Gamma parameters (k, r).
    Metadata must contain `group_label` (integer tensor [B]).
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


class HierarchicalIntegratorC(IntegratorModelC):
    """ModelC variant with fixed per-group priors.

    Like ModelC, uses five fully decoupled encoders (profile, k_i, r_i, k_bg, r_bg).
    Metadata must contain `group_label` (integer tensor [B]).
    """

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


# Metadata features used by HierarchicalIntegratorCMeta. Order is fixed;
# changing it is a breaking change for any saved checkpoint of this class.
# log-d is used instead of raw d so the feature is roughly [0, 4] rather
# than [1, 80], making LayerNorm's job easier.
_META_FEATURE_KEYS: tuple[str, ...] = (
    "d",           # resolution — log-transformed below
    "s1.0",        # scattered wavevector components (unit-vector scale)
    "s1.1",
    "s1.2",
    "lp",          # Lorentz-polarization factor, [0, 1]
    "partiality",  # fraction of reflection captured, [0, 1]
    "zeta",        # reciprocal-space velocity through Ewald, [-1, 1]
    "entering",    # 0/1: is reflection entering Ewald sphere
)


def _extract_meta_features(
    metadata: dict, device: torch.device
) -> Tensor:
    """Extract the metadata feature tensor used by HierarchicalIntegratorCMeta.

    Returns a (B, 8) tensor with d log-transformed, other features raw.
    LayerNorm inside the metadata encoder handles the scale differences.
    Raises KeyError with a clear message if any field is missing.
    """
    missing = [k for k in _META_FEATURE_KEYS if k not in metadata]
    if missing:
        raise KeyError(
            f"HierarchicalIntegratorCMeta requires metadata keys "
            f"{list(_META_FEATURE_KEYS)}; missing: {missing}"
        )
    columns = []
    for k in _META_FEATURE_KEYS:
        v = metadata[k].float().to(device)
        if k == "d":
            # d is resolution in Angstroms, physically > 0. Don't clamp —
            # a hard floor would silently compress valid values and could
            # mask a data bug. Let log(d <= 0) produce NaN loudly instead.
            v = torch.log(v)
        columns.append(v)
    return torch.stack(columns, dim=-1)  # (B, 8)


class HierarchicalIntegratorCMeta(BaseIntegrator):
    """HierarchicalC + auxiliary metadata encoder.

    Same 5 shoebox encoders as HierarchicalIntegratorC plus a dedicated
    metadata encoder that processes per-reflection geometry features
    (resolution, scattered wavevector, partiality, Lorentz-polarization,
    zeta, entering). Its output is added to each of the four intensity
    encoders' outputs — the profile encoder is left untouched (profile
    shape is a local property of the pixels).

    Why additive fusion: keeps surrogate `in_features` unchanged, so
    existing qi/qbg configs are drop-in compatible. The metadata encoder
    learns to embed the 8 raw features into the same feature space that
    intensity features live in, and the surrogate's linear head handles
    weighting.

    Required encoders:
      profile, k_i, r_i, k_bg, r_bg, metadata
    """

    REQUIRED_ENCODERS = {
        "profile": configs.ShoeboxEncoderArgs,
        "k_i": configs.IntensityEncoderArgs,
        "r_i": configs.IntensityEncoderArgs,
        "k_bg": configs.IntensityEncoderArgs,
        "r_bg": configs.IntensityEncoderArgs,
        "metadata": configs.MetadataEncoderArgs,
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
        x_k_i = self.encoders["k_i"](shoebox_reshaped)
        x_r_i = self.encoders["r_i"](shoebox_reshaped)
        x_k_bg = self.encoders["k_bg"](shoebox_reshaped)
        x_r_bg = self.encoders["r_bg"](shoebox_reshaped)

        # Metadata branch: embed per-reflection geometry features, then fold
        # additively into each intensity encoder's output.
        meta_features = _extract_meta_features(metadata, shoebox.device)
        x_meta = self.encoders["metadata"](meta_features)
        x_k_i = x_k_i + x_meta
        x_r_i = x_r_i + x_meta
        x_k_bg = x_k_bg + x_meta
        x_r_bg = x_r_bg + x_meta

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
