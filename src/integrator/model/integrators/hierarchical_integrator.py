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


def _tensor_stats(name: str, t: Tensor) -> str:
    """One-line summary: name | min max mean nan% inf%."""
    with torch.no_grad():
        flat = t.detach().float().flatten()
        nan_pct = flat.isnan().float().mean().item() * 100
        inf_pct = flat.isinf().float().mean().item() * 100
        finite = flat[flat.isfinite()]
        if finite.numel() == 0:
            return f"  {name:20s} | ALL non-finite  nan={nan_pct:.1f}%  inf={inf_pct:.1f}%"
        return (
            f"  {name:20s} | min={finite.min().item():+.4e}  max={finite.max().item():+.4e}"
            f"  mean={finite.mean().item():+.4e}  nan={nan_pct:.1f}%  inf={inf_pct:.1f}%"
        )


def _gamma_stats(name: str, dist) -> str:
    """Stats for a Gamma distribution's concentration and rate."""
    lines = [
        _tensor_stats(f"{name}.concentration", dist.concentration),
        _tensor_stats(f"{name}.rate", dist.rate),
    ]
    return "\n".join(lines)


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

    # === DEBUG: print loss components every step for first 200 steps ===
    if step == "train" and self.global_step < 200:
        parts = []
        parts.append(f"[DEBUG step={self.global_step}]")
        parts.append(f"  loss={total_loss.item():.4e}")
        for k in ("neg_ll_mean", "kl_mean", "kl_prf_mean", "kl_i_mean", "kl_bg_mean", "kl_hyper"):
            if k in loss_dict:
                v = loss_dict[k]
                parts.append(f"  {k}={v.item():.4e}")
        parts.append(_gamma_stats("qi", outputs["qi"]))
        parts.append(_gamma_stats("qbg", outputs["qbg"]))
        parts.append(_tensor_stats("rate", forward_out["rates"]))
        parts.append(_tensor_stats("counts", forward_out["counts"]))
        # Check surrogate linear layer weights for NaN
        for sname, surr in self.surrogates.items():
            for pname, p in surr.named_parameters():
                if p.grad is not None:
                    parts.append(_tensor_stats(f"{sname}.{pname}.grad", p.grad))
                if p.data.isnan().any():
                    parts.append(f"  *** NaN in {sname}.{pname} weights! ***")
        print("\n".join(parts), flush=True)
    # === END DEBUG ===

    # Log intensity surrogate stats
    qi = outputs["qi"]
    qi_mean = qi.mean
    self.log(f"{step} qi_mean_min", qi_mean.min(), on_step=False, on_epoch=True)
    self.log(f"{step} qi_mean_median", qi_mean.median(), on_step=False, on_epoch=True)

    # Gamma params — handle both regular Gamma and ZeroInflatedGamma
    qi_gamma = qi.gamma if hasattr(qi, "gamma") else qi
    self.log(f"{step} qi_k_min", qi_gamma.concentration.min(), on_step=False, on_epoch=True)
    self.log(f"{step} qi_k_median", qi_gamma.concentration.median(), on_step=False, on_epoch=True)
    self.log(f"{step} qi_r_min", qi_gamma.rate.min(), on_step=False, on_epoch=True)
    self.log(f"{step} qi_r_median", qi_gamma.rate.median(), on_step=False, on_epoch=True)

    if hasattr(qi, "pi"):
        self.log(f"{step} qi_pi_min", qi.pi.min(), on_step=False, on_epoch=True)
        self.log(f"{step} qi_pi_median", qi.pi.median(), on_step=False, on_epoch=True)
        self.log(f"{step} qi_pi_mean", qi.pi.mean(), on_step=False, on_epoch=True)

    qbg = outputs["qbg"]
    self.log(f"{step} qbg_mean_min", qbg.mean.min(), on_step=False, on_epoch=True)
    self.log(f"{step} qbg_mean_median", qbg.mean.median(), on_step=False, on_epoch=True)

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
    """Hierarchical integrator with 3 encoders: profile, k, r.

    Matches old IntegratorModelB / HierarchicalIntegratorB architecture:
    the k encoder feeds linear_mu of both qi and qbg, the r encoder
    feeds linear_fano of both qi and qbg.
    """

    from integrator import configs

    REQUIRED_ENCODERS = {
        "profile": configs.ShoeboxEncoderArgs,
        "k": configs.IntensityEncoderArgs,
        "r": configs.IntensityEncoderArgs,
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
