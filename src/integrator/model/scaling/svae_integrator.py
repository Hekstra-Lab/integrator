"""Per-observation learned-potential (Structured VAE) intensity.

The amortized sibling of `ConjugateIntegrator`. Instead of deriving the
per-pixel signal responsibilities `pi_p` by a CAVI fixed-point iteration, a head
emits a learned per-pixel signal attribution `g_p in (0, 1)` and the conjugate
potentials are summed over pixels:

    dalpha_p = g_p * c_p                 (learned expected signal photons)
    alpha_i  = alpha_W + sum_p dalpha_p
    beta_i   = tau     + sum_p prof_p    (analytic exposure)
    q(I_i)   = Gamma(alpha_i, beta_i)

`g_p * c_p` is exactly the augmented Poisson-Gamma sufficient statistic (the
expected number of signal photons at pixel p; Cemgil 2009), so `g_p -> pi_p`
recovers `ConjugateIntegrator` exactly -- a strict generalization, never worse
in principle (Johnson et al., Structured VAE, NeurIPS 2016). Bounding `g_p` to
(0, 1) caps `alpha_i <= alpha_W + sum_p c_p` (cannot attribute more signal than
photons observed) and lets the head down-weight outlier / hot pixels -- the
per-pixel analogue of the merge model's attention trust gate -- which the rigid
CAVI form cannot. A single feed-forward pass: no inner EM, no implicit-gradient
correction.

The reconstruction, the loss (`KL_i = Gamma-Gamma KL(qi || Wilson)`), the
encoders, and the KL terms are identical to `ConjugateIntegrator`; only the
`alpha` mechanism differs. That keeps it a fair comparison against the derived
conjugate model and the fully-neural gamma baseline.
"""

from math import prod
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Gamma

from integrator import configs
from integrator.model.integrators.base_integrator import (
    BaseIntegrator,
    _log_loss,
)
from integrator.model.integrators.hierarchical_integrator import (
    _add_group_outputs,
    _sample_profile,
)
from integrator.model.integrators.integrator_utils import (
    IntegratorBaseOutputs,
    _assemble_outputs,
)

# Initial bias of the responsibility head. 0.0 -> g = 0.5 uniform at init (a
# neutral cold start: half the photons attributed to signal). Lower it (e.g.
# -2.0 -> g ~ 0.12) if a very bright dataset starts alpha too high for the
# Wilson KL to pull back.
_RESP_INIT_BIAS = 0.0


class SVAEIntegrator(BaseIntegrator):
    """Per-observation learned-potential (Structured VAE) intensity.

    See the module docstring. The learned sibling of `ConjugateIntegrator`:
    same encoders, reconstruction, and loss; the per-pixel CAVI responsibility
    is replaced by a learned attribution gate `g_p`, summed in one pass.
    """

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

        self.alpha_W = float(cfg.wilson_alpha)
        self.sample_I = bool(cfg.sample_I_h)

        # Loss-consistency guards (identical to ConjugateIntegrator): the loss's
        # KL(qi || Wilson) is the closed-form Gamma-Gamma KL of our q(I) only
        # when the prior shape is alpha_W = 1 and pi_weight = 1, with no
        # learned/continuous concentration overriding the Wilson shape.
        pi_weight = float(getattr(self.loss, "pi_weight", 1.0))
        if abs(pi_weight - 1.0) > 1e-8:
            raise ValueError(
                "SVAEIntegrator requires loss pi_weight == 1.0 so the loss's "
                "KL(qi || Wilson) is exactly the per-obs Gamma-Gamma KL of the "
                f"learned-potential posterior; got {pi_weight}."
            )
        if abs(self.alpha_W - 1.0) > 1e-8:
            raise ValueError(
                "SVAEIntegrator requires wilson_alpha == 1.0: the Wilson loss "
                "hard-codes the intensity prior shape to alpha = 1, so the "
                f"potential base must match it; got alpha_W={self.alpha_W}."
            )
        if getattr(self.loss, "concentration_fn", None) is not None or getattr(
            self.loss, "learn_concentration", False
        ):
            raise ValueError(
                "SVAEIntegrator requires the default fixed Wilson prior shape "
                "(alpha = 1). A learned/continuous concentration prior makes "
                "the loss KL_i inconsistent with the potential update."
            )

        # Responsibility head: pooled profile feature -> per-pixel signal
        # attribution logit. Zero-init -> g = sigmoid(bias) uniform at init, so
        # the head starts neutral and learns the spatial responsibility from the
        # reconstruction gradient (the amortized analogue of the CAVI fixed
        # point). Shares the profile encoder: both answer "where is the signal".
        self.n_pixels = int(prod(self.shoebox_shape))
        self.resp_head = nn.Linear(cfg.encoder_out, self.n_pixels)
        nn.init.zeros_(self.resp_head.weight)
        nn.init.constant_(self.resp_head.bias, _RESP_INIT_BIAS)

    def _wilson_tau(self, metadata: dict, device: torch.device) -> Tensor:
        """Wilson prior rate tau.

        Passes the *full* metadata to the loss so that with lp_correction=True
        tau is multiplied by the per-observation LP factor -- i.e. the Wilson
        prior is transformed onto the (scale-free) *observed* intensity scale
        this integrator infers (rate = I*prf + bg, so I is the observed
        intensity and the prior must be LP-corrected). Uses the same path as the
        loss's KL_i, so the potential base and the KL prior stay consistent.
        """
        d = metadata["d"].to(device).float()
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))
        return self.loss._get_tau(metadata, s_sq, device)

    def _forward_impl(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        metadata: dict,
    ) -> dict[str, Any]:
        counts = counts.clamp(min=0)
        B = shoebox.shape[0]
        device = shoebox.device
        shoebox_masked = shoebox * mask
        shoebox_reshaped = shoebox_masked.reshape(B, 1, *self.shoebox_shape)

        # Encoders: profile + bg only (no intensity encoder, no qi surrogate).
        x_profile = self.encoders["profile"](shoebox_reshaped)
        x_k_bg = self.encoders["k_bg"](shoebox_reshaped)
        x_r_bg = self.encoders["r_bg"](shoebox_reshaped)

        qbg = self.surrogates["qbg"](x_k_bg, x_r_bg)
        qp = self.surrogates["qp"](
            x_profile,
            mc_samples=self.mc_samples,
        )

        profile_mean = qp.mean_profile  # (B, P)
        bg_mean = qbg.mean  # (B,)

        tau = self._wilson_tau(metadata, device)

        # Learned per-pixel signal attribution -> summed conjugate potentials.
        g = torch.sigmoid(self.resp_head(x_profile))  # (B, P) in (0, 1)
        delta_alpha = (g * counts * mask).sum(dim=-1)  # expected signal photons
        alpha = self.alpha_W + delta_alpha
        beta = tau + (profile_mean * mask).sum(dim=-1)  # analytic exposure

        qi = Gamma(alpha.clamp(min=1e-6), beta.clamp(min=1e-12))

        # Reconstruction rate
        if self.sample_I:
            zI = qi.rsample([self.mc_samples])  # (S, B)
            zI = zI.clamp(min=1e-10)
        else:
            zI = (alpha / beta).unsqueeze(0).expand(self.mc_samples, B)

        zp = _sample_profile(qp, self.mc_samples)
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        zI = (zI).unsqueeze(-1).permute(1, 0, 2)

        rate = zI * zp + zbg

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
        if "asu_id" in metadata:
            out["asu_id"] = metadata["asu_id"].long().to(device)
        if "group_label" in metadata:
            _add_group_outputs(out, metadata, self.loss)

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
            "alpha": alpha,
            "beta": beta,
            "tau": tau,
            "g_mean": g.mean().detach(),
            "bg_mean": bg_mean.mean().detach(),
            "profile_max_mean": profile_mean.max(dim=-1)
            .values.mean()
            .detach(),
        }

    def _step(self, batch, step: Literal["train", "val"]):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]

        group_labels = metadata["group_label"].long()

        # -ELBO = Poisson NLL + KL_prf + KL_i + KL_bg. KL_i is the closed-form
        # Gamma-Gamma KL between q(I) = Gamma(alpha, beta) and the Wilson prior.
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
                "kl_prf": loss_dict["kl_prf_mean"],
                "kl_i": loss_dict["kl_i_mean"],
                "kl_bg": loss_dict["kl_bg_mean"],
            },
        )

        penalty, penalty_components = self._profile_basis_penalty()
        for name, value in penalty_components.items():
            self.log(f"{step} {name}", value, on_step=False, on_epoch=True)
        total_loss = total_loss + penalty

        with torch.no_grad():
            alpha = outputs["alpha"]
            beta = outputs["beta"]
            I_mean = alpha / beta
            I_var = alpha / beta.pow(2)
            self.log(
                f"{step} qi_mean", I_mean.mean(), on_step=False, on_epoch=True
            )
            self.log(
                f"{step} qi_var", I_var.mean(), on_step=False, on_epoch=True
            )
            self.log(
                f"{step} alpha_mean",
                alpha.mean(),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step} beta_mean", beta.mean(), on_step=False, on_epoch=True
            )
            self.log(
                f"{step} g_mean",
                outputs["g_mean"],
                on_step=False,
                on_epoch=True,
            )
            for k in (
                "bg_mean",
                "profile_max_mean",
            ):
                self.log(
                    f"{step} {k}", outputs[k], on_step=False, on_epoch=True
                )

        # End-of-epoch model-vs-DIALS intensity/background scatters (no-op unless
        # the log_*_scatter flags are set). No scale here, so the intensity
        # scatter compares qi.mean directly to DIALS intensity.sum.value.
        if step == "train":
            self._collect_scatters(outputs, metadata, mask, counts)

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
