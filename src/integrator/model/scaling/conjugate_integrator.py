"""Per-observation conjugate Bayesian integration with Wilson prior.

Generative model. For shoebox i (HKL h(i)), pixel p:

    counts_{i,p} ~ Poisson(rate_{i,p})
    rate_{i,p}   = s_i * I_i * profile_{i,p} + bg_i
    I_i          ~ Gamma(alpha_W, tau_{h(i)})       (Wilson prior)

For each shoebox in isolation, with Poisson likelihood that is linear in
`I_i`, the conditional posterior on `I_i` (given profile, bg, scale)
is Gamma in closed form via Poisson-thinning data augmentation:

    pi_{i,p} = s_i * I_i_hat * profile_{i,p} /
               (s_i * I_i_hat * profile_{i,p} + bg_i)

    alpha_i = alpha_W + sum_p  pi_{i,p} * c_{i,p}      * mask_{i,p}
    beta_i  = tau_h   + sum_p  s_i      * profile_{i,p} * mask_{i,p}
    q(I_i)  = Gamma(alpha_i, beta_i)

`I_i_hat` for the E-step is initialized to the Wilson prior mean
`1 / tau_h` and refined for `n_em_iters` (default 3) inner EM steps —
fast convergence in practice since rate is linear in I and each step
is elementwise per pixel.

What the neural net does:
    - Encoders predict q(profile_i), q(bg_i) per shoebox.
    - Scale s_i from physics (LP + optional learnable correction).
    - I_i is *derived* per shoebox — no encoder for I, no embedding.

Unlike the merging variant, each observation is processed
independently — no per-HKL aggregation, no EMA buffer, no grouped
sampler required.
"""

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
    _get_normalized_position,
    _sample_profile,
)
from integrator.model.integrators.integrator_utils import (
    IntegratorBaseOutputs,
    _assemble_outputs,
)
from integrator.model.scaling.chebyshev_scale import (
    ChebyshevScale,
    MLPScale,
    SpatialChebyshevScale,
)


class ConjugateIntegrator(BaseIntegrator):
    """Per-observation conjugate intensity. DIALS-style Bayesian integration."""

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

        self.alpha_W = float(getattr(cfg, "wilson_alpha", 1.0))
        self.n_em_iters = int(getattr(cfg, "n_em_iters", 3))
        self.sample_I = bool(getattr(cfg, "sample_I_h", True))

        # Scale function (optional). With scale_mlp=false and scale_spatial=false,
        # the default ChebyshevScale with degree=0 gives a learnable constant
        # divided by lp (i.e. just LP correction).
        if cfg.scale_mlp:
            self.scale_fn = MLPScale(
                hidden_dim=cfg.scale_mlp_hidden,
                n_layers=cfg.scale_mlp_layers,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
                beam_center=cfg.scale_beam_center,
                r_max=cfg.scale_r_max,
                d_min=getattr(cfg, "dmin", 1.0),
                d_max=60.0,
            )
        elif cfg.scale_spatial:
            self.scale_fn = SpatialChebyshevScale(
                degree_frame=cfg.scale_degree,
                degree_radius=cfg.scale_degree_radius,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
                beam_center=cfg.scale_beam_center,
                r_min=cfg.scale_r_min,
                r_max=cfg.scale_r_max,
            )
        else:
            self.scale_fn = ChebyshevScale(
                degree=cfg.scale_degree,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
            )

    # ------------------------------------------------------------------

    def _get_scale(self, metadata: dict, device: torch.device) -> Tensor:
        frame = metadata["xyzcal.px.2"].to(device).float()
        lp = metadata["lp"].to(device).float().clamp(min=1e-8)
        if isinstance(self.scale_fn, MLPScale):
            x_det = metadata["xyzcal.px.0"].to(device).float()
            y_det = metadata["xyzcal.px.1"].to(device).float()
            d = metadata["d"].to(device).float()
            return self.scale_fn(frame, x_det, y_det, lp, d)
        elif isinstance(self.scale_fn, SpatialChebyshevScale):
            x_det = metadata["xyzcal.px.0"].to(device).float()
            y_det = metadata["xyzcal.px.1"].to(device).float()
            return self.scale_fn(frame, x_det, y_det) / lp
        else:
            return self.scale_fn(frame) / lp

    def _wilson_tau(self, d: Tensor) -> Tensor:
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))
        return self.loss._get_tau({"d": d}, s_sq, d.device)

    def _conjugate_em(
        self,
        counts: Tensor,
        profile_mean: Tensor,
        bg_mean: Tensor,
        scale: Tensor,
        tau: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run n_em_iters EM iterations. Returns (alpha, beta, pi_last).

        Gradient flows through all iterations (BPTT through the fixed-point
        update). With 3-5 iterations and elementwise pixel ops, cost is
        negligible.
        """
        # Cold start: Wilson prior mean
        I_hat = (1.0 / tau.clamp(min=1e-12)).detach()  # (B,)

        # beta_term is constant across iterations (depends only on s, profile, mask)
        beta_term = (scale.unsqueeze(-1) * profile_mean * mask).sum(dim=-1)
        beta = tau + beta_term  # (B,)

        pi = None  # will be set in loop
        alpha = None
        for it in range(self.n_em_iters):
            signal_rate = (
                scale.unsqueeze(-1) * I_hat.unsqueeze(-1) * profile_mean
            )
            total_rate = signal_rate + bg_mean.unsqueeze(-1)
            pi = signal_rate / total_rate.clamp(min=1e-12)
            alpha_signal = (pi * counts * mask).sum(dim=-1)
            alpha = self.alpha_W + alpha_signal
            if it < self.n_em_iters - 1:
                # Refine I_hat for next iteration (no need to keep this graph)
                I_hat = (alpha / beta).detach()
            # On the last iteration, alpha/beta keep their full gradient path

        assert alpha is not None and pi is not None
        return alpha, beta, pi

    # ------------------------------------------------------------------

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

        # Encoders: profile + bg only
        position = _get_normalized_position(metadata, device)
        x_profile = self.encoders["profile"](
            shoebox_reshaped, position=position
        )
        x_k_bg = self.encoders["k_bg"](shoebox_reshaped)
        x_r_bg = self.encoders["r_bg"](shoebox_reshaped)

        qbg = self.surrogates["qbg"](x_k_bg, x_r_bg)
        prf_labels = metadata.get(
            "profile_group_label", metadata.get("group_label")
        )
        prf_labels = prf_labels.long() if prf_labels is not None else None
        qp = self.surrogates["qp"](
            x_profile,
            mc_samples=self.mc_samples,
            group_labels=prf_labels,
            metadata=metadata,
        )

        profile_mean = qp.mean_profile  # (B, P)
        bg_mean = qbg.mean              # (B,)

        scale = self._get_scale(metadata, device)  # (B,)

        d_per_obs = metadata["d"].to(device).float()
        tau = self._wilson_tau(d_per_obs)

        alpha, beta, pi = self._conjugate_em(
            counts, profile_mean, bg_mean, scale, tau, mask
        )

        qi = Gamma(alpha.clamp(min=1e-6), beta.clamp(min=1e-12))

        # Reconstruction rate
        if self.sample_I:
            zI = qi.rsample([self.mc_samples])  # (S, B)
            zI = zI.clamp(min=1e-10)
        else:
            zI = (alpha / beta).unsqueeze(0).expand(self.mc_samples, B)

        zp = _sample_profile(qp, self.mc_samples)
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        zI_scaled = (scale.unsqueeze(0) * zI).unsqueeze(-1).permute(
            1, 0, 2
        )  # (B, S, 1)
        rate = zI_scaled * zp + zbg

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
            "pi_mean": pi.mean().detach(),
        }

    # ------------------------------------------------------------------

    def _step(self, batch, step: Literal["train", "val"]):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]

        group_labels = metadata["group_label"].long()

        # Standard loss: Poisson NLL + KL_prf + KL_i + KL_bg.
        # KL_i is the closed-form Gamma-Gamma KL between our conjugate
        # posterior and the Wilson prior — exactly what we want, no overrides.
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
            self.log(f"{step} qi_mean", I_mean.mean(), on_step=False, on_epoch=True)
            self.log(f"{step} qi_var", I_var.mean(), on_step=False, on_epoch=True)
            self.log(f"{step} alpha_mean", alpha.mean(), on_step=False, on_epoch=True)
            self.log(f"{step} beta_mean", beta.mean(), on_step=False, on_epoch=True)
            self.log(f"{step} pi_mean", outputs["pi_mean"], on_step=False, on_epoch=True)

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
