from typing import Any, Literal

import torch
from torch import Tensor

from integrator import configs
from integrator.model.distributions.total_fraction import (
    TotalFractionPosterior,
)
from integrator.model.integrators import BaseIntegrator
from integrator.model.integrators.base_integrator import (
    DEFAULT_PREDICT_KEYS_MODELC,
    _log_loss,
)

from .integrator_utils import (
    IntegratorBaseOutputs,
    IntegratorModelArgs,
    _assemble_outputs,
)


# %%
class IntegratorModelA(BaseIntegrator):
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

        # Reshape inputs for CNN
        b = shoebox.shape[0]
        shoebox_reshaped = shoebox.reshape(b, 1, *self.shoebox_shape)

        # Getting representations
        x_profile = self.encoders["profile"](shoebox_reshaped)
        x_intensity = self.encoders["intensity"](shoebox_reshaped)

        # Surrogate modules
        qbg = self.surrogates["qbg"](x_intensity)
        qi = self.surrogates["qi"](x_intensity)
        qp = self.surrogates["qp"](x_profile)

        # Monte Carlo samples
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qi.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        # Poisson rate
        rate = zI * zp + zbg

        # Getting outputs
        out = IntegratorBaseOutputs(
            rates=rate,
            counts=counts,
            mask=mask,
            qbg=qbg,
            qp=qp,
            qi=qi,
            zp=zp,
            zbg=zbg,
            concentration=qp.concentration,  # if using Dirichlet
            metadata=metadata,
            compute_pred_var=self.cfg.compute_pred_var,
        )
        out = _assemble_outputs(out)

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
        }


class IntegratorModelC(BaseIntegrator):
    """Integrator with a joint bivariate log-normal surrogate q(I, B).

    Replaces the independent q(I)q(B) mean-field factorization with a single
    BivariateLogNormalSurrogate that can represent the I-B anticorrelation
    induced by the Poisson likelihood.

    Surrogate keys expected in the config:
        ``qp``   – Dirichlet profile surrogate (unchanged).
        ``q_ib`` – BivariateLogNormalSurrogate for the joint (I, B) posterior.
    """

    REQUIRED_ENCODERS = {
        "profile": configs.ShoeboxEncoderArgs,
        "intensity": configs.IntensityEncoderArgs,
    }

    ARGS = IntegratorModelArgs

    def __init__(self, cfg, loss, encoders, surrogates):
        super().__init__(
            cfg=cfg, loss=loss, encoders=encoders, surrogates=surrogates
        )
        # Use the extended key list when the user hasn't set a custom predict_keys,
        # so that q_ib_loc, q_ib_scale_tril, and ib_log_space_correlation are saved.
        if cfg.predict_keys == "default":
            self.predict_keys = DEFAULT_PREDICT_KEYS_MODELC

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

        # Joint surrogate over (I, B)
        q_ib = self.surrogates["q_ib"](x_intensity)
        qp = self.surrogates["qp"](x_profile)

        # Draw joint samples: [S, B, 2]
        z_ib = q_ib.rsample([self.mc_samples])
        if isinstance(q_ib, TotalFractionPosterior):
            # (T, f) parameterization — convert to (I, bg)
            I_s, bg_s = q_ib.to_intensity_bg(z_ib)  # each [S, B]
            zI = I_s.unsqueeze(-1).permute(1, 0, 2)  # [B, S, 1]
            zbg = bg_s.unsqueeze(-1).permute(1, 0, 2)  # [B, S, 1]
        else:
            # (I, bg) parameterization — direct
            zI = z_ib[..., 0].unsqueeze(-1).permute(1, 0, 2)  # [B, S, 1]
            zbg = z_ib[..., 1].unsqueeze(-1).permute(1, 0, 2)  # [B, S, 1]
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)  # [B, S, P]

        rate = zI * zp + zbg  # [B, S, P]

        # Marginal LogNormal distributions for logging and _assemble_outputs.
        # These are the exact marginals of q_ib; they are not used for KL.
        qi_marginal = q_ib.marginal_i()
        qbg_marginal = q_ib.marginal_bg()

        out = IntegratorBaseOutputs(
            rates=rate,
            counts=counts,
            mask=mask,
            qbg=qbg_marginal,
            qp=qp,
            qi=qi_marginal,
            zp=zp,
            zbg=zbg,
            concentration=qp.concentration,
            metadata=metadata,
            compute_pred_var=self.cfg.compute_pred_var,
        )
        out = _assemble_outputs(out)

        # Store the 5 Cholesky parameters as flat [B] columns so they survive
        # the Polars DataFrame serialisation in BatchPredWriter.
        # Reconstruct with:
        #   loc = torch.stack([preds["q_ib_mu_I"], preds["q_ib_mu_B"]], dim=-1)
        #   L   = [[L11, 0], [L21, L22]]
        out["q_ib_mu_I"] = q_ib.loc[..., 0]  # [B]
        out["q_ib_mu_B"] = q_ib.loc[..., 1]  # [B]
        out["q_ib_L11"] = q_ib.scale_tril[..., 0, 0]  # [B]
        out["q_ib_L21"] = q_ib.scale_tril[..., 1, 0]  # [B]
        out["q_ib_L22"] = q_ib.scale_tril[..., 1, 1]  # [B]
        out["ib_log_space_correlation"] = q_ib.log_space_correlation  # [B]

        return {
            "forward_out": out,
            "qp": qp,
            "q_ib": q_ib,
            "qi": None,
            "qbg": None,
        }

    def _step(self, batch, step: Literal["train", "val"]):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]

        loss_dict = self.loss(
            rate=forward_out["rates"],
            counts=forward_out["counts"],
            qp=outputs["qp"],
            q_ib=outputs["q_ib"],
            mask=forward_out["mask"],
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


class IntegratorModelD(BaseIntegrator):
    """IntegratorModelB variant with fully decoupled encoders for qi and qbg.

    Uses five encoders instead of three so that the intensity and background
    surrogates receive independent latent representations, eliminating the
    gradient entanglement that biases qbg when qi and qbg share (x_k, x_r).

    Encoder keys (must match YAML order):
        profile  – shoebox encoder  → qp
        k_i      – intensity encoder → qi mean
        r_i      – intensity encoder → qi fano
        k_bg     – intensity encoder → qbg mean
        r_bg     – intensity encoder → qbg fano
    """

    REQUIRED_ENCODERS = {
        "profile": configs.ShoeboxEncoderArgs,
        "k_i": configs.IntensityEncoderArgs,
        "r_i": configs.IntensityEncoderArgs,
        "k_bg": configs.IntensityEncoderArgs,
        "r_bg": configs.IntensityEncoderArgs,
    }

    ARGS = IntegratorModelArgs

    def _forward_impl(
        self,
        counts: torch.Tensor,
        shoebox: torch.Tensor,
        mask: torch.Tensor,
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

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
        }


class IntegratorModelE(BaseIntegrator):
    """ModelD variant where the bg encoders see only border pixels.

    Like ModelD, each posterior parameter gets its own dedicated encoder,
    eliminating gradient entanglement. Unlike ModelD, the two background
    encoders (k_bg, r_bg) are BorderPixelMLPEncoders that operate only on
    the outer H×W ring across all D slices, making the inductive assumption
    explicit: background lives on the border.

    Encoder keys (must match YAML order):
        profile  – shoebox encoder  → qp
        k_i      – intensity encoder → qi shape
        r_i      – intensity encoder → qi rate
        k_bg     – border pixel MLP  → qbg shape
        r_bg     – border pixel MLP  → qbg rate
    """

    REQUIRED_ENCODERS = {
        "profile": configs.ShoeboxEncoderArgs,
        "k_i": configs.IntensityEncoderArgs,
        "r_i": configs.IntensityEncoderArgs,
        "k_bg": configs.BorderPixelMLPEncoderArgs,
        "r_bg": configs.BorderPixelMLPEncoderArgs,
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
        shoebox_reshaped = shoebox.reshape(
            b, 1, *self.shoebox_shape
        )  # (B, 1, D, H, W)

        x_profile = self.encoders["profile"](shoebox_reshaped)
        x_k_i = self.encoders["k_i"](shoebox_reshaped)
        x_r_i = self.encoders["r_i"](shoebox_reshaped)
        x_k_bg = self.encoders["k_bg"](shoebox_reshaped)
        x_r_bg = self.encoders["r_bg"](shoebox_reshaped)

        qp = self.surrogates["qp"](x_profile)
        qi = self.surrogates["qi"](x_k_i, x_r_i)
        qbg = self.surrogates["qbg"](x_k_bg, x_r_bg)

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

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
        }


class IntegratorModelF(BaseIntegrator):
    """ModelE variant where the bg encoders use summary statistics instead of raw pixels.

    Rather than learning over the full border pixel vector, each bg encoder
    computes 6 scalar statistics (mean, std, min, max, median, total) over the
    outer H×W ring, then projects them to encoder_out via a tiny MLP.  This
    eliminates spatial-pattern overfitting while retaining the signal that
    background estimation actually needs.

    Encoder keys (must match YAML order):
        profile  – shoebox encoder   → qp
        k_i      – intensity encoder → qi shape
        r_i      – intensity encoder → qi rate
        k_bg     – border stats MLP  → qbg shape
        r_bg     – border stats MLP  → qbg rate
    """

    REQUIRED_ENCODERS = {
        "profile": configs.ShoeboxEncoderArgs,
        "k_i": configs.IntensityEncoderArgs,
        "r_i": configs.IntensityEncoderArgs,
        "k_bg": configs.BorderStatsEncoderArgs,
        "r_bg": configs.BorderStatsEncoderArgs,
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

        qp = self.surrogates["qp"](x_profile)
        qi = self.surrogates["qi"](x_k_i, x_r_i)
        qbg = self.surrogates["qbg"](x_k_bg, x_r_bg)

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

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
        }


class IntegratorModelB(BaseIntegrator):
    REQUIRED_ENCODERS = {
        "profile": configs.ShoeboxEncoderArgs,
        "k": configs.IntensityEncoderArgs,
        "r": configs.IntensityEncoderArgs,
    }

    ARGS = IntegratorModelArgs

    def _forward_impl(
        self,
        counts: torch.Tensor,
        shoebox: torch.Tensor,
        mask: torch.Tensor,
        metadata: dict,
    ) -> dict[str, Any]:
        counts = torch.clamp(counts, min=0)

        # Reshape inputs for CNN
        b = shoebox.shape[0]
        shoebox_reshaped = shoebox.reshape(b, 1, *self.shoebox_shape)

        # Getting representations
        x_profile = self.encoders["profile"](shoebox_reshaped)
        x_k = self.encoders["k"](shoebox_reshaped)
        x_r = self.encoders["r"](shoebox_reshaped)

        # Surrogate modules
        qbg = self.surrogates["qbg"](x_k, x_r)
        qi = self.surrogates["qi"](x_k, x_r)
        qp = self.surrogates["qp"](x_profile)

        # Monte Carlo Samples
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qi.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        # Poisson rate
        rate = zI * zp + zbg

        # Getting outputs
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
            concentration=qp.concentration,
            compute_pred_var=self.cfg.compute_pred_var,
        )
        out = _assemble_outputs(out)

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
        }
