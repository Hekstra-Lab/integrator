import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from integrator import configs
from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)
from integrator.model.integrators.base_integrator import BaseIntegrator
from integrator.model.integrators.integrator_utils import (
    IntegratorBaseOutputs,
    _assemble_outputs,
)


def _sample_profile(qp, mc_samples: int) -> Tensor:
    if isinstance(qp, ProfileSurrogateOutput):
        return qp.zp.permute(1, 0, 2)
    return qp.rsample([mc_samples]).permute(1, 0, 2)


def _add_group_outputs(out: dict, metadata: dict, loss) -> None:
    group_labels = metadata["group_label"].long()
    out["group_label"] = group_labels
    if hasattr(loss, "log_alpha_per_group"):
        out["alpha_per_refl"] = F.softplus(
            loss.log_alpha_per_group[group_labels]
        )


class SVAEIntegrator(BaseIntegrator):
    """Per-observation SVAE: a learned responsibility gate over a conjugate Poisson-Gamma update."""

    REQUIRED_ENCODERS = {
        "profile": ("profile_encoder", configs.ProfileEncoderArgs),
        "signal": ("intensity_encoder", configs.IntensityEncoderArgs),
    }

    def __init__(self, cfg, loss, encoders, surrogates, optimizer=None):
        super().__init__(cfg, loss, encoders, surrogates, optimizer)
        self.gate_head = nn.Linear(
            self.cfg.encoder_out, math.prod(self.shoebox_shape)
        )

    def _intensity_prior(self, metadata, device):
        d = metadata["d"].to(device)  # (B,)
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))
        beta_I = self.loss._get_tau(
            metadata, s_sq, device
        )  #  Wilson Beta rate
        alpha_I = torch.ones_like(beta_I)  #
        return alpha_I, beta_I

    def _background_prior(self, metadata, device):
        conc = self.loss.bg_concentration  # () or (n_bins,)
        rate = self.loss.bg_rate
        if conc.ndim == 1:
            g = metadata["group_label"].to(device).long()  # (B,)
            conc, rate = conc[g], rate[g]  # (B,)
        return conc, rate

    def _get_pixel_probs(self, qp, x_signal):
        signal_prob = torch.sigmoid(self.gate_head(x_signal))  # g_p, (B, P)
        bg_prob = 1.0 - signal_prob
        prf = (
            qp.mean_profile
        )  # signal shape from the profile surrogate, (B, P)
        return signal_prob, bg_prob, prf

    def _get_qbg(self, bg_prob, raw_counts, metadata, mask):
        alpha_bg, beta_bg = self._background_prior(metadata, raw_counts.device)
        m = mask.squeeze(-1)
        alpha_ = alpha_bg + (bg_prob * raw_counts * m).sum(-1)
        beta_ = beta_bg + m.sum(-1)
        return torch.distributions.Gamma(alpha_, beta_)

    def _get_qi(self, signal_prob, raw_counts, metadata, mean_prf, mask):
        alpha_I, beta_I = self._intensity_prior(metadata, raw_counts.device)
        m = mask.squeeze(-1)
        alpha_ = alpha_I + (signal_prob * raw_counts * m).sum(-1)
        beta_ = beta_I + (mean_prf * m).sum(-1)
        return torch.distributions.Gamma(alpha_, beta_)

    def _forward_impl(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        metadata: dict,
    ) -> dict[str, Any]:

        # counts
        counts = torch.clamp(counts, min=0)

        # batch size
        b = shoebox.shape[0]
        shoebox_masked = shoebox * mask
        shoebox_reshaped = shoebox_masked.reshape(b, 1, *self.shoebox_shape)

        # profile representation
        x_profile = self.encoders["profile"](shoebox_reshaped)
        qp = self.surrogates["qp"](
            x_profile,
            mc_samples=self.mc_samples,
            metadata=metadata,
        )

        # get signal probs
        x_signal = self.encoders["signal"](shoebox_reshaped)
        i_prob, bg_prob, mean_prf = self._get_pixel_probs(qp, x_signal)

        # get I/bg posteriors
        qi = self._get_qi(i_prob, counts, metadata, mean_prf, mask)
        qbg = self._get_qbg(bg_prob, counts, metadata, mask)

        # samples
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = _sample_profile(qp, self.mc_samples)
        zI = qi.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        # pixel poisson rates
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
        _add_group_outputs(out, metadata, self.loss)

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
        }
