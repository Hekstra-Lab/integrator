from typing import Any

import torch
from torch import Tensor

from integrator import configs
from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)
from integrator.model.integrators import BaseIntegrator

from .integrator_utils import (
    IntegratorBaseOutputs,
    IntegratorModelArgs,
    _assemble_outputs,
)


def _sample_profile(qp, mc_samples: int) -> Tensor:
    """Sample profiles, handling both ProfileSurrogateOutput and Distribution."""
    if isinstance(qp, ProfileSurrogateOutput):
        return qp.zp.permute(1, 0, 2)  # (B, S, K)
    return qp.rsample([mc_samples]).permute(1, 0, 2)


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

        b = shoebox.shape[0]
        shoebox_reshaped = shoebox.reshape(b, 1, *self.shoebox_shape)

        x_profile = self.encoders["profile"](shoebox_reshaped)
        x_intensity = self.encoders["intensity"](shoebox_reshaped)

        qbg = self.surrogates["qbg"](x_intensity)
        qi = self.surrogates["qi"](x_intensity)
        qp = self.surrogates["qp"](x_profile, mc_samples=self.mc_samples)

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

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
        }


# %%
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

        b = shoebox.shape[0]
        shoebox_reshaped = shoebox.reshape(b, 1, *self.shoebox_shape)

        x_profile = self.encoders["profile"](shoebox_reshaped)
        x_k = self.encoders["k"](shoebox_reshaped)
        x_r = self.encoders["r"](shoebox_reshaped)

        qbg = self.surrogates["qbg"](x_k, x_r)
        qi = self.surrogates["qi"](x_k, x_r)
        qp = self.surrogates["qp"](x_profile, mc_samples=self.mc_samples)

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

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
        }


# %%
class IntegratorModelC(BaseIntegrator):
    """Integrator variant with fully decoupled encoders for qi and qbg.

    Uses five encoders instead of three so that the intensity and background
    surrogate parameters receive independent latent representations.

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
        qp = self.surrogates["qp"](x_profile, mc_samples=self.mc_samples)

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

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
        }
