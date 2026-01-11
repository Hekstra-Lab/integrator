from typing import Any

import torch
from torch import Tensor

from integrator import configs
from integrator.model.integrators import BaseIntegrator

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
            concentration=qp.concentration,  # if using Dirichlet
            metadata=metadata,
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
            metadata=metadata,
            concentration=qp.concentration,
        )
        out = _assemble_outputs(out)

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
        }
