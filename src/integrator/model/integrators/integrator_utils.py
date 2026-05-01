from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from integrator.configs.integrator import IntegratorCfg
from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)


@dataclass
class IntegratorBaseOutputs:
    rates: Tensor
    counts: Tensor
    mask: Tensor
    qbg: Any
    qp: Any
    qi: Any
    zp: Tensor
    zbg: Tensor
    metadata: dict[str, torch.Tensor]


@dataclass
class IntegratorModelArgs:
    cfg: IntegratorCfg
    loss: nn.Module
    surrogates: dict[str, nn.Module]
    encoders: dict[str, nn.Module]


def _assemble_outputs(
    out: IntegratorBaseOutputs,
) -> dict[str, Any]:
    is_profile_output = isinstance(out.qp, ProfileSurrogateOutput)

    qp_mean = out.qp.mean_profile if is_profile_output else out.qp.mean

    base = {
        "rates": out.rates,
        "counts": out.counts,
        "mask": out.mask,
        "zp": out.zp,
        "qbg_mean": out.qbg.mean,
        "qbg_var": out.qbg.variance,
        "qp_mean": qp_mean,
        "qi_mean": out.qi.mean,
        "qi_var": out.qi.variance,
        "profile": qp_mean,
    }

    if hasattr(out.qi, "pi"):
        base["qi_pi"] = out.qi.pi
        base["qi_gamma_mean"] = out.qi.gamma.mean

    if is_profile_output:
        base["qp_mu_h"] = out.qp.mu_h
        base["qp_std_h"] = out.qp.std_h

    if out.metadata is None:
        return base

    # Storing the surrogate distribution parameters
    def _dist_params(dist):
        if hasattr(dist, "arg_constraints"):
            return {name: getattr(dist, name) for name in dist.arg_constraints}
        if hasattr(dist, "gamma"):
            return {
                "concentration": dist.gamma.concentration,
                "rate": dist.gamma.rate,
                "pi": dist.pi,
            }
        return {}

    distribution_params = {
        "qbg_params": _dist_params(out.qbg),
        "qi_params": _dist_params(out.qi),
    }

    # Update base dictionary
    base.update(out.metadata)
    base.update(distribution_params)

    return base
